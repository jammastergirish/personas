#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "torch>=2.2",
#   "transformers>=4.43",
#   "matplotlib>=3.8",
#   "pandas>=2.2",
#   "python-dotenv>=1.0",
#   "accelerate",
#   "scikit-learn>=1.4",
#   "numpy>=1.26",
#   "wandb>=0.16",
#   "tqdm>=4.66",
# ]
# ///

"""
OQ3: Natural Sub-Manifold — Where do coherent trait combinations live?

This experiment maps where coherent trait combinations live in trait space by:
1. Computing global trait vectors at the best layer.
2. Sampling ~200 points in trait space by combining trait vectors with varying
   coefficients (known personas, interpolations, random).
3. Applying multi-trait steering and generating text for each sample.
4. Evaluating coherence via:
   (a) Perplexity under the base model.
   (b) Self-consistency: cosine similarity to nearest known persona centroid.
   (c) Mean probe confidence (max softmax probability from persona probe).
5. Mapping coherence scores onto trait-space PCA.
6. SVD of hidden states for all sample points reveals effective dimensionality
   of the occupied manifold.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ---- path setup so we can import from the project root and shared/ ----
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (main.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/ (shared/)

from dotenv import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from main import (
    QUESTIONS,
    Example,
    build_examples,
    collect_hidden_vectors,
    infer_device,
    sample_layers,
    set_seed,
)
from shared.trait_config import (
    PERSONAS,
    PERSONA_PROMPTS,
    PERSONA_TRAIT_MATRIX,
    TRAITS,
    TRAIT_PROMPTS,
    get_persona_trait_matrix_tensor,
    get_trait_prompt,
)
from shared.utils import (
    effective_rank,
    finish_wandb,
    get_num_layers,
    init_wandb,
    load_model_and_tokenizer,
    plot_svd_spectrum,
    save_run_config,
    svd_analysis,
)
from steer import SteeringHook, get_layer_module


# ============================================================
# Phase 1: Compute global trait vectors (same as OQ2)
# ============================================================


def build_trait_examples(
    tokenizer,
    questions: List[str],
    personas: List[str],
    max_length: int = 512,
) -> Tuple[List[Example], List[str], List[str], List[str]]:
    """
    Build examples for every (persona, trait, level) combination.

    Returns
    -------
    examples : list of Example
    trait_labels : per-example trait name
    level_labels : per-example level ("high" / "low")
    persona_labels : per-example persona name
    """
    examples: List[Example] = []
    trait_labels: List[str] = []
    level_labels: List[str] = []
    persona_labels: List[str] = []

    for persona in personas:
        for trait in TRAITS:
            for level in ("high", "low"):
                system_prompt = get_trait_prompt(persona, trait, level)
                for question in questions:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ]
                    prompt_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                    )
                    encoded = tokenizer(
                        prompt_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                    )
                    examples.append(
                        Example(
                            persona_name=persona,
                            persona_text=system_prompt,
                            question=question,
                            prompt_text=prompt_text,
                            input_ids=encoded["input_ids"],
                            attention_mask=encoded["attention_mask"],
                        )
                    )
                    trait_labels.append(trait)
                    level_labels.append(level)
                    persona_labels.append(persona)

    return examples, trait_labels, level_labels, persona_labels


def compute_global_trait_vectors_from_hidden(
    by_layer: Dict[int, List[torch.Tensor]],
    trait_labels: List[str],
    level_labels: List[str],
    layer: int,
) -> Dict[str, torch.Tensor]:
    """
    Compute one trait vector per trait at a given layer:
        trait_vec[t] = mean(high examples for t) - mean(low examples for t)

    Returns dict mapping trait name -> vector of shape [d].
    """
    vecs = torch.stack(by_layer[layer], dim=0)  # [N, d]
    trait_vectors: Dict[str, torch.Tensor] = {}

    for trait in TRAITS:
        high_mask = torch.tensor(
            [t == trait and l == "high" for t, l in zip(trait_labels, level_labels)]
        )
        low_mask = torch.tensor(
            [t == trait and l == "low" for t, l in zip(trait_labels, level_labels)]
        )
        high_mean = vecs[high_mask].mean(dim=0)
        low_mean = vecs[low_mask].mean(dim=0)
        trait_vectors[trait] = high_mean - low_mean

    return trait_vectors


def select_best_layer(
    by_layer: Dict[int, List[torch.Tensor]],
    trait_labels: List[str],
    level_labels: List[str],
) -> int:
    """
    Pick the layer whose trait vectors have the highest mean norm
    (strongest trait signal).
    """
    best_layer = -1
    best_norm = -1.0
    for layer in sorted(by_layer.keys()):
        tvecs = compute_global_trait_vectors_from_hidden(
            by_layer, trait_labels, level_labels, layer,
        )
        mean_norm = torch.stack(list(tvecs.values())).norm(dim=-1).mean().item()
        if mean_norm > best_norm:
            best_norm = mean_norm
            best_layer = layer
    print(f"  Best layer by trait vector norm: {best_layer} (mean norm = {best_norm:.4f})")
    return best_layer


# ============================================================
# Phase 2: Sample points in trait space
# ============================================================


def sample_trait_points(
    n_samples: int,
    n_traits: int,
    persona_trait_matrix: torch.Tensor,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Generate n_samples points in trait space.

    Returns
    -------
    points : [n_samples, n_traits] trait-space coordinates
    labels : per-point label ("known", "interpolation", or "random")
    """
    points: List[torch.Tensor] = []
    labels: List[str] = []

    # (a) Known persona points from the ground truth matrix
    for i in range(len(persona_trait_matrix)):
        points.append(persona_trait_matrix[i])
        labels.append("known")

    # (b) Random interpolations between pairs of personas
    n_interp = n_samples // 3
    for _ in range(n_interp):
        i, j = random.sample(range(len(persona_trait_matrix)), 2)
        t = random.random()
        points.append(t * persona_trait_matrix[i] + (1 - t) * persona_trait_matrix[j])
        labels.append("interpolation")

    # (c) Random points in [-1, 1]^n_traits
    remaining = n_samples - len(points)
    for _ in range(remaining):
        points.append(torch.rand(n_traits) * 2 - 1)
        labels.append("random")

    points = points[:n_samples]
    labels = labels[:n_samples]

    return torch.stack(points), labels


# ============================================================
# Phase 3: Multi-trait steering and generation
# ============================================================


def build_combined_steering_vector(
    alphas: torch.Tensor,
    global_trait_vecs: Dict[str, torch.Tensor],
    traits: List[str],
) -> torch.Tensor:
    """
    Build a combined steering vector from trait-space coefficients.

    alphas : [n_traits] coefficients for each trait
    global_trait_vecs : {trait: [d]} normalized trait vectors
    traits : ordered list of trait names

    Returns: [d] combined steering vector
    """
    combined = torch.zeros_like(global_trait_vecs[traits[0]])
    for alpha_i, trait in zip(alphas, traits):
        vec = global_trait_vecs[trait]
        norm = vec.norm()
        if norm > 0:
            combined = combined + alpha_i.item() * (vec / norm)
    return combined


def generate_with_multi_trait_steering(
    model,
    tokenizer,
    prompt: str,
    combined_vec: torch.Tensor,
    layer_idx: int,
    max_new_tokens: int,
    device: torch.device,
) -> Tuple[str, torch.Tensor]:
    """
    Generate text with a combined multi-trait steering vector.
    Also returns the hidden state at the steered layer for the last generated token.

    Returns
    -------
    text : generated text
    hidden : [d] hidden state at steered layer (last input token before generation)
    """
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    hook = SteeringHook(combined_vec, alpha=1.0)  # alpha already baked in
    hook.attach(get_layer_module(model, layer_idx))

    try:
        with torch.no_grad():
            # First, get hidden states from the prompt
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            seq_len = int(attention_mask.sum().item())
            last_tok_idx = seq_len - 1
            # hidden state at steered layer (+1 because index 0 is embedding)
            hidden = outputs.hidden_states[layer_idx + 1][0, last_tok_idx]
            hidden = hidden.detach().to(dtype=torch.float32).cpu()

            # Now generate
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        continuation = gen[0, input_ids.shape[1]:]
        text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
    finally:
        hook.remove()

    return text, hidden


# ============================================================
# Phase 4: Coherence metrics
# ============================================================


def compute_perplexity(
    model, tokenizer, text: str, device: torch.device,
) -> float:
    """Compute perplexity of text under the base model (no steering)."""
    if not text or len(text.strip()) == 0:
        return float("inf")
    encoded = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512,
    ).to(device)
    if encoded["input_ids"].shape[1] < 2:
        return float("inf")
    with torch.no_grad():
        outputs = model(**encoded, labels=encoded["input_ids"])
    return torch.exp(outputs.loss).item()


def compute_nearest_persona_similarity(
    hidden: torch.Tensor,
    persona_centroids: torch.Tensor,
) -> Tuple[float, int]:
    """
    Cosine similarity between a hidden state and the nearest known persona centroid.

    Parameters
    ----------
    hidden : [d]
    persona_centroids : [n_personas, d]

    Returns
    -------
    max_sim : float, highest cosine similarity
    best_idx : int, index of nearest persona
    """
    h = F.normalize(hidden.unsqueeze(0), dim=-1)  # [1, d]
    c = F.normalize(persona_centroids, dim=-1)  # [n_personas, d]
    sims = (h @ c.T).squeeze(0)  # [n_personas]
    best_idx = int(sims.argmax().item())
    return float(sims[best_idx].item()), best_idx


def compute_probe_confidence(
    hidden: torch.Tensor,
    probe: LogisticRegression,
) -> float:
    """Max softmax probability from a persona probe applied to one hidden state."""
    h = F.normalize(hidden.unsqueeze(0), dim=-1).numpy()
    probs = probe.predict_proba(h)
    return float(probs.max())


# ============================================================
# Phase 5: Visualization
# ============================================================


def plot_coherence_map_pca(
    trait_coords: np.ndarray,
    coherence_scores: np.ndarray,
    point_labels: List[str],
    outpath: Path,
) -> None:
    """
    PCA of the [n_samples, n_traits] trait-space coordinates,
    colored by coherence score.
    """
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(trait_coords)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: colored by coherence
    ax = axes[0]
    sc = ax.scatter(
        pca_coords[:, 0], pca_coords[:, 1],
        c=coherence_scores, cmap="RdYlGn", s=40, alpha=0.8,
        edgecolors="gray", linewidths=0.3,
    )
    plt.colorbar(sc, ax=ax, label="Coherence score")

    # Mark known personas
    for i, label in enumerate(point_labels):
        if label == "known":
            ax.scatter(
                pca_coords[i, 0], pca_coords[i, 1],
                marker="*", s=200, c="blue", edgecolors="black",
                linewidths=1, zorder=5,
            )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("Coherence Map in Trait Space (PCA)")

    # Right: colored by point type
    ax = axes[1]
    type_colors = {"known": "blue", "interpolation": "orange", "random": "gray"}
    for ptype in ["random", "interpolation", "known"]:
        mask = [l == ptype for l in point_labels]
        pts = pca_coords[mask]
        if len(pts) > 0:
            ax.scatter(
                pts[:, 0], pts[:, 1],
                label=ptype, alpha=0.7, s=40 if ptype != "known" else 100,
                color=type_colors[ptype],
                marker="*" if ptype == "known" else "o",
            )
    ax.legend()
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("Sample Types in Trait Space")

    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_coherence_vs_distance(
    distances: np.ndarray,
    coherence_scores: np.ndarray,
    point_labels: List[str],
    outpath: Path,
) -> None:
    """Scatter plot of coherence vs distance from nearest known persona."""
    fig, ax = plt.subplots(figsize=(8, 6))

    type_colors = {"known": "blue", "interpolation": "orange", "random": "gray"}
    for ptype in ["random", "interpolation", "known"]:
        mask = np.array([l == ptype for l in point_labels])
        if mask.any():
            ax.scatter(
                distances[mask], coherence_scores[mask],
                label=ptype, alpha=0.7, s=40 if ptype != "known" else 100,
                color=type_colors[ptype],
                marker="*" if ptype == "known" else "o",
            )

    ax.set_xlabel("Distance from nearest known persona (trait space L2)")
    ax.set_ylabel("Coherence score")
    ax.set_title("Coherence vs Distance from Known Personas")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


# ============================================================
# Main
# ============================================================


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = infer_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")

    # ---- Load model ----
    print(f"\nLoading model: {args.model_name} on {device}...")
    model, tokenizer = load_model_and_tokenizer(args.model_name, device, hf_token)
    num_layers = get_num_layers(model)
    layer_indices = sample_layers(num_layers=num_layers, stride=args.layer_stride)

    # ---- Select personas and questions ----
    personas = PERSONAS[: args.limit_personas] if args.limit_personas > 0 else PERSONAS
    questions = QUESTIONS[: args.limit_questions] if args.limit_questions > 0 else QUESTIONS

    n_traits = len(TRAITS)
    print(f"Personas: {len(personas)}, Questions: {len(questions)}, "
          f"Traits: {n_traits}, Layers: {layer_indices}")

    # ==================================================================
    # Phase 1: Compute global trait vectors
    # ==================================================================
    print("\n" + "=" * 60)
    print("Phase 1: Computing global trait vectors")
    print("=" * 60)

    trait_examples, trait_labels, level_labels, persona_labels_trait = build_trait_examples(
        tokenizer, questions, personas, max_length=512,
    )
    print(f"  Built {len(trait_examples)} trait examples "
          f"({len(personas)} personas x {n_traits} traits x 2 levels x {len(questions)} questions)")

    print("  Collecting hidden states...")
    trait_by_layer = collect_hidden_vectors(
        model=model,
        examples=trait_examples,
        device=device,
        dtype=model.dtype,
        layer_indices=layer_indices,
    )

    # Select best layer
    best_layer = select_best_layer(trait_by_layer, trait_labels, level_labels)

    # Compute trait vectors at the best layer
    global_trait_vecs = compute_global_trait_vectors_from_hidden(
        trait_by_layer, trait_labels, level_labels, best_layer,
    )
    print(f"  Trait vector norms at layer {best_layer}:")
    for trait in TRAITS:
        print(f"    {trait}: {global_trait_vecs[trait].norm().item():.4f}")

    # ==================================================================
    # Phase 1b: Collect persona centroids (for coherence metric)
    # ==================================================================
    print("\n  Collecting persona centroids for coherence metric...")
    persona_dict = {p: PERSONA_PROMPTS[p] for p in personas}
    persona_examples = build_examples(tokenizer, persona_dict, questions, max_length=512)
    persona_by_layer = collect_hidden_vectors(
        model=model,
        examples=persona_examples,
        device=device,
        dtype=model.dtype,
        layer_indices=[best_layer],
    )
    persona_vecs = torch.stack(persona_by_layer[best_layer], dim=0)  # [N, d]
    persona_names_list = [ex.persona_name for ex in persona_examples]
    unique_personas = sorted(set(persona_names_list))
    centroids = []
    for p in unique_personas:
        mask = torch.tensor([pn == p for pn in persona_names_list])
        centroids.append(persona_vecs[mask].mean(dim=0))
    persona_centroids = torch.stack(centroids, dim=0)  # [n_personas, d]
    print(f"  Computed centroids for {len(unique_personas)} personas")

    # ==================================================================
    # Phase 1c: Train persona probe (for confidence metric)
    # ==================================================================
    print("  Training persona probe for confidence metric...")
    probe_X = F.normalize(persona_vecs, dim=-1).numpy()
    persona_to_id = {p: i for i, p in enumerate(unique_personas)}
    probe_y = np.array([persona_to_id[pn] for pn in persona_names_list])
    persona_probe = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
    persona_probe.fit(probe_X, probe_y)
    print(f"  Probe trained (classes: {unique_personas})")

    # ==================================================================
    # Phase 2: Sample points in trait space
    # ==================================================================
    print("\n" + "=" * 60)
    print(f"Phase 2: Sampling {args.n_samples} points in trait space")
    print("=" * 60)

    persona_trait_matrix = get_persona_trait_matrix_tensor()
    # If we limited personas, take only matching rows
    if args.limit_personas > 0:
        persona_trait_matrix = persona_trait_matrix[: args.limit_personas]

    trait_points, point_labels = sample_trait_points(
        args.n_samples, n_traits, persona_trait_matrix,
    )
    print(f"  Sampled {len(trait_points)} points: "
          f"{point_labels.count('known')} known, "
          f"{point_labels.count('interpolation')} interpolation, "
          f"{point_labels.count('random')} random")

    # ==================================================================
    # Phase 3 + 4: Steer, generate, and measure coherence
    # ==================================================================
    print("\n" + "=" * 60)
    print("Phase 3+4: Multi-trait steering, generation, and coherence")
    print("=" * 60)

    # Build a neutral prompt (no system persona)
    test_question = questions[0]
    messages = [{"role": "user", "content": test_question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    rows = []
    all_hiddens = []

    for idx in tqdm(range(len(trait_points)), desc="Sampling trait space"):
        alphas = trait_points[idx]
        label = point_labels[idx]

        # Build combined steering vector
        combined_vec = build_combined_steering_vector(
            alphas, global_trait_vecs, TRAITS,
        )

        # Generate with multi-trait steering
        text, hidden = generate_with_multi_trait_steering(
            model, tokenizer, prompt, combined_vec, best_layer,
            args.max_new_tokens, device,
        )
        all_hiddens.append(hidden)

        # Coherence metrics
        # (a) Perplexity
        ppl = compute_perplexity(model, tokenizer, text, device)

        # (b) Self-consistency: cosine sim to nearest known persona centroid
        nearest_sim, nearest_idx = compute_nearest_persona_similarity(
            hidden, persona_centroids,
        )
        nearest_persona = unique_personas[nearest_idx]

        # (c) Probe confidence
        probe_conf = compute_probe_confidence(hidden, persona_probe)

        # Distance from nearest known persona in trait space
        dists = (persona_trait_matrix - alphas.unsqueeze(0)).norm(dim=-1)
        nearest_trait_dist = float(dists.min().item())

        # Combined coherence score: high similarity + high probe confidence + low perplexity
        # Normalize perplexity: use 1/log(ppl) clamped to [0, 1]
        if ppl < float("inf") and ppl > 0:
            ppl_score = 1.0 / (1.0 + np.log(max(ppl, 1.0)))
        else:
            ppl_score = 0.0

        coherence = (nearest_sim + probe_conf + ppl_score) / 3.0

        row = {
            "sample_idx": idx,
            "point_type": label,
            "perplexity": ppl,
            "ppl_score": ppl_score,
            "nearest_persona_sim": nearest_sim,
            "nearest_persona": nearest_persona,
            "probe_confidence": probe_conf,
            "coherence": coherence,
            "trait_dist_to_nearest": nearest_trait_dist,
            "generated_text": text[:200],
        }
        for i, trait in enumerate(TRAITS):
            row[f"alpha_{trait}"] = float(alphas[i].item())
        rows.append(row)

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx + 1:3d}/{len(trait_points)}] type={label:13s} "
                  f"ppl={ppl:8.1f} sim={nearest_sim:.3f} "
                  f"conf={probe_conf:.3f} coherence={coherence:.3f} "
                  f"nearest={nearest_persona}")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "coherence_scores.csv", index=False)
    print(f"\n  Saved coherence scores for {len(df)} samples")

    # ==================================================================
    # Phase 5: Visualization
    # ==================================================================
    print("\n" + "=" * 60)
    print("Phase 5: Visualization")
    print("=" * 60)

    trait_coords = trait_points.numpy()
    coherence_scores = df["coherence"].values

    # (a) Coherence map in trait-space PCA
    plot_coherence_map_pca(
        trait_coords, coherence_scores, point_labels,
        outpath=outdir / "coherence_map_pca.png",
    )
    print("  Saved coherence_map_pca.png")

    # (b) Coherence vs distance from nearest known persona
    distances = df["trait_dist_to_nearest"].values
    plot_coherence_vs_distance(
        distances, coherence_scores, point_labels,
        outpath=outdir / "coherence_vs_distance_from_known.png",
    )
    print("  Saved coherence_vs_distance_from_known.png")

    # ==================================================================
    # Phase 6: SVD of hidden states at all sample points
    # ==================================================================
    print("\n" + "=" * 60)
    print("Phase 6: SVD of hidden-state manifold")
    print("=" * 60)

    hidden_matrix = torch.stack(all_hiddens, dim=0)  # [n_samples, d]
    S, variance, cumvar = svd_analysis(hidden_matrix)
    eff_rank_95 = effective_rank(S, threshold=0.95)
    eff_rank_99 = effective_rank(S, threshold=0.99)

    print(f"  Hidden-state matrix shape: {list(hidden_matrix.shape)}")
    print(f"  Top-10 singular values: {S[:10].numpy()}")
    print(f"  Effective rank (95% var): {eff_rank_95}")
    print(f"  Effective rank (99% var): {eff_rank_99}")

    plot_svd_spectrum(
        S,
        title=f"Manifold SVD Spectrum ({len(all_hiddens)} steered points, layer {best_layer})",
        outpath=outdir / "manifold_svd_spectrum.png",
    )
    print("  Saved manifold_svd_spectrum.png")

    # ==================================================================
    # Save run config
    # ==================================================================
    config = {
        "experiment": "oq3_coherence_manifold",
        "model_name": args.model_name,
        "device": str(device),
        "seed": args.seed,
        "num_layers": num_layers,
        "layer_stride": args.layer_stride,
        "layer_indices": layer_indices,
        "best_layer": best_layer,
        "n_personas": len(personas),
        "n_questions": len(questions),
        "n_traits": n_traits,
        "n_samples": args.n_samples,
        "max_new_tokens": args.max_new_tokens,
        "trait_names": TRAITS,
        "persona_names": personas,
        "manifold_effective_rank_95": eff_rank_95,
        "manifold_effective_rank_99": eff_rank_99,
        "manifold_top10_singular_values": S[:10].tolist(),
        "mean_coherence_known": float(
            df[df["point_type"] == "known"]["coherence"].mean()
        ) if (df["point_type"] == "known").any() else None,
        "mean_coherence_interpolation": float(
            df[df["point_type"] == "interpolation"]["coherence"].mean()
        ) if (df["point_type"] == "interpolation").any() else None,
        "mean_coherence_random": float(
            df[df["point_type"] == "random"]["coherence"].mean()
        ) if (df["point_type"] == "random").any() else None,
    }
    save_run_config(config, outdir)
    init_wandb("oq3_coherence_manifold", config)
    print(f"\n  Saved run_config.json")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Best layer: {best_layer}")
    print(f"  Samples: {len(df)} total")
    for ptype in ["known", "interpolation", "random"]:
        sub = df[df["point_type"] == ptype]
        if len(sub) > 0:
            print(f"    {ptype:15s}: n={len(sub):3d}, "
                  f"mean_coherence={sub['coherence'].mean():.3f}, "
                  f"mean_ppl={sub['perplexity'].median():.1f}, "
                  f"mean_sim={sub['nearest_persona_sim'].mean():.3f}, "
                  f"mean_conf={sub['probe_confidence'].mean():.3f}")
    print(f"  Manifold effective rank (95%): {eff_rank_95}")
    print(f"  Manifold effective rank (99%): {eff_rank_99}")
    finish_wandb(outdir)
    print(f"\n  Outputs saved to: {outdir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OQ3: Natural Sub-Manifold — Where do coherent trait combinations live?",
    )
    parser.add_argument(
        "--model-name", type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--outdir", type=str,
        default="outputs/oq3_coherence_manifold",
    )
    parser.add_argument("--limit-personas", type=int, default=0,
                        help="Limit number of personas (0 = all 10)")
    parser.add_argument("--limit-questions", type=int, default=0,
                        help="Limit number of questions (0 = all)")
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=200,
                        help="Number of points to sample in trait space")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                        help="Max tokens to generate per sample")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
