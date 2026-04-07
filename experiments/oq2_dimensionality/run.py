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
# ]
# ///

"""
OQ2: How many independent trait dimensions matter?

This experiment answers the question by:
1. Computing 8 global trait vectors (high - low centroids) from hidden states.
2. SVD of the [8, d] trait-vector matrix to find effective rank.
3. SVD of the [10, d] persona-centroid matrix for comparison.
4. Random baseline: SVD spectrum of random unit vectors (null hypothesis).
5. Per-component probe: project hidden states onto SVD basis, train classifier
   on first k components, measure accuracy vs k.
"""

from __future__ import annotations

import argparse
import json
import os
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
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
    TRAITS,
    TRAIT_PROMPTS,
    get_trait_prompt,
)
from shared.utils import (
    effective_rank,
    finish_wandb,
    init_wandb,
    plot_svd_spectrum,
    save_run_config,
    svd_analysis,
)


# ============================================================
# Phase 1: Compute global trait vectors
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


def compute_global_trait_vectors(
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


def build_persona_examples(
    tokenizer,
    questions: List[str],
    personas: List[str],
    max_length: int = 512,
) -> Tuple[List[Example], List[str]]:
    """Build examples using the base persona prompts (no trait modifier)."""
    persona_dict = {p: PERSONA_PROMPTS[p] for p in personas}
    examples = build_examples(tokenizer, persona_dict, questions, max_length)
    persona_labels = [ex.persona_name for ex in examples]
    return examples, persona_labels


# ============================================================
# Phase 4: Random baseline
# ============================================================


def random_svd_spectrum(d: int, n_vectors: int = 8, n_repeats: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random unit vectors in R^d, compute SVD spectrum.
    Returns (mean_variance, std_variance) each of shape [n_vectors].
    """
    all_var = []
    for _ in range(n_repeats):
        vecs = torch.randn(n_vectors, d)
        vecs = F.normalize(vecs, dim=-1)
        _, variance, _ = svd_analysis(vecs)
        all_var.append(variance.numpy())

    all_var = np.stack(all_var, axis=0)  # [n_repeats, n_vectors]
    return all_var.mean(axis=0), all_var.std(axis=0)


# ============================================================
# Phase 5: Per-component probe
# ============================================================


def per_component_probe_accuracy(
    by_layer: Dict[int, List[torch.Tensor]],
    trait_labels: List[str],
    svd_basis: torch.Tensor,
    layer: int,
    max_k: int = 8,
    n_folds: int = 5,
) -> List[Dict]:
    """
    Project hidden states onto the SVD basis, train logistic regression
    on the first k components for k = 1..max_k.

    Parameters
    ----------
    svd_basis : [max_k, d] right singular vectors (rows are basis vectors)
    """
    vecs = torch.stack(by_layer[layer], dim=0)  # [N, d]
    # Center
    vecs_centered = vecs - vecs.mean(dim=0, keepdim=True)
    # Project onto all max_k components: [N, max_k]
    projections = vecs_centered @ svd_basis.T

    # Encode trait labels
    trait_vocab = sorted(set(trait_labels))
    trait_to_id = {t: i for i, t in enumerate(trait_vocab)}
    y = np.array([trait_to_id[t] for t in trait_labels])

    results = []
    for k in range(1, max_k + 1):
        X = projections[:, :k].numpy()
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        results.append({
            "k": k,
            "mean_accuracy": float(scores.mean()),
            "std_accuracy": float(scores.std()),
        })
        print(f"    k={k}: accuracy = {scores.mean():.4f} +/- {scores.std():.4f}")

    return results


# ============================================================
# Plotting
# ============================================================


def plot_comparison_spectra(
    trait_var: np.ndarray,
    persona_var: np.ndarray,
    random_mean: np.ndarray,
    random_std: np.ndarray,
    outpath: Path,
) -> None:
    """Overlay trait SVD, persona SVD, and random baseline spectra."""
    fig, ax = plt.subplots(figsize=(8, 5))
    n_trait = len(trait_var)
    n_persona = len(persona_var)
    n_random = len(random_mean)

    x_trait = np.arange(1, n_trait + 1)
    x_persona = np.arange(1, n_persona + 1)
    x_random = np.arange(1, n_random + 1)

    ax.plot(x_trait, trait_var, "o-", color="steelblue", linewidth=2, label=f"Trait vectors ({n_trait})")
    ax.plot(x_persona, persona_var, "s-", color="darkorange", linewidth=2, label=f"Persona centroids ({n_persona})")
    ax.fill_between(
        x_random,
        random_mean - random_std,
        random_mean + random_std,
        alpha=0.3, color="gray",
    )
    ax.plot(x_random, random_mean, "--", color="gray", linewidth=1.5, label="Random baseline")

    ax.set_xlabel("Component")
    ax.set_ylabel("Variance explained")
    ax.set_title("SVD Spectrum Comparison: Traits vs Personas vs Random")
    ax.legend()
    ax.set_xlim(0.5, max(n_trait, n_persona) + 0.5)
    ax.set_ylim(0, None)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_cumulative_variance(
    trait_cumvar: np.ndarray,
    persona_cumvar: np.ndarray,
    outpath: Path,
) -> None:
    """Plot cumulative variance explained for trait and persona SVD."""
    fig, ax = plt.subplots(figsize=(8, 5))

    x_trait = np.arange(1, len(trait_cumvar) + 1)
    x_persona = np.arange(1, len(persona_cumvar) + 1)

    ax.plot(x_trait, trait_cumvar, "o-", color="steelblue", linewidth=2, label="Trait vectors")
    ax.plot(x_persona, persona_cumvar, "s-", color="darkorange", linewidth=2, label="Persona centroids")
    ax.axhline(0.95, color="gray", linestyle="--", alpha=0.5, label="95% threshold")
    ax.axhline(0.99, color="gray", linestyle=":", alpha=0.5, label="99% threshold")

    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative variance explained")
    ax.set_title("Cumulative Variance: Trait Vectors vs Persona Centroids")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_probe_accuracy(
    probe_results: List[Dict],
    outpath: Path,
    chance_level: float = 0.125,
) -> None:
    """Plot per-component probe accuracy vs number of components."""
    ks = [r["k"] for r in probe_results]
    means = [r["mean_accuracy"] for r in probe_results]
    stds = [r["std_accuracy"] for r in probe_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ks, means, yerr=stds, fmt="o-", color="steelblue", linewidth=2, capsize=4)
    ax.axhline(chance_level, color="red", linestyle="--", alpha=0.6, label=f"Chance ({chance_level:.3f})")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.3)

    ax.set_xlabel("Number of SVD components (k)")
    ax.set_ylabel("Trait classification accuracy (CV)")
    ax.set_title("Per-Component Probe: Accuracy vs Dimensionality")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ks)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


# ============================================================
# Main
# ============================================================


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
        tvecs = compute_global_trait_vectors(by_layer, trait_labels, level_labels, layer)
        mean_norm = torch.stack(list(tvecs.values())).norm(dim=-1).mean().item()
        if mean_norm > best_norm:
            best_norm = mean_norm
            best_layer = layer
    print(f"  Best layer by trait vector norm: {best_layer} (mean norm = {best_norm:.4f})")
    return best_layer


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = infer_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")

    # ---- Load model ----
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model on {device}...")
    model_kwargs = {"output_hidden_states": True}
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model_kwargs["device_map"] = "auto"
    elif device.type == "mps":
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model_name, token=hf_token, **model_kwargs)
    if device.type in {"cpu", "mps"}:
        model.to(device)

    num_layers = model.config.num_hidden_layers
    layer_indices = sample_layers(num_layers=num_layers, stride=args.layer_stride)

    # ---- Select personas and questions ----
    personas = PERSONAS[: args.limit_personas] if args.limit_personas > 0 else PERSONAS
    questions = QUESTIONS[: args.limit_questions] if args.limit_questions > 0 else QUESTIONS

    print(f"Personas: {len(personas)}, Questions: {len(questions)}, Layers: {layer_indices}")

    # ==================================================================
    # Phase 1: Compute global trait vectors
    # ==================================================================
    trait_vectors_by_layer: Dict[int, Dict[str, torch.Tensor]] = {}

    if args.trait_vectors_path:
        # Option A: Load pre-computed trait vectors
        print(f"\nLoading pre-computed trait vectors from {args.trait_vectors_path}")
        loaded = torch.load(args.trait_vectors_path, map_location="cpu", weights_only=True)
        # Expect keys like "layer_{i}_{trait}" -> tensor
        for key, vec in loaded.items():
            parts = key.split("_", 2)  # "layer", str(i), trait_name
            layer = int(parts[1])
            trait = parts[2]
            if layer not in trait_vectors_by_layer:
                trait_vectors_by_layer[layer] = {}
            trait_vectors_by_layer[layer][trait] = vec
        print(f"  Loaded {len(loaded)} vectors across {len(trait_vectors_by_layer)} layers")
        # We still need hidden states for the probe, so collect those
        trait_by_layer = None
        trait_labels = None
        level_labels = None
    else:
        # Option B: Recompute (self-contained)
        print("\n" + "=" * 60)
        print("Phase 1: Computing global trait vectors")
        print("=" * 60)

        trait_examples, trait_labels, level_labels, persona_labels_trait = build_trait_examples(
            tokenizer, questions, personas, max_length=512,
        )
        print(f"  Built {len(trait_examples)} trait examples "
              f"({len(personas)} personas x {len(TRAITS)} traits x 2 levels x {len(questions)} questions)")

        print("  Collecting hidden states...")
        trait_by_layer = collect_hidden_vectors(
            model=model,
            examples=trait_examples,
            device=device,
            dtype=model.dtype,
            layer_indices=layer_indices,
        )

        # Compute trait vectors at each layer
        for layer in layer_indices:
            trait_vectors_by_layer[layer] = compute_global_trait_vectors(
                trait_by_layer, trait_labels, level_labels, layer,
            )
            norms = {t: f"{v.norm().item():.3f}" for t, v in trait_vectors_by_layer[layer].items()}
            print(f"  Layer {layer:2d} trait vector norms: {norms}")

    # ==================================================================
    # Phase 2: SVD of trait vectors
    # ==================================================================
    print("\n" + "=" * 60)
    print("Phase 2: SVD of trait vectors")
    print("=" * 60)

    # Select best layer
    if len(trait_vectors_by_layer) == 0:
        raise RuntimeError("No trait vectors computed or loaded.")

    if trait_labels is not None and trait_by_layer is not None:
        best_layer = select_best_layer(trait_by_layer, trait_labels, level_labels)
    else:
        # If loaded, pick the layer with highest mean trait vector norm
        best_layer = max(
            trait_vectors_by_layer.keys(),
            key=lambda l: torch.stack(list(trait_vectors_by_layer[l].values())).norm(dim=-1).mean().item(),
        )
        print(f"  Best layer (from loaded vectors): {best_layer}")

    trait_vecs = trait_vectors_by_layer[best_layer]
    trait_matrix = torch.stack([trait_vecs[t] for t in TRAITS], dim=0)  # [8, d]
    # Normalize
    trait_matrix_normed = F.normalize(trait_matrix, dim=-1)

    trait_S, trait_var, trait_cumvar = svd_analysis(trait_matrix_normed)
    trait_eff_rank_95 = effective_rank(trait_S, threshold=0.95)
    trait_eff_rank_99 = effective_rank(trait_S, threshold=0.99)

    print(f"  Trait SVD at layer {best_layer}:")
    print(f"    Singular values: {trait_S.numpy()}")
    print(f"    Variance explained: {trait_var.numpy()}")
    print(f"    Effective rank (95%): {trait_eff_rank_95}")
    print(f"    Effective rank (99%): {trait_eff_rank_99}")

    # Plot trait SVD spectrum
    plot_svd_spectrum(
        trait_S, title=f"Trait Vector SVD Spectrum (layer {best_layer})",
        outpath=outdir / "trait_svd_spectrum.png",
    )

    # ==================================================================
    # Phase 3: SVD of persona centroids
    # ==================================================================
    print("\n" + "=" * 60)
    print("Phase 3: SVD of persona centroids")
    print("=" * 60)

    persona_examples, persona_labels = build_persona_examples(
        tokenizer, questions, personas, max_length=512,
    )
    print(f"  Built {len(persona_examples)} persona examples")

    print("  Collecting persona hidden states...")
    persona_by_layer = collect_hidden_vectors(
        model=model,
        examples=persona_examples,
        device=device,
        dtype=model.dtype,
        layer_indices=[best_layer],
    )

    # Compute persona centroids at best_layer
    persona_vecs = torch.stack(persona_by_layer[best_layer], dim=0)  # [N, d]
    unique_personas = sorted(set(persona_labels))
    centroids = []
    for p in unique_personas:
        mask = torch.tensor([pl == p for pl in persona_labels])
        centroids.append(persona_vecs[mask].mean(dim=0))
    centroid_matrix = torch.stack(centroids, dim=0)  # [n_personas, d]
    centroid_matrix_normed = F.normalize(centroid_matrix, dim=-1)

    persona_S, persona_var, persona_cumvar = svd_analysis(centroid_matrix_normed)
    persona_eff_rank_95 = effective_rank(persona_S, threshold=0.95)
    persona_eff_rank_99 = effective_rank(persona_S, threshold=0.99)

    print(f"  Persona centroid SVD at layer {best_layer}:")
    print(f"    Singular values: {persona_S.numpy()}")
    print(f"    Variance explained: {persona_var.numpy()}")
    print(f"    Effective rank (95%): {persona_eff_rank_95}")
    print(f"    Effective rank (99%): {persona_eff_rank_99}")

    # Plot persona SVD spectrum
    plot_svd_spectrum(
        persona_S, title=f"Persona Centroid SVD Spectrum (layer {best_layer})",
        outpath=outdir / "persona_svd_spectrum.png",
    )

    # ==================================================================
    # Phase 4: Random baseline
    # ==================================================================
    print("\n" + "=" * 60)
    print("Phase 4: Random baseline")
    print("=" * 60)

    d = trait_matrix.shape[1]
    random_var_mean, random_var_std = random_svd_spectrum(d, n_vectors=len(TRAITS), n_repeats=100)
    print(f"  Random baseline (d={d}, {len(TRAITS)} vectors, 100 repeats):")
    print(f"    Mean variance: {random_var_mean}")

    # ==================================================================
    # Comparison plots
    # ==================================================================
    print("\nGenerating comparison plots...")

    plot_comparison_spectra(
        trait_var=trait_var.numpy(),
        persona_var=persona_var.numpy(),
        random_mean=random_var_mean,
        random_std=random_var_std,
        outpath=outdir / "trait_svd_spectrum.png",  # overwrite with comparison
    )

    plot_cumulative_variance(
        trait_cumvar=trait_cumvar.numpy(),
        persona_cumvar=persona_cumvar.numpy(),
        outpath=outdir / "trait_cumulative_variance.png",
    )

    # ==================================================================
    # Phase 5: Per-component probe
    # ==================================================================
    print("\n" + "=" * 60)
    print("Phase 5: Per-component probe")
    print("=" * 60)

    # Get SVD basis from the trait matrix (right singular vectors)
    trait_centered = trait_matrix_normed - trait_matrix_normed.mean(dim=0, keepdim=True)
    _, _, Vh = torch.linalg.svd(trait_centered, full_matrices=False)
    # Vh: [min(8,d), d] — rows are the right singular vectors
    max_k = min(len(TRAITS), Vh.shape[0])
    svd_basis = Vh[:max_k]  # [max_k, d]

    # We need per-example hidden states with trait labels for the probe
    if trait_by_layer is not None and trait_labels is not None:
        print(f"  Using trait hidden states for probe (layer {best_layer})")
        probe_results = per_component_probe_accuracy(
            trait_by_layer, trait_labels, svd_basis, best_layer, max_k=max_k,
        )
    elif args.trait_vectors_path:
        # We loaded trait vectors but don't have per-example states.
        # Recompute the trait examples for the probe phase.
        print("  Recomputing trait examples for per-component probe...")
        trait_examples, trait_labels, level_labels, _ = build_trait_examples(
            tokenizer, questions, personas, max_length=512,
        )
        trait_by_layer = collect_hidden_vectors(
            model=model,
            examples=trait_examples,
            device=device,
            dtype=model.dtype,
            layer_indices=[best_layer],
        )
        probe_results = per_component_probe_accuracy(
            trait_by_layer, trait_labels, svd_basis, best_layer, max_k=max_k,
        )
    else:
        raise RuntimeError("No hidden states available for probe.")

    chance_level = 1.0 / len(TRAITS)
    plot_probe_accuracy(
        probe_results,
        outpath=outdir / "per_component_probe_accuracy.png",
        chance_level=chance_level,
    )

    # ==================================================================
    # Save outputs
    # ==================================================================
    print("\nSaving outputs...")

    # Dimensionality comparison CSV
    rows = []
    for i in range(max(len(trait_var), len(persona_var), len(random_var_mean))):
        row = {"component": i + 1}
        if i < len(trait_var):
            row["trait_variance"] = float(trait_var[i])
            row["trait_cumulative"] = float(trait_cumvar[i])
        if i < len(persona_var):
            row["persona_variance"] = float(persona_var[i])
            row["persona_cumulative"] = float(persona_cumvar[i])
        if i < len(random_var_mean):
            row["random_variance_mean"] = float(random_var_mean[i])
            row["random_variance_std"] = float(random_var_std[i])
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(outdir / "dimensionality_comparison.csv", index=False)

    # Probe results
    probe_df = pd.DataFrame(probe_results)
    probe_df.to_csv(outdir / "probe_accuracy_vs_k.csv", index=False)

    # Run config
    config = {
        "experiment": "oq2_dimensionality",
        "model_name": args.model_name,
        "device": str(device),
        "seed": args.seed,
        "num_layers": num_layers,
        "layer_stride": args.layer_stride,
        "layer_indices": layer_indices,
        "best_layer": best_layer,
        "n_personas": len(personas),
        "n_questions": len(questions),
        "n_traits": len(TRAITS),
        "trait_names": TRAITS,
        "persona_names": personas,
        "trait_effective_rank_95": trait_eff_rank_95,
        "trait_effective_rank_99": trait_eff_rank_99,
        "persona_effective_rank_95": persona_eff_rank_95,
        "persona_effective_rank_99": persona_eff_rank_99,
        "trait_singular_values": trait_S.tolist(),
        "persona_singular_values": persona_S.tolist(),
        "trait_vectors_path": args.trait_vectors_path,
    }
    save_run_config(config, outdir)
    init_wandb("oq2_dimensionality", config)

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Best layer: {best_layer}")
    print(f"  Trait effective rank (95%): {trait_eff_rank_95} / {len(TRAITS)}")
    print(f"  Trait effective rank (99%): {trait_eff_rank_99} / {len(TRAITS)}")
    print(f"  Persona effective rank (95%): {persona_eff_rank_95} / {len(personas)}")
    print(f"  Persona effective rank (99%): {persona_eff_rank_99} / {len(personas)}")
    print(f"\n  Probe accuracy by #components:")
    for r in probe_results:
        marker = " <-- 95% var" if r["k"] == trait_eff_rank_95 else ""
        print(f"    k={r['k']}: {r['mean_accuracy']:.4f} +/- {r['std_accuracy']:.4f}{marker}")
    finish_wandb(outdir)
    print(f"\n  Outputs saved to: {outdir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OQ2: How many independent trait dimensions matter?",
    )
    parser.add_argument(
        "--model-name", type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--outdir", type=str,
        default="outputs/oq2_dimensionality",
    )
    parser.add_argument("--limit-personas", type=int, default=0,
                        help="Limit number of personas (0 = all 10)")
    parser.add_argument("--limit-questions", type=int, default=0,
                        help="Limit number of questions (0 = all)")
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument(
        "--trait-vectors-path", type=str, default=None,
        help="Path to pre-computed trait vectors (.pt file). "
             "If not provided, trait vectors are recomputed.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
