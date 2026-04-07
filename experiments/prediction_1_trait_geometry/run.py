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
Prediction 1 — Trait Geometry
=============================

Tests whether trait vectors (high - low activation centroids) are mostly
shared across personas, i.e. the "honesty direction" extracted from the
pirate persona points roughly the same way as the one from the scientist.

Steps
-----
1. For each (persona, trait, level=high/low) combination, build prompts
   that combine the persona system prompt with a trait modifier and a
   question.  Forward-pass to extract hidden states.
2. Compute per-persona trait vectors:
       trait_vec[persona][trait] = centroid(high) - centroid(low)
3. Compute global (persona-averaged) trait vectors.
4. Measure cross-persona cosine similarity per trait.
5. Compute residual norm ratio: ||persona_vec - global_vec|| / ||global_vec||.
6. SVD per trait: stack the N persona-specific vectors -> rank analysis.
7. PCA across all persona x trait vectors, colored by trait.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Path manipulation so we can import project modules
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _SCRIPT_DIR.parent                # experiments/
_PROJECT_ROOT = _EXPERIMENTS_DIR.parent              # personas/

sys.path.insert(0, str(_PROJECT_ROOT))               # main.py, steer.py
sys.path.insert(0, str(_EXPERIMENTS_DIR))             # shared/

from dotenv import load_dotenv

load_dotenv()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Project imports
from main import (
    QUESTIONS,
    build_examples,
    collect_hidden_vectors,
    infer_device,
    pca_2d_torch,
    sample_layers,
    set_seed,
)
from shared.trait_config import (
    PERSONA_PROMPTS,
    PERSONAS as TRAIT_PERSONAS,
    TRAIT_PROMPTS,
    TRAITS,
    get_trait_prompt,
)
from shared.trait_vectors import (
    compute_global_trait_vectors,
    compute_trait_residuals,
    compute_trait_vectors_per_persona,
    cross_persona_cosine_similarity,
)
from shared.utils import (
    finish_wandb,
    get_num_layers,
    init_wandb,
    load_model_and_tokenizer,
    plot_heatmap,
    plot_pca_scatter,
    plot_svd_spectrum,
    save_run_config,
    svd_analysis,
)


# ============================================================
# Helpers
# ============================================================


def parse_trait_key(key: str) -> Tuple[str, str, str]:
    """Parse a composite key ``persona__trait__level`` back into parts."""
    parts = key.split("__")
    if len(parts) != 3:
        raise ValueError(f"Cannot parse trait key: {key!r}")
    return parts[0], parts[1], parts[2]


def build_trait_personas(
    personas_to_use: List[str],
    traits_to_use: List[str],
) -> Dict[str, str]:
    """
    Build a persona dict keyed by ``persona__trait__level`` whose values
    are the combined system prompts.  This dict can be passed directly
    to ``build_examples`` from main.py.
    """
    trait_personas: Dict[str, str] = {}
    for persona_name in personas_to_use:
        for trait in traits_to_use:
            for level in ("high", "low"):
                key = f"{persona_name}__{trait}__{level}"
                trait_personas[key] = get_trait_prompt(persona_name, trait, level)
    return trait_personas


def decompose_example_metadata(
    examples,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Given a list of ``Example`` objects whose ``persona_name`` fields are
    composite keys, return three parallel lists:
    ``(persona_names, trait_names, trait_labels)`` (one per example).
    """
    persona_names: List[str] = []
    trait_names: List[str] = []
    trait_labels: List[str] = []
    for ex in examples:
        persona, trait, level = parse_trait_key(ex.persona_name)
        persona_names.append(persona)
        trait_names.append(trait)
        trait_labels.append(level)
    return persona_names, trait_names, trait_labels


# ============================================================
# SVD per trait
# ============================================================


def svd_per_trait(
    per_persona: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
    layer: int,
    personas: List[str],
    traits: List[str],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    For each trait, stack persona-specific vectors into [N_personas, d],
    run SVD, and return ``{trait: (S, variance, cumvar)}``.
    """
    layer_data = per_persona[layer]
    results: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for trait in traits:
        vecs = [layer_data[p][trait] for p in personas]
        mat = torch.stack(vecs, dim=0)  # [N, d]
        S, variance, cumvar = svd_analysis(mat)
        results[trait] = (S, variance, cumvar)
    return results


# ============================================================
# Plotting helpers specific to this experiment
# ============================================================


def plot_cosine_by_layer(
    cosine_by_layer: Dict[int, Dict[str, float]],
    traits: List[str],
    outdir: Path,
) -> None:
    """Line plot: cross-persona cosine similarity per trait across layers."""
    layers = sorted(cosine_by_layer.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.get_cmap("tab10")
    for i, trait in enumerate(traits):
        vals = [cosine_by_layer[l].get(trait, 0.0) for l in layers]
        ax.plot(layers, vals, marker="o", markersize=4, label=trait,
                color=cmap(i / len(traits)))
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean pairwise cosine similarity")
    ax.set_title("Cross-persona cosine similarity per trait")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "cosine_by_layer.png", dpi=180)
    plt.close()


def plot_residual_by_layer(
    residuals_by_layer: Dict[int, Dict[str, Dict[str, float]]],
    personas: List[str],
    traits: List[str],
    outdir: Path,
) -> None:
    """Line plot: mean residual norm ratio across layers (averaged over traits)."""
    layers = sorted(residuals_by_layer.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.get_cmap("tab10")
    for i, persona in enumerate(personas):
        vals = []
        for l in layers:
            mean_ratio = np.mean([
                residuals_by_layer[l][persona][t]
                for t in traits
            ])
            vals.append(mean_ratio)
        ax.plot(layers, vals, marker="s", markersize=4, label=persona,
                color=cmap(i / len(personas)))
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean residual norm ratio")
    ax.set_title("Persona-specific residual (lower = more shared)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "residual_by_layer.png", dpi=180)
    plt.close()


def plot_svd_per_trait_summary(
    svd_by_trait: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    traits: List[str],
    outdir: Path,
) -> None:
    """Bar chart: number of components to reach 95 % variance per trait."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ranks = []
    for trait in traits:
        _, _, cumvar = svd_by_trait[trait]
        idx = int((cumvar >= 0.95).nonzero(as_tuple=True)[0][0].item()) + 1 \
            if (cumvar >= 0.95).any() else len(cumvar)
        ranks.append(idx)
    ax.bar(range(len(traits)), ranks, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(traits)))
    ax.set_xticklabels(traits, rotation=45, ha="right")
    ax.set_ylabel("Components for 95% variance")
    ax.set_title("SVD effective rank per trait (across personas)")
    plt.tight_layout()
    plt.savefig(outdir / "svd_rank_per_trait.png", dpi=180)
    plt.close()


def plot_pca_trait_vectors(
    per_persona: Dict[str, Dict[str, torch.Tensor]],
    personas: List[str],
    traits: List[str],
    outdir: Path,
) -> None:
    """
    Stack all persona x trait vectors, PCA to 2D, color by trait.
    Also create a version colored by persona.
    """
    vecs_list: List[torch.Tensor] = []
    trait_labels: List[str] = []
    persona_labels: List[str] = []
    for persona in personas:
        for trait in traits:
            vecs_list.append(per_persona[persona][trait])
            trait_labels.append(trait)
            persona_labels.append(persona)

    mat = torch.stack(vecs_list, dim=0)  # [N*T, d]
    coords = pca_2d_torch(mat).numpy()

    # Color by trait
    plot_pca_scatter(
        coords=coords,
        labels=trait_labels,
        title="Trait vectors in PCA space (colored by trait)",
        outpath=outdir / "pca_by_trait.png",
        label_points=True,
    )

    # Color by persona
    plot_pca_scatter(
        coords=coords,
        labels=persona_labels,
        title="Trait vectors in PCA space (colored by persona)",
        outpath=outdir / "pca_by_persona.png",
        label_points=True,
    )


# ============================================================
# Main logic
# ============================================================


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = infer_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")

    # ---- Determine subsets -----------------------------------------------
    personas_to_use = TRAIT_PERSONAS[:]
    if args.limit_personas > 0:
        personas_to_use = personas_to_use[: args.limit_personas]

    traits_to_use = TRAITS[:]

    questions = QUESTIONS[: args.limit_questions] if args.limit_questions > 0 else QUESTIONS

    n_combos = len(personas_to_use) * len(traits_to_use) * 2
    print(f"Personas: {len(personas_to_use)}, Traits: {len(traits_to_use)}, "
          f"Questions: {len(questions)}")
    print(f"Total forward passes: {n_combos * len(questions)} "
          f"({n_combos} combos x {len(questions)} questions)")

    # ---- Load model & tokenizer ------------------------------------------
    print(f"\nLoading model: {args.model_name} on {device}...")
    model, tokenizer = load_model_and_tokenizer(args.model_name, device, hf_token)
    num_layers = get_num_layers(model)
    layer_indices = sample_layers(num_layers=num_layers, stride=args.layer_stride)
    print(f"Layers ({len(layer_indices)}): {layer_indices}")

    # ---- Build examples --------------------------------------------------
    print("\nBuilding trait-modulated examples...")
    trait_personas_dict = build_trait_personas(personas_to_use, traits_to_use)
    examples = build_examples(
        tokenizer=tokenizer,
        personas=trait_personas_dict,
        questions=questions,
        max_length=args.max_length,
    )
    print(f"Built {len(examples)} examples")

    # Decompose composite keys
    persona_names, trait_names_list, trait_labels = decompose_example_metadata(examples)

    # ---- Collect hidden states -------------------------------------------
    print("\nCollecting hidden states (this may take a while)...")
    by_layer = collect_hidden_vectors(
        model=model,
        examples=examples,
        device=device,
        dtype=model.dtype,
        layer_indices=layer_indices,
    )

    # Free model memory
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- Step 3: Per-persona trait vectors --------------------------------
    print("\nComputing per-persona trait vectors...")
    per_persona_vecs = compute_trait_vectors_per_persona(
        by_layer=by_layer,
        persona_names=persona_names,
        trait_labels=trait_labels,
        trait_names=trait_names_list,
        personas=personas_to_use,
        traits=traits_to_use,
    )

    # ---- Step 4: Global trait vectors ------------------------------------
    print("Computing global (persona-averaged) trait vectors...")
    global_vecs = compute_global_trait_vectors(
        per_persona_vecs, personas_to_use, traits_to_use,
    )

    # ---- Step 5: Cross-persona cosine similarity -------------------------
    print("Computing cross-persona cosine similarity...")
    cosine_by_layer: Dict[int, Dict[str, float]] = {}
    for layer in layer_indices:
        cosine_by_layer[layer] = cross_persona_cosine_similarity(
            per_persona_vecs, layer, personas_to_use, traits_to_use,
        )

    # Save CSV: rows = layers, columns = traits
    cosine_rows = []
    for layer in layer_indices:
        row = {"layer": layer}
        row.update(cosine_by_layer[layer])
        cosine_rows.append(row)
    cosine_df = pd.DataFrame(cosine_rows)
    cosine_df.to_csv(outdir / "cosine_similarity.csv", index=False)
    print(f"  Saved cosine_similarity.csv ({len(cosine_df)} rows)")

    # ---- Step 6: Residual norm ratio -------------------------------------
    print("Computing residual norm ratios...")
    residuals_by_layer: Dict[int, Dict[str, Dict[str, float]]] = {}
    for layer in layer_indices:
        residuals_by_layer[layer] = compute_trait_residuals(
            per_persona_vecs, global_vecs, layer, personas_to_use, traits_to_use,
        )

    # Save CSV: rows = (layer, persona), columns = traits
    residual_rows = []
    for layer in layer_indices:
        for persona in personas_to_use:
            row = {"layer": layer, "persona": persona}
            row.update(residuals_by_layer[layer][persona])
            residual_rows.append(row)
    residual_df = pd.DataFrame(residual_rows)
    residual_df.to_csv(outdir / "residual_norm_ratios.csv", index=False)
    print(f"  Saved residual_norm_ratios.csv ({len(residual_df)} rows)")

    # ---- Step 7: SVD per trait -------------------------------------------
    print("Running SVD per trait...")
    # Use the deepest layer for SVD analysis
    analysis_layer = layer_indices[-1]
    svd_by_trait = svd_per_trait(
        per_persona_vecs, analysis_layer, personas_to_use, traits_to_use,
    )

    # Save per-trait SVD spectra
    for trait in traits_to_use:
        S, variance, cumvar = svd_by_trait[trait]
        plot_svd_spectrum(
            S,
            title=f"SVD spectrum: {trait} (layer {analysis_layer})",
            outpath=outdir / f"svd_spectrum_{trait}.png",
        )

    # Also save a summary CSV
    svd_rows = []
    for trait in traits_to_use:
        S, variance, cumvar = svd_by_trait[trait]
        for k in range(len(S)):
            svd_rows.append({
                "trait": trait,
                "component": k + 1,
                "singular_value": S[k].item(),
                "variance_explained": variance[k].item(),
                "cumulative_variance": cumvar[k].item(),
            })
    svd_df = pd.DataFrame(svd_rows)
    svd_df.to_csv(outdir / "svd_per_trait.csv", index=False)
    print(f"  Saved svd_per_trait.csv ({len(svd_df)} rows)")

    # ---- Step 8: PCA of all trait vectors --------------------------------
    print("Running PCA on all trait vectors...")
    plot_pca_trait_vectors(
        per_persona=per_persona_vecs[analysis_layer],
        personas=personas_to_use,
        traits=traits_to_use,
        outdir=outdir,
    )

    # ---- Plots -----------------------------------------------------------
    print("\nGenerating plots...")

    # Cosine similarity line plot across layers
    plot_cosine_by_layer(cosine_by_layer, traits_to_use, outdir)

    # Residual norm ratio line plot across layers
    plot_residual_by_layer(residuals_by_layer, personas_to_use, traits_to_use, outdir)

    # SVD rank summary bar chart
    plot_svd_per_trait_summary(svd_by_trait, traits_to_use, outdir)

    # Cosine similarity heatmap at the analysis layer
    cosine_at_layer = cosine_by_layer[analysis_layer]
    # Build a full pairwise cosine matrix for one representative trait
    # Instead, build a trait x layer heatmap
    cosine_matrix = np.array([
        [cosine_by_layer[l].get(t, 0.0) for l in layer_indices]
        for t in traits_to_use
    ])
    plot_heatmap(
        cosine_matrix,
        row_labels=traits_to_use,
        col_labels=[str(l) for l in layer_indices],
        title="Cross-persona cosine similarity (trait x layer)",
        outpath=outdir / "cosine_heatmap.png",
        cmap="RdYlGn",
        vmin=-1.0,
        vmax=1.0,
    )

    # Residual heatmap at the analysis layer
    residual_matrix = np.array([
        [residuals_by_layer[analysis_layer][p][t] for t in traits_to_use]
        for p in personas_to_use
    ])
    plot_heatmap(
        residual_matrix,
        row_labels=personas_to_use,
        col_labels=traits_to_use,
        title=f"Residual norm ratio (layer {analysis_layer})",
        outpath=outdir / "residual_heatmap.png",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=None,
    )

    # ---- Save run config -------------------------------------------------
    config = {
        "experiment": "prediction_1_trait_geometry",
        "model_name": args.model_name,
        "device": str(device),
        "seed": args.seed,
        "num_layers": num_layers,
        "layer_indices": layer_indices,
        "layer_stride": args.layer_stride,
        "analysis_layer": analysis_layer,
        "personas": personas_to_use,
        "traits": traits_to_use,
        "n_personas": len(personas_to_use),
        "n_traits": len(traits_to_use),
        "n_questions": len(questions),
        "n_examples": len(examples),
        "max_length": args.max_length,
        "limit_personas": args.limit_personas,
        "limit_questions": args.limit_questions,
    }
    save_run_config(config, outdir)
    init_wandb("prediction_1_trait_geometry", config)

    # ---- Summary ---------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Summary (deepest layer = {})".format(analysis_layer))
    print(f"{'=' * 60}")
    print("\nCross-persona cosine similarity per trait:")
    for trait in traits_to_use:
        val = cosine_by_layer[analysis_layer][trait]
        print(f"  {trait:20s}  {val:+.4f}")

    print("\nMean residual norm ratio per persona:")
    for persona in personas_to_use:
        mean_ratio = np.mean([
            residuals_by_layer[analysis_layer][persona][t]
            for t in traits_to_use
        ])
        print(f"  {persona:20s}  {mean_ratio:.4f}")

    print(f"\nSVD effective rank (95% variance) per trait at layer {analysis_layer}:")
    for trait in traits_to_use:
        _, _, cumvar = svd_by_trait[trait]
        if (cumvar >= 0.95).any():
            rank = int((cumvar >= 0.95).nonzero(as_tuple=True)[0][0].item()) + 1
        else:
            rank = len(cumvar)
        print(f"  {trait:20s}  rank = {rank}")

    finish_wandb(outdir)
    print(f"\nAll outputs saved to: {outdir.resolve()}")


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prediction 1: Trait Geometry — test whether trait vectors "
                    "are mostly shared across personas",
    )
    parser.add_argument(
        "--model-name", type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--outdir", type=str,
        default="outputs/prediction_1_trait_geometry",
        help="Directory for output files",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (auto-detected if omitted)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--layer-stride", type=int, default=4,
        help="Stride for sampling transformer layers",
    )
    parser.add_argument(
        "--limit-personas", type=int, default=0,
        help="Limit to first N personas (0 = use all)",
    )
    parser.add_argument(
        "--limit-questions", type=int, default=0,
        help="Limit to first N questions (0 = use all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
