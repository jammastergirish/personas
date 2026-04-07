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
OQ1 — Coupling Coefficients
============================

Measures cross-trait coupling: when steering for trait T, how much do the
other traits shift?

Steps
-----
1. Compute global trait vectors from trait-modulated examples (same approach
   as prediction_1).
2. For each persona, collect unsteered hidden states using the base persona
   prompt (no trait modifier) + questions.
3. For each (persona, steer_trait T):
   - Apply SteeringHook with global_trait_vec[T] at the analysis layer.
   - Collect steered hidden states via collect_steered_hidden.
   - Project both unsteered and steered onto the 8-trait basis.
   - coupling[persona][T][j] = mean(steered_coord[j] - unsteered_coord[j])
4. Build 8x8 global coupling matrix (averaged over personas).
5. Build 10 per-persona coupling matrices.
6. SVD of global coupling matrix reveals independent "coupling modes."
7. PCA of per-persona coupling matrices (flattened) to see if personas
   cluster by coupling pattern.
"""

from __future__ import annotations

import argparse
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
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Project imports
from main import (
    QUESTIONS,
    build_examples,
    collect_hidden_vectors,
    infer_device,
    sample_layers,
    set_seed,
)
from steer import (
    SteeringHook,
    collect_steered_hidden,
    get_layer_module,
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
    compute_trait_vectors_per_persona,
    project_onto_trait_basis,
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
# Helpers (reused from prediction_1)
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
    are the combined system prompts.
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
    Given examples with composite persona_name keys, return three parallel
    lists: (persona_names, trait_names, trait_labels).
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

    print(f"Personas: {len(personas_to_use)}, Traits: {len(traits_to_use)}, "
          f"Questions: {len(questions)}")

    # ---- Load model & tokenizer ------------------------------------------
    print(f"\nLoading model: {args.model_name} on {device}...")
    model, tokenizer = load_model_and_tokenizer(args.model_name, device, hf_token)
    num_layers = get_num_layers(model)
    layer_indices = sample_layers(num_layers=num_layers, stride=args.layer_stride)
    print(f"Layers ({len(layer_indices)}): {layer_indices}")

    # Use the deepest sampled layer as the analysis / steering layer
    analysis_layer = layer_indices[-1]
    print(f"Analysis / steering layer: {analysis_layer}")

    # ==================================================================
    # Phase 1: Compute global trait vectors
    # ==================================================================
    print("\n" + "=" * 60)
    print("Phase 1: Computing global trait vectors")
    print("=" * 60)

    print("Building trait-modulated examples...")
    trait_personas_dict = build_trait_personas(personas_to_use, traits_to_use)
    trait_examples = build_examples(
        tokenizer=tokenizer,
        personas=trait_personas_dict,
        questions=questions,
        max_length=512,
    )
    print(f"Built {len(trait_examples)} trait-modulated examples")

    persona_names_t, trait_names_t, trait_labels_t = decompose_example_metadata(
        trait_examples,
    )

    print("Collecting hidden states for trait examples...")
    trait_by_layer = collect_hidden_vectors(
        model=model,
        examples=trait_examples,
        device=device,
        dtype=model.dtype,
        layer_indices=[analysis_layer],
    )

    print("Computing per-persona trait vectors...")
    per_persona_vecs = compute_trait_vectors_per_persona(
        by_layer=trait_by_layer,
        persona_names=persona_names_t,
        trait_labels=trait_labels_t,
        trait_names=trait_names_t,
        personas=personas_to_use,
        traits=traits_to_use,
    )

    print("Computing global (persona-averaged) trait vectors...")
    global_trait_vecs = compute_global_trait_vectors(
        per_persona_vecs, personas_to_use, traits_to_use,
    )

    # Print trait vector norms
    print("\nGlobal trait vector norms at analysis layer:")
    for trait in traits_to_use:
        norm = global_trait_vecs[analysis_layer][trait].norm().item()
        print(f"  {trait:20s}  {norm:.4f}")

    # Free trait examples from memory
    del trait_examples, trait_by_layer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ==================================================================
    # Phase 2: Collect unsteered hidden states per persona
    # ==================================================================
    print("\n" + "=" * 60)
    print("Phase 2: Collecting unsteered hidden states per persona")
    print("=" * 60)

    # Build examples using each persona's base prompt (no trait modifier)
    base_personas_dict = {p: PERSONA_PROMPTS[p] for p in personas_to_use}
    base_examples = build_examples(
        tokenizer=tokenizer,
        personas=base_personas_dict,
        questions=questions,
        max_length=512,
    )
    print(f"Built {len(base_examples)} base (unsteered) examples")

    # Collect unsteered hidden states at the analysis layer
    print("Collecting unsteered hidden states...")
    unsteered_by_layer = collect_hidden_vectors(
        model=model,
        examples=base_examples,
        device=device,
        dtype=model.dtype,
        layer_indices=[analysis_layer],
    )

    # Organize unsteered hidden states by persona
    # base_examples are ordered: persona_0 x questions, persona_1 x questions, ...
    n_questions = len(questions)
    unsteered_by_persona: Dict[str, torch.Tensor] = {}
    for i, persona in enumerate(personas_to_use):
        start = i * n_questions
        end = start + n_questions
        vecs = unsteered_by_layer[analysis_layer][start:end]
        unsteered_by_persona[persona] = torch.stack(vecs, dim=0)  # [n_q, d]

    # Also organize examples by persona for steering
    examples_by_persona: Dict[str, list] = {}
    for i, persona in enumerate(personas_to_use):
        start = i * n_questions
        end = start + n_questions
        examples_by_persona[persona] = base_examples[start:end]

    # Project unsteered states onto trait basis
    trait_vecs_at_layer = global_trait_vecs[analysis_layer]
    unsteered_coords_by_persona: Dict[str, torch.Tensor] = {}
    for persona in personas_to_use:
        coords = project_onto_trait_basis(
            unsteered_by_persona[persona], trait_vecs_at_layer, traits_to_use,
        )  # [n_q, n_traits]
        unsteered_coords_by_persona[persona] = coords
        mean_coords = coords.mean(dim=0)
        print(f"  {persona:20s} unsteered mean coords: "
              f"[{', '.join(f'{v:.3f}' for v in mean_coords.tolist())}]")

    # ==================================================================
    # Phase 3: Steering and coupling measurement
    # ==================================================================
    print("\n" + "=" * 60)
    print("Phase 3: Measuring cross-trait coupling")
    print("=" * 60)

    alpha = args.steering_alpha
    print(f"Steering alpha: {alpha}")

    # coupling[persona][steer_trait][measure_trait] = mean shift
    coupling: Dict[str, Dict[str, Dict[str, float]]] = {}

    for persona in tqdm(personas_to_use, desc="Personas"):
        coupling[persona] = {}
        persona_examples = examples_by_persona[persona]
        unsteered_coords = unsteered_coords_by_persona[persona]  # [n_q, n_traits]

        for steer_trait in tqdm(traits_to_use, desc=f"  {persona} traits", leave=False):
            steer_vec = trait_vecs_at_layer[steer_trait]

            # Collect steered hidden states
            steered_hidden = collect_steered_hidden(
                model=model,
                tokenizer=tokenizer,
                examples=persona_examples,
                device=device,
                dtype=model.dtype,
                probe_layer=analysis_layer,
                steer_layer=analysis_layer,
                steering_vector=steer_vec,
                alpha=alpha,
            )  # [n_q, d]

            # Project steered states onto trait basis
            steered_coords = project_onto_trait_basis(
                steered_hidden, trait_vecs_at_layer, traits_to_use,
            )  # [n_q, n_traits]

            # Compute coupling: mean shift in each trait coordinate
            shift = (steered_coords - unsteered_coords).mean(dim=0)  # [n_traits]

            coupling[persona][steer_trait] = {}
            for j, measure_trait in enumerate(traits_to_use):
                coupling[persona][steer_trait][measure_trait] = shift[j].item()

            # Print diagonal (self-coupling)
            self_val = coupling[persona][steer_trait][steer_trait]
            print(f"    steer={steer_trait:20s}  self_coupling={self_val:+.4f}")

    # ==================================================================
    # Phase 4: Global coupling matrix (averaged over personas)
    # ==================================================================
    print("\n" + "=" * 60)
    print("Phase 4: Building coupling matrices")
    print("=" * 60)

    n_traits = len(traits_to_use)
    n_personas = len(personas_to_use)

    # Global coupling matrix: [n_traits, n_traits]
    # coupling_matrix[i][j] = mean over personas of coupling when steering
    # trait i, measured on trait j
    global_coupling = np.zeros((n_traits, n_traits))
    for i, steer_trait in enumerate(traits_to_use):
        for j, measure_trait in enumerate(traits_to_use):
            vals = [
                coupling[persona][steer_trait][measure_trait]
                for persona in personas_to_use
            ]
            global_coupling[i, j] = np.mean(vals)

    # Save global coupling matrix
    coupling_df = pd.DataFrame(
        global_coupling,
        index=traits_to_use,
        columns=traits_to_use,
    )
    coupling_df.to_csv(outdir / "global_coupling_matrix.csv")
    print(f"Saved global_coupling_matrix.csv")

    # Print global coupling matrix
    print("\nGlobal coupling matrix (rows=steered trait, cols=measured trait):")
    print(coupling_df.to_string(float_format=lambda x: f"{x:+.4f}"))

    # Plot global coupling heatmap
    # Determine symmetric color range
    vmax = max(abs(global_coupling.min()), abs(global_coupling.max()))
    plot_heatmap(
        global_coupling,
        row_labels=traits_to_use,
        col_labels=traits_to_use,
        title=f"Global coupling matrix (alpha={alpha}, layer={analysis_layer})",
        outpath=outdir / "global_coupling_matrix.png",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        fmt="+.3f",
    )
    print("Saved global_coupling_matrix.png")

    # ==================================================================
    # Phase 5: Per-persona coupling matrices
    # ==================================================================
    per_persona_coupling: Dict[str, np.ndarray] = {}
    for persona in personas_to_use:
        mat = np.zeros((n_traits, n_traits))
        for i, steer_trait in enumerate(traits_to_use):
            for j, measure_trait in enumerate(traits_to_use):
                mat[i, j] = coupling[persona][steer_trait][measure_trait]
        per_persona_coupling[persona] = mat

        # Save per-persona heatmap
        p_vmax = max(abs(mat.min()), abs(mat.max())) if mat.max() != mat.min() else 1.0
        plot_heatmap(
            mat,
            row_labels=traits_to_use,
            col_labels=traits_to_use,
            title=f"Coupling matrix: {persona} (alpha={alpha})",
            outpath=outdir / f"coupling_{persona}.png",
            cmap="RdBu_r",
            vmin=-p_vmax,
            vmax=p_vmax,
            fmt="+.3f",
        )

    print(f"Saved {len(personas_to_use)} per-persona coupling heatmaps")

    # ==================================================================
    # Phase 5b: Per-persona variance in coupling
    # ==================================================================
    # For each (steer_trait, measure_trait), compute variance across personas
    variance_matrix = np.zeros((n_traits, n_traits))
    for i, steer_trait in enumerate(traits_to_use):
        for j, measure_trait in enumerate(traits_to_use):
            vals = [
                coupling[persona][steer_trait][measure_trait]
                for persona in personas_to_use
            ]
            variance_matrix[i, j] = np.var(vals)

    variance_df = pd.DataFrame(
        variance_matrix,
        index=traits_to_use,
        columns=traits_to_use,
    )
    variance_df.to_csv(outdir / "coupling_persona_variance.csv")
    print("Saved coupling_persona_variance.csv")

    # ==================================================================
    # Phase 6: SVD of global coupling matrix
    # ==================================================================
    print("\n" + "=" * 60)
    print("Phase 6: SVD of global coupling matrix")
    print("=" * 60)

    coupling_tensor = torch.tensor(global_coupling, dtype=torch.float32)
    S, variance, cumvar = svd_analysis(coupling_tensor)

    print("\nSingular values:")
    for k in range(len(S)):
        print(f"  Component {k+1}: S={S[k].item():.4f}  "
              f"var={variance[k].item():.4f}  "
              f"cumvar={cumvar[k].item():.4f}")

    plot_svd_spectrum(
        S,
        title=f"SVD spectrum of global coupling matrix (layer={analysis_layer})",
        outpath=outdir / "coupling_svd_spectrum.png",
    )
    print("Saved coupling_svd_spectrum.png")

    # ==================================================================
    # Phase 7: PCA of per-persona coupling matrices
    # ==================================================================
    print("\n" + "=" * 60)
    print("Phase 7: PCA of per-persona coupling patterns")
    print("=" * 60)

    # Flatten each persona's coupling matrix into a vector
    flattened = []
    persona_labels = []
    for persona in personas_to_use:
        flattened.append(per_persona_coupling[persona].flatten())
        persona_labels.append(persona)

    flattened_matrix = np.stack(flattened, axis=0)  # [n_personas, n_traits^2]

    if len(personas_to_use) >= 2:
        n_components = min(2, len(personas_to_use), flattened_matrix.shape[1])
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(flattened_matrix)

        if n_components >= 2:
            plot_pca_scatter(
                coords=coords,
                labels=persona_labels,
                title=f"PCA of per-persona coupling matrices (alpha={alpha})",
                outpath=outdir / "coupling_pca.png",
                label_points=True,
            )
            print("Saved coupling_pca.png")
            print(f"PCA explained variance: "
                  f"PC1={pca.explained_variance_ratio_[0]:.3f}, "
                  f"PC2={pca.explained_variance_ratio_[1]:.3f}")
        else:
            # Only 1 component possible, save a simple bar chart
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(range(len(persona_labels)), coords[:, 0], color="steelblue")
            ax.set_xticks(range(len(persona_labels)))
            ax.set_xticklabels(persona_labels, rotation=45, ha="right")
            ax.set_ylabel("PC1")
            ax.set_title("PCA of per-persona coupling matrices (1 component)")
            plt.tight_layout()
            plt.savefig(outdir / "coupling_pca.png", dpi=180)
            plt.close()
            print("Saved coupling_pca.png (1D)")
    else:
        print("Skipping PCA: need at least 2 personas")

    # ==================================================================
    # Save run config
    # ==================================================================
    config = {
        "experiment": "oq1_coupling_coefficients",
        "model_name": args.model_name,
        "device": str(device),
        "seed": args.seed,
        "num_layers": num_layers,
        "analysis_layer": analysis_layer,
        "layer_stride": args.layer_stride,
        "steering_alpha": alpha,
        "personas": personas_to_use,
        "traits": traits_to_use,
        "n_personas": len(personas_to_use),
        "n_traits": len(traits_to_use),
        "n_questions": len(questions),
        "limit_personas": args.limit_personas,
        "limit_questions": args.limit_questions,
    }
    save_run_config(config, outdir)
    init_wandb("oq1_coupling_coefficients", config)

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")

    # Diagonal (self-coupling) vs off-diagonal
    diag = np.diag(global_coupling)
    off_diag_mask = ~np.eye(n_traits, dtype=bool)
    off_diag = global_coupling[off_diag_mask]

    print(f"\nGlobal coupling matrix diagnostics:")
    print(f"  Mean diagonal (self-coupling):     {diag.mean():+.4f}")
    print(f"  Mean |diagonal|:                   {np.abs(diag).mean():.4f}")
    print(f"  Mean off-diagonal:                 {off_diag.mean():+.4f}")
    print(f"  Mean |off-diagonal|:               {np.abs(off_diag).mean():.4f}")
    print(f"  Ratio |diag|/|off-diag|:           {np.abs(diag).mean() / (np.abs(off_diag).mean() + 1e-10):.2f}")

    print(f"\nSVD effective rank (95% variance): ", end="")
    if (cumvar >= 0.95).any():
        rank = int((cumvar >= 0.95).nonzero(as_tuple=True)[0][0].item()) + 1
    else:
        rank = len(cumvar)
    print(f"{rank}")

    print(f"\nStrongest off-diagonal couplings:")
    for i in range(n_traits):
        for j in range(n_traits):
            if i != j and abs(global_coupling[i, j]) > np.abs(off_diag).mean():
                print(f"  steer {traits_to_use[i]:20s} -> {traits_to_use[j]:20s}: "
                      f"{global_coupling[i, j]:+.4f}")

    finish_wandb(outdir)
    print(f"\nAll outputs saved to: {outdir.resolve()}")


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OQ1: Coupling Coefficients -- measure cross-trait coupling "
                    "under steering",
    )
    parser.add_argument(
        "--model-name", type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--outdir", type=str,
        default="outputs/oq1_coupling_coefficients",
        help="Directory for output files",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (auto-detected if omitted)",
    )
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument(
        "--steering-alpha", type=float, default=5.0,
        help="Steering coefficient alpha",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
