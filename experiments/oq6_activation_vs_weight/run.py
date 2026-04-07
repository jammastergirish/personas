#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "torch>=2.2",
#   "transformers>=4.43",
#   "matplotlib>=3.8",
#   "pandas>=2.2",
#   "python-dotenv>=1.0",
#   "accelerate",
#   "numpy>=1.26",
#   "scikit-learn>=1.4",
#   "wandb>=0.16",
# ]
# ///

"""
OQ6: Activation vs Weight Space

Compares steering vectors (activation-space directions) to the SVD structure
of weight matrices at the same layer.  For each steering vector we measure its
alignment with the top-k right singular vectors of every weight matrix
(Q/K/V/O projections + MLP gate/up/down), revealing whether persona directions
live in the high-variance or low-variance subspace of the weights.

This is a FAST experiment — mostly matrix math on already-loaded weights.
No text generation is required.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# sys.path so we can import from the repo root and the shared experiments pkg
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from main import (
    PERSONAS,
    QUESTIONS,
    build_examples,
    collect_hidden_vectors,
    infer_device,
    sample_layers,
    set_seed,
)
from steer import compute_steering_vectors, get_layer_module
from shared.utils import finish_wandb, init_wandb, save_run_config


# ============================================================
# Weight matrix extraction
# ============================================================


def extract_weight_matrices(layer_module) -> Dict[str, torch.Tensor]:
    """
    Extract named weight matrices from a transformer layer.

    Returns a dict  name -> Tensor[out, in]  (float32, CPU).
    Works for Llama-style architectures; silently skips missing sub-modules.
    """
    matrices: Dict[str, torch.Tensor] = {}

    if hasattr(layer_module, "self_attn"):
        attn = layer_module.self_attn
        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if hasattr(attn, proj_name):
                matrices[proj_name] = getattr(attn, proj_name).weight.data.float().cpu()

    if hasattr(layer_module, "mlp"):
        mlp = layer_module.mlp
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            if hasattr(mlp, proj_name):
                matrices[proj_name] = getattr(mlp, proj_name).weight.data.float().cpu()

    return matrices


# ============================================================
# Alignment computation
# ============================================================


def compute_alignment_profile(
    W: torch.Tensor,
    steering_vec: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Compute the alignment of a steering vector with the SVD of a weight matrix.

    Parameters
    ----------
    W : Tensor [out_dim, in_dim]
        Weight matrix.
    steering_vec : Tensor [d]
        Steering vector.  Must live in the *input* space of W (dim = in_dim).

    Returns
    -------
    weighted_alignment : Tensor [k]
        sigma_k * |v_k . s_norm|^2  for each singular index k
    singular_values : Tensor [k]
        The singular values themselves (for reference / plotting).
    total_alignment : float
        sum(sigma_i * |v_i . s|^2) / (||s||^2 * sum(sigma_i))
    """
    in_dim = W.shape[1]
    sv = steering_vec[:in_dim]  # trim if necessary (should be exact for matching spaces)
    s_norm = F.normalize(sv.unsqueeze(0), dim=-1).squeeze(0)  # unit vector

    # Economy SVD:  W = U @ diag(S) @ Vh,  Vh is [min(out,in), in]
    _U, S, Vh = torch.linalg.svd(W, full_matrices=False)

    dots_sq = (Vh @ s_norm).pow(2)  # [min(out,in)]
    weighted = S * dots_sq           # sigma_k * |v_k . s|^2

    total = weighted.sum().item() / (S.sum().item() + 1e-12)
    return weighted, S, total


# ============================================================
# Plotting helpers
# ============================================================


def plot_alignment_by_singular_idx(
    records: List[dict],
    outpath: Path,
    top_k: int = 64,
) -> None:
    """
    Line plot: x = singular-value index, y = weighted alignment,
    one line per persona, averaged over all weight matrices whose
    input dimension matches the hidden size.
    """
    personas = sorted({r["persona"] for r in records})
    cmap = plt.cm.get_cmap("tab10" if len(personas) <= 10 else "tab20")

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, persona in enumerate(personas):
        # Gather profiles from records where the steering vec dimension matched
        profiles = [
            np.array(r["weighted_alignment"][:top_k])
            for r in records
            if r["persona"] == persona and r["dim_match"]
        ]
        if not profiles:
            continue
        # Pad to uniform length and average
        max_len = max(len(p) for p in profiles)
        padded = np.zeros((len(profiles), max_len))
        for j, p in enumerate(profiles):
            padded[j, : len(p)] = p
        mean_profile = padded.mean(axis=0)[:top_k]
        ax.plot(range(len(mean_profile)), mean_profile, label=persona,
                color=cmap(i / max(len(personas) - 1, 1)), alpha=0.8)

    ax.set_xlabel("Singular value index k")
    ax.set_ylabel(r"$\sigma_k \cdot |v_k \cdot \hat{s}|^2$  (mean over weight matrices)")
    ax.set_title("Steering-vector alignment by singular-value index")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_alignment_by_weight_matrix(
    df: pd.DataFrame,
    outpath: Path,
) -> None:
    """
    Grouped bar chart: x = weight matrix name, groups = personas,
    y = total alignment score.
    """
    matrices = sorted(df["weight_matrix"].unique())
    personas = sorted(df["persona"].unique())
    cmap = plt.cm.get_cmap("tab10" if len(personas) <= 10 else "tab20")

    x = np.arange(len(matrices))
    width = 0.8 / max(len(personas), 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, persona in enumerate(personas):
        vals = []
        for m in matrices:
            sub = df[(df["persona"] == persona) & (df["weight_matrix"] == m)]
            vals.append(sub["total_alignment"].values[0] if len(sub) else 0.0)
        ax.bar(x + i * width, vals, width, label=persona,
               color=cmap(i / max(len(personas) - 1, 1)), alpha=0.85)

    ax.set_xticks(x + width * len(personas) / 2)
    ax.set_xticklabels(matrices, rotation=30, ha="right")
    ax.set_ylabel("Total alignment (normalized)")
    ax.set_title("Steering-vector alignment with weight-matrix SVD, by matrix")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_steerability_vs_alignment(
    df: pd.DataFrame,
    outpath: Path,
) -> None:
    """
    Scatter: x = steering-vector norm (steerability proxy),
             y = mean total alignment across weight matrices.
    """
    personas = sorted(df["persona"].unique())
    cmap = plt.cm.get_cmap("tab10" if len(personas) <= 10 else "tab20")

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, persona in enumerate(personas):
        sub = df[df["persona"] == persona]
        mean_align = sub["total_alignment"].mean()
        sv_norm = sub["steering_vec_norm"].iloc[0]
        ax.scatter(sv_norm, mean_align,
                   color=cmap(i / max(len(personas) - 1, 1)),
                   s=80, zorder=3)
        ax.annotate(persona, (sv_norm, mean_align),
                    fontsize=7, textcoords="offset points", xytext=(5, 5))

    ax.set_xlabel("Steering vector norm (steerability proxy)")
    ax.set_ylabel("Mean alignment with weight SVD (normalized)")
    ax.set_title("Steerability vs. weight-space alignment")
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

    # ---- Load model & tokenizer ----
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model on {device}...")
    model_kwargs = {"output_hidden_states": True}
    if device.type == "cuda":
        model_kwargs["dtype"] = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        model_kwargs["device_map"] = "auto"
    elif device.type == "mps":
        model_kwargs["dtype"] = torch.float16
    else:
        model_kwargs["dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, token=hf_token, **model_kwargs
    )
    if device.type in {"cpu", "mps"}:
        model.to(device)

    if hasattr(model.config, "text_config"):
        num_layers = model.config.text_config.num_hidden_layers
    else:
        num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    layer_indices = sample_layers(num_layers=num_layers, stride=args.layer_stride)

    # ---- Build examples & collect activations ----
    personas = PERSONAS.copy()
    if args.limit_personas > 0:
        personas = dict(list(personas.items())[: args.limit_personas])

    questions = QUESTIONS[: args.limit_questions] if args.limit_questions > 0 else QUESTIONS
    examples = build_examples(
        tokenizer=tokenizer,
        personas=personas,
        questions=questions,
        max_length=512,
    )

    persona_names = [ex.persona_name for ex in examples]
    persona_vocab = sorted(set(persona_names))

    print(f"Built {len(examples)} prompts "
          f"({len(personas)} personas x {len(questions)} questions)")
    print(f"Sampled layers: {layer_indices}")

    print("\nCollecting hidden states for steering vectors...")
    by_layer = collect_hidden_vectors(
        model=model,
        examples=examples,
        device=device,
        dtype=model.dtype,
        layer_indices=layer_indices,
    )

    # ---- Compute steering vectors ----
    print("Computing persona steering vectors (baseline=mean)...")
    steering_vectors = compute_steering_vectors(
        by_layer, persona_names, persona_vocab, baseline="mean",
    )

    # ---- Pick analysis layer ----
    best_layer = args.layer if args.layer is not None else max(layer_indices)
    print(f"\nAnalysis layer: {best_layer}")

    if best_layer not in steering_vectors:
        raise ValueError(
            f"Layer {best_layer} not in sampled layers {layer_indices}. "
            "Pass --layer explicitly or adjust --layer-stride."
        )

    # ---- Extract weight matrices at the analysis layer ----
    print(f"Extracting weight matrices from layer {best_layer}...")
    layer_module = get_layer_module(model, best_layer)
    weight_matrices = extract_weight_matrices(layer_module)
    print(f"  Found {len(weight_matrices)} weight matrices: {list(weight_matrices.keys())}")
    for name, W in weight_matrices.items():
        print(f"    {name}: {list(W.shape)}")

    # ---- Compute alignment for every (persona, weight_matrix) pair ----
    print("\nComputing SVD alignment...")
    sv_at_layer = steering_vectors[best_layer]
    rows = []
    alignment_records = []  # for the per-index plot

    for persona in persona_vocab:
        sv = sv_at_layer[persona].float().cpu()
        sv_norm = sv.norm().item()
        print(f"  {persona:20s}  ||sv|| = {sv_norm:.4f}")

        for mat_name, W in weight_matrices.items():
            in_dim = W.shape[1]
            # Check whether steering vector lives in the input space of W
            dim_match = (sv.shape[0] == in_dim)

            if not dim_match:
                # For down_proj: input dim = intermediate_size != hidden_size.
                # Skip alignment computation — the spaces are incomparable.
                rows.append({
                    "persona": persona,
                    "weight_matrix": mat_name,
                    "total_alignment": float("nan"),
                    "steering_vec_norm": sv_norm,
                    "dim_match": False,
                    "sv_dim": sv.shape[0],
                    "weight_in_dim": in_dim,
                })
                alignment_records.append({
                    "persona": persona,
                    "weight_matrix": mat_name,
                    "weighted_alignment": [],
                    "dim_match": False,
                })
                print(f"    {mat_name:12s}: dim mismatch "
                      f"(sv={sv.shape[0]}, W_in={in_dim}) — skipped")
                continue

            weighted, S, total = compute_alignment_profile(W, sv)
            rows.append({
                "persona": persona,
                "weight_matrix": mat_name,
                "total_alignment": total,
                "steering_vec_norm": sv_norm,
                "dim_match": True,
                "sv_dim": sv.shape[0],
                "weight_in_dim": in_dim,
            })
            alignment_records.append({
                "persona": persona,
                "weight_matrix": mat_name,
                "weighted_alignment": weighted.numpy().tolist(),
                "dim_match": True,
            })
            print(f"    {mat_name:12s}: total_alignment = {total:.6f}")

    df = pd.DataFrame(rows)

    # ---- Save CSV ----
    csv_path = outdir / "trait_weight_alignment.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved alignment table to {csv_path}")

    # ---- Plots ----
    print("Generating plots...")

    # 1) Alignment by singular-value index
    plot_alignment_by_singular_idx(
        alignment_records,
        outdir / "trait_alignment_by_singular_idx.png",
    )

    # 2) Alignment by weight matrix (bar chart)
    df_matched = df[df["dim_match"]].copy()
    if len(df_matched) > 0:
        plot_alignment_by_weight_matrix(
            df_matched,
            outdir / "alignment_by_weight_matrix.png",
        )

    # 3) Steerability vs alignment scatter
    if len(df_matched) > 0:
        plot_steerability_vs_alignment(
            df_matched,
            outdir / "steerability_vs_alignment.png",
        )

    # ---- Save run config ----
    config = {
        "experiment": "oq6_activation_vs_weight",
        "model_name": args.model_name,
        "device": str(device),
        "seed": args.seed,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "analysis_layer": best_layer,
        "layer_indices": layer_indices,
        "n_personas": len(persona_vocab),
        "persona_vocab": persona_vocab,
        "n_questions": len(questions),
        "n_examples": len(examples),
        "weight_matrices": {
            name: list(W.shape) for name, W in weight_matrices.items()
        },
    }
    save_run_config(config, outdir)
    init_wandb("oq6_activation_vs_weight", config)
    finish_wandb(outdir)
    print(f"\nDone. Outputs in {outdir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OQ6: Activation vs Weight Space — "
        "align persona steering vectors with weight-matrix SVD structure",
    )
    parser.add_argument(
        "--model-name", type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--outdir", type=str,
        default="outputs/oq6_activation_vs_weight",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Layer to analyse (default: deepest sampled layer)",
    )
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument("--limit-personas", type=int, default=0)
    parser.add_argument("--limit-questions", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
