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
#   "scipy>=1.12",
#   "wandb>=0.16",
#   "tqdm>=4.66",
# ]
# ///

"""
Prediction 2 — Basin Transitions
=================================
Tests whether adversarial steering causes directional reorientation with
discrete transition moments.  We steer a source persona (default: pirate)
toward a target persona (default: lawyer) at increasing alpha values and
track:
  (a) angular displacement (arccos of cosine similarity) vs alpha,
  (b) hidden-state norm vs alpha,
  (c) sigmoid fit to the angle curve (detecting transition sharpness),
  (d) SVD dimensionality of the full transition manifold,
  (e) PCA trajectory in the global persona space (basin-hopping).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Path setup so we can import from the repo root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

from main import (
    PERSONAS,
    QUESTIONS,
    Example,
    build_examples,
    collect_hidden_vectors,
    infer_device,
    pca_2d_torch,
    sample_layers,
    set_seed,
)
from tqdm import tqdm
from steer import (
    SteeringHook,
    collect_steered_hidden,
    compute_steering_vectors,
    get_layer_module,
)
from shared.utils import finish_wandb, get_num_layers, init_wandb, load_model_and_tokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALPHAS = [0, 1, 2, 3, 5, 8, 10, 15, 20, 30]


# ---------------------------------------------------------------------------
# Sigmoid model for curve fitting
# ---------------------------------------------------------------------------


def sigmoid_model(x, a, b, c):
    """angle = a / (1 + exp(-b * (x - c)))"""
    return a / (1.0 + np.exp(-b * (x - c)))


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def plot_angle_vs_alpha(
    alphas: List[float],
    mean_angles: List[float],
    std_angles: List[float],
    sigmoid_params: dict | None,
    outpath: Path,
    source: str,
    target: str,
) -> None:
    """Plot angle-vs-alpha with optional sigmoid overlay."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        alphas, mean_angles, yerr=std_angles,
        fmt="o-", capsize=4, color="tab:blue", label="Mean angle",
    )

    if sigmoid_params is not None:
        x_smooth = np.linspace(min(alphas), max(alphas), 200)
        y_fit = sigmoid_model(
            x_smooth,
            sigmoid_params["a"],
            sigmoid_params["b"],
            sigmoid_params["c"],
        )
        ax.plot(x_smooth, y_fit, "--", color="tab:red", linewidth=2,
                label=(f"Sigmoid fit (a={sigmoid_params['a']:.1f}, "
                       f"b={sigmoid_params['b']:.2f}, "
                       f"c={sigmoid_params['c']:.1f})"))

    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Angle from unsteered (degrees)")
    ax.set_title(f"Angular displacement: {source} -> {target}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_norm_vs_alpha(
    alphas: List[float],
    mean_norms: List[float],
    std_norms: List[float],
    outpath: Path,
    source: str,
    target: str,
) -> None:
    """Plot hidden-state norm vs alpha."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        alphas, mean_norms, yerr=std_norms,
        fmt="s-", capsize=4, color="tab:green", label="Mean norm",
    )
    ax.set_xlabel("Steering alpha")
    ax.set_ylabel("Hidden-state L2 norm")
    ax.set_title(f"Hidden-state norm: {source} steered toward {target}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_pca_trajectories(
    all_pca_coords: np.ndarray,
    all_labels: List[str],
    trajectory_coords: np.ndarray,
    trajectory_alphas: List[float],
    outpath: Path,
    source: str,
    target: str,
) -> None:
    """PCA scatter of all personas + steered trajectory as a connected line."""
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_labels = sorted(set(all_labels))
    cmap = plt.cm.get_cmap("tab10" if len(unique_labels) <= 10 else "tab20")
    color_map = {l: cmap(i / max(len(unique_labels), 1)) for i, l in enumerate(unique_labels)}

    # Plot unsteered persona clusters
    for label in unique_labels:
        mask = np.array([l == label for l in all_labels])
        pts = all_pca_coords[mask]
        ax.scatter(pts[:, 0], pts[:, 1], label=label, alpha=0.5, s=40,
                   color=color_map[label], edgecolors="none")

    # Plot steered trajectory as connected line with markers
    ax.plot(
        trajectory_coords[:, 0], trajectory_coords[:, 1],
        "k-", linewidth=2, alpha=0.8, zorder=5,
    )
    scatter = ax.scatter(
        trajectory_coords[:, 0], trajectory_coords[:, 1],
        c=trajectory_alphas, cmap="coolwarm", s=80, edgecolors="black",
        linewidths=1.0, zorder=6,
    )
    cbar = plt.colorbar(scatter, ax=ax, label="Steering alpha")

    # Annotate start and end
    ax.annotate(
        f"alpha={trajectory_alphas[0]}",
        trajectory_coords[0], fontsize=8,
        xytext=(10, 10), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    ax.annotate(
        f"alpha={trajectory_alphas[-1]}",
        trajectory_coords[-1], fontsize=8,
        xytext=(10, -15), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    ax.legend(bbox_to_anchor=(1.25, 1), loc="upper left", fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Transition trajectory: {source} -> {target} (PCA space)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = infer_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    source_persona = args.source_persona
    target_persona = args.target_persona
    alphas = ALPHAS

    hf_token = os.environ.get("HF_TOKEN")

    # ---- Load model and tokenizer ----
    print(f"\nLoading model: {args.model_name} on {device}...")
    model, tokenizer = load_model_and_tokenizer(args.model_name, device, hf_token)
    num_layers = get_num_layers(model)

    dtype = model.dtype
    layer_indices = sample_layers(num_layers=num_layers, stride=args.layer_stride)

    questions = QUESTIONS[: args.limit_questions] if args.limit_questions > 0 else QUESTIONS
    n_questions = len(questions)

    print(f"Layers: {layer_indices}")
    print(f"Questions: {n_questions}")
    print(f"Source persona: {source_persona}")
    print(f"Target persona: {target_persona}")
    print(f"Alphas: {alphas}")

    # ================================================================
    # 1. Build examples for ALL personas (needed for steering vectors)
    # ================================================================
    print("\nBuilding examples for all personas...")
    all_examples = build_examples(tokenizer, PERSONAS, questions, max_length=512)
    all_persona_names = [ex.persona_name for ex in all_examples]
    all_persona_vocab = sorted(set(all_persona_names))

    # Build source-persona-only examples (for steering)
    source_personas = {source_persona: PERSONAS[source_persona]}
    pirate_examples = build_examples(tokenizer, source_personas, questions, max_length=512)

    # ================================================================
    # 2. Collect unsteered hidden states at all sampled layers
    # ================================================================
    print("\nCollecting unsteered hidden states (all personas)...")
    by_layer_all = collect_hidden_vectors(model, all_examples, device, dtype, layer_indices)

    # ================================================================
    # 3. Compute steering vectors
    # ================================================================
    print("\nComputing steering vectors...")
    steering_vectors = compute_steering_vectors(
        by_layer_all, all_persona_names, all_persona_vocab, baseline="mean",
    )

    # ================================================================
    # 4. Pick best (deepest) layer for steering
    # ================================================================
    best_layer = max(layer_indices)
    sv = steering_vectors[best_layer][target_persona]
    print(f"\nUsing layer {best_layer} for steering")
    print(f"Steering vector norm: {sv.norm().item():.4f}")

    # ================================================================
    # 5. Extract unsteered pirate hidden states at the best layer
    # ================================================================
    pirate_mask = [n == source_persona for n in all_persona_names]
    all_vecs_best_layer = torch.stack(by_layer_all[best_layer], dim=0)
    pirate_unsteered = all_vecs_best_layer[pirate_mask]
    print(f"Unsteered {source_persona} vectors: {pirate_unsteered.shape}")

    # ================================================================
    # 6. Sweep alphas: collect steered hidden states
    # ================================================================
    print("\nSweeping alphas...")
    angle_records: List[dict] = []
    norm_records: List[dict] = []
    all_steered_vecs: List[torch.Tensor] = []  # for SVD / PCA later
    all_steered_alphas: List[float] = []

    mean_angles: List[float] = []
    std_angles: List[float] = []
    mean_norms: List[float] = []
    std_norms: List[float] = []

    for alpha in tqdm(alphas, desc="Alpha sweep"):
        steered = collect_steered_hidden(
            model, tokenizer, pirate_examples, device, dtype,
            probe_layer=best_layer, steer_layer=best_layer,
            steering_vector=sv, alpha=alpha,
        )

        # Angle: arccos(cosine_similarity)
        cos_sim = F.cosine_similarity(pirate_unsteered, steered, dim=-1)
        angles_deg = torch.acos(cos_sim.clamp(-1.0, 1.0)) * (180.0 / np.pi)

        # Norm
        norms = steered.norm(dim=-1)

        mean_angle = angles_deg.mean().item()
        std_angle = angles_deg.std().item()
        mean_norm = norms.mean().item()
        std_norm = norms.std().item()

        mean_angles.append(mean_angle)
        std_angles.append(std_angle)
        mean_norms.append(mean_norm)
        std_norms.append(std_norm)

        print(f"angle={mean_angle:.2f} +/- {std_angle:.2f} deg, "
              f"norm={mean_norm:.2f} +/- {std_norm:.2f}")

        for i in range(steered.shape[0]):
            angle_records.append({
                "alpha": alpha,
                "example_idx": i,
                "angle_deg": angles_deg[i].item(),
                "cos_sim": cos_sim[i].item(),
            })
            norm_records.append({
                "alpha": alpha,
                "example_idx": i,
                "norm": norms[i].item(),
            })

        all_steered_vecs.append(steered)
        all_steered_alphas.extend([alpha] * steered.shape[0])

    # ================================================================
    # 7. Sigmoid fit to mean angle vs alpha
    # ================================================================
    print("\nFitting sigmoid to angle-vs-alpha curve...")
    alpha_arr = np.array(alphas, dtype=np.float64)
    angle_arr = np.array(mean_angles, dtype=np.float64)

    sigmoid_params = None
    try:
        # Initial guesses: a = max angle, b = 0.5, c = midpoint alpha
        p0 = [angle_arr.max(), 0.5, alpha_arr[len(alpha_arr) // 2]]
        bounds = ([0, 0, 0], [360, 10, alpha_arr.max() * 2])
        popt, pcov = curve_fit(
            sigmoid_model, alpha_arr, angle_arr,
            p0=p0, bounds=bounds, maxfev=10000,
        )
        perr = np.sqrt(np.diag(pcov))
        sigmoid_params = {
            "a": float(popt[0]),
            "b": float(popt[1]),
            "c": float(popt[2]),
            "a_se": float(perr[0]),
            "b_se": float(perr[1]),
            "c_se": float(perr[2]),
        }
        # R-squared
        y_pred = sigmoid_model(alpha_arr, *popt)
        ss_res = np.sum((angle_arr - y_pred) ** 2)
        ss_tot = np.sum((angle_arr - angle_arr.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        sigmoid_params["r_squared"] = float(r_squared)

        # Transition sharpness: derivative at inflection point (alpha=c)
        # d(angle)/d(alpha) at alpha=c = a*b/4
        sigmoid_params["transition_sharpness"] = float(popt[0] * popt[1] / 4.0)

        print(f"  Sigmoid: a={popt[0]:.2f} (+/-{perr[0]:.2f}), "
              f"b={popt[1]:.3f} (+/-{perr[1]:.3f}), "
              f"c={popt[2]:.2f} (+/-{perr[2]:.2f})")
        print(f"  R^2 = {r_squared:.4f}")
        print(f"  Transition sharpness (a*b/4) = {sigmoid_params['transition_sharpness']:.2f} deg/alpha")
    except (RuntimeError, ValueError) as exc:
        warnings.warn(f"Sigmoid fit failed: {exc}")
        print(f"  Sigmoid fit failed: {exc}")

    # Save sigmoid params
    sigmoid_df = pd.DataFrame([sigmoid_params] if sigmoid_params else [{"fit_failed": True}])
    sigmoid_df.to_csv(outdir / "sigmoid_fit_params.csv", index=False)

    # ================================================================
    # 8. SVD on the full transition manifold
    # ================================================================
    print("\nSVD analysis of transition manifold...")
    stacked_steered = torch.cat(all_steered_vecs, dim=0)  # [n_alphas * n_examples, d]
    centered = stacked_steered - stacked_steered.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

    variance_explained = (S ** 2) / (S ** 2).sum()
    cumvar = torch.cumsum(variance_explained, dim=0)

    # Effective rank at various thresholds
    svd_rows = []
    for threshold in [0.80, 0.90, 0.95, 0.99]:
        rank = int((cumvar >= threshold).nonzero(as_tuple=True)[0][0].item()) + 1
        svd_rows.append({"threshold": threshold, "effective_rank": rank})
        print(f"  {threshold:.0%} variance captured by {rank} components")

    svd_df = pd.DataFrame(svd_rows)
    svd_df.to_csv(outdir / "transition_svd_ranks.csv", index=False)

    # ================================================================
    # 9. PCA: project all personas + steered trajectory
    # ================================================================
    print("\nPCA visualization...")

    # Collect all persona unsteered vecs
    all_unsteered = all_vecs_best_layer  # [n_all_examples, d]
    # Per-alpha steered means (trajectory centroids)
    trajectory_vecs = torch.stack(
        [v.mean(dim=0) for v in all_steered_vecs], dim=0,
    )  # [n_alphas, d]

    # Combine for PCA
    combined = torch.cat([all_unsteered, stacked_steered], dim=0)  # [N_total, d]
    combined_np = combined.numpy()

    # Fit PCA on everything
    pca = PCA(n_components=2)
    all_coords = pca.fit_transform(combined_np)

    n_unsteered = all_unsteered.shape[0]
    unsteered_coords = all_coords[:n_unsteered]
    steered_coords = all_coords[n_unsteered:]

    # Build trajectory coords (means per alpha)
    trajectory_coords_list = []
    offset = 0
    for v in all_steered_vecs:
        n = v.shape[0]
        chunk = steered_coords[offset: offset + n]
        trajectory_coords_list.append(chunk.mean(axis=0))
        offset += n
    trajectory_coords = np.stack(trajectory_coords_list, axis=0)

    plot_pca_trajectories(
        all_pca_coords=unsteered_coords,
        all_labels=all_persona_names,
        trajectory_coords=trajectory_coords,
        trajectory_alphas=alphas,
        outpath=outdir / "transition_trajectories_pca.png",
        source=source_persona,
        target=target_persona,
    )

    # ================================================================
    # 10. Plots
    # ================================================================
    print("\nGenerating plots...")

    plot_angle_vs_alpha(
        alphas, mean_angles, std_angles, sigmoid_params,
        outpath=outdir / "angle_vs_alpha_curves.png",
        source=source_persona, target=target_persona,
    )

    plot_norm_vs_alpha(
        alphas, mean_norms, std_norms,
        outpath=outdir / "norm_vs_alpha.png",
        source=source_persona, target=target_persona,
    )

    # ================================================================
    # 11. Save run config
    # ================================================================
    config = {
        "model_name": args.model_name,
        "device": str(device),
        "seed": args.seed,
        "source_persona": source_persona,
        "target_persona": target_persona,
        "alphas": alphas,
        "best_layer": best_layer,
        "layer_indices": layer_indices,
        "layer_stride": args.layer_stride,
        "n_questions": n_questions,
        "num_layers": num_layers,
        "steering_vector_norm": float(sv.norm().item()),
        "sigmoid_params": sigmoid_params,
    }
    with open(outdir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)
    init_wandb("prediction_2_basin_transitions", config)

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'=' * 60}")
    print("Prediction 2: Basin Transitions -- Summary")
    print(f"{'=' * 60}")
    print(f"Source: {source_persona}, Target: {target_persona}")
    print(f"Layer: {best_layer}")
    print(f"Alphas tested: {alphas}")
    print(f"Angle range: {mean_angles[0]:.2f} -> {mean_angles[-1]:.2f} degrees")
    if sigmoid_params:
        print(f"Sigmoid inflection at alpha={sigmoid_params['c']:.2f}")
        print(f"Transition sharpness: {sigmoid_params['transition_sharpness']:.2f} deg/alpha")
        print(f"R^2: {sigmoid_params['r_squared']:.4f}")
    print(f"SVD effective rank (95%): {svd_df[svd_df['threshold'] == 0.95]['effective_rank'].iloc[0]}")
    finish_wandb(outdir)
    print(f"\nOutputs written to: {outdir.resolve()}")
    print("  - angle_vs_alpha_curves.png")
    print("  - norm_vs_alpha.png")
    print("  - sigmoid_fit_params.csv")
    print("  - transition_trajectories_pca.png")
    print("  - transition_svd_ranks.csv")
    print("  - run_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prediction 2: Basin Transitions — angular displacement under steering",
    )
    parser.add_argument(
        "--model-name", type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--outdir", type=str,
        default="outputs/prediction_2_basin_transitions",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--source-persona", type=str, default="pirate",
        help="Persona to steer FROM",
    )
    parser.add_argument(
        "--target-persona", type=str, default="lawyer",
        help="Persona to steer TOWARD",
    )
    parser.add_argument("--limit-questions", type=int, default=0)
    parser.add_argument("--layer-stride", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
