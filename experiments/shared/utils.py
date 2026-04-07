"""Shared utilities: metrics, plotting, SVD helpers, W&B integration, model loading."""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---- W&B helpers ----

WANDB_PROJECT = "personas_original"


def init_wandb(experiment_name: str, config: dict) -> None:
    """Initialize a W&B run for an experiment."""
    model_tag = config.get("model_name", "unknown").split("/")[-1]
    wandb.init(
        project=WANDB_PROJECT,
        name=f"{model_tag}/{experiment_name}",
        config=config,
        tags=[f"model:{model_tag}", f"experiment:{experiment_name}"],
    )


def log_wandb_metrics(metrics: dict, step: Optional[int] = None) -> None:
    """Log scalar metrics to the active W&B run."""
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def log_wandb_image(key: str, path: Path) -> None:
    """Log a single image to W&B."""
    if wandb.run is not None and path.exists():
        wandb.log({key: wandb.Image(str(path))})


def log_wandb_table(key: str, df: pd.DataFrame) -> None:
    """Log a DataFrame as a W&B table."""
    if wandb.run is not None:
        wandb.log({key: wandb.Table(dataframe=df)})


def finish_wandb(outdir: Path) -> None:
    """Log all PNGs and CSVs from outdir, then finish the W&B run."""
    if wandb.run is None:
        return
    for png in sorted(outdir.glob("*.png")):
        wandb.log({png.stem: wandb.Image(str(png))})
    for csv_file in sorted(outdir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file)
            wandb.log({csv_file.stem: wandb.Table(dataframe=df)})
        except Exception:
            pass
    wandb.finish()


# ---- Model helpers ----


def load_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    hf_token: Optional[str] = None,
):
    """Load model and tokenizer with correct dtype for the device."""
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        model_name, token=hf_token, **model_kwargs,
    )
    if device.type in {"cpu", "mps"}:
        model.to(device)

    return model, tokenizer


def get_num_layers(model) -> int:
    """Get number of hidden layers, handling nested configs (e.g. Gemma 4)."""
    if hasattr(model.config, "text_config"):
        return model.config.text_config.num_hidden_layers
    return model.config.num_hidden_layers


def cohens_d(group1: torch.Tensor, group2: torch.Tensor) -> float:
    """
    Compute Cohen's d effect size between two groups.
    group1, group2: [n, d] tensors
    Uses pooled standard deviation on the L2 norms.
    """
    norms1 = group1.norm(dim=-1) if group1.dim() > 1 else group1
    norms2 = group2.norm(dim=-1) if group2.dim() > 1 else group2

    n1, n2 = len(norms1), len(norms2)
    mean1, mean2 = norms1.mean().item(), norms2.mean().item()
    var1, var2 = norms1.var().item(), norms2.var().item()

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (mean1 - mean2) / pooled_std


def cohens_d_multivariate(group1: torch.Tensor, group2: torch.Tensor) -> float:
    """
    Multivariate Cohen's d using cosine distance from centroid.
    group1, group2: [n, d] tensors
    """
    centroid1 = group1.mean(dim=0)
    centroid2 = group2.mean(dim=0)
    diff = centroid1 - centroid2

    # Pool covariance using traces (simplified)
    var1 = (group1 - centroid1).pow(2).sum(dim=-1).mean().item()
    var2 = (group2 - centroid2).pow(2).sum(dim=-1).mean().item()
    pooled_var = (var1 + var2) / 2

    if pooled_var < 1e-10:
        return 0.0
    return (diff.norm().item()) / np.sqrt(pooled_var)


def svd_analysis(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Centered SVD with variance analysis.
    matrix: [n, d]
    Returns: (singular_values, variance_explained, cumulative_variance)
    """
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    U, S, V = torch.linalg.svd(centered, full_matrices=False)
    variance = (S ** 2) / (S ** 2).sum()
    cumvar = torch.cumsum(variance, dim=0)
    return S, variance, cumvar


def effective_rank(singular_values: torch.Tensor, threshold: float = 0.95) -> int:
    """Number of singular values needed to capture `threshold` of variance."""
    variance = (singular_values ** 2) / (singular_values ** 2).sum()
    cumvar = torch.cumsum(variance, dim=0)
    return int((cumvar >= threshold).nonzero(as_tuple=True)[0][0].item()) + 1


def plot_heatmap(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    outpath: Path,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    fmt: str = ".2f",
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """Generic annotated heatmap."""
    if figsize is None:
        figsize = (max(8, len(col_labels) * 0.8 + 2), max(5, len(row_labels) * 0.5 + 1))
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            color = "white" if abs(val - (vmin or matrix.min())) > 0.6 * ((vmax or matrix.max()) - (vmin or matrix.min())) else "black"
            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", color=color, fontsize=8)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_svd_spectrum(
    singular_values: torch.Tensor,
    title: str,
    outpath: Path,
    cumulative: bool = True,
) -> None:
    """Plot singular value spectrum with optional cumulative variance."""
    S = singular_values.numpy()
    variance = S**2 / (S**2).sum()
    cumvar = np.cumsum(variance)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(1, len(S) + 1)

    ax1.bar(x, variance, alpha=0.7, color="steelblue", label="Variance explained")
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Variance explained", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    if cumulative:
        ax2 = ax1.twinx()
        ax2.plot(x, cumvar, "r-o", markersize=4, label="Cumulative")
        ax2.axhline(0.95, color="gray", linestyle="--", alpha=0.5)
        ax2.set_ylabel("Cumulative variance", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        ax2.set_ylim(0, 1.05)

    ax1.set_title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_pca_scatter(
    coords: np.ndarray,
    labels: List[str],
    colors: Optional[List[str]] = None,
    title: str = "PCA",
    outpath: Optional[Path] = None,
    label_points: bool = False,
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.7,
) -> None:
    """2D scatter plot with categorical coloring."""
    unique_labels = sorted(set(labels))
    if colors is None:
        cmap = plt.cm.get_cmap("tab10" if len(unique_labels) <= 10 else "tab20")
        color_map = {l: cmap(i / len(unique_labels)) for i, l in enumerate(unique_labels)}
    else:
        color_map = dict(zip(unique_labels, colors))

    fig, ax = plt.subplots(figsize=figsize)
    for label in unique_labels:
        mask = [l == label for l in labels]
        pts = coords[mask]
        ax.scatter(pts[:, 0], pts[:, 1], label=label, alpha=alpha, s=40, color=color_map[label])
        if label_points:
            for pt in pts:
                ax.annotate(label, pt, fontsize=6, alpha=0.6)

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=180)
    plt.close()


def save_run_config(config: dict, outdir: Path) -> None:
    """Save run configuration as JSON."""
    with open(outdir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)
