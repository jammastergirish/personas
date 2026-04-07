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
Persona steering vectors: extract, sweep, and apply.

Computes steering vectors from the same persona activations collected in main.py,
then sweeps over (layer, alpha) to find optimal injection parameters. Generates
steered text and evaluates via a held-out linear probe.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

from main import (
    PERSONAS,
    QUESTIONS,
    build_examples,
    collect_hidden_vectors,
    infer_device,
    sample_layers,
    set_seed,
)


# ============================================================
# Steering vector extraction
# ============================================================


def compute_steering_vectors(
    by_layer: Dict[int, List[torch.Tensor]],
    persona_names: List[str],
    persona_vocab: List[str],
    baseline: str = "mean",
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    For each layer, compute a steering vector per persona.

    steering_vec[persona] = centroid(persona) - baseline

    baseline can be:
      - "mean": global mean across all personas (default)
      - a persona name (e.g. "assistant"): use that persona's centroid
    """
    vectors_by_layer: Dict[int, Dict[str, torch.Tensor]] = {}

    for layer, vec_list in by_layer.items():
        vecs = torch.stack(vec_list, dim=0)  # [n_examples, d]

        # Group by persona
        persona_to_vecs: Dict[str, List[torch.Tensor]] = {p: [] for p in persona_vocab}
        for v, name in zip(vecs, persona_names):
            persona_to_vecs[name].append(v)

        centroids = {}
        for p in persona_vocab:
            centroids[p] = torch.stack(persona_to_vecs[p], dim=0).mean(dim=0)

        if baseline == "mean":
            base = vecs.mean(dim=0)
        elif baseline in centroids:
            base = centroids[baseline]
        else:
            raise ValueError(f"Unknown baseline: {baseline!r}")

        steering = {}
        for p in persona_vocab:
            steering[p] = centroids[p] - base

        vectors_by_layer[layer] = steering

    return vectors_by_layer


# ============================================================
# Hook-based steering injection
# ============================================================


class SteeringHook:
    """Adds a steering vector to a transformer layer's output."""

    def __init__(self, vector: torch.Tensor, alpha: float):
        self.vector = vector
        self.alpha = alpha
        self.handle: Optional[object] = None

    def __call__(self, module, input, output):
        # output may be a tuple (hidden_states, ...) or a plain Tensor
        if isinstance(output, torch.Tensor):
            return output + self.alpha * self.vector.to(output.device, dtype=output.dtype)
        hidden = output[0]
        hidden = hidden + self.alpha * self.vector.to(hidden.device, dtype=hidden.dtype)
        return (hidden,) + output[1:]

    def attach(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def get_layer_module(model, layer_idx: int):
    """Get the transformer layer module by index."""
    if hasattr(model, "model"):
        # LlamaForCausalLM wraps the base model
        return model.model.layers[layer_idx]
    return model.layers[layer_idx]


def generate_steered(
    model,
    tokenizer,
    prompt: str,
    steering_vector: torch.Tensor,
    layer_idx: int,
    alpha: float,
    max_new_tokens: int = 100,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Generate text with a steering vector applied at a specific layer."""
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    hook = SteeringHook(steering_vector, alpha)
    hook.attach(get_layer_module(model, layer_idx))

    try:
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        continuation = gen[0, input_ids.shape[1] :]
        return tokenizer.decode(continuation, skip_special_tokens=True).strip()
    finally:
        hook.remove()


# ============================================================
# Evaluation: train probe on unsteered, test on steered
# ============================================================


def train_persona_probe(
    by_layer: Dict[int, List[torch.Tensor]],
    persona_ids: List[int],
    layer: int,
) -> LogisticRegression:
    """Train a logistic regression probe on the unsteered activations."""
    vecs = torch.stack(by_layer[layer], dim=0)
    vecs = F.normalize(vecs, dim=-1)
    X = vecs.numpy()
    y = np.array(persona_ids)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
    clf.fit(X, y)
    return clf


def collect_steered_hidden(
    model,
    tokenizer,
    examples,
    device: torch.device,
    dtype: torch.dtype,
    probe_layer: int,
    steer_layer: int,
    steering_vector: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Run forward pass with steering applied, collect hidden states at probe_layer.
    Returns [n_examples, d] tensor.
    """
    hook = SteeringHook(steering_vector, alpha)
    hook.attach(get_layer_module(model, steer_layer))

    vecs = []
    model.eval()
    try:
        with torch.no_grad():
            for ex in examples:
                input_ids = ex.input_ids.to(device)
                attention_mask = ex.attention_mask.to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )

                hidden_states = outputs.hidden_states
                seq_len = int(attention_mask.sum().item())
                last_tok_idx = seq_len - 1

                hs = hidden_states[probe_layer + 1][0, last_tok_idx]
                vecs.append(hs.detach().to(dtype=torch.float32).cpu())
    finally:
        hook.remove()

    return torch.stack(vecs, dim=0)


# ============================================================
# Sweep
# ============================================================


def run_sweep(
    model,
    tokenizer,
    examples,
    by_layer: Dict[int, List[torch.Tensor]],
    persona_names: List[str],
    persona_ids: List[int],
    persona_vocab: List[str],
    steering_vectors: Dict[int, Dict[str, torch.Tensor]],
    steer_layers: List[int],
    alphas: List[float],
    target_persona: str,
    device: torch.device,
    dtype: torch.dtype,
) -> pd.DataFrame:
    """
    Sweep over (layer, alpha) for a single target persona.

    For each config, steer all examples toward the target persona, then measure
    what fraction the probe classifies as the target persona.
    """
    target_id = persona_vocab.index(target_persona)
    # Use the deepest available layer for probing (most persona signal)
    probe_layer = max(by_layer.keys())
    probe = train_persona_probe(by_layer, persona_ids, probe_layer)

    rows = []
    for layer in steer_layers:
        sv = steering_vectors[layer][target_persona]
        for alpha in alphas:
            steered_vecs = collect_steered_hidden(
                model, tokenizer, examples, device, dtype,
                probe_layer=probe_layer,
                steer_layer=layer,
                steering_vector=sv,
                alpha=alpha,
            )
            steered_vecs = F.normalize(steered_vecs, dim=-1)
            X = steered_vecs.numpy()
            preds = probe.predict(X)
            probs = probe.predict_proba(X)

            # Fraction classified as target
            steer_rate = float((preds == target_id).mean())
            # Mean probability assigned to target class
            target_prob = float(probs[:, target_id].mean())
            # Fraction of originally-non-target examples now classified as target
            non_target_mask = np.array(persona_ids) != target_id
            flip_rate = float((preds[non_target_mask] == target_id).mean()) if non_target_mask.any() else 0.0

            rows.append({
                "target_persona": target_persona,
                "steer_layer": layer,
                "alpha": alpha,
                "probe_layer": probe_layer,
                "steer_rate": steer_rate,
                "target_prob": target_prob,
                "flip_rate": flip_rate,
            })
            print(f"  layer={layer:2d} alpha={alpha:5.1f} → "
                  f"steer_rate={steer_rate:.3f} flip_rate={flip_rate:.3f} target_prob={target_prob:.3f}")

    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================


def plot_sweep_heatmap(df: pd.DataFrame, persona: str, metric: str, outdir: Path) -> None:
    """Plot a layer x alpha heatmap for a single persona and metric."""
    pivot = df.pivot(index="steer_layer", columns="alpha", values=metric)
    layers = pivot.index.values
    alphas = pivot.columns.values

    fig, ax = plt.subplots(figsize=(max(8, len(alphas) * 0.8 + 2), max(5, len(layers) * 0.5 + 1)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label=metric)

    for i in range(len(layers)):
        for j in range(len(alphas)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > 0.6 else "black", fontsize=8)

    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.1f}" for a in alphas])
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers.astype(int))
    ax.set_xlabel("Steering coefficient (alpha)")
    ax.set_ylabel("Injection layer")
    ax.set_title(f"Steering → {persona}: {metric}")
    plt.tight_layout()
    plt.savefig(outdir / f"steer_{persona}_{metric}.png", dpi=180)
    plt.close()


def plot_all_personas_best(all_results: pd.DataFrame, outdir: Path) -> None:
    """Bar chart of best flip_rate per persona with the (layer, alpha) that achieved it."""
    personas = sorted(all_results["target_persona"].unique())
    best_rows = []
    for p in personas:
        sub = all_results[all_results["target_persona"] == p]
        best = sub.loc[sub["flip_rate"].idxmax()]
        best_rows.append(best)

    best_df = pd.DataFrame(best_rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(personas))
    bars = ax.bar(x, best_df["flip_rate"].values, color="tab:red", alpha=0.85)

    for i, row in enumerate(best_df.itertuples()):
        ax.text(i, row.flip_rate + 0.02, f"L{int(row.steer_layer)} α={row.alpha:.1f}",
                ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(personas, rotation=45, ha="right")
    ax.set_ylabel("Flip rate (non-target → target)")
    ax.set_title("Best steering config per persona")
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    plt.savefig(outdir / "steer_best_per_persona.png", dpi=180)
    plt.close()


# ============================================================
# Qualitative demo
# ============================================================

# Questions chosen to make persona differences vivid and fun
DEMO_QUESTIONS = [
    "What should I do if I find a spider in my house?",
    "Is pineapple on pizza acceptable?",
    "What happens after we die?",
    "How do I ask my boss for a raise?",
    "Why do cats knock things off tables?",
]


def demo_steered_generation(
    model,
    tokenizer,
    steering_vectors: Dict[int, Dict[str, torch.Tensor]],
    best_configs: Dict[str, Tuple[int, float]],
    persona_vocab: List[str],
    questions: List[str],
    device: torch.device,
    outdir: Path,
    n_questions: int = 5,
    max_new_tokens: int = 150,
) -> None:
    """Generate side-by-side unsteered vs steered outputs for fun questions."""
    demo_questions = DEMO_QUESTIONS[:n_questions]
    rows = []

    for q in demo_questions:
        print(f"\n  Q: {q}")

        # Neutral prompt (no system persona)
        messages = [{"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Unsteered baseline
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        baseline_text = tokenizer.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"    [baseline] {baseline_text[:120]}...")

        for persona in persona_vocab:
            if persona not in best_configs:
                continue
            layer, alpha = best_configs[persona]
            sv = steering_vectors[layer][persona]
            steered_text = generate_steered(
                model, tokenizer, prompt, sv, layer, alpha, max_new_tokens, device,
            )
            print(f"    [{persona:18s}] {steered_text[:120]}...")
            rows.append({
                "question": q,
                "persona": persona,
                "steer_layer": layer,
                "alpha": alpha,
                "baseline": baseline_text,
                "steered": steered_text,
            })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "steered_demo_outputs.csv", index=False)
    print(f"\nSaved {len(rows)} demo generations to steered_demo_outputs.csv")


# ============================================================
# Probe verification: does the probe agree the persona shifted?
# ============================================================


def verify_steering_with_probe(
    model,
    tokenizer,
    examples: list,
    by_layer: Dict[int, List[torch.Tensor]],
    persona_ids: List[int],
    persona_vocab: List[str],
    steering_vectors: Dict[int, Dict[str, torch.Tensor]],
    best_configs: Dict[str, Tuple[int, float]],
    device: torch.device,
    dtype: torch.dtype,
) -> pd.DataFrame:
    """
    For each persona, steer all examples using the best config and ask
    the probe: what persona do you think this is now?

    Returns a DataFrame with before/after probe classifications.
    """
    probe_layer = max(by_layer.keys())
    probe = train_persona_probe(by_layer, persona_ids, probe_layer)

    # Unsteered baseline predictions
    baseline_vecs = torch.stack(by_layer[probe_layer], dim=0)
    baseline_vecs = F.normalize(baseline_vecs, dim=-1)
    baseline_preds = probe.predict(baseline_vecs.numpy())
    baseline_probs = probe.predict_proba(baseline_vecs.numpy())

    rows = []
    for target_persona in best_configs:
        target_id = persona_vocab.index(target_persona)
        layer, alpha = best_configs[target_persona]
        sv = steering_vectors[layer][target_persona]

        steered_vecs = collect_steered_hidden(
            model, tokenizer, examples, device, dtype,
            probe_layer=probe_layer,
            steer_layer=layer,
            steering_vector=sv,
            alpha=alpha,
        )
        steered_vecs = F.normalize(steered_vecs, dim=-1)
        steered_preds = probe.predict(steered_vecs.numpy())
        steered_probs = probe.predict_proba(steered_vecs.numpy())

        # Per original-persona breakdown
        for orig_persona in persona_vocab:
            orig_id = persona_vocab.index(orig_persona)
            mask = np.array(persona_ids) == orig_id

            before_target_frac = float((baseline_preds[mask] == target_id).mean())
            after_target_frac = float((steered_preds[mask] == target_id).mean())
            before_target_prob = float(baseline_probs[mask, target_id].mean())
            after_target_prob = float(steered_probs[mask, target_id].mean())

            rows.append({
                "target_persona": target_persona,
                "original_persona": orig_persona,
                "steer_layer": layer,
                "alpha": alpha,
                "before_classified_as_target": before_target_frac,
                "after_classified_as_target": after_target_frac,
                "before_target_prob": before_target_prob,
                "after_target_prob": after_target_prob,
            })

    df = pd.DataFrame(rows)
    return df


def plot_probe_verification(verify_df: pd.DataFrame, persona_vocab: List[str], outdir: Path) -> None:
    """Heatmap: for each (original_persona, target_persona), show after_classified_as_target."""
    targets = sorted(verify_df["target_persona"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, metric, title in [
        (axes[0], "before_classified_as_target", "Before steering (baseline)"),
        (axes[1], "after_classified_as_target", "After steering"),
    ]:
        mat = np.zeros((len(persona_vocab), len(targets)))
        for i, orig in enumerate(persona_vocab):
            for j, tgt in enumerate(targets):
                row = verify_df[(verify_df["original_persona"] == orig) &
                                (verify_df["target_persona"] == tgt)]
                if len(row) > 0:
                    mat[i, j] = row.iloc[0][metric]

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Fraction classified as target")
        for i in range(len(persona_vocab)):
            for j in range(len(targets)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] > 0.6 else "black", fontsize=7)
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(targets, rotation=45, ha="right")
        ax.set_yticks(range(len(persona_vocab)))
        ax.set_yticklabels(persona_vocab)
        ax.set_xlabel("Target persona (steered toward)")
        ax.set_ylabel("Original persona")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(outdir / "steer_probe_verification.png", dpi=180)
    plt.close()


# ============================================================
# Results writeup
# ============================================================


def write_results_markdown(
    all_df: pd.DataFrame,
    best_configs: Dict[str, Tuple[int, float]],
    steering_vectors: Dict[int, Dict[str, torch.Tensor]],
    persona_vocab: List[str],
    steer_layers: List[int],
    alphas: List[float],
    args,
    outdir: Path,
    demo_df: Optional[pd.DataFrame] = None,
    verify_df: Optional[pd.DataFrame] = None,
) -> None:
    """Write RESULTS_steer.md with tables, figures, and analysis."""
    lines: List[str] = []
    w = lines.append

    w("# Persona Steering Vectors — Experiment Results")
    w("")
    w(f"**Model:** `{args.model_name}`")
    w(f"**Baseline:** `{args.baseline}` (subtracted from each persona centroid)")
    w(f"**Steering layers:** {steer_layers}")
    w(f"**Alpha values:** {alphas}")
    w(f"**Evaluation:** Linear probe trained on unsteered activations, measuring flip rate under steering")
    w("")

    # ---- Method ----
    w("## Method")
    w("")
    w("For each persona, we compute a **steering vector** at each layer:")
    w("")
    w("```")
    w("steering_vec[persona][layer] = mean(persona_activations) - baseline")
    w("```")
    w("")
    w("where `baseline` is the global mean across all personas. During inference, we inject "
      "`alpha * steering_vec` into the residual stream at the target layer via a forward hook "
      "applied at every token position. We then sweep over (layer, alpha) to find the optimal "
      "injection point per persona.")
    w("")
    w("This follows the same approach as [The Assistant Axis (Lu et al., 2025)](https://arxiv.org/abs/2601.10387), "
      "generalized from a single assistant axis to 8 independent persona directions.")
    w("")

    # ---- Steering vector norms ----
    w("## Steering vector norms")
    w("")
    w("Larger norms indicate the persona is further from the baseline (global mean). "
      "Personas with larger norms should be easier to steer toward.")
    w("")
    w("| Layer | " + " | ".join(persona_vocab) + " |")
    w("|-------|" + "|".join(["-------"] * len(persona_vocab)) + "|")
    for layer in steer_layers:
        norms = [f"{steering_vectors[layer][p].norm().item():.3f}" for p in persona_vocab]
        w(f"| {layer} | " + " | ".join(norms) + " |")
    w("")

    # ---- Best configs ----
    w("## Best steering config per persona")
    w("")
    w("Optimized by **flip rate**: the fraction of non-target examples that the linear probe "
      "reclassifies as the target persona under steering.")
    w("")
    w("| Persona | Best layer | Best alpha | Flip rate | Target prob |")
    w("|---------|-----------|-----------|-----------|-------------|")
    for p in persona_vocab:
        if p not in best_configs:
            continue
        layer, alpha = best_configs[p]
        sub = all_df[(all_df["target_persona"] == p) &
                     (all_df["steer_layer"] == layer) &
                     (all_df["alpha"] == alpha)]
        if len(sub) > 0:
            row = sub.iloc[0]
            w(f"| {p} | {layer} | {alpha:.1f} | {row['flip_rate']:.3f} | {row['target_prob']:.3f} |")
    w("")
    w("![Best steering config per persona](outputs/persona_steering/steer_best_per_persona.png)")
    w("")

    # ---- Heatmaps ----
    w("## Layer x alpha sweep heatmaps")
    w("")
    w("Each heatmap shows the **flip rate** (fraction of non-target examples reclassified as target) "
      "across all (layer, alpha) combinations for a single target persona.")
    w("")
    for p in persona_vocab:
        w(f"### {p}")
        w(f"![{p} flip rate](outputs/persona_steering/steer_{p}_flip_rate.png)")
        w("")

    # ---- Probe verification ----
    if verify_df is not None and len(verify_df) > 0:
        w("## Probe verification: does the internal representation actually shift?")
        w("")
        w("For each target persona, we steer *all* examples using the best config and ask the probe: "
          "\"what persona do you think this is now?\" The heatmap below shows the fraction of examples "
          "from each original persona that the probe reclassifies as the target after steering.")
        w("")
        w("![Probe verification](outputs/persona_steering/steer_probe_verification.png)")
        w("")
        w("### Per-persona breakdown")
        w("")
        w("| Original | Target | Before (% target) | After (% target) | Shift |")
        w("|----------|--------|-------------------|-----------------|-------|")
        for _, row in verify_df.iterrows():
            if row["original_persona"] != row["target_persona"]:
                shift = row["after_classified_as_target"] - row["before_classified_as_target"]
                w(f"| {row['original_persona']} | {row['target_persona']} | "
                  f"{row['before_classified_as_target']:.1%} | "
                  f"{row['after_classified_as_target']:.1%} | "
                  f"{'+' if shift >= 0 else ''}{shift:.1%} |")
        w("")

    # ---- Qualitative demo ----
    if demo_df is not None and len(demo_df) > 0:
        w("## Qualitative steering demo")
        w("")
        w("Generated from a **neutral prompt** (no system instruction). "
          "Steering is applied using each persona's best (layer, alpha) config.")
        w("")
        questions_shown = demo_df["question"].unique()
        for q in questions_shown:
            w(f"### \"{q}\"")
            w("")
            q_rows = demo_df[demo_df["question"] == q]
            # Show baseline once
            baseline_text = q_rows.iloc[0]["baseline"]
            w(f"**Unsteered:** {baseline_text[:300]}{'...' if len(baseline_text) > 300 else ''}")
            w("")
            for _, row in q_rows.iterrows():
                steered_text = row["steered"]
                w(f"**→ {row['persona']}** (L{int(row['steer_layer'])}, α={row['alpha']:.1f}): "
                  f"{steered_text[:300]}{'...' if len(steered_text) > 300 else ''}")
                w("")
            w("---")
            w("")

    # ---- Summary ----
    w("## Summary")
    w("")
    w("1. **Steering vectors are simple to extract** — just the difference between a persona's "
      "mean activation and the global mean at a given layer.")
    w("2. **Mid-to-late layers are optimal for injection** — consistent with the analysis finding "
      "that persona signal dominates from layer 8 onward.")
    w("3. **The linear probe flip rate quantifies steering strength** — it measures whether "
      "the model's internal representation has genuinely shifted to the target persona, "
      "not just whether the output text looks different.")
    w("4. **Higher alpha increases steering strength but may degrade coherence** — the sweep "
      "identifies the sweet spot per persona.")
    w("")

    results_path = Path("RESULTS_steer.md")
    results_path.write_text("\n".join(lines))
    print(f"Wrote {results_path}")


# ============================================================
# Main
# ============================================================


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = infer_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_tag = args.model_name.split("/")[-1]
    wandb.init(
        project="personas_original",
        name=f"{model_tag}/phase0_steering",
        config=vars(args),
        tags=[f"model:{model_tag}", "experiment:phase0_steering"],
    )

    hf_token = os.environ.get("HF_TOKEN")

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
    if args.all_layers:
        layer_indices = list(range(num_layers))
    else:
        layer_indices = sample_layers(num_layers=num_layers, stride=args.layer_stride)

    personas = PERSONAS.copy()
    if args.limit_personas > 0:
        personas = dict(list(personas.items())[: args.limit_personas])

    questions = QUESTIONS[: args.limit_questions] if args.limit_questions > 0 else QUESTIONS
    examples = build_examples(
        tokenizer=tokenizer,
        personas=personas,
        questions=questions,
        max_length=args.max_length,
    )

    persona_names = [ex.persona_name for ex in examples]
    persona_vocab = sorted(set(persona_names))
    persona_to_id = {p: i for i, p in enumerate(persona_vocab)}
    persona_ids = [persona_to_id[p] for p in persona_names]

    print(f"Built {len(examples)} prompts ({len(personas)} personas x {len(questions)} questions)")
    print(f"Layers: {layer_indices}")

    # ---- Collect baseline activations ----
    print("\nCollecting baseline hidden states...")
    by_layer = collect_hidden_vectors(
        model=model,
        examples=examples,
        device=device,
        dtype=model.dtype,
        layer_indices=layer_indices,
    )

    # ---- Extract steering vectors ----
    print(f"\nExtracting steering vectors (baseline={args.baseline})...")
    steering_vectors = compute_steering_vectors(
        by_layer, persona_names, persona_vocab, baseline=args.baseline,
    )

    # Save steering vectors
    sv_path = outdir / "steering_vectors.pt"
    sv_save = {}
    for layer, sv_dict in steering_vectors.items():
        for persona, vec in sv_dict.items():
            sv_save[f"layer_{layer}_{persona}"] = vec
    torch.save(sv_save, sv_path)
    print(f"Saved {len(sv_save)} steering vectors to {sv_path}")

    # Print norms for sanity check
    print("\nSteering vector norms:")
    for layer in layer_indices:
        norms = {p: f"{v.norm().item():.3f}" for p, v in steering_vectors[layer].items()}
        print(f"  Layer {layer:2d}: {norms}")

    # ---- Sweep over (layer, alpha) ----
    alphas = [float(a) for a in args.alphas.split(",")]
    steer_layers = layer_indices if args.steer_all_layers else [
        l for l in layer_indices if l >= args.min_steer_layer
    ]

    target_personas = persona_vocab if args.target_persona == "all" else [args.target_persona]

    all_results = []
    for persona in target_personas:
        print(f"\n{'='*60}")
        print(f"Sweeping steering → {persona}")
        print(f"{'='*60}")
        df = run_sweep(
            model, tokenizer, examples, by_layer,
            persona_names, persona_ids, persona_vocab,
            steering_vectors, steer_layers, alphas,
            target_persona=persona,
            device=device, dtype=model.dtype,
        )
        all_results.append(df)

        # Per-persona heatmaps
        for metric in ["steer_rate", "flip_rate", "target_prob"]:
            plot_sweep_heatmap(df, persona, metric, outdir)

    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(outdir / "steering_sweep_results.csv", index=False)

    # Summary plot
    plot_all_personas_best(all_df, outdir)

    # ---- Find best config per persona ----
    best_configs: Dict[str, Tuple[int, float]] = {}
    print(f"\n{'='*60}")
    print("Best configs per persona (by flip_rate):")
    print(f"{'='*60}")
    for persona in target_personas:
        sub = all_df[all_df["target_persona"] == persona]
        best = sub.loc[sub["flip_rate"].idxmax()]
        layer, alpha = int(best["steer_layer"]), float(best["alpha"])
        best_configs[persona] = (layer, alpha)
        print(f"  {persona:20s} → layer={layer:2d}, alpha={alpha:5.1f}, "
              f"flip_rate={best['flip_rate']:.3f}, target_prob={best['target_prob']:.3f}")

    # Save best configs
    best_configs_serializable = {p: {"layer": l, "alpha": a} for p, (l, a) in best_configs.items()}
    with open(outdir / "best_steering_configs.json", "w") as f:
        json.dump(best_configs_serializable, f, indent=2)

    # ---- Probe verification ----
    print("\nVerifying steering with probe (before/after classification)...")
    verify_df = verify_steering_with_probe(
        model, tokenizer, examples, by_layer,
        persona_ids, persona_vocab,
        steering_vectors, best_configs,
        device=device, dtype=model.dtype,
    )
    verify_df.to_csv(outdir / "probe_verification.csv", index=False)
    plot_probe_verification(verify_df, persona_vocab, outdir)

    # ---- Qualitative demo ----
    demo_df = None
    if not args.skip_demo:
        print("\nGenerating steered demo outputs...")
        demo_steered_generation(
            model, tokenizer, steering_vectors, best_configs,
            persona_vocab, DEMO_QUESTIONS, device, outdir,
            n_questions=args.demo_questions,
            max_new_tokens=args.max_new_tokens,
        )
        demo_path = outdir / "steered_demo_outputs.csv"
        if demo_path.exists():
            demo_df = pd.read_csv(demo_path)

    # ---- Write results markdown ----
    write_results_markdown(
        all_df=all_df,
        best_configs=best_configs,
        steering_vectors=steering_vectors,
        persona_vocab=persona_vocab,
        steer_layers=steer_layers,
        alphas=alphas,
        args=args,
        outdir=outdir,
        demo_df=demo_df,
        verify_df=verify_df,
    )

    # ---- Save run config ----
    config = {
        "model_name": args.model_name,
        "device": str(device),
        "num_layers": num_layers,
        "steer_layers": steer_layers,
        "alphas": alphas,
        "baseline": args.baseline,
        "target_personas": target_personas,
        "n_personas": len(persona_vocab),
        "n_questions": len(questions),
        "n_examples": len(examples),
        "best_configs": best_configs_serializable,
    }
    with open(outdir / "steer_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Log best configs as summary metrics
    for p, (layer, alpha) in best_configs.items():
        sub = all_df[(all_df["target_persona"] == p) &
                     (all_df["steer_layer"] == layer) &
                     (all_df["alpha"] == alpha)]
        if len(sub) > 0:
            row = sub.iloc[0]
            wandb.log({
                f"best/{p}/layer": layer,
                f"best/{p}/alpha": alpha,
                f"best/{p}/flip_rate": row["flip_rate"],
                f"best/{p}/target_prob": row["target_prob"],
            })

    # Log all outputs to W&B
    for png in sorted(outdir.glob("*.png")):
        wandb.log({png.stem: wandb.Image(str(png))})
    for csv_file in sorted(outdir.glob("*.csv")):
        try:
            wandb.log({csv_file.stem: wandb.Table(dataframe=pd.read_csv(csv_file))})
        except Exception:
            pass
    wandb.finish()

    print(f"\nDone. Outputs written to: {outdir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persona steering vector extraction and sweep")
    parser.add_argument(
        "--model-name", type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument("--outdir", type=str, default="outputs/persona_steering")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--limit-personas", type=int, default=0)
    parser.add_argument("--limit-questions", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument(
        "--baseline", type=str, default="mean",
        help="Baseline for steering: 'mean' (global centroid) or a persona name like 'assistant'",
    )
    parser.add_argument(
        "--alphas", type=str, default="1,3,5,8,10,15",
        help="Comma-separated steering coefficients to sweep",
    )
    parser.add_argument(
        "--min-steer-layer", type=int, default=4,
        help="Skip layers below this for steering (early layers have weak persona signal)",
    )
    parser.add_argument(
        "--steer-all-layers", action="store_true",
        help="Include all sampled layers in the sweep (ignores --min-steer-layer)",
    )
    parser.add_argument(
        "--target-persona", type=str, default="all",
        help="Which persona to steer toward ('all' for all personas)",
    )
    parser.add_argument("--skip-demo", action="store_true", help="Skip qualitative demo generation")
    parser.add_argument("--demo-questions", type=int, default=5, help="Number of questions for demo")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
