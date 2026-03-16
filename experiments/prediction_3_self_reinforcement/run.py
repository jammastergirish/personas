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
# ]
# ///

"""
Prediction 3 — Self-Reinforcement
===================================
Tests whether self-generation reinforces basin position, making adversarial
steering less effective at later turns.

Claim: adversarial steering at turn 4 is less effective than at turn 1
(Cohen's d drops).

Method:
  1. For each persona x question, generate a 6-turn conversation (unsteered).
  2. At each turn, collect hidden states both with and without adversarial
     steering (toward the maximally distant persona).
  3. Measure Cohen's d of hidden-state shift at each turn.
  4. Measure probe flip rate at each turn.
  5. SVD of steered-minus-unsteered difference vectors per turn to track
     "vulnerability subspace" dimensionality shrinking at later turns.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
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
from steer import (
    SteeringHook,
    compute_steering_vectors,
    get_layer_module,
    train_persona_probe,
)
from experiments.shared.multi_turn import (
    build_multi_turn_prompt,
    collect_multi_turn_hidden,
)
from experiments.shared.utils import (
    cohens_d_multivariate,
    save_run_config,
    svd_analysis,
    effective_rank,
)

# ---------------------------------------------------------------------------
# Steering alpha used for adversarial injection
# ---------------------------------------------------------------------------
STEERING_ALPHA = 10.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_most_distant_persona(
    source: str,
    centroids: Dict[str, torch.Tensor],
) -> str:
    """Return the persona whose centroid is maximally cosine-distant from source."""
    src_vec = centroids[source]
    best_name, best_dist = None, -1.0
    for name, vec in centroids.items():
        if name == source:
            continue
        cos_sim = F.cosine_similarity(src_vec.unsqueeze(0), vec.unsqueeze(0)).item()
        dist = 1.0 - cos_sim
        if dist > best_dist:
            best_dist = dist
            best_name = name
    return best_name


def generate_response(
    model,
    tokenizer,
    system_prompt: str,
    conversation_history: List[Dict[str, str]],
    device: torch.device,
    max_new_tokens: int,
) -> str:
    """Generate a single assistant response given the conversation so far."""
    input_ids, attention_mask = build_multi_turn_prompt(
        tokenizer, system_prompt, conversation_history,
        add_generation_prompt=True,
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

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
    return tokenizer.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_cohens_d_by_turn(df: pd.DataFrame, outpath: Path) -> None:
    """Line plot of Cohen's d (mean +/- std across personas) vs turn."""
    agg = df.groupby("turn")["cohens_d"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        agg["turn"], agg["mean"], yerr=agg["std"],
        fmt="o-", capsize=4, color="tab:blue", linewidth=2, label="Mean Cohen's d",
    )
    ax.set_xlabel("Conversation turn")
    ax.set_ylabel("Cohen's d (steered vs unsteered)")
    ax.set_title("Prediction 3: Adversarial steering effectiveness by turn")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_flip_rate_by_turn(df: pd.DataFrame, outpath: Path) -> None:
    """Line plot of probe flip rate vs turn."""
    agg = df.groupby("turn")["flip_rate"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        agg["turn"], agg["mean"], yerr=agg["std"],
        fmt="s-", capsize=4, color="tab:red", linewidth=2, label="Mean flip rate",
    )
    ax.set_xlabel("Conversation turn")
    ax.set_ylabel("Probe flip rate (classified as adversarial)")
    ax.set_title("Prediction 3: Probe flip rate by turn")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def plot_turn_trajectories_pca(
    all_unsteered: Dict[int, List[torch.Tensor]],
    all_steered: Dict[int, List[torch.Tensor]],
    persona_labels: Dict[int, List[str]],
    n_turns: int,
    outpath: Path,
) -> None:
    """
    PCA scatter showing unsteered (circles) and steered (X markers) hidden states
    per turn, colored by turn index.
    """
    # Stack all vectors for joint PCA
    vecs_list = []
    labels_list = []  # (turn, "unsteered"/"steered")
    persona_list = []

    for t in range(n_turns):
        if t in all_unsteered and all_unsteered[t]:
            u_vecs = torch.stack(all_unsteered[t], dim=0)
            vecs_list.append(u_vecs)
            labels_list.extend([(t, "unsteered")] * u_vecs.shape[0])
            persona_list.extend(persona_labels.get(t, ["?"] * u_vecs.shape[0]))
        if t in all_steered and all_steered[t]:
            s_vecs = torch.stack(all_steered[t], dim=0)
            vecs_list.append(s_vecs)
            labels_list.extend([(t, "steered")] * s_vecs.shape[0])
            persona_list.extend(persona_labels.get(t, ["?"] * s_vecs.shape[0]))

    if not vecs_list:
        return

    combined = torch.cat(vecs_list, dim=0).numpy()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(combined)

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap("viridis", n_turns)

    for t in range(n_turns):
        # Unsteered
        mask_u = [i for i, (turn, cond) in enumerate(labels_list) if turn == t and cond == "unsteered"]
        if mask_u:
            pts = coords[mask_u]
            ax.scatter(pts[:, 0], pts[:, 1], color=cmap(t), marker="o",
                       alpha=0.5, s=30, label=f"Turn {t} unsteered" if t < 3 else None)

        # Steered
        mask_s = [i for i, (turn, cond) in enumerate(labels_list) if turn == t and cond == "steered"]
        if mask_s:
            pts = coords[mask_s]
            ax.scatter(pts[:, 0], pts[:, 1], color=cmap(t), marker="x",
                       alpha=0.7, s=50, label=f"Turn {t} steered" if t < 3 else None)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("Hidden-state trajectories: unsteered (o) vs steered (x) by turn")

    # Colorbar for turns
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_turns - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label="Turn")
    cbar.set_ticks(range(n_turns))

    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = infer_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    hf_token = os.environ.get("HF_TOKEN")

    # ================================================================
    # 1. Load model and tokenizer
    # ================================================================
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model on {device}...")
    model_kwargs = {"output_hidden_states": True}
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        model_kwargs["device_map"] = "auto"
    elif device.type == "mps":
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, token=hf_token, **model_kwargs,
    )
    if device.type in {"cpu", "mps"}:
        model.to(device)

    dtype = model.dtype
    num_layers = model.config.num_hidden_layers
    layer_indices = sample_layers(num_layers=num_layers, stride=args.layer_stride)
    best_layer = max(layer_indices)

    # ================================================================
    # 2. Select personas and questions
    # ================================================================
    personas = PERSONAS.copy()
    if args.limit_personas > 0:
        personas = dict(list(personas.items())[:args.limit_personas])

    questions = QUESTIONS[:args.limit_questions] if args.limit_questions > 0 else QUESTIONS
    n_turns = args.n_turns
    persona_vocab = sorted(personas.keys())

    print(f"Personas ({len(persona_vocab)}): {persona_vocab}")
    print(f"Questions: {len(questions)}")
    print(f"Turns: {n_turns}")
    print(f"Layers: {layer_indices}")
    print(f"Best layer (deepest): {best_layer}")

    # ================================================================
    # 3. Collect baseline hidden states and compute steering vectors
    # ================================================================
    print("\nBuilding single-turn examples for steering vector extraction...")
    all_examples = build_examples(tokenizer, personas, questions, max_length=512)
    all_persona_names = [ex.persona_name for ex in all_examples]

    print("Collecting baseline hidden states...")
    by_layer = collect_hidden_vectors(model, all_examples, device, dtype, layer_indices)

    print("Computing steering vectors...")
    steering_vectors = compute_steering_vectors(
        by_layer, all_persona_names, persona_vocab, baseline="mean",
    )

    # Compute per-persona centroids at best_layer
    all_vecs = torch.stack(by_layer[best_layer], dim=0)
    centroids: Dict[str, torch.Tensor] = {}
    for p in persona_vocab:
        mask = [n == p for n in all_persona_names]
        centroids[p] = all_vecs[mask].mean(dim=0)

    # Find adversarial targets (maximally distant persona for each)
    adversarial_targets: Dict[str, str] = {}
    for p in persona_vocab:
        adversarial_targets[p] = find_most_distant_persona(p, centroids)
        print(f"  {p} -> adversarial target: {adversarial_targets[p]}")

    # ================================================================
    # 4. Train persona probe on single-turn baseline data
    # ================================================================
    print("\nTraining persona probe on baseline hidden states...")
    persona_to_id = {p: i for i, p in enumerate(persona_vocab)}
    persona_ids = [persona_to_id[n] for n in all_persona_names]
    probe = train_persona_probe(by_layer, persona_ids, best_layer)
    print(f"  Probe trained on {len(all_examples)} examples, {len(persona_vocab)} classes")

    # ================================================================
    # 5. Multi-turn experiment loop
    # ================================================================
    print(f"\n{'='*60}")
    print("Running multi-turn self-reinforcement experiment")
    print(f"{'='*60}")

    # Storage: per-turn hidden state collections
    # all_unsteered_by_turn[turn] = list of [d] tensors
    # all_steered_by_turn[turn] = list of [d] tensors
    all_unsteered_by_turn: Dict[int, List[torch.Tensor]] = {t: [] for t in range(n_turns)}
    all_steered_by_turn: Dict[int, List[torch.Tensor]] = {t: [] for t in range(n_turns)}
    persona_labels_by_turn: Dict[int, List[str]] = {t: [] for t in range(n_turns)}

    # Per-turn Cohen's d rows
    cohens_d_rows: List[dict] = []
    # Per-turn flip rate rows
    flip_rate_rows: List[dict] = []
    # Per-turn cosine distance from persona centroid
    cosine_dist_rows: List[dict] = []

    total_combos = len(persona_vocab) * len(questions)
    combo_idx = 0

    for persona in persona_vocab:
        system_prompt = personas[persona]
        adv_persona = adversarial_targets[persona]
        sv = steering_vectors[best_layer][adv_persona]
        adv_id = persona_to_id[adv_persona]

        for q_idx, first_question in enumerate(questions):
            combo_idx += 1
            print(f"\n[{combo_idx}/{total_combos}] Persona={persona}, Q={q_idx} "
                  f"(adv={adv_persona})")

            conversation_history: List[Dict[str, str]] = []

            for turn in range(n_turns):
                # Pick the question for this turn (cycle through questions)
                q = questions[(q_idx + turn) % len(questions)]
                conversation_history.append({"role": "user", "content": q})

                # Generate unsteered response
                response = generate_response(
                    model, tokenizer, system_prompt, conversation_history,
                    device, args.max_new_tokens,
                )
                conversation_history.append({"role": "assistant", "content": response})

                # Collect hidden state WITHOUT steering
                unsteered_h = collect_multi_turn_hidden(
                    model, tokenizer, system_prompt, conversation_history,
                    device, [best_layer],
                )
                u_vec = unsteered_h[best_layer]  # [d]

                # Collect hidden state WITH adversarial steering
                hook = SteeringHook(sv, alpha=STEERING_ALPHA)
                hook.attach(get_layer_module(model, best_layer))
                try:
                    steered_h = collect_multi_turn_hidden(
                        model, tokenizer, system_prompt, conversation_history,
                        device, [best_layer],
                    )
                finally:
                    hook.remove()
                s_vec = steered_h[best_layer]  # [d]

                # Store for aggregation
                all_unsteered_by_turn[turn].append(u_vec)
                all_steered_by_turn[turn].append(s_vec)
                persona_labels_by_turn[turn].append(persona)

                # Probe prediction on steered vector
                s_norm = F.normalize(s_vec.unsqueeze(0), dim=-1).numpy()
                probe_pred = probe.predict(s_norm)[0]
                flipped = int(probe_pred == adv_id)

                # Cosine distance from own persona centroid
                cos_from_own = F.cosine_similarity(
                    u_vec.unsqueeze(0), centroids[persona].unsqueeze(0),
                ).item()
                cos_steered_from_own = F.cosine_similarity(
                    s_vec.unsqueeze(0), centroids[persona].unsqueeze(0),
                ).item()

                cosine_dist_rows.append({
                    "persona": persona,
                    "adversarial": adv_persona,
                    "question_idx": q_idx,
                    "turn": turn,
                    "cos_unsteered_from_centroid": cos_from_own,
                    "cos_steered_from_centroid": cos_steered_from_own,
                })

                print(f"  Turn {turn}: flipped={flipped}, "
                      f"cos_own={cos_from_own:.3f}->{cos_steered_from_own:.3f}")

    # ================================================================
    # 6. Aggregate: Cohen's d per turn
    # ================================================================
    print(f"\n{'='*60}")
    print("Aggregating Cohen's d by turn")
    print(f"{'='*60}")

    for turn in range(n_turns):
        if not all_unsteered_by_turn[turn]:
            continue
        u_mat = torch.stack(all_unsteered_by_turn[turn], dim=0)
        s_mat = torch.stack(all_steered_by_turn[turn], dim=0)
        cd = cohens_d_multivariate(u_mat, s_mat)

        # Per-persona Cohen's d
        for persona in persona_vocab:
            mask = [l == persona for l in persona_labels_by_turn[turn]]
            if not any(mask):
                continue
            u_sub = u_mat[mask]
            s_sub = s_mat[mask]
            cd_persona = cohens_d_multivariate(u_sub, s_sub)
            cohens_d_rows.append({
                "turn": turn,
                "persona": persona,
                "cohens_d": cd_persona,
                "n_samples": int(sum(mask)),
            })

        # Aggregate row
        cohens_d_rows.append({
            "turn": turn,
            "persona": "ALL",
            "cohens_d": cd,
            "n_samples": u_mat.shape[0],
        })
        print(f"  Turn {turn}: Cohen's d = {cd:.4f} (n={u_mat.shape[0]})")

    # ================================================================
    # 7. Aggregate: probe flip rate per turn
    # ================================================================
    print(f"\n{'='*60}")
    print("Aggregating probe flip rate by turn")
    print(f"{'='*60}")

    for turn in range(n_turns):
        if not all_steered_by_turn[turn]:
            continue
        s_mat = torch.stack(all_steered_by_turn[turn], dim=0)
        s_norm_mat = F.normalize(s_mat, dim=-1).numpy()
        preds = probe.predict(s_norm_mat)

        labels = persona_labels_by_turn[turn]

        # Per-persona flip rate
        for persona in persona_vocab:
            adv_persona = adversarial_targets[persona]
            adv_id = persona_to_id[adv_persona]
            mask = [l == persona for l in labels]
            if not any(mask):
                continue
            p_preds = preds[mask]
            fr = float((p_preds == adv_id).mean())
            flip_rate_rows.append({
                "turn": turn,
                "persona": persona,
                "adversarial": adv_persona,
                "flip_rate": fr,
                "n_samples": int(sum(mask)),
            })

        # Aggregate: overall fraction that flipped to their adversarial target
        flipped_count = 0
        total_count = 0
        for i, persona in enumerate(labels):
            adv_id = persona_to_id[adversarial_targets[persona]]
            if preds[i] == adv_id:
                flipped_count += 1
            total_count += 1
        overall_fr = flipped_count / max(total_count, 1)
        flip_rate_rows.append({
            "turn": turn,
            "persona": "ALL",
            "adversarial": "respective",
            "flip_rate": overall_fr,
            "n_samples": total_count,
        })
        print(f"  Turn {turn}: flip rate = {overall_fr:.4f} (n={total_count})")

    # ================================================================
    # 8. SVD of difference vectors per turn (vulnerability subspace)
    # ================================================================
    print(f"\n{'='*60}")
    print("SVD analysis of steered-minus-unsteered per turn")
    print(f"{'='*60}")

    svd_rows: List[dict] = []
    for turn in range(n_turns):
        if not all_unsteered_by_turn[turn]:
            continue
        u_mat = torch.stack(all_unsteered_by_turn[turn], dim=0)
        s_mat = torch.stack(all_steered_by_turn[turn], dim=0)
        diff = s_mat - u_mat  # [n, d]

        if diff.shape[0] < 2:
            continue

        S, variance, cumvar = svd_analysis(diff)

        for threshold in [0.80, 0.90, 0.95, 0.99]:
            rank = effective_rank(S, threshold)
            svd_rows.append({
                "turn": turn,
                "threshold": threshold,
                "effective_rank": rank,
            })

        rank_95 = effective_rank(S, 0.95)
        print(f"  Turn {turn}: effective rank (95%) = {rank_95}")

    # ================================================================
    # 9. Save CSVs
    # ================================================================
    cohens_d_df = pd.DataFrame(cohens_d_rows)
    cohens_d_df.to_csv(outdir / "cohens_d_by_turn.csv", index=False)

    flip_rate_df = pd.DataFrame(flip_rate_rows)
    flip_rate_df.to_csv(outdir / "flip_rate_by_turn.csv", index=False)

    svd_df = pd.DataFrame(svd_rows)
    svd_df.to_csv(outdir / "vulnerability_subspace_rank.csv", index=False)

    cosine_df = pd.DataFrame(cosine_dist_rows)
    cosine_df.to_csv(outdir / "cosine_distances_by_turn.csv", index=False)

    # ================================================================
    # 10. Plots
    # ================================================================
    print("\nGenerating plots...")

    # Cohen's d by turn (aggregate only)
    agg_cd = cohens_d_df[cohens_d_df["persona"] == "ALL"].copy()
    plot_cohens_d_by_turn(agg_cd, outdir / "cohens_d_by_turn.png")

    # Per-persona Cohen's d overlay
    per_persona_cd = cohens_d_df[cohens_d_df["persona"] != "ALL"].copy()
    if not per_persona_cd.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        for persona in persona_vocab:
            sub = per_persona_cd[per_persona_cd["persona"] == persona]
            if not sub.empty:
                ax.plot(sub["turn"], sub["cohens_d"], "o-", label=persona, alpha=0.7)
        # Also plot aggregate
        if not agg_cd.empty:
            ax.plot(agg_cd["turn"], agg_cd["cohens_d"], "k-s", linewidth=2.5,
                    markersize=8, label="ALL (aggregate)", zorder=10)
        ax.set_xlabel("Conversation turn")
        ax.set_ylabel("Cohen's d (steered vs unsteered)")
        ax.set_title("Cohen's d by turn and persona")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / "cohens_d_by_turn_per_persona.png", dpi=180)
        plt.close()

    # Flip rate by turn (aggregate only)
    agg_fr = flip_rate_df[flip_rate_df["persona"] == "ALL"].copy()
    plot_flip_rate_by_turn(agg_fr, outdir / "flip_rate_by_turn.png")

    # Vulnerability subspace rank by turn
    if not svd_df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for threshold in [0.90, 0.95, 0.99]:
            sub = svd_df[svd_df["threshold"] == threshold]
            if not sub.empty:
                ax.plot(sub["turn"], sub["effective_rank"], "o-",
                        label=f"{threshold:.0%} threshold")
        ax.set_xlabel("Conversation turn")
        ax.set_ylabel("Effective rank (vulnerability subspace)")
        ax.set_title("Vulnerability subspace dimensionality by turn")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / "vulnerability_subspace_rank.png", dpi=180)
        plt.close()

    # PCA turn trajectories
    plot_turn_trajectories_pca(
        all_unsteered_by_turn, all_steered_by_turn,
        persona_labels_by_turn, n_turns,
        outdir / "turn_trajectories_pca.png",
    )

    # ================================================================
    # 11. Save run config
    # ================================================================
    elapsed = time.time() - t_start
    config = {
        "model_name": args.model_name,
        "device": str(device),
        "seed": args.seed,
        "n_turns": n_turns,
        "max_new_tokens": args.max_new_tokens,
        "n_personas": len(persona_vocab),
        "personas": persona_vocab,
        "n_questions": len(questions),
        "layer_stride": args.layer_stride,
        "best_layer": best_layer,
        "layer_indices": layer_indices,
        "steering_alpha": STEERING_ALPHA,
        "adversarial_targets": adversarial_targets,
        "elapsed_seconds": round(elapsed, 1),
    }
    save_run_config(config, outdir)

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*60}")
    print("Prediction 3: Self-Reinforcement -- Summary")
    print(f"{'='*60}")
    print(f"Personas: {len(persona_vocab)}, Questions: {len(questions)}, Turns: {n_turns}")
    print(f"Layer: {best_layer}, Alpha: {STEERING_ALPHA}")

    if not agg_cd.empty:
        cd_t0 = agg_cd[agg_cd["turn"] == 0]["cohens_d"].values
        cd_last = agg_cd[agg_cd["turn"] == n_turns - 1]["cohens_d"].values
        if len(cd_t0) > 0 and len(cd_last) > 0:
            print(f"Cohen's d: turn 0 = {cd_t0[0]:.4f}, turn {n_turns-1} = {cd_last[0]:.4f}")
            drop = cd_t0[0] - cd_last[0]
            print(f"  Drop: {drop:.4f} ({'supports' if drop > 0 else 'contradicts'} prediction)")

    if not agg_fr.empty:
        fr_t0 = agg_fr[agg_fr["turn"] == 0]["flip_rate"].values
        fr_last = agg_fr[agg_fr["turn"] == n_turns - 1]["flip_rate"].values
        if len(fr_t0) > 0 and len(fr_last) > 0:
            print(f"Flip rate: turn 0 = {fr_t0[0]:.4f}, turn {n_turns-1} = {fr_last[0]:.4f}")

    svd_95 = svd_df[svd_df["threshold"] == 0.95]
    if not svd_95.empty:
        rank_t0 = svd_95[svd_95["turn"] == 0]["effective_rank"].values
        rank_last = svd_95[svd_95["turn"] == n_turns - 1]["effective_rank"].values
        if len(rank_t0) > 0 and len(rank_last) > 0:
            print(f"Vulnerability rank (95%): turn 0 = {rank_t0[0]}, "
                  f"turn {n_turns-1} = {rank_last[0]}")

    print(f"\nElapsed: {elapsed:.0f}s")
    print(f"Outputs written to: {outdir.resolve()}")
    print("  - cohens_d_by_turn.csv / .png")
    print("  - cohens_d_by_turn_per_persona.png")
    print("  - flip_rate_by_turn.csv / .png")
    print("  - vulnerability_subspace_rank.csv / .png")
    print("  - turn_trajectories_pca.png")
    print("  - cosine_distances_by_turn.csv")
    print("  - run_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prediction 3: Self-Reinforcement — adversarial steering effectiveness over turns",
    )
    parser.add_argument(
        "--model-name", type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--outdir", type=str,
        default="outputs/prediction_3_self_reinforcement",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-turns", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--limit-personas", type=int, default=0)
    parser.add_argument("--limit-questions", type=int, default=0)
    parser.add_argument("--layer-stride", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
