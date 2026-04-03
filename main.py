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

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from dotenv import load_dotenv

load_dotenv()  # loads HF_TOKEN (and any other vars) from .env

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Idea
# ============================================================
# We hold the semantic question fixed and vary only the persona.
# For each persona x question pair, we:
#   1. Build a chat prompt with that persona instruction.
#   2. Run the model with output_hidden_states=True.
#   3. Extract the hidden state of the LAST INPUT TOKEN at each layer.
#      This is the cleanest cheap representation of "the model state
#      right before it begins answering".
#   4. Compare these vectors across personas/questions.
#   5. Plot 2D PCA projections and compute simple clustering metrics.
#
# V2 improvements:
#   - Confusion matrices for k-means clusters (improvement 1)
#   - 40 diverse questions across many domains (improvement 2)
#   - Null baseline with rephrased persona instructions (improvement 3)
#   - Linear probe (logistic regression) accuracy (improvement 4)
#   - Generated-token hidden state analysis (improvement 5)
# ============================================================


PERSONAS: Dict[str, str] = {
    "assistant": "You are a helpful, careful AI assistant. Answer clearly and truthfully.",
    "pirate": "You are a pirate. Answer in the voice of a pirate, with nautical language and swagger.",
    "lawyer": "You are a meticulous lawyer. Answer with legalistic precision and caveats.",
    "scientist": "You are a scientist. Answer analytically, cautiously, and with explicit reasoning.",
    "comedian": "You are a comedian. Answer with wit, playfulness, and punchy phrasing.",
    "stoic": "You are a stoic philosopher. Answer calmly, tersely, and with emotional restraint.",
    "conspiracy_host": "You are a sensational conspiracy talk-show host. Answer with suspicious, dramatic framing.",
    "kind_teacher": "You are a warm teacher. Answer gently, clearly, and pedagogically.",
}

# Rephrased persona instructions with identical meaning but different surface tokens.
# Used for the null-baseline experiment (improvement 3).
PERSONAS_ALT: Dict[str, str] = {
    "assistant": "You are a reliable and considerate AI helper. Respond with clarity and honesty.",
    "pirate": "You are a seafaring buccaneer. Respond using nautical slang and bold, swashbuckling flair.",
    "lawyer": "You are a precise attorney. Respond with legal exactness, hedging, and careful qualifications.",
    "scientist": "You are a research scientist. Respond with analytical rigor, caution, and step-by-step logic.",
    "comedian": "You are a stand-up comic. Respond with humor, wordplay, and snappy one-liners.",
    "stoic": "You are a philosopher in the Stoic tradition. Respond with brevity, composure, and detachment.",
    "conspiracy_host": "You are a dramatic conspiracy-theory broadcaster. Respond with paranoid suspicion and theatrical flair.",
    "kind_teacher": "You are a gentle, encouraging educator. Respond with patience, warmth, and clear explanations.",
}

# Expanded question set: 40 questions across diverse domains (improvement 2)
QUESTIONS: List[str] = [
    # Science / nature
    "Why do eclipses happen?",
    "Is nuclear energy a good idea?",
    "What causes inflation?",
    "How does a vaccine work?",
    "Why is the sky blue?",
    # Policy / ethics
    "Should governments regulate advanced AI systems?",
    "Should children learn more than one language?",
    "Why do people disagree about moral issues?",
    "Is universal basic income a realistic policy?",
    "Should social media platforms censor misinformation?",
    # Personal / emotional
    "What is the best way to comfort a friend after a failure?",
    "How do you deal with loneliness?",
    "What advice would you give someone going through a breakup?",
    "How can a person build self-confidence?",
    "What should you do when you feel overwhelmed?",
    # History / culture
    "Why did the Roman Empire fall?",
    "What caused the French Revolution?",
    "Why is the Mona Lisa so famous?",
    "How did the printing press change the world?",
    "What lessons can we learn from the Cold War?",
    # Practical / everyday
    "How should a journalist verify a leaked document?",
    "What should someone do before investing in a volatile stock?",
    "How do I make a good cup of coffee?",
    "What is the best way to prepare for a job interview?",
    "How do you fix a leaky faucet?",
    # Technical / analytical
    "Explain how a binary search algorithm works.",
    "What is the difference between TCP and UDP?",
    "How does encryption keep data secure?",
    "What makes a good database index?",
    "Why is recursion useful in programming?",
    # Philosophy / abstract
    "What is consciousness?",
    "Can machines truly think?",
    "Is free will an illusion?",
    "What makes an action morally right?",
    "Does objective truth exist?",
    # Creative / open-ended
    "If you could redesign education from scratch, what would it look like?",
    "What would a perfect city look like?",
    "How would you explain music to someone who has never heard sound?",
    "What is the most important invention in human history?",
    "If animals could talk, how would society change?",
]


@dataclass
class Example:
    persona_name: str
    persona_text: str
    question: str
    prompt_text: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_device(explicit_device: str | None = None) -> torch.device:
    if explicit_device:
        return torch.device(explicit_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_examples(
    tokenizer,
    personas: Dict[str, str],
    questions: List[str],
    max_length: int,
) -> List[Example]:
    examples: List[Example] = []

    for persona_name, persona_text in personas.items():
        for question in questions:
            messages = [
                {"role": "system", "content": persona_text},
                {"role": "user", "content": question},
            ]

            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            examples.append(
                Example(
                    persona_name=persona_name,
                    persona_text=persona_text,
                    question=question,
                    prompt_text=prompt_text,
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
            )
    return examples


def sample_layers(num_layers: int, stride: int, include_last: bool = True) -> List[int]:
    layers = list(range(0, num_layers, stride))
    if include_last and (num_layers - 1) not in layers:
        layers.append(num_layers - 1)
    return sorted(set(layers))


def pca_2d_torch(x: torch.Tensor) -> torch.Tensor:
    """
    x: [n, d]
    returns [n, 2]
    """
    x = x - x.mean(dim=0, keepdim=True)
    q = min(4, min(x.shape[0], x.shape[1]))
    u, s, v = torch.pca_lowrank(x, q=q, center=False)
    coords = x @ v[:, :2]
    return coords


def kmeans_torch(
    x: torch.Tensor,
    k: int,
    n_iters: int = 50,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Very small/simple torch K-means.
    Returns:
      labels: [n]
      centroids: [k, d]
    """
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)

    n = x.shape[0]
    perm = torch.randperm(n, generator=g, device=x.device)
    centroids = x[perm[:k]].clone()

    for _ in range(n_iters):
        dists = torch.cdist(x, centroids)  # [n, k]
        labels = dists.argmin(dim=1)

        new_centroids = []
        for i in range(k):
            mask = labels == i
            if mask.any():
                new_centroids.append(x[mask].mean(dim=0))
            else:
                new_centroids.append(centroids[i])
        new_centroids = torch.stack(new_centroids, dim=0)

        if torch.allclose(new_centroids, centroids, atol=1e-5, rtol=1e-5):
            centroids = new_centroids
            break
        centroids = new_centroids

    dists = torch.cdist(x, centroids)
    labels = dists.argmin(dim=1)
    return labels, centroids


def clustering_purity(pred: torch.Tensor, gold: List[int], k: int) -> float:
    gold_t = torch.tensor(gold, device=pred.device)
    total = 0
    for cluster_id in range(k):
        mask = pred == cluster_id
        if not mask.any():
            continue
        counts = torch.bincount(gold_t[mask])
        total += counts.max().item()
    return total / len(gold)


def pairwise_metrics_by_group(
    x: torch.Tensor,
    persona_ids: List[int],
    question_ids: List[int],
) -> Dict[str, float]:
    x = F.normalize(x, dim=-1)
    sim = x @ x.T
    n = x.shape[0]

    same_persona, diff_persona = [], []
    same_question, diff_question = [], []

    for i in range(n):
        for j in range(i + 1, n):
            sij = sim[i, j].item()
            if persona_ids[i] == persona_ids[j]:
                same_persona.append(sij)
            else:
                diff_persona.append(sij)
            if question_ids[i] == question_ids[j]:
                same_question.append(sij)
            else:
                diff_question.append(sij)

    def mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    return {
        "cosine_same_persona": mean(same_persona),
        "cosine_diff_persona": mean(diff_persona),
        "cosine_same_question": mean(same_question),
        "cosine_diff_question": mean(diff_question),
        "persona_separation_gap": mean(same_persona) - mean(diff_persona),
        "question_separation_gap": mean(same_question) - mean(diff_question),
    }


def collect_hidden_vectors(
    model,
    examples: List[Example],
    device: torch.device,
    dtype: torch.dtype,
    layer_indices: List[int],
) -> Dict[int, List[torch.Tensor]]:
    by_layer: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layer_indices}

    model.eval()
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
            # hidden_states[0] is embedding output; transformer blocks start at 1
            seq_len = int(attention_mask.sum().item())
            last_tok_idx = seq_len - 1

            for layer in layer_indices:
                hs = hidden_states[layer + 1][0, last_tok_idx].detach().to(dtype=torch.float32).cpu()
                by_layer[layer].append(hs)

    return by_layer


# ============================================================
# Improvement 5: collect hidden states from generated tokens
# ============================================================

def collect_generated_hidden_vectors(
    model,
    tokenizer,
    examples: List[Example],
    device: torch.device,
    layer_indices: List[int],
    max_new_tokens: int = 20,
    n_gen_positions: int = 5,
) -> Dict[int, List[torch.Tensor]]:
    """
    Generate tokens, then run a forward pass on the full sequence
    (input + generated) and extract hidden states at the first
    n_gen_positions generated token positions.

    Returns mean-pooled vector across those positions per example per layer.
    """
    by_layer: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layer_indices}

    model.eval()
    with torch.no_grad():
        for ex in examples:
            input_ids = ex.input_ids.to(device)
            attention_mask = ex.attention_mask.to(device)
            input_len = input_ids.shape[1]

            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

            full_ids = gen[:, :input_len + n_gen_positions]
            full_mask = torch.ones_like(full_ids)

            outputs = model(
                input_ids=full_ids,
                attention_mask=full_mask,
                output_hidden_states=True,
                use_cache=False,
            )

            hidden_states = outputs.hidden_states
            actual_gen_len = full_ids.shape[1] - input_len
            if actual_gen_len < 1:
                # Model generated nothing — use last input token as fallback
                for layer in layer_indices:
                    hs = hidden_states[layer + 1][0, -1].detach().to(dtype=torch.float32).cpu()
                    by_layer[layer].append(hs)
            else:
                for layer in layer_indices:
                    gen_hidden = hidden_states[layer + 1][0, input_len:input_len + actual_gen_len]
                    pooled = gen_hidden.mean(dim=0).detach().to(dtype=torch.float32).cpu()
                    by_layer[layer].append(pooled)

    return by_layer


# ============================================================
# Improvement 1: confusion matrix
# ============================================================

def plot_confusion_matrix(
    pred_labels: torch.Tensor,
    true_ids: List[int],
    persona_vocab: List[str],
    layer: int,
    outdir: Path,
) -> None:
    k = len(persona_vocab)
    # Build confusion: rows = k-means cluster, cols = true persona
    mat = np.zeros((k, k), dtype=int)
    for pred, true in zip(pred_labels.tolist(), true_ids):
        mat[pred][true] += 1

    # Reorder rows by best-match to make it more readable:
    # greedily assign each cluster to its dominant persona
    row_order = []
    used = set()
    for _ in range(k):
        best_score, best_row, best_col = -1, 0, 0
        for r in range(k):
            if r in used:
                continue
            for c in range(k):
                if mat[r][c] > best_score:
                    best_score = mat[r][c]
                    best_row = r
                    best_col = c
            # Actually, just pick the row with the highest single-cell count
        # Simpler: pick the unassigned row with the highest max
        remaining = [r for r in range(k) if r not in used]
        best_row = max(remaining, key=lambda r: mat[r].max())
        row_order.append(best_row)
        used.add(best_row)

    mat_reordered = mat[row_order]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat_reordered, aspect="auto", cmap="Blues")
    plt.colorbar(im, ax=ax, label="count")

    # Annotate cells
    for i in range(k):
        for j in range(k):
            val = mat_reordered[i, j]
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                        color="white" if val > mat_reordered.max() / 2 else "black",
                        fontsize=9)

    ax.set_xticks(range(k))
    ax.set_xticklabels(persona_vocab, rotation=45, ha="right")
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"cluster {row_order[i]}" for i in range(k)])
    ax.set_xlabel("True persona")
    ax.set_ylabel("K-means cluster")
    ax.set_title(f"K-means confusion matrix - layer {layer}")
    plt.tight_layout()
    plt.savefig(outdir / f"layer_{layer:02d}_confusion.png", dpi=180)
    plt.close()


# ============================================================
# Improvement 4: linear probe
# ============================================================

def linear_probe_accuracy(
    vectors: torch.Tensor,
    labels: List[int],
    n_folds: int = 5,
    seed: int = 0,
) -> float:
    """
    Stratified k-fold logistic regression accuracy.
    """
    X = vectors.numpy()
    y = np.array(labels)

    if len(set(labels)) < 2:
        return float("nan")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
        clf.fit(X[train_idx], y[train_idx])
        accs.append(clf.score(X[test_idx], y[test_idx]))
    return float(np.mean(accs))


# ============================================================
# Multi-seed evaluation (improvement 6)
# ============================================================

def multi_seed_eval(
    vectors: torch.Tensor,
    labels: List[int],
    persona_vocab: List[str],
    k: int,
    n_seeds: int = 10,
    n_folds: int = 5,
) -> Dict[str, any]:
    """
    Run k-means and linear probe across multiple seeds.
    Returns mean, std for each metric.
    """
    purities = []
    probe_accs = []

    for seed in range(n_seeds):
        # K-means
        pred_labels, _ = kmeans_torch(vectors, k=k, n_iters=50, seed=seed)
        purity = clustering_purity(pred_labels, labels, k=k)
        purities.append(purity)

        # Linear probe
        acc = linear_probe_accuracy(vectors, labels, n_folds=n_folds, seed=seed)
        probe_accs.append(acc)

    return {
        "kmeans_purity_mean": float(np.mean(purities)),
        "kmeans_purity_std": float(np.std(purities)),
        "probe_accuracy_mean": float(np.mean(probe_accs)),
        "probe_accuracy_std": float(np.std(probe_accs)),
    }


# ============================================================
# Per-persona F1 scores (improvement 7)
# ============================================================

def per_persona_f1(
    vectors: torch.Tensor,
    labels: List[int],
    persona_vocab: List[str],
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """
    Train a logistic regression on 80% of data, evaluate per-class
    precision/recall/F1 on the held-out 20%.
    """
    X = vectors.numpy()
    y = np.array(labels)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    # Use just the first fold for a clean report
    train_idx, test_idx = next(iter(skf.split(X, y)))

    clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
    clf.fit(X[train_idx], y[train_idx])
    y_pred = clf.predict(X[test_idx])

    report = classification_report(
        y[test_idx], y_pred,
        target_names=persona_vocab,
        output_dict=True,
        zero_division=0,
    )
    return report


def plot_per_persona_f1(
    report: Dict,
    persona_vocab: List[str],
    layer: int,
    outdir: Path,
) -> None:
    personas = persona_vocab
    f1s = [report[p]["f1-score"] for p in personas]
    precisions = [report[p]["precision"] for p in personas]
    recalls = [report[p]["recall"] for p in personas]

    x = np.arange(len(personas))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, precisions, width, label="Precision", alpha=0.85)
    ax.bar(x, recalls, width, label="Recall", alpha=0.85)
    ax.bar(x + width, f1s, width, label="F1", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(personas, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(f"Per-persona classification metrics - layer {layer}")
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"layer_{layer:02d}_per_persona_f1.png", dpi=180)
    plt.close()


# ============================================================
# Persona subspace dimensionality (improvement 8)
# ============================================================

def persona_subspace_analysis(
    vectors: torch.Tensor,
    persona_names: List[str],
    layer: int,
    outdir: Path,
) -> Dict[str, float]:
    """
    Compute PCA on the persona centroids to find how many dimensions
    capture the persona variance. Also compute variance explained
    on all vectors projected into persona-subspace.
    """
    personas = sorted(set(persona_names))
    persona_to_vecs = {p: [] for p in personas}
    for v, p in zip(vectors, persona_names):
        persona_to_vecs[p].append(v)

    centroids = []
    for p in personas:
        mat = torch.stack(persona_to_vecs[p], dim=0)
        centroids.append(mat.mean(dim=0))
    centroid_mat = torch.stack(centroids, dim=0)  # [k, d]

    # PCA on centroids
    centroid_centered = centroid_mat - centroid_mat.mean(dim=0, keepdim=True)
    U, S, V = torch.linalg.svd(centroid_centered, full_matrices=False)
    variance = (S ** 2) / (S ** 2).sum()
    cumvar = torch.cumsum(variance, dim=0)

    # Also do PCA on all vectors to see total vs persona variance
    all_centered = vectors - vectors.mean(dim=0, keepdim=True)
    _, S_all, _ = torch.linalg.svd(all_centered, full_matrices=False)
    variance_all = (S_all ** 2) / (S_all ** 2).sum()
    cumvar_all = torch.cumsum(variance_all, dim=0)

    # How many dims to capture 95% / 99% of persona variance?
    dims_95 = int((cumvar >= 0.95).nonzero(as_tuple=True)[0][0].item()) + 1
    dims_99 = int((cumvar >= 0.99).nonzero(as_tuple=True)[0][0].item()) + 1

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_show = min(len(S), 8)
    axes[0].bar(range(n_show), variance[:n_show].numpy(), color="tab:blue", alpha=0.8)
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Variance explained")
    axes[0].set_title(f"Persona centroid PCA - layer {layer}\n({dims_95} dims for 95%, {dims_99} dims for 99%)")
    axes[0].set_xticks(range(n_show))

    n_show_all = min(50, len(S_all))
    axes[1].plot(range(n_show_all), cumvar_all[:n_show_all].numpy(), marker=".", label="All vectors", color="tab:orange")
    axes[1].plot(range(n_show), cumvar[:n_show].numpy(), marker="o", label="Persona centroids", color="tab:blue")
    axes[1].axhline(0.95, color="gray", linestyle="--", alpha=0.5, label="95%")
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative variance explained")
    axes[1].set_title(f"Cumulative variance - layer {layer}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(outdir / f"layer_{layer:02d}_subspace_dims.png", dpi=180)
    plt.close()

    return {
        "persona_dims_95": dims_95,
        "persona_dims_99": dims_99,
        "pc1_variance": float(variance[0].item()),
        "pc2_variance": float(variance[1].item()) if len(variance) > 1 else 0.0,
        "pc3_variance": float(variance[2].item()) if len(variance) > 2 else 0.0,
    }


# ============================================================
# Plotting helpers (unchanged + new)
# ============================================================

def maybe_generate_answers(
    model,
    tokenizer,
    examples: List[Example],
    device: torch.device,
    max_new_tokens: int,
    out_path: Path,
) -> None:
    rows = []
    model.eval()
    with torch.no_grad():
        for ex in examples:
            input_ids = ex.input_ids.to(device)
            attention_mask = ex.attention_mask.to(device)
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
            answer = tokenizer.decode(continuation, skip_special_tokens=True)
            rows.append({
                "persona": ex.persona_name,
                "question": ex.question,
                "answer": answer.strip(),
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def plot_layer_projection(
    coords: torch.Tensor,
    persona_names: List[str],
    questions: List[str],
    layer: int,
    outdir: Path,
    suffix: str = "",
) -> None:
    plt.figure(figsize=(9, 7))
    unique_personas = sorted(set(persona_names))

    for persona in unique_personas:
        idxs = [i for i, p in enumerate(persona_names) if p == persona]
        plt.scatter(
            coords[idxs, 0].numpy(),
            coords[idxs, 1].numpy(),
            label=persona,
            alpha=0.8,
        )

        for i in idxs:
            label = questions[i][:12]
            plt.annotate(label, (coords[i, 0].item(), coords[i, 1].item()), fontsize=7, alpha=0.75)

    title_suffix = f" ({suffix})" if suffix else ""
    plt.title(f"Persona activation clusters - layer {layer}{title_suffix}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fname = f"layer_{layer:02d}_pca{('_' + suffix) if suffix else ''}.png"
    plt.savefig(outdir / fname, dpi=180)
    plt.close()


def save_centroid_heatmap(
    vectors: torch.Tensor,
    persona_names: List[str],
    layer: int,
    outdir: Path,
    suffix: str = "",
) -> None:
    personas = sorted(set(persona_names))
    persona_to_vecs: Dict[str, List[torch.Tensor]] = {p: [] for p in personas}
    for v, p in zip(vectors, persona_names):
        persona_to_vecs[p].append(v)

    centroids = []
    for p in personas:
        mat = torch.stack(persona_to_vecs[p], dim=0)
        c = F.normalize(mat.mean(dim=0), dim=0)
        centroids.append(c)
    centroid_mat = torch.stack(centroids, dim=0)
    sim = centroid_mat @ centroid_mat.T

    plt.figure(figsize=(8, 6))
    plt.imshow(sim.numpy(), aspect="auto")
    plt.colorbar(label="cosine similarity")
    plt.xticks(range(len(personas)), personas, rotation=45, ha="right")
    plt.yticks(range(len(personas)), personas)
    title_suffix = f" ({suffix})" if suffix else ""
    plt.title(f"Persona centroid similarity - layer {layer}{title_suffix}")
    plt.tight_layout()
    fname = f"layer_{layer:02d}_centroid_similarity{('_' + suffix) if suffix else ''}.png"
    plt.savefig(outdir / fname, dpi=180)
    plt.close()


# ============================================================
# Improvement 3: null baseline — plot original vs alt centroids
# ============================================================

def plot_null_baseline_similarity(
    orig_vectors: torch.Tensor,
    alt_vectors: torch.Tensor,
    orig_names: List[str],
    alt_names: List[str],
    layer: int,
    outdir: Path,
) -> Dict[str, float]:
    """
    For each persona, compute centroid from original wording and alt wording,
    then measure cosine similarity between matching pairs vs non-matching pairs.
    """
    personas = sorted(set(orig_names))

    def compute_centroids(vecs, names):
        persona_to_vecs = {p: [] for p in personas}
        for v, p in zip(vecs, names):
            persona_to_vecs[p].append(v)
        centroids = {}
        for p in personas:
            mat = torch.stack(persona_to_vecs[p], dim=0)
            centroids[p] = F.normalize(mat.mean(dim=0), dim=0)
        return centroids

    orig_c = compute_centroids(orig_vectors, orig_names)
    alt_c = compute_centroids(alt_vectors, alt_names)

    # Build similarity matrix: orig centroids (rows) vs alt centroids (cols)
    k = len(personas)
    sim_mat = np.zeros((k, k))
    for i, p1 in enumerate(personas):
        for j, p2 in enumerate(personas):
            sim_mat[i, j] = (orig_c[p1] @ alt_c[p2]).item()

    same_persona_sims = [sim_mat[i, i] for i in range(k)]
    diff_persona_sims = [sim_mat[i, j] for i in range(k) for j in range(k) if i != j]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sim_mat, aspect="auto", vmin=min(sim_mat.min(), 0), vmax=1.0)
    plt.colorbar(im, ax=ax, label="cosine similarity")
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{sim_mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if sim_mat[i, j] > 0.7 else "black", fontsize=8)
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"{p} (alt)" for p in personas], rotation=45, ha="right")
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"{p} (orig)" for p in personas])
    ax.set_title(f"Null baseline: original vs rephrased centroids - layer {layer}")
    plt.tight_layout()
    plt.savefig(outdir / f"layer_{layer:02d}_null_baseline.png", dpi=180)
    plt.close()

    return {
        "null_same_persona_sim": float(np.mean(same_persona_sims)),
        "null_diff_persona_sim": float(np.mean(diff_persona_sims)),
        "null_gap": float(np.mean(same_persona_sims) - np.mean(diff_persona_sims)),
    }


# ============================================================
# Main
# ============================================================

def analyze_layer(
    vectors: torch.Tensor,
    persona_names: List[str],
    question_texts: List[str],
    persona_ids: List[int],
    question_ids: List[int],
    persona_vocab: List[str],
    layer: int,
    outdir: Path,
    seed: int,
    n_seeds: int = 10,
    suffix: str = "",
) -> Dict:
    """Run all per-layer analyses and return a metrics dict."""
    coords = pca_2d_torch(vectors)
    plot_layer_projection(coords, persona_names, question_texts, layer, outdir, suffix=suffix)
    save_centroid_heatmap(vectors, persona_names, layer, outdir, suffix=suffix)

    pred_labels, _ = kmeans_torch(
        vectors,
        k=len(persona_vocab),
        n_iters=50,
        seed=seed,
    )
    purity = clustering_purity(pred_labels, persona_ids, k=len(persona_vocab))
    pair_metrics = pairwise_metrics_by_group(vectors, persona_ids, question_ids)

    # Improvement 1: confusion matrix
    plot_confusion_matrix(pred_labels, persona_ids, persona_vocab, layer, outdir)

    # Improvement 4: linear probe
    probe_acc = linear_probe_accuracy(vectors, persona_ids, n_folds=5, seed=seed)

    # Improvement 6: multi-seed evaluation
    ms = multi_seed_eval(vectors, persona_ids, persona_vocab, k=len(persona_vocab), n_seeds=n_seeds)

    # Improvement 7: per-persona F1
    report = per_persona_f1(vectors, persona_ids, persona_vocab, seed=seed)
    plot_per_persona_f1(report, persona_vocab, layer, outdir)
    per_persona = {f"f1_{p}": report[p]["f1-score"] for p in persona_vocab}

    # Improvement 8: persona subspace dimensionality
    subspace = persona_subspace_analysis(vectors, persona_names, layer, outdir)

    row = {
        "layer": layer,
        "n_examples": vectors.shape[0],
        "n_personas": len(persona_vocab),
        "kmeans_persona_purity": purity,
        "linear_probe_accuracy": probe_acc,
        **ms,
        **subspace,
        **pair_metrics,
        **per_persona,
    }
    return row


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = infer_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model on {device}...")
    model_kwargs = {
        "output_hidden_states": True,
    }
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

    print(f"Using layers: {layer_indices}")

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

    print(f"Built {len(examples)} persona-question prompts ({len(personas)} personas x {len(questions)} questions)")

    # ---- Collect hidden vectors for main experiment ----
    print("Collecting input-token hidden states...")
    by_layer = collect_hidden_vectors(
        model=model,
        examples=examples,
        device=device,
        dtype=model.dtype,
        layer_indices=layer_indices,
    )

    persona_names = [ex.persona_name for ex in examples]
    question_texts = [ex.question for ex in examples]

    persona_vocab = sorted(set(persona_names))
    question_vocab = sorted(set(question_texts))
    persona_to_id = {p: i for i, p in enumerate(persona_vocab)}
    question_to_id = {q: i for i, q in enumerate(question_vocab)}
    persona_ids = [persona_to_id[p] for p in persona_names]
    question_ids = [question_to_id[q] for q in question_texts]

    # ---- Main per-layer analysis ----
    metrics_rows = []
    for layer in layer_indices:
        vectors = torch.stack(by_layer[layer], dim=0)
        vectors = F.normalize(vectors, dim=-1)

        row = analyze_layer(
            vectors, persona_names, question_texts,
            persona_ids, question_ids, persona_vocab,
            layer, outdir, args.seed, n_seeds=args.n_seeds,
        )
        metrics_rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if not k.startswith("f1_")}, indent=2))

    # ---- Improvement 3: null baseline with rephrased personas ----
    if not args.skip_null_baseline:
        print("\nRunning null baseline (rephrased persona instructions)...")
        personas_alt = {k: v for k, v in PERSONAS_ALT.items() if k in personas}
        alt_examples = build_examples(
            tokenizer=tokenizer,
            personas=personas_alt,
            questions=questions,
            max_length=args.max_length,
        )
        alt_by_layer = collect_hidden_vectors(
            model=model,
            examples=alt_examples,
            device=device,
            dtype=model.dtype,
            layer_indices=layer_indices,
        )
        alt_persona_names = [ex.persona_name for ex in alt_examples]

        null_rows = []
        for layer in layer_indices:
            orig_vecs = torch.stack(by_layer[layer], dim=0)
            orig_vecs = F.normalize(orig_vecs, dim=-1)
            alt_vecs = torch.stack(alt_by_layer[layer], dim=0)
            alt_vecs = F.normalize(alt_vecs, dim=-1)

            null_metrics = plot_null_baseline_similarity(
                orig_vecs, alt_vecs, persona_names, alt_persona_names, layer, outdir,
            )
            null_metrics["layer"] = layer
            null_rows.append(null_metrics)
            print(f"  Layer {layer}: same={null_metrics['null_same_persona_sim']:.4f} "
                  f"diff={null_metrics['null_diff_persona_sim']:.4f} "
                  f"gap={null_metrics['null_gap']:.4f}")

        null_df = pd.DataFrame(null_rows).sort_values("layer")
        null_df.to_csv(outdir / "null_baseline_metrics.csv", index=False)

        # Plot null baseline gap across layers
        plt.figure(figsize=(8, 5))
        plt.plot(null_df["layer"], null_df["null_same_persona_sim"], marker="o", label="same persona (orig vs alt)")
        plt.plot(null_df["layer"], null_df["null_diff_persona_sim"], marker="o", label="diff persona (orig vs alt)")
        plt.xlabel("Layer")
        plt.ylabel("Cosine similarity")
        plt.title("Null baseline: do rephrased personas map to same representation?")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "null_baseline_by_layer.png", dpi=180)
        plt.close()

    # ---- Improvement 5: generated-token activations ----
    if not args.skip_gen_activations:
        print("\nCollecting generated-token hidden states...")
        gen_by_layer = collect_generated_hidden_vectors(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            device=device,
            layer_indices=layer_indices,
            max_new_tokens=args.max_new_tokens,
            n_gen_positions=5,
        )

        gen_metrics_rows = []
        for layer in layer_indices:
            gen_vectors = torch.stack(gen_by_layer[layer], dim=0)
            gen_vectors = F.normalize(gen_vectors, dim=-1)

            coords = pca_2d_torch(gen_vectors)
            plot_layer_projection(coords, persona_names, question_texts, layer, outdir, suffix="gen")
            save_centroid_heatmap(gen_vectors, persona_names, layer, outdir, suffix="gen")

            pred_labels, _ = kmeans_torch(gen_vectors, k=len(persona_vocab), n_iters=50, seed=args.seed)
            purity = clustering_purity(pred_labels, persona_ids, k=len(persona_vocab))
            probe_acc = linear_probe_accuracy(gen_vectors, persona_ids, n_folds=5, seed=args.seed)
            pair_metrics = pairwise_metrics_by_group(gen_vectors, persona_ids, question_ids)

            row = {
                "layer": layer,
                "source": "generated_tokens",
                "kmeans_persona_purity": purity,
                "linear_probe_accuracy": probe_acc,
                **pair_metrics,
            }
            gen_metrics_rows.append(row)
            print(f"  Layer {layer} (gen): purity={purity:.3f} probe={probe_acc:.3f} "
                  f"persona_gap={pair_metrics['persona_separation_gap']:.4f}")

        gen_df = pd.DataFrame(gen_metrics_rows).sort_values("layer")
        gen_df.to_csv(outdir / "gen_token_metrics.csv", index=False)

        # Comparison plot: input-token vs generated-token purity & probe accuracy
        metrics_df_tmp = pd.DataFrame(metrics_rows).sort_values("layer")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(metrics_df_tmp["layer"], metrics_df_tmp["kmeans_persona_purity"], marker="o", label="input token")
        axes[0].plot(gen_df["layer"], gen_df["kmeans_persona_purity"], marker="s", label="generated tokens")
        axes[0].set_xlabel("Layer")
        axes[0].set_ylabel("K-means persona purity")
        axes[0].set_title("Purity: input token vs generated tokens")
        axes[0].legend()

        axes[1].plot(metrics_df_tmp["layer"], metrics_df_tmp["linear_probe_accuracy"], marker="o", label="input token")
        axes[1].plot(gen_df["layer"], gen_df["linear_probe_accuracy"], marker="s", label="generated tokens")
        axes[1].set_xlabel("Layer")
        axes[1].set_ylabel("Linear probe accuracy")
        axes[1].set_title("Linear probe: input token vs generated tokens")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(outdir / "input_vs_gen_comparison.png", dpi=180)
        plt.close()

    # ---- Save metrics and summary plots ----
    metrics_df = pd.DataFrame(metrics_rows).sort_values("layer")
    metrics_df.to_csv(outdir / "layer_metrics.csv", index=False)

    # Purity + linear probe plot WITH ERROR BARS
    fig, ax1 = plt.subplots(figsize=(8, 5))
    layers = metrics_df["layer"].values
    ax1.errorbar(
        layers, metrics_df["kmeans_purity_mean"], yerr=metrics_df["kmeans_purity_std"],
        marker="o", color="tab:blue", capsize=4, label=f"k-means purity (mean ± std, n={args.n_seeds} seeds)",
    )
    ax1.errorbar(
        layers, metrics_df["probe_accuracy_mean"], yerr=metrics_df["probe_accuracy_std"],
        marker="s", color="tab:red", capsize=4, label=f"linear probe (mean ± std, n={args.n_seeds} seeds)",
    )
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Score")
    ax1.set_title("Persona classification by layer (with error bars)")
    ax1.legend()
    plt.tight_layout()
    plt.savefig(outdir / "persona_purity_by_layer.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["layer"], metrics_df["persona_separation_gap"], marker="o", label="persona gap")
    plt.plot(metrics_df["layer"], metrics_df["question_separation_gap"], marker="o", label="question gap")
    plt.xlabel("Layer")
    plt.ylabel("Mean cosine gap")
    plt.title("Persona vs question separation by layer")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "separation_gaps_by_layer.png", dpi=180)
    plt.close()

    # Per-persona F1 heatmap across layers
    f1_cols = [c for c in metrics_df.columns if c.startswith("f1_")]
    if f1_cols:
        f1_data = metrics_df[f1_cols].values  # [n_layers, n_personas]
        persona_labels = [c.replace("f1_", "") for c in f1_cols]

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(f1_data.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="F1 score")
        for i in range(len(persona_labels)):
            for j in range(len(layers)):
                ax.text(j, i, f"{f1_data[j, i]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers.astype(int))
        ax.set_yticks(range(len(persona_labels)))
        ax.set_yticklabels(persona_labels)
        ax.set_xlabel("Layer")
        ax.set_title("Per-persona F1 score across layers")
        plt.tight_layout()
        plt.savefig(outdir / "per_persona_f1_heatmap.png", dpi=180)
        plt.close()

    # Subspace dimensionality summary
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, metrics_df["persona_dims_95"], marker="o", label="dims for 95% persona variance")
    ax.plot(layers, metrics_df["persona_dims_99"], marker="s", label="dims for 99% persona variance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Number of dimensions")
    ax.set_title("Persona subspace dimensionality by layer")
    ax.legend()
    ax.set_ylim(0, max(metrics_df["persona_dims_99"].max() + 1, 8))
    plt.tight_layout()
    plt.savefig(outdir / "subspace_dimensionality.png", dpi=180)
    plt.close()

    if args.generate_answers:
        maybe_generate_answers(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            device=device,
            max_new_tokens=args.max_new_tokens,
            out_path=outdir / "generated_answers.csv",
        )

    config = {
        "model_name": args.model_name,
        "device": str(device),
        "num_layers": num_layers,
        "analyzed_layers": layer_indices,
        "n_personas": len(persona_vocab),
        "n_questions": len(question_vocab),
        "n_examples": len(examples),
        "representation": "last input token hidden state before generation",
        "n_seeds": args.n_seeds,
        "improvements": [
            "confusion_matrix",
            "expanded_questions_40",
            "null_baseline_rephrased_personas",
            "linear_probe_logreg",
            "generated_token_activations",
            "multi_seed_error_bars",
            "per_persona_f1",
            "persona_subspace_dimensionality",
        ],
    }
    with open(outdir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. Outputs written to: {outdir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persona activation clustering experiment")
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/gemma-4-E2B-it",
        help="HF model name.",
    )
    parser.add_argument("--outdir", type=str, default="outputs/persona_clusters")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, mps, or leave unset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--limit-personas", type=int, default=0)
    parser.add_argument("--limit-questions", type=int, default=0)
    parser.add_argument("--generate-answers", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of seeds for multi-seed evaluation")
    parser.add_argument("--skip-null-baseline", action="store_true", help="Skip rephrased-persona baseline")
    parser.add_argument("--skip-gen-activations", action="store_true", help="Skip generated-token analysis")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
