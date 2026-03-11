#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "torch>=2.2",
#   "transformers>=4.43",
#   "matplotlib>=3.8",
#   "pandas>=2.2",
#   "python-dotenv>=1.0",
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
import pandas as pd
import torch
import torch.nn.functional as F
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
# This is deliberately a FIRST PASS experiment. If you see separation,
# the natural follow-up is to inspect generated-token activations too.
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

QUESTIONS: List[str] = [
    "Why do eclipses happen?",
    "Should governments regulate advanced AI systems?",
    "What is the best way to comfort a friend after a failure?",
    "Why did the Roman Empire fall?",
    "Is nuclear energy a good idea?",
    "How should a journalist verify a leaked document?",
    "What causes inflation?",
    "Should children learn more than one language?",
    "Why do people disagree about moral issues?",
    "What should someone do before investing in a volatile stock?",
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
    u, s, _ = torch.pca_lowrank(x, q=q, center=False)
    coords = x @ _[:, :2]
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

        # Label only one point per question initial to avoid chaos.
        for i in idxs:
            label = questions[i][:12]
            plt.annotate(label, (coords[i, 0].item(), coords[i, 1].item()), fontsize=7, alpha=0.75)

    plt.title(f"Persona activation clusters - layer {layer}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / f"layer_{layer:02d}_pca.png", dpi=180)
    plt.close()


def save_centroid_heatmap(
    vectors: torch.Tensor,
    persona_names: List[str],
    layer: int,
    outdir: Path,
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
    plt.title(f"Persona centroid similarity - layer {layer}")
    plt.tight_layout()
    plt.savefig(outdir / f"layer_{layer:02d}_centroid_similarity.png", dpi=180)
    plt.close()


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

    print(f"Built {len(examples)} persona-question prompts")

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

    metrics_rows = []

    for layer in layer_indices:
        vectors = torch.stack(by_layer[layer], dim=0)  # [n_examples, d]
        vectors = F.normalize(vectors, dim=-1)

        coords = pca_2d_torch(vectors)
        plot_layer_projection(coords, persona_names, question_texts, layer, outdir)
        save_centroid_heatmap(vectors, persona_names, layer, outdir)

        pred_labels, _ = kmeans_torch(
            vectors,
            k=len(persona_vocab),
            n_iters=50,
            seed=args.seed,
        )
        purity = clustering_purity(pred_labels, persona_ids, k=len(persona_vocab))
        pair_metrics = pairwise_metrics_by_group(vectors, persona_ids, question_ids)

        row = {
            "layer": layer,
            "n_examples": vectors.shape[0],
            "n_personas": len(persona_vocab),
            "kmeans_persona_purity": purity,
            **pair_metrics,
        }
        metrics_rows.append(row)
        print(json.dumps(row, indent=2))

    metrics_df = pd.DataFrame(metrics_rows).sort_values("layer")
    metrics_df.to_csv(outdir / "layer_metrics.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["layer"], metrics_df["kmeans_persona_purity"], marker="o")
    plt.xlabel("Layer")
    plt.ylabel("K-means persona purity")
    plt.title("Persona clustering strength by layer")
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
    }
    with open(outdir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. Outputs written to: {outdir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persona activation clustering experiment")
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HF model name. Note: official Llama 3 release is 8B/70B, not 7B.",
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
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
