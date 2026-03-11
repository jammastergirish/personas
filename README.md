# Persona Activation Clustering

Do language models build internal representations of *who they're pretending to be*?

This experiment probes **Llama 3 8B Instruct** to find out. We give the model 8 different persona instructions (pirate, lawyer, scientist, comedian, etc.) across 40 diverse questions, extract hidden states at every layer, and ask: do personas form distinct geometric clusters in activation space?

**The answer is yes** — and the structure is richer than expected.

## Key findings

- **99.3% linear probe accuracy from layer 8 onward.** Persona identity is fully linearly separable well before the model's midpoint.
- **The model encodes semantic roles, not surface tokens.** Rephrased persona instructions ("You are a pirate" vs "You are a seafaring buccaneer") converge to the same representation (0.96 cosine similarity).
- **Persona identity lives in a 6-7 dimensional subspace** out of 4096 dimensions.
- **A crossover occurs at layer 8-12:** early layers organize by question content, later layers organize by persona identity.

See [RESULTS.md](RESULTS.md) for the full writeup with figures.

## Quick start

```bash
# Requires a HuggingFace token with Llama 3 access
echo "HF_TOKEN=hf_your_token" > .env

# Full experiment (takes ~40 min on a single GPU)
uv run main.py

# Quick run — skip the slower analyses
uv run main.py --skip-null-baseline --skip-gen-activations

# Fewer questions for a fast sanity check
uv run main.py --limit-questions 10 --skip-null-baseline --skip-gen-activations
```

## What it does

For each of 8 personas × 40 questions = 320 prompts:

1. Builds a chat prompt with the persona as the system message
2. Runs a forward pass with `output_hidden_states=True`
3. Extracts the **last input token's hidden state** at each sampled layer (the model's representation right before it starts generating)
4. Compares these vectors across personas and questions

### Analyses

| Analysis | What it measures |
|----------|-----------------|
| **K-means clustering** | Do personas form compact clusters? (with multi-seed error bars) |
| **Linear probe** | Is persona identity linearly decodable? (logistic regression, 5-fold CV) |
| **Confusion matrix** | Which persona pairs does k-means confuse? |
| **Null baseline** | Do rephrased persona instructions map to the same representation? |
| **Per-persona F1** | Which personas are hardest/easiest to classify? |
| **Subspace dimensionality** | How many PCA dimensions capture persona variance? |
| **Generated-token activations** | Does persona signal persist after generation begins? |

### Outputs

All outputs go to `outputs/persona_clusters/`:

- `layer_XX_pca.png` — 2D PCA projections colored by persona
- `layer_XX_confusion.png` — K-means confusion matrices
- `layer_XX_centroid_similarity.png` — Cosine similarity between persona centroids
- `layer_XX_null_baseline.png` — Original vs rephrased persona centroid similarity
- `layer_XX_per_persona_f1.png` — Per-persona precision/recall/F1
- `layer_XX_subspace_dims.png` — PCA variance explained for persona centroids
- `persona_purity_by_layer.png` — K-means purity + linear probe accuracy with error bars
- `separation_gaps_by_layer.png` — Persona vs question cosine separation
- `per_persona_f1_heatmap.png` — F1 scores for all personas across all layers
- `subspace_dimensionality.png` — Dims needed for 95%/99% persona variance
- `input_vs_gen_comparison.png` — Input-token vs generated-token metrics
- `layer_metrics.csv` — All metrics in tabular form

## CLI options

```
--model-name        HF model name (default: meta-llama/Meta-Llama-3-8B-Instruct)
--outdir            Output directory (default: outputs/persona_clusters)
--device            cpu, cuda, mps, or auto-detect
--seed              Random seed (default: 0)
--n-seeds           Number of seeds for multi-seed evaluation (default: 10)
--layer-stride      Sample every N layers (default: 4)
--all-layers        Analyze every layer
--limit-personas    Use only the first N personas
--limit-questions   Use only the first N questions
--generate-answers  Also generate and save text answers
--max-new-tokens    Max tokens for generation (default: 80)
--skip-null-baseline   Skip the rephrased-persona analysis
--skip-gen-activations Skip the generated-token analysis
```

## Personas

| Persona | System prompt |
|---------|--------------|
| assistant | Helpful, careful AI assistant |
| pirate | Nautical language and swagger |
| lawyer | Legalistic precision and caveats |
| scientist | Analytical, cautious, explicit reasoning |
| comedian | Wit, playfulness, punchy phrasing |
| stoic | Calm, terse, emotionally restrained |
| conspiracy_host | Suspicious, dramatic framing |
| kind_teacher | Gentle, clear, pedagogical |

## Requirements

- Python 3.10+
- A GPU with ~16GB VRAM (for Llama 3 8B in bf16)
- [uv](https://docs.astral.sh/uv/) for dependency management
- HuggingFace token with Llama 3 access

Dependencies are managed inline via `uv run --script` — no separate install step needed.
