# Persona Landscape Experiments

Empirical tests of the Persona Landscape framework across multiple language models. The framework proposes that LMs learn a continuous space of behavioural traits, and personas are points in that space — with basin structure, trait coupling, and self-reinforcing dynamics.

## What this does

Given an instruction-tuned model, this codebase:

1. **Extracts persona representations** — runs forward passes with 10 persona system prompts across 40 questions, pulling hidden states at the last input token across sampled layers.
2. **Probes for structure** — trains linear classifiers, measures clustering purity, computes cosine separation, runs PCA and SVD on persona/trait subspaces.
3. **Builds and applies steering vectors** — computes persona centroids, sweeps (layer, alpha) combinations, measures flip rates and cosine displacement under steering.
4. **Tests framework predictions** — trait geometry universality, basin transitions, coupling coefficients, coherence manifolds, self-reinforcement dynamics.

All results are logged to [Weights & Biases](https://wandb.ai) and saved locally.

## Quick start

```bash
# 1. Set up credentials
echo "HF_TOKEN=hf_your_token" > .env
echo "WANDB_API_KEY=your_key" >> .env

# 2. Run everything for a model
./run_all_experiments.sh --model meta-llama/Meta-Llama-3-8B-Instruct

# Smoke test first (2 personas, 5 questions — minutes not hours)
./run_all_experiments.sh --model google/gemma-4-E4B-it --smoke

# Run a single phase
./run_all_experiments.sh --model google/gemma-4-E4B-it --phase 0

# Run a single experiment
./run_all_experiments.sh --model google/gemma-4-E4B-it --only oq1_coupling_coefficients

# Or run scripts individually
uv run main.py --model-name google/gemma-4-E4B-it --outdir outputs/gemma-4-E4B-it/persona_clusters
```

The `--model` flag is required. It determines both the HuggingFace model to load and the output directory structure.

## Experiments

Experiments are grouped into phases that run sequentially:

| Phase | Experiment | Script | What it tests |
|---|---|---|---|
| 0 | Clustering | `main.py` | Do personas form linearly separable clusters? At which layers? |
| 0 | Steering | `steer.py` | Can steering vectors shift persona classification? What's the optimal (layer, alpha)? |
| 1 | Trait Geometry | `experiments/prediction_1_trait_geometry/run.py` | Are trait directions (honesty, humor, etc.) shared across personas? |
| 1 | Dimensionality | `experiments/oq2_dimensionality/run.py` | How many independent dimensions does the persona/trait space have? |
| 1 | Activation vs. Weight | `experiments/oq6_activation_vs_weight/run.py` | Do steering vectors align with weight matrix SVD structure? |
| 1 | Basin Transitions | `experiments/prediction_2_basin_transitions/run.py` | Do personas occupy discrete basins with sigmoid transitions under steering? |
| 2 | Coupling Coefficients | `experiments/oq1_coupling_coefficients/run.py` | When you steer one trait, how much do others move? |
| 2 | Coherence Manifold | `experiments/oq3_coherence_manifold/run.py` | Do arbitrary trait-space points produce coherent output? |
| 2 | Self-Reinforcement | `experiments/prediction_3_self_reinforcement/run.py` | Does multi-turn generation reinforce the current basin position? |

Phase 2 experiments depend on trait vectors computed in Phase 1. Phase 3 (fine-tuning, cross-model) is stubbed out.

## Personas

10 personas with distinct behavioural profiles:

| Persona | Style |
|---------|-------|
| assistant | Helpful, careful, truthful |
| pirate | Nautical language, swagger |
| lawyer | Legalistic precision, caveats |
| scientist | Analytical, cautious, explicit reasoning |
| comedian | Wit, playfulness, punchy phrasing |
| stoic | Calm, terse, emotionally restrained |
| conspiracy_host | Suspicious, dramatic framing |
| kind_teacher | Gentle, clear, pedagogical |
| drill_sergeant | Commanding, direct, no-nonsense |
| diplomat | Careful, balanced, tactful |

## Traits

8 behavioural dimensions extracted via HIGH/LOW modifiers in the system prompt:

`honesty, assertiveness, warmth, deference, analytical_rigor, humor, suspicion, impulsivity`

Trait vectors are computed as `centroid(HIGH activations) - centroid(LOW activations)`, then averaged across personas for global vectors.

## Output structure

Outputs are organized by model, then by experiment:

```
outputs/
├── Meta-Llama-3-8B-Instruct/
│   ├── persona_clusters/           # Phase 0: probes, PCA, confusion matrices
│   ├── persona_steering/           # Phase 0: steering vectors, sweep heatmaps
│   ├── prediction_1_trait_geometry/ # Cosine similarity, residual norms, SVD
│   ├── oq2_dimensionality/         # SVD spectra, probe accuracy curves
│   ├── oq6_activation_vs_weight/   # Weight alignment tables
│   ├── prediction_2_basin_transitions/  # Sigmoid fits, PCA trajectories
│   ├── oq1_coupling_coefficients/  # Coupling matrices, per-persona heatmaps
│   ├── oq3_coherence_manifold/     # Coherence scores, manifold SVD
│   └── prediction_3_self_reinforcement/ # Cohen's d, flip rates, vulnerability ranks
├── gemma-4-E4B-it/
│   └── ...
```

Each experiment directory contains:
- `run_config.json` — full parameters used
- `*.csv` — raw metric tables
- `*.png` — visualizations

## Experiment tracking (W&B)

All scripts log to the `personas_original` W&B project.

Each run includes:
- **Name**: `<model>/<experiment>` (e.g. `gemma-4-E4B-it/phase0_clustering`)
- **Tags**: `model:<name>` and `experiment:<name>` for filtering
- **Config**: all CLI args and derived parameters
- **Artifacts**: every PNG as a W&B Image, every CSV as a W&B Table

Set `WANDB_API_KEY` in `.env` or run `wandb login` before your first run.

## How steering works

1. Extract activations for all (persona, question) pairs at each layer
2. Compute steering vectors: `steering_vec = mean(persona_activations) - global_mean`
3. Inject via forward hook: add `alpha * steering_vec` to the residual stream at the target layer
4. Evaluate: measure persona classifier flip rate, cosine displacement, generated text quality

## Requirements

- Python 3.10+
- GPU with ~16GB VRAM (for 8B models in fp16), or Apple Silicon Mac with sufficient unified memory
- [uv](https://docs.astral.sh/uv/) for dependency management
- HuggingFace token (for gated models like Llama 3)

Dependencies are managed inline via `uv run --script` — no separate install step needed. Each script declares its own deps in its header.

## CLI reference

### `run_all_experiments.sh`

```
--model MODEL       HuggingFace model ID (required)
--smoke             Smoke test: 2 personas, 5 questions
--phase N           Run only phase N (0, 1, 2, 3, or all)
--only NAME         Run only the named experiment
```

### `main.py` (Phase 0: clustering)

```
--model-name        HF model ID
--outdir            Output directory
--device            cpu, cuda, mps, or auto-detect
--seed              Random seed (default: 0)
--n-seeds           Seeds for multi-seed evaluation (default: 10)
--layer-stride      Sample every N layers (default: 4)
--limit-personas    Use only first N personas
--limit-questions   Use only first N questions
--skip-null-baseline   Skip rephrased-persona analysis
--skip-gen-activations Skip generated-token analysis
```

### `steer.py` (Phase 0: steering)

```
--model-name        HF model ID
--outdir            Output directory
--alphas            Comma-separated steering coefficients (default: 1,3,5,8,10,15)
--target-persona    Steer toward this persona, or 'all' (default: all)
--skip-demo         Skip qualitative demo generation
```

All experiment scripts accept `--model-name`, `--outdir`, `--device`, `--seed`, `--layer-stride`, `--limit-personas`, and `--limit-questions`.
