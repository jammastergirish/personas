# Persona Landscape Experiments

Empirical tests of the [Persona Landscape framework](BLOGPOST.md) on **Llama 3 8B Instruct**. The framework proposes that language models learn a continuous space of behavioural traits, and personas are points in that space — with basin structure, trait coupling, and self-reinforcing dynamics.

## Key findings

See [RESULTS_ALL.md](RESULTS_ALL.md) for the full analysis with per-experiment Hypothesis / Operationalization / Results / Analysis / Limitations.

**Supported:**
- Personas occupy distinct basins (sigmoid transitions, R² = 0.98)
- Traits are coupled in interpretable clusters ("seriousness" vs. "playfulness", rank-3 coupling matrix)
- Self-reinforcement is real (vulnerability subspace rank drops 91 → 26 over 6 turns)
- The space is low-dimensional (5-7 dimensions consistently)

**Challenged:**
- Trait universality is weaker than claimed (cosine 0.35-0.51 vs. reported 0.82)
- The coherence manifold is flat — no incoherent off-manifold regions detected
- Traits capture only ~36% of persona identity

**Untested:**
- Level 1 (weight-space) vs. Level 2 (activation-space) durability — requires fine-tuning experiments
- Cross-model universality — all results from a single model

## Experiments

| Experiment | Script | What it tests |
|---|---|---|
| Phase 0: Clustering & Steering | `main.py`, `steer.py` | Do personas form geometric clusters? Can steering vectors shift persona? |
| Prediction 1: Trait Geometry | `experiments/prediction_1_trait_geometry/run.py` | Are trait directions shared across personas? |
| OQ2: Dimensionality | `experiments/oq2_dimensionality/run.py` | How many dimensions does the landscape have? |
| OQ6: Activation vs. Weight | `experiments/oq6_activation_vs_weight/run.py` | Do steering vectors align with weight matrix structure? |
| Prediction 2: Basin Transitions | `experiments/prediction_2_basin_transitions/run.py` | Do personas occupy discrete basins with sigmoid transitions? |
| OQ1: Coupling Coefficients | `experiments/oq1_coupling_coefficients/run.py` | When you steer one trait, how much do others move? |
| OQ3: Coherence Manifold | `experiments/oq3_coherence_manifold/run.py` | Do arbitrary trait-space points produce coherent output? |
| Prediction 3: Self-Reinforcement | `experiments/prediction_3_self_reinforcement/run.py` | Does generation reinforce the current basin position? |

## Quick start

```bash
# Requires a HuggingFace token with Llama 3 access
echo "HF_TOKEN=hf_your_token" > .env

# Phase 0: clustering & probing (~40 min on a single GPU)
uv run main.py

# Phase 0: steering vectors
uv run steer.py

# Run all Persona Landscape experiments (hours)
./run_all_experiments.sh

# Smoke test (2 personas, 5 questions)
./run_all_experiments.sh --smoke

# Run a single experiment
./run_all_experiments.sh --only oq1_coupling_coefficients

# Run by phase
./run_all_experiments.sh --phase 1   # foundational
./run_all_experiments.sh --phase 2   # coupling, coherence, multi-turn
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
| drill_sergeant | Commanding, direct, no-nonsense |
| diplomat | Careful, balanced, tactful |

## Traits

honesty, assertiveness, warmth, deference, analytical_rigor, humor, suspicion, impulsivity

Extracted via HIGH/LOW modifiers in the system prompt. Trait vectors = centroid(HIGH) - centroid(LOW), averaged across personas for global vectors.

## Outputs

```
outputs/
├── persona_clusters/              # Phase 0: probes, PCA, confusion matrices, null baseline
├── persona_steering/              # Phase 0: steering vectors, sweep results
├── prediction_1_trait_geometry/    # Cosine similarity, residual norms, SVD
├── oq2_dimensionality/            # SVD spectra, probe accuracy curves
├── oq6_activation_vs_weight/      # Weight alignment tables
├── prediction_2_basin_transitions/ # Sigmoid fits, transition trajectories
├── oq1_coupling_coefficients/     # Coupling matrices, SVD, per-persona heatmaps
├── oq3_coherence_manifold/        # Coherence scores, manifold SVD
└── prediction_3_self_reinforcement/ # Cohen's d, flip rates, vulnerability ranks
```

## How steering works

1. Extract activations for all (persona, question) pairs at each layer
2. Compute steering vectors: `steering_vec = mean(persona_activations) - baseline`
3. Inject via forward hook: add `alpha * steering_vec` to the residual stream at the target layer
4. Evaluate: measure persona classifier flip rate, cosine displacement, generated text quality

## Requirements

- Python 3.10+
- A GPU with ~16GB VRAM (for Llama 3 8B in bf16)
- [uv](https://docs.astral.sh/uv/) for dependency management
- HuggingFace token with Llama 3 access

Dependencies are managed inline via `uv run --script` — no separate install step needed.

## CLI options

### `main.py` (Phase 0 analysis)

```
--model-name        HF model name (default: meta-llama/Meta-Llama-3-8B-Instruct)
--outdir            Output directory (default: outputs/persona_clusters)
--device            cpu, cuda, mps, or auto-detect
--seed              Random seed (default: 0)
--n-seeds           Number of seeds for multi-seed evaluation (default: 10)
--layer-stride      Sample every N layers (default: 4)
--limit-personas    Use only the first N personas
--limit-questions   Use only the first N questions
--skip-null-baseline   Skip the rephrased-persona analysis
--skip-gen-activations Skip the generated-token analysis
```

### `steer.py` (Phase 0 steering)

```
--model-name        HF model name (default: meta-llama/Meta-Llama-3-8B-Instruct)
--outdir            Output directory (default: outputs/persona_steering)
--alphas            Comma-separated steering coefficients (default: 1,3,5,8,10,15)
--target-persona    Steer toward this persona, or 'all' (default: all)
--skip-demo         Skip qualitative demo generation
```
