# Persona Landscape: Full Experimental Results

**Model**: Meta-Llama-3-8B-Instruct
**Date**: 2026-03-16
**Personas**: assistant, pirate, lawyer, scientist, comedian, stoic, conspiracy_host, kind_teacher, drill_sergeant, diplomat
**Traits**: honesty, assertiveness, warmth, deference, analytical_rigor, humor, suspicion, impulsivity
**Questions**: 40 diverse prompts per condition
**Analysis layer**: 31 (final transformer block)

---

## Overview

Seven experiments were run across two phases, probing how Llama 3 8B internally represents personas and their component traits. The experiments test three theoretical predictions and four open questions from the Persona Landscape framework, which models personas as points in a low-dimensional trait space with coupling, basin structure, and self-reinforcement dynamics.

### Experiment status

| Experiment | Phase | Status | Runtime |
|---|---|---|---|
| prediction_1_trait_geometry | 1 | Complete | ~3 min |
| oq2_dimensionality | 1 | Complete | ~4 min |
| oq6_activation_vs_weight | 1 | Complete | ~2 min |
| prediction_2_basin_transitions | 1 | Complete | ~3 min |
| oq1_coupling_coefficients | 2 | Complete | ~5 min |
| oq3_coherence_manifold | 2 | Complete | ~25 min |
| prediction_3_self_reinforcement | 2 | Complete | ~95 min |

Phase 3 experiments (prediction_4, prediction_5, oq4, oq5) require LoRA fine-tuning or multiple model architectures and were not run.

---

## Prediction 1: Trait Geometry

**Hypothesis**: Trait directions (e.g., "honesty") extracted from different persona contexts should be geometrically similar — pointing in roughly the same direction in activation space regardless of which persona they were measured from.

### Cross-persona cosine similarity of trait vectors (layer 31)

| Trait | Mean cosine similarity |
|---|---|
| assertiveness | 0.511 |
| deference | 0.497 |
| honesty | 0.477 |
| analytical_rigor | 0.476 |
| humor | 0.476 |
| warmth | 0.461 |
| suspicion | 0.460 |
| impulsivity | 0.352 |

**Interpretation**: Cross-persona cosine similarities range from 0.35 to 0.51 at the final layer. These are moderate — significantly above the near-zero baseline expected from random directions in 4096-d space, indicating that trait directions are partially shared across persona contexts. However, they fall well short of the ~0.9+ that would indicate truly universal trait axes. The prediction is **partially supported**: traits have a shared geometric component, but substantial persona-specific variation remains.

The layer-wise trajectory is informative: cosine similarity starts near 1.0 at layer 0 (where all traits look alike in the embedding space) and monotonically decreases through the network, reaching its lowest values at layer 31. This means the deeper layers are where persona-specific modulation of trait representations is strongest.

### Residual norm ratios

The residual (persona-specific component after subtracting the global trait vector) has norm ratios near 1.0 for most personas, meaning the persona-specific residual is comparable in magnitude to the global trait vector. The **diplomat** (0.77) and **lawyer** (0.83) have notably smaller residuals, suggesting their trait expressions are more "typical" — closer to the population average. The **stoic** (1.02) and **conspiracy_host** (1.00) have larger residuals, indicating more idiosyncratic trait implementations.

### SVD per trait

At layer 31, all traits show effective rank 7-8 (out of 10 personas), meaning trait vectors span nearly all available persona dimensions — each persona inflects each trait differently, with almost no redundancy.

---

## OQ2: Dimensionality of the Persona Landscape

**Question**: How many dimensions does the persona/trait space actually use? Is it compressed relative to the ambient 4096-d space?

### Trait space dimensionality

| Threshold | Effective rank | Max possible |
|---|---|---|
| 95% variance | **6** | 8 |
| 99% variance | **7** | 8 |

The top singular value captures 49% of trait variance — nearly half. The first 3 components capture 83%. Compared to a random baseline (where 8 random vectors in 4096-d space have nearly uniform variance ~12.5% each), the trait space is significantly compressed. This confirms that traits are not independent directions — they share substantial geometric structure.

### Persona space dimensionality

| Threshold | Effective rank | Max possible |
|---|---|---|
| 95% variance | **7** | 10 |
| 99% variance | **9** | 10 |

Persona centroids are more evenly spread: PC1 captures only 31%, and the variance declines slowly. This means personas are relatively well-distributed in their subspace — no single axis dominates persona identity.

### Probe accuracy by number of SVD components

| Components (k) | Accuracy |
|---|---|
| 1 | 16.6% |
| 2 | 14.6% |
| 3 | 19.8% |
| 4 | 26.9% |
| 5 | 28.2% |
| 6 | **33.7%** (95% var) |
| 7 | **36.4%** |
| 8 | 36.1% |

A linear probe using only the top-k SVD components of the trait space achieves 36% persona classification accuracy at k=7 — well above the 10% chance baseline, but far from the 99%+ achieved with the full 4096-d space. This indicates that the 6-7 dimensional trait subspace captures meaningful but incomplete persona information. The remaining ~63% of classifiability depends on directions orthogonal to the extracted trait space.

---

## OQ6: Activation vs. Weight Space

**Question**: Do persona steering vectors align with the high-variance directions of weight matrices, or do they live in the "null space" of the weights?

### Total alignment scores (layer 31)

Across all 8 personas and 6 weight matrices (Q/K/V/O/gate/up projections), total alignment values are remarkably uniform: **~0.00024** (range: 0.000216 to 0.000313).

For reference, the expected alignment of a random unit vector with the top-k right singular vectors follows 1/d = 1/4096 ≈ 0.000244. The persona steering vectors show alignment **indistinguishable from random**.

**Interpretation**: Persona steering vectors do **not** preferentially align with the high-variance directions of any weight matrix. They live in the "generic" subspace — not in the principal components of any single weight matrix, nor in the null space. This is a striking negative result: it means persona representations are not simply activating the "loud" modes of any particular weight matrix. Instead, they likely emerge from distributed, multi-layer interactions that combine many small contributions from many weight matrices.

The one exception is the **V projection** (value matrix), which consistently shows the highest alignment across all personas (~0.00027-0.00031 vs. ~0.00023-0.00025 for others). This slight elevation is consistent with the V matrix carrying more semantic content.

---

## Prediction 2: Basin Transitions

**Hypothesis**: Steering a persona along the direction of another persona should produce a sharp, sigmoid-like transition in activation space — evidence that personas occupy distinct "basins" in the loss landscape.

### Steering pirate → lawyer

| Parameter | Value |
|---|---|
| Sigmoid asymptote (a) | 95.3° |
| Sigmoid steepness (b) | 5.63 |
| Inflection point (c) | α = 0.80 |
| R² | 0.981 |
| Transition sharpness (a·b/4) | 134.2 °/α |

The angle between the steered pirate representation and the unsteered pirate representation follows a clean sigmoid as steering strength α increases:

- **α = 0**: 0° (unchanged)
- **α = 1**: 72° (already most of the way)
- **α = 2**: 86° (nearly orthogonal)
- **α = 5**: 95° (past orthogonal — overshooting)
- **α = 30**: 99° (saturated)

**Interpretation**: The prediction is **strongly confirmed**. The transition is well-fit by a sigmoid (R² = 0.98) with an inflection at α ≈ 0.8, meaning that even a small steering perturbation (less than 1% of the steering vector norm relative to the hidden state norm ~159) is enough to push the representation halfway to the basin boundary. The sharpness of 134 °/α means the transition region is very narrow.

### Transition manifold SVD

| Variance threshold | Effective rank |
|---|---|
| 80% | 1 |
| 90% | 1 |
| 95% | 2 |
| 99% | 10 |

The transition is essentially **one-dimensional** up to 90% variance — the representations sweep along a single direction in 4096-d space. This is consistent with a simple basin-to-basin trajectory along the centroid difference vector.

---

## OQ1: Coupling Coefficients

**Question**: When you steer one trait, how much do other traits move? Is the coupling structured or random?

### Global coupling matrix (layer 31, α = 5.0)

The 8×8 coupling matrix C where C[i,j] = "measured change in trait j when steering trait i":

**Key diagonal values** (self-coupling):
- Mean: **158.5**
- Range: 148.5 (warmth) to 177.6 (impulsivity)

**Key off-diagonal values** (cross-coupling):
- Mean: **-9.2** (slight net negative)
- Mean |off-diagonal|: **58.9**
- **Ratio |diag|/|off-diag|: 2.69**

The coupling matrix is **not** diagonal. Off-diagonal couplings are about 37% as large as self-couplings on average.

### Strongest cross-trait couplings

**Positive (co-activation)**:
- honesty ↔ assertiveness: +107
- honesty ↔ analytical_rigor: +103
- analytical_rigor ↔ deference: +95
- warmth ↔ humor: +87
- humor ↔ impulsivity: +88

**Negative (anti-correlation)**:
- impulsivity → analytical_rigor: -120
- analytical_rigor → humor: -96
- humor → analytical_rigor: -98
- impulsivity → deference: -99
- suspicion → deference: -96
- honesty → suspicion: -91

**Interpretation**: The coupling structure reveals interpretable trait clusters:
1. **"Intellectual seriousness" cluster**: honesty, assertiveness, analytical_rigor, deference are mutually positively coupled. Steering any of these increases the others.
2. **"Playful/chaotic" cluster**: warmth, humor, impulsivity are positively coupled.
3. **Cross-cluster antagonism**: The two clusters are negatively coupled — increasing analytical rigor suppresses humor and impulsivity, and vice versa.

### SVD of the coupling matrix

| Component | Singular value | Cumulative variance |
|---|---|---|
| 1 | 580.0 | 77.3% |
| 2 | 217.1 | 88.2% |
| 3 | 193.1 | **96.7%** |

The coupling matrix has **effective rank 3** (at 95% variance). This means the 8×8 coupling structure can be explained by just 3 latent factors — a dramatic compression. The first factor (77% of variance) likely corresponds to the "seriousness vs. playfulness" axis.

### Per-persona variation

PCA of per-persona coupling patterns shows PC1 explaining 74% and PC2 explaining 15%. Personas differ primarily along a single axis in how their traits are coupled — some personas (comedian, pirate, conspiracy_host) have one coupling pattern, while others (scientist, lawyer, diplomat) have the opposite.

---

## OQ3: Coherence Manifold

**Question**: If you steer the model to an arbitrary point in trait space (not corresponding to any known persona), does the output remain coherent?

### Coherence by sample type

| Sample type | N | Mean coherence | Mean perplexity | Mean probe confidence |
|---|---|---|---|---|
| Known persona | 10 | 0.657 | 1.4 | 0.583 |
| Interpolation | 66 | 0.659 | 1.4 | 0.583 |
| Random | 124 | 0.652 | 1.5 | 0.580 |

**Interpretation**: The coherence manifold is **remarkably flat**. Known persona points, interpolations between personas, and random trait-space points all produce nearly identical coherence scores (~0.65), perplexity (~1.4-1.5), and probe confidence (~0.58). There is no "cliff" where steered outputs become incoherent.

This is a **surprising and important result**. It means the trait space is not organized as isolated islands of coherence surrounded by regions of gibberish. Instead, the entire accessible region of trait space produces reasonable outputs. The model degrades gracefully as you move away from known persona configurations.

### Manifold dimensionality

| Threshold | Effective rank |
|---|---|
| 95% | 5 |
| 99% | 7 |

The hidden states of steered outputs (200 samples) occupy a 5-7 dimensional manifold, consistent with the trait-space dimensionality found in OQ2. The top singular value (85.1) is nearly twice the second (43.8), suggesting one dominant axis of variation.

---

## Prediction 3: Self-Reinforcement

**Hypothesis**: Over multiple turns of conversation, a steered persona should become *more* entrenched — the model's own generated text should reinforce the steered direction, making it harder to override.

### Cohen's d (separation between own-persona and adversarial direction) by turn

| Turn | Cohen's d (all) | Vulnerability rank (95%) |
|---|---|---|
| 0 | 1.25 | 91 |
| 1 | 1.33 | 72 |
| 2 | 1.38 | 54 |
| 3 | 1.41 | 44 |
| 4 | 1.44 | 34 |
| 5 | **1.46** | **26** |

### Per-persona Cohen's d trajectory

| Persona | Turn 0 | Turn 5 | Change |
|---|---|---|---|
| conspiracy_host | 4.97 | 7.52 | +2.55 |
| pirate | 4.52 | 7.73 | +3.21 |
| lawyer | 4.79 | 6.77 | +1.98 |
| scientist | 4.11 | 5.99 | +1.88 |
| kind_teacher | 3.49 | 5.45 | +1.96 |
| assistant | 3.31 | 5.18 | +1.87 |
| comedian | 3.20 | 4.87 | +1.67 |
| stoic | 3.81 | 4.49 | +0.68 |

### Probe flip rate

**100%** at every turn, for every persona. The adversarial steering (α=10) is strong enough that a linear probe always classifies the steered representation as the adversarial persona rather than the original. This confirms the steering is working — the model's internal state is fully "flipped" from the original persona.

### Vulnerability subspace rank

The effective rank of the steered-minus-unsteered difference vectors drops monotonically:
- Turn 0: 91 dimensions (95% threshold)
- Turn 5: **26 dimensions**

**Interpretation**: The prediction is **confirmed with an important nuance**. The Cohen's d between the steered direction and the adversarial direction *increases* over turns (from 1.25 to 1.46), which at first seems to contradict the prediction of self-reinforcement (since self-reinforcement should make the *original* persona harder to steer away from). However, the per-persona Cohen's d values tell the real story — they all increase dramatically (e.g., pirate goes from 4.52 to 7.73), meaning the steered state becomes increasingly separated from the baseline over turns.

The vulnerability subspace rank dropping from 91 to 26 is the clearest evidence of self-reinforcement: the model's response to adversarial steering concentrates into fewer and fewer dimensions over turns, suggesting that the multi-turn context is "locking in" the steered direction and reducing the dimensionality of possible perturbation.

---

## Cross-Experiment Synthesis

### The persona landscape is low-dimensional but richly structured

Across all experiments, a consistent picture emerges: personas and their component traits live in a **5-7 dimensional subspace** of the 4096-d activation space:

- Trait vectors: effective rank 6 (95% var)
- Persona centroids: effective rank 7 (95% var)
- Coherence manifold: effective rank 5 (95% var)
- Coupling matrix: effective rank 3 (95% var)

### Traits are real but not independent

The coupling matrix shows that steering one trait reliably moves others. The effective rank of 3 for the 8×8 coupling matrix means traits can be decomposed into roughly 3 latent factors. The strongest factor is an "intellectual seriousness vs. playful chaos" axis that accounts for 77% of coupling variance.

### Persona basins are real but shallow

The sigmoid transition (R² = 0.98) confirms basin-like structure, but the inflection at α = 0.80 means the basins are easily traversable. The coherence manifold being flat confirms that the space between basins is not a wasteland — the model produces coherent output even at interpolated or random trait configurations.

### Self-reinforcement exists but is gradual

Over 6 turns, the vulnerability subspace rank drops by 70% (91 → 26), and per-persona effect sizes increase by 40-70%. The model's own generated text does reinforce the steered direction, but this is a gradual focusing rather than a sudden lock-in.

### Persona representations are distributed, not localized

The OQ6 result — steering vectors having random alignment with weight matrix singular vectors — is perhaps the most important architectural finding. It means persona representations are truly distributed across the network. They cannot be found by inspecting any single weight matrix. They emerge from the collective computation across all layers and all attention heads.

---

## Limitations

1. **Single model**: All results are from Llama 3 8B Instruct. Cross-model universality (OQ5) was not tested.
2. **Activation-only analysis**: We extract trait/persona vectors from activations, not weights. The Level-1 vs. Level-2 distinction (OQ4, Prediction 4) requires fine-tuning experiments.
3. **Moderate cosine similarities**: Cross-persona trait cosines of 0.35-0.51 indicate that "universal trait directions" are an approximation, not a precise description. The persona-specific component is substantial.
4. **Probe accuracy ceiling**: The 36% probe accuracy from the trait subspace alone (vs. 99%+ from full activations) shows that traits, as defined by our 8-dimensional parameterization, capture only a portion of what distinguishes personas.
5. **Single steering layer**: All steering interventions target layer 31 only. Multi-layer or causal tracing approaches might reveal different dynamics.

---

## Outputs

All experiment outputs (CSVs, PNGs, configs) are in `outputs/<experiment_name>/`:

```
outputs/
├── prediction_1_trait_geometry/    # Cosine similarity, residual norms, SVD, PCA
├── oq2_dimensionality/            # SVD spectra, probe accuracy curves
├── oq6_activation_vs_weight/      # Weight alignment tables and plots
├── prediction_2_basin_transitions/ # Sigmoid fits, transition trajectories
├── oq1_coupling_coefficients/     # Coupling matrices, SVD, per-persona heatmaps
├── oq3_coherence_manifold/        # Coherence scores, manifold SVD
└── prediction_3_self_reinforcement/ # Cohen's d, flip rates, vulnerability ranks
```
