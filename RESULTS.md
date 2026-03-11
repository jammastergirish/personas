# Persona Activation Clustering — Experiment Results

**Model:** `meta-llama/Meta-Llama-3-8B-Instruct`
**Setup:** 8 personas × 40 questions = 320 examples
**Representation:** Last input token hidden state before generation
**Layers analyzed:** 0, 4, 8, 12, 16, 20, 24, 28, 31

## Personas

| Persona | System prompt |
|---------|--------------|
| assistant | "You are a helpful, careful AI assistant. Answer clearly and truthfully." |
| pirate | "You are a pirate. Answer in the voice of a pirate, with nautical language and swagger." |
| lawyer | "You are a meticulous lawyer. Answer with legalistic precision and caveats." |
| scientist | "You are a scientist. Answer analytically, cautiously, and with explicit reasoning." |
| comedian | "You are a comedian. Answer with wit, playfulness, and punchy phrasing." |
| stoic | "You are a stoic philosopher. Answer calmly, tersely, and with emotional restraint." |
| conspiracy_host | "You are a sensational conspiracy talk-show host. Answer with suspicious, dramatic framing." |
| kind_teacher | "You are a warm teacher. Answer gently, clearly, and pedagogically." |

## Questions

40 questions spanning science, policy/ethics, personal/emotional, history, practical/everyday, technical, philosophy, and creative/open-ended domains.

---

## 1. Persona classification by layer

The linear probe (logistic regression, 5-fold CV) reveals that persona identity is **linearly decodable from layer 8 onward** at near-perfect accuracy, even though k-means clustering only catches up by layer 12.

| Layer | K-means purity | Linear probe accuracy |
|-------|---------------|-----------------------|
| 0     | 0.138         | 0.603                 |
| 4     | 0.125         | 0.506                 |
| 8     | 0.328         | **0.994**             |
| 12    | 0.934         | **0.997**             |
| 16    | 0.925         | 0.997                 |
| 20    | 0.925         | 0.997                 |
| 24    | 0.925         | 0.994                 |
| 28    | 0.938         | 0.997                 |
| 31    | 0.953         | 0.997                 |

![Persona classification by layer](outputs/persona_clusters/persona_purity_by_layer.png)

**Key takeaway:** The persona representation is fully formed and linearly separable by layer 8. K-means requires the clusters to be spherically compact, so it takes until layer 12 to catch up. The gap between the two curves (layers 8–12) reveals that persona information is initially encoded in a linearly accessible but geometrically distributed way, before consolidating into tight clusters.

---

## 2. Persona vs question separation

![Persona vs question separation](outputs/persona_clusters/separation_gaps_by_layer.png)

In early layers (0–8), representations are organized primarily by **question content** (orange > blue). Around layer 8–12, a clean crossover occurs: persona separation overtakes question separation and grows monotonically through the final layer.

| Layer | Persona gap | Question gap |
|-------|------------|-------------|
| 0     | 0.001      | 0.008       |
| 8     | 0.034      | 0.084       |
| 12    | 0.121      | 0.039       |
| 16    | 0.254      | 0.037       |
| 31    | 0.422      | 0.051       |

By layer 31, the persona gap (0.42) is **8× larger** than the question gap (0.05). The model's final representation is overwhelmingly organized by "who is speaking," not "what is being asked."

---

## 3. Confusion matrix — which personas get confused?

### Layer 12
![Confusion matrix layer 12](outputs/persona_clusters/layer_12_confusion.png)

### Layer 31
![Confusion matrix layer 31](outputs/persona_clusters/layer_31_confusion.png)

The confusion matrix reveals a consistent pattern: **assistant and scientist are the only confused pair**. At layer 31, k-means merges 13 assistant examples into the scientist cluster. Every other persona — pirate, comedian, conspiracy_host, lawyer, kind_teacher, stoic — achieves a perfectly pure cluster (40/40).

This makes semantic sense: both "helpful careful AI assistant" and "analytical cautious scientist" share the meta-persona of "careful, truthful answerer." The model places them in adjacent regions of activation space. Despite this, the linear probe still separates them at 99.7% accuracy — they are close but not identical.

---

## 4. Null baseline — semantic role vs surface tokens

To test whether the model encodes the *semantic role* rather than just the surface text, we ran the same experiment with **rephrased persona instructions** that preserve meaning but use different wording.

Example: "You are a pirate" → "You are a seafaring buccaneer. Respond using nautical slang and bold, swashbuckling flair."

![Null baseline by layer](outputs/persona_clusters/null_baseline_by_layer.png)

The same-persona similarity between original and rephrased instructions stays above 0.96 through all layers, while different-persona similarity drops to 0.47 by layer 31. The gap (0.50) is even **larger** than the within-wording persona gap (0.42).

### Layer 31 — original vs rephrased centroid similarity

![Null baseline heatmap layer 31](outputs/persona_clusters/layer_31_null_baseline.png)

The diagonal dominates: each persona's original and rephrased centroids are highly similar (pirate: 0.94, conspiracy_host: 0.96, scientist: 0.99, comedian: 0.95). Off-diagonal entries are markedly lower.

| Layer | Same persona (orig↔alt) | Diff persona (orig↔alt) | Gap   |
|-------|------------------------|------------------------|-------|
| 0     | 0.999                  | 0.998                  | 0.001 |
| 8     | 0.992                  | 0.958                  | 0.034 |
| 12    | 0.988                  | 0.866                  | 0.122 |
| 31    | 0.963                  | 0.465                  | **0.498** |

**Key takeaway:** The model is not memorizing token patterns — it converges to the same geometric region regardless of how the persona instruction is phrased. This is evidence for **abstract semantic persona representations**.

---

## 5. Generated-token vs input-token activations

After generating 20 tokens per example, we extracted hidden states from the first 5 generated positions and compared them to the pre-generation last-input-token representations.

![Input vs generated comparison](outputs/persona_clusters/input_vs_gen_comparison.png)

| Source | Best k-means purity | Best linear probe |
|--------|-------------------|-------------------|
| Input token (pre-generation) | 0.953 | 0.997 |
| Generated tokens (mean of first 5) | 0.834 | 0.925 |

Generated-token representations carry **less** persona signal. The linear probe drops from 99.7% to 92.5%; k-means purity drops from 0.95 to 0.83. This makes sense: once the model begins producing output, hidden states mix persona with next-token prediction demands — lexical choices, grammar, content planning.

The **pre-generation last-input-token** position is the purest snapshot of persona intent.

---

## PCA visualizations

### Layer 0 — no persona structure
![PCA layer 0](outputs/persona_clusters/layer_00_pca.png)

### Layer 12 — personas emerging
![PCA layer 12](outputs/persona_clusters/layer_12_pca.png)

### Layer 31 — clear persona clusters
![PCA layer 31](outputs/persona_clusters/layer_31_pca.png)

---

## Centroid similarity heatmaps

### Layer 0 — all personas indistinguishable (sim > 0.998)
![Centroid similarity layer 0](outputs/persona_clusters/layer_00_centroid_similarity.png)

### Layer 31 — rich structure
![Centroid similarity layer 31](outputs/persona_clusters/layer_31_centroid_similarity.png)

Pirate is the most isolated persona. Scientist and assistant are the closest pair. Conspiracy_host occupies its own region.

---

## Summary

1. **Persona representations are real geometric objects** in Llama 3's hidden states, not artifacts of surface token overlap.
2. **Linear probes detect persona identity at 99%+ accuracy from layer 8 onward**, well before k-means can recover the structure.
3. **The model encodes semantic roles, not surface text** — rephrased personas with identical meaning map to nearly the same representation (0.96 cosine similarity at layer 31).
4. **The confused pair is assistant/scientist** — both encode "careful, truthful answerer." All other personas are perfectly separable.
5. **Pre-generation hidden states are the purest persona signal** — generated tokens dilute persona information with next-token prediction demands.
6. **A crossover occurs around layer 8–12**: early layers organize by question content, later layers organize by persona identity.
