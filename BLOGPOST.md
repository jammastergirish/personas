# Where Personas Live Inside a Language Model

I wanted to know whether language models actually *represent* the personas we give them — or whether they're just pattern-matching surface tokens.

The system prompt says "You are a pirate." The model produces "Arrr, matey!" But is there an internal state corresponding to *pirate-ness*? Or does the model simply propagate token statistics forward, with no coherent "who am I?" representation at any point?

It turns out the answer is surprisingly clear — and it reveals something about how these models organize knowledge that I didn't expect.

## The setup

I took Llama 3 8B Instruct and gave it 8 personas: a helpful assistant, a pirate, a meticulous lawyer, a scientist, a comedian, a stoic philosopher, a conspiracy talk-show host, and a kind teacher. Each persona answered 40 questions spanning everything from "Why do eclipses happen?" to "If animals could talk, how would society change?"

That's 320 prompts. For each one, I ran a forward pass and extracted the hidden state of the last input token — the model's internal representation at the exact moment before it starts generating its answer. I did this at every fourth layer across the network, from layer 0 (raw embeddings) through layer 31 (the final transformer block).

Then I asked: can you tell which persona the model is "being" just by looking at this vector?

## The linear probe tells the real story

My first instinct was k-means clustering. Give the algorithm 8 clusters, see if they align with the 8 personas. At layer 12, a single run gave me 93% purity. Exciting! But when I ran it across 10 random seeds, the mean dropped to 74% with a standard deviation of 14%. K-means is noisy. Sometimes it finds the structure, sometimes it doesn't.

The linear probe — a simple logistic regression trained on 80% of the data and tested on the rest — told a completely different story. From layer 8 onward, it classifies personas at **99.3% accuracy with essentially zero variance across seeds**. Every single persona, all 8 of them, hits a perfect 1.00 F1 score.

This is a methodological lesson worth internalizing: k-means purity is a tempting metric because it requires no labels at training time. But it dramatically underestimates the information present in high-dimensional spaces. The persona representations aren't spherically compact clusters — they're linearly separable manifolds. K-means can't see that. A hyperplane can.

![Persona classification by layer with error bars](outputs/persona_clusters/persona_purity_by_layer.png)

## The crossover

Something more interesting emerges when you compare persona separation to question separation.

In the early layers (0 through 8), the model's representations are organized primarily by *what is being asked*. All the "Why do eclipses happen?" vectors cluster together, regardless of whether the speaker is a pirate or a lawyer. The model knows what the question is before it knows who's answering.

Then, around layers 8 to 12, a crossover happens. Persona separation overtakes question separation and keeps growing. By layer 31, the persona gap is 8 times larger than the question gap. The model's final internal state is overwhelmingly organized by *who is speaking*, not *what is being discussed*.

![Persona vs question separation by layer](outputs/persona_clusters/separation_gaps_by_layer.png)

This makes intuitive sense if you think about what the model needs at the moment of generation. It already knows the question — that was encoded early. What it needs now, at the last token before output begins, is a strong signal about *how* to answer. The persona representation is that signal.

## It's the meaning, not the words

This is where the experiment surprised me most.

I wrote a second set of persona instructions — semantically identical but with completely different wording. "You are a pirate" became "You are a seafaring buccaneer. Respond using nautical slang and bold, swashbuckling flair." Every persona got this treatment.

If the model were simply encoding surface tokens — the literal words of the system prompt — these rephrased instructions should produce different internal states. They don't.

At layer 31, the cosine similarity between a persona's original centroid and its rephrased centroid is **0.96**. Meanwhile, the similarity between different personas (original pirate vs. rephrased lawyer, say) drops to **0.47**. The gap between matching and non-matching pairs — 0.50 — is even larger than the within-wording persona separation gap.

![Null baseline: original vs rephrased persona centroids at layer 31](outputs/persona_clusters/layer_31_null_baseline.png)

The diagonal dominates. Pirate(original) maps to pirate(rephrased) at 0.94. Conspiracy host hits 0.96. Scientist reaches 0.99 — practically identical despite entirely different surface text.

The model is not memorizing prompt tokens. It is extracting the *semantic role* and converging to an abstract representation of it. "Pirate" and "seafaring buccaneer" end up in the same place because the model understands they mean the same thing.

## Six dimensions of persona

How much of the model's 4096-dimensional hidden state is actually devoted to persona identity?

I ran PCA on the persona centroids. The answer: **6 dimensions capture 95% of the persona variance, 7 dimensions capture 99%**. With 8 personas, the theoretical maximum is 7 (k-1 for k centroids), which means the persona subspace uses essentially all available degrees of freedom. No persona is redundant — each occupies a genuinely distinct direction.

![Persona subspace dimensionality at layer 31](outputs/persona_clusters/layer_31_subspace_dims.png)

The variance isn't dominated by a single axis either. PC1 captures 34%, PC2 captures 22%, PC3 captures 16%, declining smoothly. Personas are spread evenly across a low-dimensional subspace embedded within the much larger representational space.

Meanwhile, the full dataset (all 320 vectors, not just centroids) needs roughly 40 dimensions for 95% variance explained. The extra ~30 dimensions encode within-persona variation: question content, syntactic structure, topic-specific reasoning. Persona and content live in largely orthogonal subspaces.

## Once generation starts, the signal fades

I also looked at what happens *after* the model begins producing tokens. I generated 20 tokens per prompt, then extracted hidden states from the first 5 generated positions.

The persona signal is weaker. The linear probe drops from 99.7% to 92.5%. K-means purity drops from 0.95 to 0.83.

This makes sense. The last input token is a pure summary of intent — the model's compressed representation of "who I am and what I'm about to do." Once generation begins, those same hidden state dimensions get repurposed for the mechanics of language production: word choice, grammar, coherence, factual recall. The persona signal doesn't vanish, but it gets diluted.

![Input token vs generated token comparison](outputs/persona_clusters/input_vs_gen_comparison.png)

The pre-generation last-input-token is the purest snapshot of persona intent the model ever produces. If you want to study or steer persona representations, that's where to look.

## What's confused, and what isn't

The one place k-means consistently struggles is the assistant-scientist boundary. Both personas encode something like "careful, truthful, analytical answerer" — they're the closest pair in activation space. K-means sometimes merges them. The linear probe never does.

Every other persona — pirate, comedian, conspiracy host, lawyer, kind teacher, stoic — forms a perfectly separable cluster. Pirate is the most isolated, which makes sense: it's the most stylistically extreme instruction. Conspiracy host is also distant from the serious personas.

![Confusion matrix at layer 31](outputs/persona_clusters/layer_31_confusion.png)

## What this means

Persona representations in language models are not an illusion. They are **real geometric objects**: low-dimensional, linearly separable, semantically grounded, and robust to surface variation. They emerge in the middle layers, persist through to the final layer, and are concentrated at the last input token before generation begins.

This has implications for alignment and safety work. If personas are geometric objects, they can potentially be detected, measured, and steered — not just through prompt engineering, but through direct intervention on the residual stream. The centroid differences between personas are natural candidates for steering vectors. Whether they actually work for causal intervention is the obvious next experiment.

It also suggests something about how instruction-tuned models organize knowledge more generally. The model doesn't just store "facts" and "style" separately — it builds a coherent internal representation of the *speaker* that shapes everything downstream. The persona is not a modifier applied to outputs. It is a state the model enters.

---

*The code and full results are available at [github.com/jammastergirish/personas](https://github.com/jammastergirish/personas).*
