# Interpretability methods in NeuralScope V1

Every quantity NeuralScope shows carries one of four **epistemic statuses**:

| status | meaning | examples in V1 |
|---|---|---|
| **OBSERVED** | a direct tensor / model measurement | tokens, generated text, residual-stream vector at (depth, token), attention probabilities, final logits, SymPy oracle label |
| **DERIVED** | a mathematical transformation of observed quantities | Logit Lens projections and ranks, norms, cosines, entropies, direct logit attribution, cross-run differences |
| **INFERRED** | a semantic interpretation or hypothesis | user-written hypotheses and notes ("possible carry head") — never generated automatically |
| **CAUSALLY TESTED** | the measured effect of an actual intervention, with its scope | "zero-ablating L22.H10 changed log P('9') by −7.96 on this prompt (n = 1)"; benchmark-wide intervention evaluations (n = N) |

A causal label always states its scope (single prompt, a counterfactual pair, N problems). A single-prompt
effect shows that a component *was necessary for this output on this input*; it does not establish a general
mechanism. Nothing in NeuralScope is described as "what the model is thinking".

## Conventions

* **Depth** `d ∈ {0 … L}` indexes residual-stream sites: `d = 0` is the token embedding (input to block 0,
  labelled `emb`), `d = l + 1` is the residual stream **after** block `l` (labelled `Ll`), before the final norm.
  Qwen2.5-1.5B has 29 depths.
* **Layer** `l ∈ {0 … L−1}` indexes blocks (attention heads `Ll.Hh`, MLPs).
* **Position** `t`: logits at position `t` predict token `t + 1`. A run's *target* is (position, token); by default
  the last prompt token predicting the first answer token.

## Residual stream (OBSERVED)

Captured by forward hooks on the embedding and on each decoder block. Pre-norm decoders satisfy exactly

```
h_{l+1} = h_l + Σ_h W_O^{l,h} z^{l,h} + MLP_l(RMSNorm(h_l + attn_l))
```

and NeuralScope verifies the decomposition numerically (error 0.0 on the fixture model in fp32).

## Logit Lens (DERIVED)

`lens(h_d) = W_U · RMSNorm_final(h_d)` using the model's **own** final norm (learned gain γ and the per-vector
1/rms of `h_d`) and unembedding. At the final depth this equals the model's logits exactly (tested). At earlier
depths it reads an intermediate state through a readout the model was not trained to use there, so it is
**biased early** (Belrose et al. 2023) — ranks in early layers are not evidence of absence. NeuralScope shows
top-k tokens, entropy, and a chosen target token's logit / probability / rank at every depth
(rank = 1 + #tokens with a strictly larger logit).

## Attention (OBSERVED patterns; DERIVED summaries)

Eager attention returns the post-softmax probabilities `A[l, h, query, key]`. Summaries: entropy of a query row,
top source tokens, weight on the first token (attention sink), head output norm `‖W_O^{l,h} z^{l,h}_t‖`.
**Attention is not explanation**: a large weight says where information was read from, not whether it mattered —
test with ablation.

## Direct logit attribution (DERIVED)

With the final-norm scale `s = 1/rms(h_final)` frozen at its actual value, the readout is linear and

```
logit_τ − logit_κ = s · Σ_components ⟨c, γ ⊙ (W_U[τ] − W_U[κ])⟩
```

over the embedding, every head write and every MLP output. The sum reproduces the real logit (error 4e-6 on the
flagship run). DLA measures **direct** effects only; a head that matters by changing later components can have
near-zero DLA (L22.H10 has DLA +0.83 but an ablation effect of −7.96 log-prob).

## Interventions (CAUSALLY TESTED when run)

| intervention | implementation | notes |
|---|---|---|
| head ablation / scaling | multiply head `h`'s slice of the `o_proj` input by `factor` (0 = zero-ablation) at all or selected positions | exact removal of that head's residual write; zero-ablation can push activations off-distribution (mean-ablation planned) |
| MLP ablation / scaling | multiply the MLP block output | |
| residual patching | `h[d, t] ← (1−α) h + α h_src[d, t']` from another stored run | standard activation patching (Meng et al. 2022; Heimersheim & Nanda 2024) |
| steering | `h[d, T] += s · (h_A[d, t_A] − h_B[d, t_B])` | difference-of-activations; shows sufficiency to move the output, not that the direction "is" a concept |

Every intervention run reports: target logit / probability / log-prob / rank before → after and Δ, KL(P_base ‖ P_int)
at the target position and at every generated position (teacher-forced on the baseline tokens), top-10 before vs
after, free-running output before vs after with changed token indices, the per-depth relative residual change
(showing where the effect starts), and correctness before → after for Math Lab runs.

**Sweeps.** Exhaustive single-head ablation (Δ metric for every head) and activation patching
(restoration for every depth × position) produce CAUSALLY TESTED maps for one prompt / one pair.

## Math Lab

Problems come from a seeded generator; answers are labelled by a SymPy oracle through a conservative parser
(ambiguous and unparseable outputs are never scored correct; outputs cut off by the token budget are
`truncated` and excluded from correct/incorrect comparisons). The oracle label is OBSERVED ground truth about the
**output**; it says nothing about the internal mechanism. Correct-vs-incorrect trace comparisons (target rank by
depth, residual norms, successive cosine change, lens entropy, cross-run cosine, attention-entropy differences)
are DERIVED and correlational. Candidate differences become CAUSALLY TESTED only through patching / ablation,
e.g. the benchmark-wide intervention evaluation (accuracy before → after, flips, mean Δ log P(oracle answer) ± SEM).

## Known scientific limitations (V1)

* One model family; CPU float32 only in the reference environment.
* Logit Lens is not a tuned lens; early-layer readouts are biased.
* Zero-ablation only (no mean/resample ablation yet); single-head effects miss redundancy (backup heads) and
  interactions; head sweeps are per prompt.
* Patching sweeps require equal-length, position-aligned counterfactual pairs.
* Attention patterns are shown for one layer at a time; Q/K/V, MLP neurons and features are not yet captured.
* No SAE / transcoder features, attribution graphs or automated circuit discovery (V2–V3).
* Correct-vs-incorrect comparisons across different problems confound problem identity with correctness.
