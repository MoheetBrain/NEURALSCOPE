# NeuralScope Research Log

Findings, decisions and evidence accumulated while building NeuralScope. Newest entries last.
Every claim carries its epistemic status: **OBSERVED**, **DERIVED**, **INFERRED** or **CAUSALLY TESTED**
(with its scope). Numbers are reproducible: greedy decoding, eager attention, float32, CPU.

---

## 2026-09-29 — Environment audit and stack decision

**Finding.** The environment is 4 CPU cores, 15 GB RAM, no GPU (no CUDA/MPS). Installed: Python 3.11, torch
2.14.0+cpu, transformers 5.17.0, TransformerLens 4.0.0, nnsight 0.7.0, Node 22.

**Finding.** `from transformer_lens import HookedTransformer` fails in TransformerLens 4.0.0 (the 4.x API moved to
`TransformerBridge`). nnsight 0.7's tracer raised `OSError: could not get source code` when run from a
non-file context (it inspects caller source to build its intervention graph).

**Decision.** Instrument Hugging Face models directly with **native PyTorch module hooks** behind our own
`ModelAdapter` interface (`packages/neuralscope/neuralscope/instrumentation/`). Rationale:
1. exact to the HF forward pass — no re-implementation of the architecture (TransformerLens re-implements and,
   by default, folds LayerNorm/centres weights, which changes what "the residual stream" numerically is);
2. no tracing DSL between the researcher and the tensors;
3. the adapter boundary keeps the backend free to adopt TransformerLens/nnsight/SAELens later per model.

**Decision.** V1 model = **Qwen/Qwen2.5-1.5B-Instruct** (Apache-2.0, ungated; 28 layers, 12 query heads,
2 KV heads (GQA), d_model 1536, d_head 128, tied embeddings, RMSNorm, vocab 151,936).
Why: ungated (Gemma-2 / Llama-3.2 need licence acceptance + tokens), standard Llama-family pre-norm block,
supported across the interpretability ecosystem, fits comfortably in float32 on CPU (6.2 GB), and — decisive for
Math Lab — **its tokenizer splits every digit into its own token**, so a multi-digit answer is produced one digit
per step and "the answer token" is a single digit. Qwen2.5-0.5B-Instruct (same family, 24 layers) is the fast
real-model test target.

**Evidence (OBSERVED).** CPU float32: load 14–16 s; greedy generation ≈ 5 tok/s; a 44-token forward pass with
attention materialised ≈ 0.3–0.8 s. `What is 37 + 58? Give only the answer.` → `95`;
`What is 73 * 14?` → `1022`; `What is 12345 + 67890?` → `70235` (**incorrect**, oracle 80235 — leading-digit
carry error).

---

## 2026-09-29 — Where to hook a head, and why it is exact

**Finding.** In transformers 5.x, `Qwen2Attention.forward` computes `attn_output` (concatenated per-head
`z_h = A_h V_h`, shape `[B, T, H·d_head]`) and then `o_proj(attn_output)`. `o_proj` is `bias=False`.
Eager attention returns the attention probabilities as the second output of the attention module.

**Decision.** Head-level capture and intervention use a **forward pre-hook on `o_proj`**. Scaling the slice
`[h·d_head, (h+1)·d_head)` of its input scales exactly head *h*'s residual write `W_O^h z_h` and nothing else —
correct under GQA too (K/V are shared across a query-head group, but each query head has its own `z_h` and
`W_O` slice). All runs use `attn_implementation="eager"` so every run in an experiment uses the same kernels and
attention probabilities exist.

**Evidence (DERIVED, tested).** On the fixture model and on real weights:
`resid[l+1] − resid[l] − Σ_h W_O^h z_h − mlp_out[l] = 0` exactly (max error 0.0 in fp32 on the fixture);
zero-ablating a head at layer *l* leaves depths ≤ *l* bit-identical and changes the post-attention residual at
layer *l* by exactly `−W_O^h z_h`; batched edit sets equal individual runs; resuming the forward pass from a
cached residual at depth *d* equals the full pass (max diff 0.0). See `tests/test_instrumentation.py`.

**Finding.** In transformers 5.x, `output_hidden_states` is collected by a decorator and the *last* element has
the final norm applied. We therefore capture the residual stream with our own hooks on each decoder block
(pre-final-norm), and the Logit Lens applies the model's final norm explicitly — so the lens at the last depth
reproduces the model logits exactly (tested).

---

## 2026-09-29 — Flagship trace: 37 + 58 (Qwen2.5-1.5B-Instruct)

Prompt (chat template, 44 prompt tokens): `What is 37 + 58? Give only the answer.` → output `95<|im_end|>`.
Teacher-forced capture reproduces every greedy token (**OBSERVED**, `trace_consistent = true`).

**Logit Lens (DERIVED).** Rank of the target `9` at the answer position by depth
(emb, L0 … L27): 70, 620, 668, 609, 2139, 1768, 7819, 525, 508, 224, 545, 727, 131, 38, 106, 18, 33, 36, 44,
219, 6515, 187, 71, 24, **2** (L23), 3, 2, **1** (L26), 1. The lens top-1 at L19–L20 is `答案` ("answer") — a
format/intent token — before digits take over at L22+. (Lens is biased at early depths; not a tuned lens.)

**Direct logit attribution (DERIVED).** Components sum to the real logit 27.43957 with error 4e-6. Largest direct
contributors: L25.H3 (+1.27), L27.H2 (−1.06), L20.H2 (+0.84), L22.H10 (+0.83), L23.H9 (+0.82).

**Exhaustive single-head zero-ablation (CAUSALLY TESTED, n = 1 prompt, 336 heads).**
Δ log P("9") at the answer position:
L22.H10 **−7.96**, L22.H11 **−7.71**, L0.H5 −1.26, L22.H8 −0.11, L20.H10 −0.10; every other head |Δ| < 0.1.
Ablating L22.H10 alone: P("9") 0.99985 → 0.00035, KL 7.96 nats, **free-running output `95` → `85`**.

`85` is the sum computed **without the units carry** (3 + 5 = 8; 7 + 8 = 15 carries 1). H10 and H11 belong to
the same KV group (heads 6–11 share KV head 1). **INFERRED hypothesis:** L22.H10/H11 move information needed to
add the units carry into the tens digit. *Not established:* one prompt cannot distinguish "carry" from
"any tens-digit computation"; see E1.

**Activation patching (CAUSALLY TESTED, n = 1 counterfactual pair).** Clean `37 + 58` (→95) vs corrupt
`37 + 48` (→85), metric logit("9") − logit("8") at the answer position (clean 10.11, corrupt −17.50).
Restoration from patching the clean residual:
at the differing operand token `4`→`5`: ≈ 1.00 at the embedding, 0.86–0.89 through L0–L20, 0.00 from L24 on;
at the answer position: 0.00 through L20, **1.00 from L23/L24 on**.
The information that distinguishes the two answers leaves the operand position and arrives at the answer
position between L21 and L24 — the same layer range as the two critical heads. Two independent causal methods
(ablation and patching) agree on *where*.

**Cross-model observation (OBSERVED).** Qwen2.5-0.5B-Instruct answers `37 + 58` with **`85`** — the same
carry-dropping error that ablating L22.H10 induces in the 1.5B model.

---

## 2026-09-29 — Math Lab first benchmark (default 25-problem mix, seed 0)

**OBSERVED (SymPy oracle).** 17 correct, 3 incorrect, 5 truncated. By kind: add 8/9, sub 3/3, mul 4/6, mod 2/2,
linear equations 0/3 and fraction comparisons 0/2 — **all five of the latter were chain-of-thought outputs cut off
at the token budget**, not wrong answers. Incorrect: `76150 + 28254 → 104374` (oracle 104404),
`673 * 22 → 14846` (14806), `462 * 65 → 29430` (30030).

**Decision.** Outputs that stop at the token budget are labelled `truncated` and excluded from correct/incorrect
comparisons (a truncated `10` from `1022` would otherwise be scored "incorrect"). Equation and fraction prompts
now say "Reply with only …, no explanation.", which elicits direct answers (e.g. `7x − 71 = 545` → `73`, oracle 88;
`40/83 vs 33/47` → `40/83`, oracle 33/47) — more incorrect traces for correct-vs-incorrect analysis.

**Decision.** For an incorrect run the Math Lab target is placed at the **first divergent answer token**
(the "decision point"): context = prompt + the model's correct prefix, target = oracle digit, contrast = the
model's digit. This isolates the step where the computation went wrong.

---

## 2026-09-29 — Performance notes

Exhaustive head sweep (336 heads) on a 44-token prompt: 136 s with full forward passes → **82 s** after resuming
each layer's batch from the cached residual (blocks before the ablated layer cannot change). Activation-patching
sweep over 29 depths × 13 positions: ≈ 105 s. Logit-Lens summaries for all 29 × 47 cells ≈ 3–5 s (computed
eagerly for interactive runs, lazily for Math Lab runs). A standard-level trace for a 47-token run is ≈ 20 MB
(float32 residuals + head outputs, float16 attention, float32 output log-probs at generated positions).

---

## 2026-09-29 — The Logit Lens sees the no-carry digit first (37 + 58)

**DERIVED (Logit Lens, answer position, flagship run).** Lens probability of `8` (the tens digit *without* the
carry, 3 + 5) vs `9` (the correct tens digit) by depth, from L19 to L27:

| depth | L19 | L20 | L21 | L22 | **L23** | **L24** | **L25** | **L26** | L27 |
|---|---|---|---|---|---|---|---|---|---|
| P(`8`) | 0.000 | 0.000 | 0.000 | 0.000 | **0.996** | **0.936** | **0.523** | 0.027 | 0.000 |
| P(`9`) | 0.000 | 0.001 | 0.002 | 0.000 | 0.003 | 0.002 | 0.273 | **0.964** | 1.000 |
| lens top-1 | `答案` | `答案` | `getDb` | `6` (0.90) | `8` | `8` | `8` | `9` | `9` |

(columns are the residual stream after block Lk). The readout first commits to the **no-carry** digit
(L23–L25) and flips to the carried digit only at L26. Together with the CAUSAL result that ablating L22.H10 yields
the output `85`, this suggests (INFERRED) a two-stage computation: tens operands summed by ~L23, carry applied
late, with L22.H10/H11 supplying information the late correction needs. Caveat: the lens is biased and this is one
prompt; E1 tests the ablation half across controlled problem sets, and the lens half should be checked across
carry problems (does P(no-carry digit) peak before P(correct digit) on every carry problem?).

---

## 2026-09-29 — E1: L22.H10/H11 are necessary, specifically, for the units→tens carry

Script `scripts/experiments/carry_heads.py --n 40 --seed 0`; raw data `docs/research/E1_carry_heads.json`.
Qwen2.5-1.5B-Instruct, 80 prompts `What is A + B? Give only the answer.` (A, B two-digit, A + B < 100;
40 with a units carry, 40 without; identical token length), readout = tens digit at the answer position.
Baseline tens-digit accuracy 1.00 / 1.00; full answers 40/40 and 40/40.

**CAUSALLY TESTED (n = 80, one template).** Δ = change in logit(correct tens) − logit(carry-mishandled tens)
(carry: a_t + b_t; no-carry: a_t + b_t + 1), mean ± SEM.

| zero-ablation | carry: tens acc | carry: → mishandled digit | carry: Δ | no-carry: tens acc | no-carry: Δ |
|---|---|---|---|---|---|
| L22.H10 | 0.35 | 0.65 | −17.7 ± 1.8 | 1.00 | −0.5 ± 0.4 |
| L22.H11 | 0.40 | 0.60 | −17.8 ± 1.5 | 0.95 | −0.9 ± 0.4 |
| **L22.H10 + H11** | **0.05** | **0.95** | **−27.1 ± 1.0** | 0.93 | −0.9 ± 0.6 |
| other 10 heads of L22 (mean; worst H8) | 0.98 | – | −0.9 (−4.2) | 0.98 | −0.6 (−2.6) |
| 10 random heads in other layers (mean; worst L0.H6) | 0.99 | – | −0.4 (−5.1) | 1.00 | −0.6 (−4.3) |

Free-running outputs with H10 + H11 ablated: **38/40 carry problems output exactly the carry-dropped sum**
(e.g. 37 + 58 → 85); no-carry problems 37/40 fully correct, 0/40 carry-type errors.

**Mean-ablation control** (head output replaced by its dataset-mean at each position): H10 + H11 → carry tens
accuracy 0.40 (Δ −15.3 ± 1.1) vs no-carry 0.95 (Δ −5.9 ± 0.7). Selectivity survives the off-distribution control.
Mean-ablation of H10 alone is *not* selective (−4.9 vs −4.7) and makes 10% of no-carry problems *gain* a spurious
carry — the errors of mean-ablation fall on the carry-mishandled digit in **both** directions, as expected if the
head's output encodes "carry present" and the dataset mean is a half-carry value (INFERRED).

**DERIVED (attention).** From the answer position, H10 puts 0.58 / 0.40 of its attention on the units digits of
A / B (carry problems; 0.68 / 0.29 no-carry); H11 0.37 / 0.60. Together ≈ 98% on the two units digits — the only
tokens that determine whether a carry exists. Weight on tens digits < 0.04.

**DERIVED (Logit Lens).** Carry problems: the carry-mishandled digit dominates the lens (P > 0.5) before the
correct digit in **34/40** problems (mean P_alt peaks 0.79 at L23; mean P_true 0.77 at L25, 0.97 at L26). No-carry
problems: 0/40. With H10 + H11 ablated the lens stays on the carry-mishandled digit through the final layer
(P ≈ 0.95).

## 2026-09-29 — E2: their output at one position is sufficient to inject or remove a carry

Script `scripts/experiments/carry_patch.py --n 40 --seed 0`; raw data `docs/research/E2_carry_patch.json`.
40 matched pairs sharing tens digits: N (units do not carry) and C (units carry). Only the outputs z of the two
heads **at the answer position** are transplanted (HeadSetEdit), everything else untouched.

**CAUSALLY TESTED (n = 40 pairs).**

| transplant | tens digit flips | Δ logit(t1+t2+1) − logit(t1+t2), mean ± SEM |
|---|---|---|
| z_{L22.H10,H11}(C) → N (inject) | **35/40** to t1 + t2 + 1 | **+23.5 ± 1.3** |
| z_{L22.H10,H11}(N) → C (remove) | **37/40** to t1 + t2 | **−27.3 ± 1.1** |
| control z_{L22.H8,H9}, inject | 0/40 | +0.04 ± 0.05 |
| control z_{L22.H8,H9}, remove | 0/40 | −0.18 ± 0.09 |

**Conclusion (scope: Qwen2.5-1.5B-Instruct, this prompt template, two-digit sums < 100, tens-digit readout).**
The outputs of attention heads L22.H10 and L22.H11 (one KV group) at the answer position carry a signal that is
necessary (E1) and sufficient (E2) for adding the units carry into the tens digit; the heads read almost
exclusively from the units digits; the lens shows the no-carry sum formed first and the carry applied at L25–L26.
**Not yet shown:** the dimensionality of the signal (is there a single carry direction in the 256-d head
output?), which downstream components consume it, whether the same heads serve carries into the hundreds digit,
chained carries (e.g. the 5-digit failure 76150 + 28254 → 104374) or other prompt formats, and why Qwen2.5-0.5B
fails the carry on 37 + 58.
