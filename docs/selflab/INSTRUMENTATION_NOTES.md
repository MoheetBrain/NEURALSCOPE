# SelfLab instrumentation notes (SA2)

Scope: the instrumentation additions in INTERFACES §3 — `tokenize_messages`, `selflab/measure.py`, chat runs in
`NeuralScopeService.create_run`, hook hit counts — and the measurements behind the device recommendation.
Every number below was measured on this machine on 2026-10-17 (Apple M5 Max, 48 GB, macOS 26.6.2, torch 2.14.1,
transformers 5.20.1, Python 3.11). The commands are listed so the numbers can be re-measured.

## Recommendation: reference device for the pilot

**Run the pilot on CPU fp32 (the documented reference).**
- MPS fp32 is about 7× faster and agrees with CPU to about 1e-3 in logits. It does **not** give identical
  decisions: in a 32-trial sweep, one greedy answer flipped at a near-tie, and it flipped between the document
  value and the hint value, in the condition the experiment is about.
- Use MPS for development, smoke runs and exploratory sweeps, labelled with the device. Do not use it for
  confirmatory numbers unless the same trials are re-run on CPU.
- Never mix devices within an experiment or within a paired comparison. Provenance records the device.
- **We do not promise cross-hardware bitwise determinism.** On one device, one code path and one batch composition,
  repeated calls were bit-identical in our tests (fixture). Different code paths agree only to about 1e-5:
  KV-cached vs teacher-forced, padded batch vs single, resumed vs full forward.
- Record the per-step decision margin (`GenerationResult.generated_margins`, top-1 minus top-2 logit). Flag trials
  with a margin below 1e-3 as **near-ties**, report how many there are, and check that conclusions survive
  dropping them.

## (a) MPS vs CPU parity — Qwen2.5-1.5B-Instruct, fp32, eager attention

Measured under `scripts/model_lock.py`, with `HFDecoderAdapter(device="cpu"|"mps", dtype="float32")` on identical
token inputs. The two measurement scripts are session scratch files (not committed) and are summarised here:
- `parity.py`: 4 SelfLab-style prompts (system + document + hint + question, 282–289 tokens).
- `parity2.py`: 32 Experiment-A-style trials, 8 invented entities × {doc_v, no_doc, doc_v_hint_conflict,
  no_doc_hint}.
  - Turn 1: greedy decoding, max 16 tokens.
  - Turn 2: a source-report question, with greedy decoding (max 24 tokens) and an A–D option readout.

**Logits on identical inputs** (4 prompts, all positions, full vocabulary):

| quantity | value |
|---|---|
| max \|Δlogit\| (CPU − MPS), per prompt | 7.7e-4, 1.7e-3, 1.3e-3, 8.4e-4 |
| mean \|Δlogit\| | ≈ 1.8e-5 |
| max \|Δlogit\| at the last prompt position | 1.3e-4 – 4.9e-4 |
| argmax agreement over all positions | 100 % (4/4 prompts) |
| greedy output equal (turn 1) | 4/4, and 4/4 under prompt-position head ablations |
| max \|Δ sum log-prob\| of 4 candidate continuations (2–4 tokens incl. `<\|im_end\|>`) | ≤ 4.0e-4; candidate ranking identical 4/4 |
| max \|Δ option log-prob\| (A–D) | ≤ 2.8e-4 |
| `patch_restoration`: max \|Δ restoration\| over 29 depths × 2 positions + joint | 3.3e-6 |
| KV-cached vs teacher-forced log-probs under prompt-position edits (same device) | CPU ≤ 2.0e-5, MPS ≤ 1.6e-5 |

**Decisions** (32-trial sweep; the run was repeated and the results were identical):

| quantity | value |
|---|---|
| turn-1 greedy answer equal | **31/32** |
| turn-2 greedy report equal | 32/32 |
| option argmax equal | 32/32 |
| max \|Δ option log-prob\|, over the 31 trials with identical turn-2 inputs | 2.9e-4 |
| trials whose smallest greedy-step margin is < 0.01 / < 1e-3 | 1 / 1 |

The single mismatch was a `doc_v_hint_conflict` trial. CPU answered "Red" (the document value) and MPS answered
"Ochre" (the hint). The CPU top-1/top-2 margin at that step was 4.6e-5 logits teacher-forced and 1.7e-5 on the
KV-cached path. That is a genuine near-tie, well inside the ~1e-3 cross-device noise. Every other trial had a
minimum margin ≥ 0.0116 and agreed. The lesson: logit-level parity does not guarantee decision-level parity at
near-ties, and near-ties are exactly where "document vs hint" is decided. Hence the CPU recommendation and the
near-tie flag above.

## (b) Timing and memory (Qwen2.5-1.5B-Instruct, fp32)

| operation | CPU (torch default 6 threads) | MPS |
|---|---|---|
| load (warm HF cache) | 3.3 s | 1.2 s |
| one teacher-forced forward, ~285 tokens, full logits | 0.61 – 0.74 s | 0.088 – 0.093 s |
| `continuation_logprobs`, 4 candidates in one padded batch, ~285-token prefix | 1.88 – 2.02 s | 0.26 – 0.28 s |
| `greedy`, max 16 (stopped at EOS after 2–4 tokens) | 0.66 – 0.84 s | 0.11 – 0.15 s |
| `patch_restoration`, 282 tokens, 29 depths × 3 rows (2 positions + joint) = 87 rows, resumed | 23.6 s | 3.5 s |
| 32-trial sweep (turn 1 + turn 2 + option readout + margin recompute), 56–233 tokens | 93 – 101 s (~3 s/trial) | 23 – 24 s (~0.75 s/trial) |

- **CPU threads.** A 173-token forward took 0.379 s with 4 threads, 0.40 s with 6, 0.406 s with 12 and 0.425 s
  with 18. There is no gain beyond 4 threads at this size; the M5 Max has 6 performance cores and 12 efficiency
  cores.
- **Memory.** The weights are 1.54 B params × 4 B ≈ 6.2 GB. Peak process RSS was 9.7 GB after the CPU phase,
  which includes the retained full-vocabulary logits of the parity script. MPS driver-allocated memory was 7.5 GB.
  Activation storage is the variable cost of a Lab run. A `standard` trace stores attention as fp16
  [28 layers × 12 heads × T × T], about 60 MB at T = 300 and about 700 MB at the 1024-token hard cap. Use
  `trace_level="minimal"` for bulk SelfLab runs and upgrade lazily.
- **Pilot throughput estimate.** This is derived, not measured end to end. One Experiment-A trial is turn-1
  greedy, 4-candidate scoring, two label orders × (option readout + 24-token report). That is about 6 s on CPU
  and about 1 s on MPS, so 1000 trials take about 1.7 h on CPU.
- **Optional speed-up, not implemented.** `continuation_logprobs` re-reads the prefix once per candidate in a
  padded batch. Reusing the prefix KV cache would cut 4-candidate scoring on CPU from about 2.0 s to about 0.8 s.
  It was left out on purpose: correctness first, and it would put the KV-vs-full ~1e-5 path difference into
  every score.

## Tests and results (exact commands)

| command | result |
|---|---|
| `.venv/bin/python -m pytest tests/test_instrumentation_selflab.py -m "not slow" -q` | 26 passed, 3 deselected |
| `.venv/bin/python -m pytest -m "not slow" -q` (whole fast suite, shared tree, 2026-10-17) | 569 passed, 1 failed, 10 xfailed. The failure is `test_selflab_validation_persistence.py::test_insert_or_replace_cannot_overwrite_a_trial`, a strict-xfail (SA6) on SA4's store that now XPASSes. It is not in SA2's files. |
| `.venv/bin/python scripts/model_lock.py -- .venv/bin/python -m pytest -m slow -q tests/test_instrumentation_selflab.py` | 3 passed (real Qwen2.5-0.5B-Instruct, CPU fp32) |
| `.venv/bin/python scripts/model_lock.py -- .venv/bin/python -m pytest -m slow -q` | 8 passed (5 baseline + 3 new) |

## (c) External backends (NNsight)

The earlier audit (`SELFLAB_AUDIT_AND_PLAN.md`, `PRIOR_ART_MATRIX.md`) ran NNsight 0.7.0 on Qwen2.5-0.5B and found it
bit-exact with our hooks in five checks, at the same speed, including resume-from-depth via `.skip()`.

No current blocker requires migrating. Everything SelfLab v1 needs runs on the existing native-hook adapter:
chat tokenisation, full continuation scoring, option readouts, position-aligned patching, prompt-position edits
with hook hit counts, and KV-cached generation under edits. All of it is covered by the tests listed below.
NNsight was not installed for this work. The planned migration (thin NNsight adapter after a parity test that
reproduces E2 exactly) stays a separate milestone. If it happens, `tests/test_instrumentation.py` and
`tests/test_instrumentation_selflab.py` are the parity suite it must pass. The edit vocabulary, `n_edits_applied`
semantics and absolute-position rules defined here are what the new provider has to reproduce.

## (d) What an Experiment C prompt-position edit touches

Experiment C manipulates internal state at prompt positions and asks whether the model's later report reflects it.
For that to be interpretable we need to know exactly where the manipulation enters. The rules below are the
implemented behaviour, and each one is asserted by a test in `tests/test_instrumentation_selflab.py`.

**Positions are absolute.** An edit's `positions` are indices into the full token sequence: system prompt,
template tokens, every earlier turn, then the current turn. The hook session tracks the absolute position of the
first token in each forward call (`offset`), so the same edit means the same token whether the sequence is
processed in one teacher-forced pass, in a batched pass, or as a prefill followed by KV-cached decode steps. Map
a character span of a message to positions with `tokenize_messages(messages).char_span_to_tokens(i, a, b)` on the
*exact* message list that is fed to the model. A turn-2 prompt contains turn 1. With the Qwen template, the turn-1
messages occupy the same token positions in the turn-2 sequence (prefix-stable; tested). Even so, take positions
from the turn-2 `TokenizedChat` rather than reusing turn-1 numbers. The assistant answer inside turn 2 is the
template's canonical tokenization of the answer text, which is not guaranteed to equal the generated ids.

**Which forward call.** `generate` runs one prefill over positions `0..P-1` and then one call per generated token
(position `P + k - 1` for the call that produces token `k+1`). An edit restricted to prompt positions fires **only
in the prefill**. Decode calls process positions ≥ P, so the edit never fires there: `n_edits_applied` equals
the number of prompt-position edits (×1 row), not edits × steps (`test_kv_cached_greedy_under_prompt_edits_…`,
`test_hook_hit_counts`).

**Which layers and positions change** (edit at prompt position set S, s = min S):

| edit | first modified tensor | unchanged (bit-identical) |
|---|---|---|
| `ResidAddEdit/ResidSetEdit(depth d)` | residual entering block d at S | all depths < d; every depth at positions < s; depth d at positions ∉ S |
| `HeadScaleEdit/HeadSetEdit(layer l)` | head z of layer l at S → residual depth l+1 at S | depths ≤ l; positions < s; depth l+1 at positions ∉ S |
| `MLPScaleEdit(layer l)` | MLP output of layer l at S → residual depth l+1 at S | depths ≤ l; positions < s; depth l+1 at positions ∉ S |

From the first modified depth onward, attention carries the change to every later position (> s) at every later
layer, so the final-layer residual at positions ≥ s generally changes. This is checked bit-for-bit in
`test_prompt_position_edit_changes_only_downstream_states`.

**Which KV-cache entries carry it.** Keys and values of layer k are computed from the residual entering block k
(after its input norm). So after the prefill:
- a residual edit at depth d changes the cached K/V of layers ≥ d at positions in S, and of layers ≥ d+1 at
  positions > s;
- a head or MLP edit at layer l changes the cached K/V of layers ≥ l+1 at positions ≥ s. Layer l's own K/V are
  computed before its o_proj and MLP and are unchanged.

**How the manipulation reaches the report.**
1. The **first** generated token is read off the prefill's logits at position P−1. If P−1 ≥ s, the edit reaches
   it directly through the residual stream of the prefill, not only through the cache.
2. Every **later** generated token is computed in a decode call that processes only its own position. The edit
   does not fire there. It reaches that token **only through the cached K/V** of the prompt positions (and of
   generated positions, whose own K/V were computed by attending to the edited cache).

`test_kv_cached_greedy_under_prompt_edits_equals_teacher_forced_rescoring` checks that this cached path equals one
teacher-forced pass over prompt + generated tokens with the same edits. Generated token ids match, and per-token
log-probs agree within the test tolerances: 1e-4 on the fixture and 2e-3 on Qwen2.5-0.5B. The gap measured on
Qwen2.5-1.5B was ≤ 2e-5 (see (a)). So `continuation_logprobs(prefix, [generated], edits)` is a valid re-scoring of a
report produced under prompt edits. The exception is a near-tie step, where the two paths may pick different
tokens; `generated_margins` flags those.

**Sham control.** `HeadScaleEdit(factor=1.0)`, a zero-vector `ResidAddEdit` and `MLPScaleEdit(factor=1.0)` run the
same hooks (they do fire: `n_edits_applied > 0`) and leave logits, residuals, attention, head z, generations and
continuation scores bit-identical (`test_sham_edits_are_bit_identical`). Use them as the sham arm. The hook
machinery itself does not perturb the computation.

**Hit-count semantics.** `n_edits_applied` counts (edit, batch row, forward call) triples in which the edit's
positions intersected the positions processed by that call. It is exposed as `GenerationResult.n_edits_applied`
and, for every instrumented call (generate, capture, `final_logits_batch`, `sequence_logits`, and so every
`measure.py` function), as `adapter.last_session_stats = {n_edits_applied, n_forward_calls, n_rows, n_edits}`.
Read it under `adapter.lock`. `measure.py` functions hold the lock for their whole duration, and
`option_logprobs` also returns `n_edits_applied`. One edit with `positions=[a, b, c]` counts once per call, not
three times. A resumed `final_logits_batch(start_depth>0)` counts as one forward call.

## API added (all additive)

`instrumentation/adapter.py`
- `TokenizedChat(text, input_ids, message_token_spans, message_char_spans=[], offsets=[], span_exact=[],
  add_generation_prompt=True)` with `char_span_to_tokens(message_index, char_start, char_end)`.
- `ModelAdapter.tokenize_messages(messages, add_generation_prompt=True)`. The base raises NotImplementedError and
  `HFDecoderAdapter` implements it.
- `ModelAdapter.sequence_logits(sequences, positions, edits=None) -> list[Tensor[len(pos), V]]`. The generic
  fallback loops; `HFDecoderAdapter` runs one right-padded batch.
- `final_logits_batch(..., logit_positions=None)`: unembed only the requested positions.
- `GenerationResult.n_edits_applied`, `GenerationResult.generated_logprobs` (log-softmax of the cached decode
  logits at each chosen token).
- `ModelAdapter.last_session_stats`.

`selflab/measure.py`: `ContinuationScore`, `continuation_logprobs`, `assistant_continuation_ids`, `option_logprobs`,
`greedy`, `differing_positions`, `patch_restoration` (as in §3), plus `last_hook_stats(adapter)`.
`patch_restoration` has keyword-only extras: `joint=True` (adds `restoration_joint[i]`, all positions patched
together, which multi-token values need), `batch_size=16`, `resume=True`, and `max_diff_fraction=0.25`.

`experiments/service.py`: `create_run(prompt=None, …, messages=None, max_prompt_tokens=None)`.
- `max_prompt_tokens` is a per-call override within `1..MAX_PROMPT_TOKENS_HARD_CAP` (1024). The global default
  (160) is unchanged.
- With `messages`, the last message must be a user message, a system message may only come first, and
  `system_prompt` must not also be passed.
- Intervention children copy `messages` from their base run.

`experiments/store.py`: new nullable `runs.messages` JSON column, added by an idempotent `ALTER TABLE` migration on
open. Old rows read `messages = None`.

### Behavioural notes for callers (SA3/SA4)

- **Locating content.** `tokenize_messages` re-renders the conversation with message i's content replaced by a
  private-use sentinel. The sentinel's offset is where the content starts, and the real text is verified to hold
  the content verbatim there. If that fails, it falls back to an ordered verbatim search. A naive search would
  wrongly place content like "You" inside Qwen's injected default system prompt (tested).
- **Token boundaries.** Spans contain every token overlapping the content. `span_exact[i] = False` flags a token
  that straddles the template/content boundary (e.g. content starting with `\n`, which merges with the header's
  newline).
- **`assistant_continuation_ids` is derived from the template.** It checks prefix stability, verbatim rendering
  and token alignment, and that the token after the content is a stop token (`<|im_end|>` = 151645 for Qwen). It
  raises ValueError instead of guessing. On Qwen2.5-0.5B, `assistant_continuation_ids(decode(greedy_output))`
  reproduced the greedy token ids exactly (slow test). That is one prompt, not a guarantee: a model can emit a
  non-canonical tokenization. Score candidates by these canonical ids, and keep the generated ids for the
  model's own answer.
- **`continuation_logprobs`.** Scores all candidates in one right-padded batch. This is exact under causal
  attention. On the fixture, batched and one-at-a-time scores were bit-identical. Log-softmax is computed in
  float64 over the full vocabulary from float32 logits.
- **`patch_restoration`.**
  - It refuses unequal lengths, positions after the readout, depths outside `0..n_layers`, `tok_clean ==
    tok_corrupt`, and pairs that differ at ≥ 25 % of positions (misaligned).
  - Identical inputs are allowed and return `defined=False` with `restoration=None`. The no-op patch leaves
    `ld_patched == ld_corrupt`.
  - It truncates both sequences to `readout_pos + 1`, because nothing later can matter causally.
  - Each depth is one batched call, resumed from the cached corrupt residual at that depth.
