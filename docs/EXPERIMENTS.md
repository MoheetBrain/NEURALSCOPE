# Experiments in NeuralScope

## The experiment record

Every run is persisted with enough information to re-execute it deterministically:

```
run
├── id, created_at, kind (baseline | intervention | math), model_id, is_demo
├── prompt, prompt_text (exact templated string), chat, system_prompt
├── generation   {decoding: greedy, max_new_tokens, stopped_by, kv_cache, attn_implementation: eager, dtype, timings}
├── input_ids, prompt_len, user_span, generated_ids, generated_text
├── free_generated_ids/text          (intervention runs: free-running output under the intervention)
├── target       {position, token_id, token, contrast_id, source, note}
├── trace_level, tensor_path, tensor_meta {keys: shapes/dtypes, bytes}
├── parent_run_id, interventions[]   (Pydantic specs; compiled to hook edits at run time)
├── metrics      {comparison (intervention runs), sweeps[], lazy_recapture}
├── math         {problem, verification, oracle_answer_logprob, decision_point, truncated, label}
├── correctness, notes, tags, benchmark_id, trace_consistent
experiment  = {id, name, notes (INFERRED hypotheses live here), run_ids[], payload (UI selection state)}
benchmark   = {id, model_id, spec (problems), results[], summary (accuracy, by_kind, intervention_evals[])}
```

`trace_consistent` checks that a teacher-forced capture reproduces every greedy token of the generation.

## Built-in flagship demo — 37 + 58

1. `What is 37 + 58? Give only the answer.` → `95` (tokenisation shows each digit as its own token).
2. TOKEN × LAYER matrix, metric *Logit Lens rank of target* with target `9` at the answer position.
3. Logit Lens: `9` climbs from rank ~70 (emb) to rank 2 at L23 and rank 1 at L26 (DERIVED).
4. Attention for layer 22: select head 10 and inspect its query row from the answer position.
5. Head sweep: all 336 heads, one at a time → L22.H10 and L22.H11 stand out (Δ log P ≈ −8).
6. ABLATE L22.H10 + RERUN → P(`9`) 0.9998 → 0.0003, KL 7.96 nats, output `95` → `85` (CAUSALLY TESTED, n = 1).
7. DIFF: the residual change begins at depth 23 (after block 22), exactly where the edit acts.

## Math Lab workflow

1. **Run a benchmark** (default: 25 seeded problems — additions 2d/3d/5d, subtraction, multiplication 2d×2d and
   3d×2d, linear equations, fraction comparison, modular arithmetic). Each problem is generated greedily, parsed,
   verified by SymPy, traced (minimal level, lazily upgraded), and given a target at its *decision point*.
2. **Compare** a correct run A with an incorrect run B: target rank by depth, residual norms, successive cosine
   change, lens entropy, cross-run cosine, attention-entropy differences (DERIVED, correlational).
3. **Test** candidate components across the whole benchmark: benchmark intervention evaluation reports accuracy
   before → after, flips in both directions and mean Δ log P(oracle answer) ± SEM (CAUSALLY TESTED, n = N).
4. **Patch** between equal-length counterfactual problems (e.g. `37 + 58` vs `37 + 48`) with the patching sweep.

## Experiment E1 — carry heads (script: `scripts/experiments/carry_heads.py`)

*Question.* Are L22.H10 / L22.H11 of Qwen2.5-1.5B-Instruct necessary specifically for adding the units carry into
the tens digit in two-digit addition?

*Design.* 2 × N problems `A + B < 100` (N with a units carry, N without), identical token length, readout at the
tens digit. Conditions: baseline, zero-ablate H10, H11, H10+H11, every other head in layer 22 (same-layer
controls), K random heads in other layers. Metrics per group: tens-digit accuracy, fraction predicting the
carry-mishandled digit (`a_t + b_t` for carry problems, `a_t + b_t + 1` for no-carry problems), mean
P(correct), mean Δ logit-difference (correct − carry-mishandled) ± SEM; free-running outputs for the ablated
condition; mean attention weight of H10/H11 from the answer position to each operand digit (DERIVED).

*Predictions if the hypothesis holds.* Large drop for carry problems, small for no-carry; errors concentrated on
the carry-mishandled digit; same-layer and random controls near zero.

*Run.* `.venv/bin/python scripts/experiments/carry_heads.py --n 40` (needs ~7 GB RAM; stop the API server first on
a 16 GB machine). Results → `docs/research/E1_carry_heads.json` and RESEARCH_LOG.md.

## Reproducibility

* Greedy decoding with first-index argmax tie-breaking; eager attention; float32; `torch.no_grad`.
* Runs store exact token ids, so re-tokenisation drift cannot change a stored experiment.
* Hero snapshot: `scripts/export_hero_snapshot.py` regenerates `apps/web/public/snapshots/hero-trace.json` from the
  live API (the head shown is selected by the exhaustive sweep, not by hand).
