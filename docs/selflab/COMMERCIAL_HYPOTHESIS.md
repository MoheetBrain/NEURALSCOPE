# Commercial hypothesis (untested)

**Hypothesis.** Some model and post-training teams will pay for a repeatable audit that does three things:
1. takes the claims their model makes about its own behaviour: "I checked the document", "my answer was based on
   the evidence", "I used the tool result";
2. runs those claims through controlled counterfactual tests;
3. reports, per checkpoint, where those claims stop tracking measured behaviour.

The hypothesis is that such an audit catches unsupported verification claims, or explanation regressions between
checkpoints, that the team's current evaluations miss.

There is no market validation. No customer, design partner, price or demand signal exists. Nothing in this
repository should be read as evidence of commercial demand. Billing is out of scope.

**Why it might be wrong.**
- Teams may already catch these failures with existing faithfulness or groundedness evals.
- The audit may only rediscover what a transcript-only check finds. Experiment A explicitly measures this: if the
  transcript observer matches the self-report, the audit adds little beyond groundedness checking.
- White-box access to production checkpoints is rarely given to outside tools.
- The behaviour of interest may not appear in small open models.

## Design-partner test (one)

1. **Partner.** One team that ships an instruction-tuned open-weight model (≤ 8B) with retrieval or document
   grounding.
2. **Input.**
   - Two of their checkpoints, A and B, run locally on their hardware.
   - 50 of their own grounded tasks, adapted to the Experiment A template: document present, absent or replaced,
     and a conflicting user hint.
3. **Procedure.**
   - Before running anything, the team writes down which checkpoint they expect to make more unsupported "based
     on the document" claims, and their current eval's verdict on that.
   - We run the locked Experiment A protocol on both checkpoints. Settings are fixed in advance: N, families,
     metrics and the held-out family.
4. **Success criterion, fixed in advance.** The audit reports a checkpoint difference in source-report error
   whose clustered 95% CI excludes 0 and which the team's existing eval did not flag. The team also states, in
   writing, that the finding would change a release or training decision.
5. **Failure modes recorded as results.**
   - No difference.
   - A difference their eval already caught.
   - A difference fully explained by the transcript observer: the audit adds nothing over output checking.
   - Inability to run white-box locally.

Cost to run: about one engineer-day of setup and under one CPU-hour per checkpoint at N = 50. This estimate is
scaled from the pilot timings in `docs/selflab/results/` and must be re-measured on the partner's model.
