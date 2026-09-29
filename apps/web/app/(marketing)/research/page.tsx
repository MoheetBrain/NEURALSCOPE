import type { Metadata } from "next";
import { ArrowRight } from "lucide-react";
import { Callout, DocHeading, PageHeader, Prose, Toc, type TocItem } from "@/components/docs/DocPrimitives";
import { Container } from "@/components/marketing/Section";
import { ResearchLoop } from "@/components/marketing/ResearchLoop";
import { ButtonLink } from "@/components/ui/Button";
import { EpistemicBadge } from "@/components/ui/EpistemicBadge";
import { StatusChip } from "@/components/ui/StatusChip";
import { EPISTEMIC, EPISTEMIC_ORDER, type EpistemicStatus } from "@/lib/epistemic";

export const metadata: Metadata = {
  title: "Research",
  description:
    "NeuralScope's research programme: recover the algorithms that learning discovered, by measurement, interpretation and causal intervention.",
};

const TOC: TocItem[] = [
  { id: "philosophy", label: "Philosophy" },
  { id: "loop", label: "The research loop" },
  { id: "epistemology", label: "Epistemology" },
  { id: "first-question", label: "First scientific question" },
  { id: "experiments", label: "Planned experiments" },
  { id: "non-claims", label: "What we will not claim" },
];

const PRINCIPLES = [
  {
    k: "Measurement before interpretation",
    v: "Start from tensors captured during a real forward pass of the real model, not from a story about what the model might be doing.",
  },
  {
    k: "Interpretation is a hypothesis",
    v: "A readout such as the Logit Lens, or a description written by another language model, proposes what a component does. It does not establish it.",
  },
  {
    k: "Intervention before claim",
    v: "A mechanistic claim is only as strong as the intervention that tested it: which component, which edit, which metric, over how many controlled examples, against which baseline.",
  },
  {
    k: "Negative results are results",
    v: "A hypothesis that fails its intervention is recorded with the run, not quietly dropped.",
  },
];

const EPISTEMOLOGY: Record<EpistemicStatus, { producedBy: string; allowed: string; notAllowed: string }> = {
  observed: {
    producedBy: "Forward hooks on the actual run: residual stream, attention weights, MLP outputs, logits.",
    allowed: "“The residual state h[L14, t7] was recorded for this prompt.”",
    notAllowed: "“This neuron detects carries.”",
  },
  derived: {
    producedBy: "Deterministic maths on observed tensors: Logit Lens, norms, KL divergence, cosine change, direct logit attribution.",
    allowed: "“Projected through the unembedding, h[L14, t7] ranks “8” first.”",
    notAllowed: "“At layer 14 the model already believes the answer is 85.”",
  },
  inferred: {
    producedBy: "A researcher or an LLM proposing a meaning for a component or direction.",
    allowed: "“L17.H5 may carry tens-digit information (hypothesis).”",
    notAllowed: "Presenting the description as a finding, or as ground truth.",
  },
  causal: {
    producedBy: "A rerun with a specified intervention, measured against the clean run and against baselines.",
    allowed: "“Zero-ablating L17.H5 lowered p(correct) across N controlled problems, more than random heads did.”",
    notAllowed: "“L17.H5 is the carry circuit.” (Necessity for one metric is not a full mechanism.)",
  },
};

const EXPERIMENTS: {
  id: string;
  title: string;
  question: string;
  method: string;
  metric: string;
  yields: EpistemicStatus;
}[] = [
  {
    id: "E1",
    title: "Where does the answer appear?",
    question: "At what depth does the correct answer token become decodable, and does that depth differ between correct and incorrect runs?",
    method: "Logit Lens rank and probability of the first answer token across all depths, for oracle-labelled correct vs incorrect generations.",
    metric: "First depth at rank 1; rank trajectories; entropy by depth.",
    yields: "derived",
  },
  {
    id: "E2",
    title: "Head ablation sweep",
    question: "Which attention heads are necessary for multi-digit addition on this model?",
    method: "Zero-ablate each head in turn (all positions) and measure at the answer position; compare to the clean run and to the distribution over all heads.",
    metric: "Δ p(correct), Δ rank, KL(clean ‖ ablated); oracle accuracy on held-out problems.",
    yields: "causal",
  },
  {
    id: "E3",
    title: "Carry vs no-carry",
    question: "Do the components found in E2 matter selectively for problems that require a carry?",
    method: "Matched problem sets that differ only in whether a carry is needed; repeat the E2 ablations on both.",
    metric: "Interaction between ablation effect and carry condition.",
    yields: "causal",
  },
  {
    id: "E4",
    title: "Rescue by patching",
    question: "Can a failed computation be rescued by patching internal state from a successful one?",
    method: "Patch residual states at the first-divergence depth from a correct run into a failed run of a matched problem; re-verify with the oracle.",
    metric: "Rescue rate vs patches at random depths and positions.",
    yields: "causal",
  },
  {
    id: "E5",
    title: "Attention to operands",
    question: "Does attention from the answer position to operand digits differ between correct and failed runs?",
    method: "Compare observed attention weights per head; follow up any difference with E2-style ablations.",
    metric: "Attention mass on operand tokens; effect size between groups.",
    yields: "observed",
  },
  {
    id: "E6",
    title: "Robustness controls",
    question: "Do the findings survive changes that should not matter?",
    method: "Repeat key results under prompt paraphrase, operand magnitude, chat vs raw prompts, and zero vs scaled ablation.",
    metric: "Stability of effect sizes and component rankings.",
    yields: "causal",
  },
];

export default function ResearchPage() {
  return (
    <>
      <PageHeader kicker="Research" title="Recover the algorithms that learning discovered.">
        <p>
          Training produces computations that no one wrote down. NeuralScope treats them as objects of experiment: measured
          directly, decoded carefully, and tested by intervention.
        </p>
        <p className="flex flex-wrap items-center gap-2 pt-2 font-mono text-[12px] text-ns-text-faint">
          <StatusChip tone="neutral">V1 model</StatusChip>
          Qwen2.5-1.5B-Instruct · runs locally · PyTorch forward hooks · float32 by default
        </p>
      </PageHeader>

      <Container className="py-16 sm:py-20">
        <div className="grid gap-12 lg:grid-cols-[220px_minmax(0,1fr)] lg:gap-16">
          <aside className="hidden lg:block">
            <div className="sticky top-24">
              <Toc items={TOC} />
            </div>
          </aside>

          <div className="min-w-0 space-y-24">
            {/* ------------------------------------------------ philosophy */}
            <section aria-labelledby="philosophy">
              <DocHeading id="philosophy" kicker="01">
                Philosophy
              </DocHeading>
              <Prose>
                <p>
                  Modern networks are not programmed; they are optimised. Gradient descent searches for parameters that satisfy
                  an objective, and in doing so it discovers internal procedures: ways of representing numbers, of routing
                  information between tokens, of composing intermediate results. Those procedures are real, they are
                  executed on every forward pass, and they are only partially understood.
                </p>
                <p>
                  <strong>Our mission is to recover the algorithms that learning discovered.</strong> That requires an
                  instrument that can capture internal state precisely, decode it without overclaiming, and change it to test
                  whether a proposed mechanism is actually used.
                </p>
              </Prose>
              <ol className="mt-10 grid gap-px overflow-hidden rounded-[var(--ns-radius)] border border-ns-border bg-ns-border sm:grid-cols-2">
                {PRINCIPLES.map((p, i) => (
                  <li key={p.k} className="bg-ns-panel p-5">
                    <p className="font-mono text-[11px] text-ns-text-faint">P{i + 1}</p>
                    <h3 className="mt-3 text-[16px] font-medium text-ns-text">{p.k}</h3>
                    <p className="mt-2 text-[14px] leading-relaxed text-ns-text-muted">{p.v}</p>
                  </li>
                ))}
              </ol>
            </section>

            {/* ------------------------------------------------ loop */}
            <section aria-labelledby="loop">
              <DocHeading id="loop" kicker="02">
                The research loop
              </DocHeading>
              <Prose className="mb-10">
                <p>
                  Observe → Interpret → Hypothesise → Intervene → Measure → Causally validate → Improve. Each step produces a
                  labelled artefact, so it is always clear which parts of a result were measured, which were computed, which
                  were guessed, and which were tested.
                </p>
              </Prose>
              <ResearchLoop />
            </section>

            {/* ------------------------------------------------ epistemology */}
            <section aria-labelledby="epistemology">
              <DocHeading id="epistemology" kicker="03">
                Epistemology
              </DocHeading>
              <Prose>
                <p>
                  Every quantity in NeuralScope carries one of four epistemic labels. The label constrains the sentence you are
                  allowed to write about it.
                </p>
              </Prose>
              <div className="mt-8 overflow-hidden rounded-[var(--ns-radius)] border border-ns-border">
                <div className="hidden grid-cols-[170px_minmax(0,1.1fr)_minmax(0,1fr)_minmax(0,1fr)] gap-6 border-b border-ns-border bg-ns-panel px-5 py-3 md:grid">
                  {["Status", "Produced by", "Permitted claim", "Not permitted"].map((h) => (
                    <span key={h} className="ns-label">
                      {h}
                    </span>
                  ))}
                </div>
                <ul className="divide-y divide-ns-border">
                  {EPISTEMIC_ORDER.map((s) => (
                    <li key={s} className="grid gap-3 px-5 py-5 md:grid-cols-[170px_minmax(0,1.1fr)_minmax(0,1fr)_minmax(0,1fr)] md:gap-6">
                      <div>
                        <EpistemicBadge status={s} />
                        <p className="mt-2 text-[12.5px] leading-snug text-ns-text-faint">{EPISTEMIC[s].definition}</p>
                      </div>
                      <p className="text-[14px] leading-relaxed text-ns-text-muted">
                        <span className="ns-label mr-2 md:hidden">Produced by</span>
                        {EPISTEMOLOGY[s].producedBy}
                      </p>
                      <p className="text-[14px] leading-relaxed text-ns-text">
                        <span className="ns-label mr-2 md:hidden">Permitted</span>
                        {EPISTEMOLOGY[s].allowed}
                      </p>
                      <p className="text-[14px] leading-relaxed text-ns-text-faint">
                        <span className="ns-label mr-2 md:hidden">Not permitted</span>
                        <span aria-hidden="true" className="mr-1.5 font-mono text-ns-danger/80">✕</span>
                        {EPISTEMOLOGY[s].notAllowed}
                      </p>
                    </li>
                  ))}
                </ul>
              </div>
              <Callout title="LLM-generated descriptions" tone="accent">
                Natural-language descriptions produced by a language model are always labelled INFERRED. They are useful for
                generating hypotheses and are never displayed as ground truth.
              </Callout>
            </section>

            {/* ------------------------------------------------ first question */}
            <section aria-labelledby="first-question">
              <DocHeading id="first-question" kicker="04">
                First scientific question
              </DocHeading>
              <blockquote className="max-w-[40ch] border-l-2 border-ns-accent-2/70 pl-6 text-[22px] font-light leading-snug text-ns-text sm:text-[28px]">
                Can we identify internal differences between correct and incorrect mathematical computations and causally test
                whether any of those differences matter?
              </blockquote>
              <div className="mt-10 grid gap-px overflow-hidden rounded-[var(--ns-radius)] border border-ns-border bg-ns-border md:grid-cols-2">
                {[
                  { k: "Model", v: "Qwen2.5-1.5B-Instruct, run locally with PyTorch forward hooks (float32 by default)." },
                  { k: "Problems", v: "Generated arithmetic and short maths problems with machine-checkable answers." },
                  { k: "Labels", v: "A SymPy oracle marks each generation correct or incorrect. The label is an observation about the output." },
                  { k: "Contrast", v: "Matched correct vs incorrect runs, aligned by token position and depth." },
                  { k: "Interventions", v: "Head and MLP ablation; residual-stream patching from correct into incorrect runs." },
                  { k: "Bar for a claim", v: "Effects that exceed random-component baselines on held-out problems, with controls reported." },
                ].map((row) => (
                  <div key={row.k} className="bg-ns-panel p-5">
                    <p className="ns-label">{row.k}</p>
                    <p className="mt-2 text-[14px] leading-relaxed text-ns-text-muted">{row.v}</p>
                  </div>
                ))}
              </div>
              <p className="mt-4 font-mono text-[11px] text-ns-text-faint">Study design · subject to revision as V1 matures.</p>
            </section>

            {/* ------------------------------------------------ experiments */}
            <section aria-labelledby="experiments">
              <DocHeading id="experiments" kicker="05">
                Planned experiments
              </DocHeading>
              <Prose>
                <p>
                  None of these have been run yet. They are listed with their question, method, metric and the epistemic label a
                  successful result would earn. Results, including null results, will be published here.
                </p>
              </Prose>
              <ol className="mt-8 grid gap-3 md:grid-cols-2">
                {EXPERIMENTS.map((e) => (
                  <li key={e.id} className="flex flex-col rounded-[var(--ns-radius)] border border-ns-border bg-ns-panel">
                    <div className="flex h-9 items-center gap-2 border-b border-ns-border px-4">
                      <span className="font-mono text-[11px] text-ns-text-faint">{e.id}</span>
                      <StatusChip tone="future" className="ml-auto">
                        Planned
                      </StatusChip>
                    </div>
                    <div className="flex flex-1 flex-col p-4">
                      <h3 className="text-[16px] font-medium text-ns-text">{e.title}</h3>
                      <p className="mt-2 text-[14px] leading-relaxed text-ns-text-muted">{e.question}</p>
                      <dl className="mt-4 space-y-2 border-t border-dashed border-ns-border pt-4 text-[13px] leading-relaxed">
                        <div className="grid grid-cols-[64px_minmax(0,1fr)] gap-3">
                          <dt className="ns-label pt-1">Method</dt>
                          <dd className="text-ns-text-muted">{e.method}</dd>
                        </div>
                        <div className="grid grid-cols-[64px_minmax(0,1fr)] gap-3">
                          <dt className="ns-label pt-1">Metric</dt>
                          <dd className="text-ns-text-muted">{e.metric}</dd>
                        </div>
                      </dl>
                      <p className="mt-auto flex items-center gap-2 pt-4 font-mono text-[10px] uppercase tracking-[0.12em] text-ns-text-faint">
                        Would yield <EpistemicBadge status={e.yields} size="xs" />
                      </p>
                    </div>
                  </li>
                ))}
              </ol>
            </section>

            {/* ------------------------------------------------ non-claims */}
            <section aria-labelledby="non-claims">
              <DocHeading id="non-claims" kicker="06">
                What we will not claim
              </DocHeading>
              <ul className="max-w-[68ch] divide-y divide-ns-border border-y border-ns-border">
                {[
                  "That a view shows “what the model is thinking”. Views show measured or derived quantities.",
                  "That attention weights are explanations. Attention is observed; its importance is tested by intervention.",
                  "That one ablation on one prompt reveals a mechanism. It is a single data point (n = 1) until repeated under controls.",
                  "That a component “is” a concept because steering along it changes behaviour. Sufficiency to move an output is not identity.",
                  "That findings on Qwen2.5-1.5B-Instruct transfer to other models without testing them there.",
                ].map((t) => (
                  <li key={t} className="flex gap-4 py-4 text-[15px] leading-relaxed text-ns-text-muted">
                    <span aria-hidden="true" className="mt-[11px] h-px w-3 shrink-0 bg-ns-danger/70" />
                    {t}
                  </li>
                ))}
              </ul>
              <div className="mt-12 flex flex-col gap-3 sm:flex-row">
                <ButtonLink href="/app" variant="primary" size="lg" caps>
                  Open Neural Debugger <ArrowRight aria-hidden="true" />
                </ButtonLink>
                <ButtonLink href="/docs" variant="secondary" size="lg" caps>
                  Read the docs
                </ButtonLink>
              </div>
            </section>
          </div>
        </div>
      </Container>
    </>
  );
}
