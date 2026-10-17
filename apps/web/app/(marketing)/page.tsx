import type { Metadata } from "next";
import { ArrowRight, ArrowUpRight } from "lucide-react";
import { EpistemicLadder, LabelledViewExample } from "@/components/marketing/EpistemicLadder";
import { FeatureStatusList } from "@/components/marketing/FeatureStatusList";
import { HeroTrace } from "@/components/marketing/HeroTrace";
import { MathLabMethod, MathLabPipeline, ProofFuture } from "@/components/marketing/MathLab";
import { Pillars } from "@/components/marketing/Pillars";
import { ReticleBackdrop } from "@/components/marketing/ReticleBackdrop";
import { PremiseDiagram } from "@/components/marketing/PremiseDiagram";
import { Roadmap } from "@/components/marketing/Roadmap";
import { Container, Section } from "@/components/marketing/Section";
import { SelfLabExample, SelfLabFlow, SelfLabNotClaimed } from "@/components/marketing/SelfLabSection";
import { ButtonLink } from "@/components/ui/Button";
import { EpistemicBadge } from "@/components/ui/EpistemicBadge";
import { EPISTEMIC_ORDER } from "@/lib/epistemic";
import { parseHeroTrace } from "@/lib/hero-trace";
import { SELFLAB_EXAMPLE } from "@/lib/selflab-example";
import heroTraceJson from "@/public/snapshots/hero-trace.json";

const heroTrace = parseHeroTrace(heroTraceJson);

export const metadata: Metadata = {
  title: { absolute: "NeuralScope — Inspect the model. Audit its claims." },
  description:
    "An experimental workbench for tracing AI responses, testing self-reports, and comparing explanations against measured evidence.",
};

const SPECS = [
  { k: "Model", v: "Qwen2.5-1.5B-Instruct, local" },
  { k: "Capture", v: "PyTorch forward hooks" },
  { k: "Views", v: "Token × layer · Logit Lens · attention" },
  { k: "Tests", v: "Causal ablation, rerun and measured" },
  { k: "Claims", v: "SelfLab audits · research prototype" },
];

export default function LandingPage() {
  return (
    <>
      {/* ================================================================ HERO */}
      <section aria-labelledby="hero-title" className="relative overflow-hidden">
        <div
          aria-hidden="true"
          className="ns-grid-bg pointer-events-none absolute inset-0 [mask-image:radial-gradient(ellipse_80%_70%_at_30%_0%,black,transparent_75%)]"
        />
        <Container className="relative grid items-center gap-12 pb-20 pt-14 sm:pt-20 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:gap-14 lg:pb-28 lg:pt-24">
          <div className="min-w-0">
            <p className="flex items-center gap-3">
              <span aria-hidden="true" className="h-px w-8 bg-ns-accent" />
              <span className="ns-label text-ns-text-muted">Interpretability workbench · research preview</span>
            </p>
            <h1
              id="hero-title"
              className="mt-7 text-[44px] font-medium uppercase leading-none tracking-[0.04em] text-ns-text sm:text-[60px] lg:text-[clamp(52px,4.7vw,68px)]"
            >
              NEURALSCOPE
            </h1>
            <p className="mt-6 text-[24px] font-light leading-tight text-ns-text sm:text-[30px]">
              Inspect the model. Audit its claims.
            </p>
            <p className="mt-5 max-w-[46ch] text-[16px] leading-relaxed text-ns-text-muted sm:text-[17px]">
              An experimental workbench for tracing AI responses, testing self-reports, and comparing explanations against measured evidence.
            </p>
            <div className="mt-9 flex flex-col gap-3 sm:flex-row sm:flex-wrap">
              <ButtonLink href="/app" variant="primary" size="lg" caps>
                Open Neural Debugger
                <ArrowRight aria-hidden="true" />
              </ButtonLink>
              <ButtonLink href="/app/selflab" variant="secondary" size="lg" caps>
                Open SelfLab
              </ButtonLink>
              <ButtonLink href="/research" variant="ghost" size="lg" caps>
                View Research
              </ButtonLink>
            </div>
            <dl className="mt-12 grid max-w-md grid-cols-[76px_minmax(0,1fr)] gap-x-4 gap-y-2.5 border-t border-ns-border pt-5 font-mono text-[11.5px]">
              {SPECS.map((s) => (
                <div key={s.k} className="contents">
                  <dt className="uppercase tracking-[0.12em] text-ns-text-faint">{s.k}</dt>
                  <dd className="text-ns-text-muted">{s.v}</dd>
                </div>
              ))}
            </dl>
          </div>

          <HeroTrace snapshot={heroTrace} />
        </Container>
      </section>

      {/* ============================================================= PREMISE */}
      <Section
        id="premise"
        index="01"
        kicker="Premise"
        title={
          <>
            Training discovers algorithms. <span className="text-ns-text-muted">NeuralScope helps us recover them.</span>
          </>
        }
        lede={
          <>
            <p>Modern neural networks learn internal computations that were never explicitly programmed.</p>
            <p>
              NeuralScope exposes those computations as experimentally accessible objects: activations, representations,
              attention pathways, candidate circuits and causal interventions.
            </p>
          </>
        }
      >
        <PremiseDiagram />
        <div className="mt-16 lg:grid lg:grid-cols-12 lg:gap-x-8">
          <div className="grid gap-6 border-l-2 border-ns-accent-2/70 pl-6 sm:pl-8 lg:col-span-9 lg:col-start-4">
            <p className="ns-label text-ns-accent-2">Mission</p>
            <p className="text-balance text-[28px] font-light leading-[1.15] text-ns-text sm:text-[40px]">
              Recover the algorithms that learning discovered.
            </p>
          </div>
        </div>
      </Section>

      {/* ============================================================= PILLARS */}
      <Section
        id="method"
        index="02"
        kicker="Method"
        title={
          <>
            Observe. Decode. Intervene. <span className="text-ns-text-muted">Verify.</span>
          </>
        }
        lede={
          <p>
            Four operations, one loop. Every measurement NeuralScope shows can be traced to a captured tensor, and every
            mechanistic claim can be sent back to the model as an intervention.
          </p>
        }
      >
        <Pillars />
      </Section>

      {/* ========================================================== EPISTEMICS */}
      <Section
        id="epistemics"
        index="03"
        kicker="Epistemics"
        title="From correlation to causation"
        lede={
          <>
            <p>
              Every view in NeuralScope carries one of four labels. Observed and derived quantities are facts about tensors.
              Inferred claims are hypotheses about meaning. Only an intervention promotes a hypothesis to causally tested.
            </p>
            <p>
              LLM-generated descriptions are never shown as ground truth. At most they are labelled{" "}
              <EpistemicBadge status="inferred" size="xs" /> and wait for an experiment.
            </p>
          </>
        }
      >
        <EpistemicLadder />
        <div className="mt-16 grid grid-cols-1 gap-8 lg:grid-cols-12 lg:gap-8">
          <div className="min-w-0 lg:col-span-4">
            <p className="ns-label text-ns-text">In the debugger</p>
            <p className="mt-4 max-w-[40ch] text-[15px] leading-relaxed text-ns-text-muted">
              Labels travel with the data: a panel header, a table row, a tooltip. You always know whether you are looking at
              a measurement, a transformation of one, a guess, or a result.
            </p>
          </div>
          <div className="min-w-0 lg:col-span-8">
            <LabelledViewExample />
          </div>
        </div>
      </Section>

      {/* ============================================================ MATH LAB */}
      <Section
        id="math-lab"
        index="04"
        kicker="Math Lab"
        title="A microscope for mathematical reasoning"
        lede={
          <>
            <p>NeuralScope Math Lab aligns mathematical correctness with neural computation.</p>
            <p>
              Compare successful and failed reasoning trajectories, locate where their internal states diverge, intervene on
              those states and test whether failed computation can be rescued.
            </p>
          </>
        }
      >
        <MathLabPipeline />
        <div className="mt-12">
          <MathLabMethod />
        </div>
        <div className="mt-12">
          <ProofFuture />
        </div>
      </Section>

      {/* ============================================================= SELFLAB */}
      <Section
        id="selflab"
        index="05"
        kicker="SelfLab"
        title={
          <>
            Audit what a model says about its own answer. <span className="text-ns-text-muted">Against what it measurably did.</span>
          </>
        }
        lede={
          <>
            <p>
              Models explain themselves: &ldquo;I used the document&rdquo;, &ldquo;I checked the arithmetic&rdquo;. SelfLab treats such a sentence as a
              claim to test, not as evidence.
            </p>
            <p>
              Select the claim in a response, commit a pre-registered experiment, rerun the target with its context changed, and compare the claim with the
              recorded outcome. SelfLab is a research prototype; every template states its own status.
            </p>
          </>
        }
      >
        <SelfLabFlow />
        <div className="mt-12 grid grid-cols-1 gap-8 lg:grid-cols-12">
          <div className="min-w-0 lg:col-span-4">
            <p className="ns-label text-ns-text">Recorded example</p>
            <p className="mt-4 max-w-[40ch] text-[15px] leading-relaxed text-ns-text-muted">
              Shown only from an exported run record, with its model, hashes and scope. No illustrative numbers.
            </p>
          </div>
          <div className="min-w-0 lg:col-span-8">
            <SelfLabExample example={SELFLAB_EXAMPLE} />
          </div>
        </div>
        <div className="mt-12 border-t border-ns-border pt-8">
          <p className="ns-label mb-5 text-ns-text">What a SelfLab result does not claim</p>
          <SelfLabNotClaimed />
        </div>
      </Section>

      {/* ============================================================== STATUS */}
      <Section
        id="status"
        index="06"
        kicker="Status"
        title="What runs today, and what does not"
        lede={
          <p>
            Every capability carries a marking: available, research prototype, fixture, API-only view, unsupported or roadmap. The same markings appear
            inside the workspace.
          </p>
        }
      >
        <FeatureStatusList />
      </Section>

      {/* ============================================================= ROADMAP */}
      <Section
        id="roadmap"
        index="07"
        kicker="Roadmap"
        title="The roadmap"
        lede={
          <p>
            From a neural debugger to a circuit compiler. V1 is being built now; everything after it is a plan, labelled as
            such.
          </p>
        }
      >
        <Roadmap />
      </Section>

      {/* ================================================================= CTA */}
      <section aria-labelledby="cta-title" className="relative overflow-hidden border-t border-ns-border">
        <div
          aria-hidden="true"
          className="ns-grid-bg pointer-events-none absolute inset-0 [mask-image:radial-gradient(ellipse_60%_80%_at_50%_100%,black,transparent_75%)]"
        />
        <ReticleBackdrop className="absolute left-1/2 top-1/2 size-[640px] -translate-x-1/2 -translate-y-1/2 text-ns-border sm:size-[760px]" />
        <Container className="relative flex flex-col items-center py-28 text-center sm:py-36">
          <p className="flex flex-wrap justify-center gap-1.5" aria-label="Epistemic statuses">
            {EPISTEMIC_ORDER.map((s) => (
              <EpistemicBadge key={s} status={s} size="xs" />
            ))}
          </p>
          <h2
            id="cta-title"
            className="mt-8 text-[40px] font-normal uppercase leading-none tracking-[0.03em] text-ns-text sm:text-[64px] lg:text-[80px]"
          >
            Open the black box.
          </h2>
          <p className="mt-6 max-w-[48ch] text-[16px] leading-relaxed text-ns-text-muted">
            Capture a forward pass, decode it, cut a component, and measure what changed.
          </p>
          <div className="mt-10 flex flex-col items-center gap-4 sm:flex-row">
            <ButtonLink href="/app" variant="primary" size="lg" caps>
              Launch NeuralScope
              <ArrowRight aria-hidden="true" />
            </ButtonLink>
            <ButtonLink href="/app/selflab" variant="secondary" size="lg" caps>
              Open SelfLab
            </ButtonLink>
            <ButtonLink href="/docs" variant="ghost" size="lg" caps>
              Read the docs
              <ArrowUpRight aria-hidden="true" />
            </ButtonLink>
          </div>
        </Container>
      </section>
    </>
  );
}
