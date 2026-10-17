import { ArrowRight } from "lucide-react";
import { ButtonLink } from "@/components/ui/Button";
import { StatusChip, type StatusChipTone } from "@/components/ui/StatusChip";
import type { EvidenceKind, TestStatus } from "@/lib/api/selflab";
import type { SelfLabRecordedExample } from "@/lib/selflab-example";
import { cn } from "@/lib/cn";

const STEPS = [
  {
    n: "1",
    title: "Select a claim",
    body: "Ask the local target a question, with or without a document or a hint. Select the span of its answer you want to audit.",
  },
  {
    n: "2",
    title: "Commit the test",
    body: "Pick a registered template. Its hypotheses, metric, controls and budget are hashed and committed before any trial runs.",
  },
  {
    n: "3",
    title: "Change the context",
    body: "The target is rerun with the document removed, replaced, or a hint added. Every trial is stored, including failures.",
  },
  {
    n: "4",
    title: "Read the evidence",
    body: "Each item keeps its kind, its test status and its scope apart. Open any trial's tensors in the debugger.",
  },
];

const KIND_LABEL: Record<EvidenceKind, string> = {
  direct_observation: "Direct observation",
  deterministic_derivation: "Deterministic derivation",
  model_report: "Model report",
  interpretive_hypothesis: "Interpretive hypothesis",
};
const STATUS_TONE: Record<TestStatus, StatusChipTone> = {
  untested: "future",
  supported: "ok",
  contradicted: "danger",
  inconclusive: "neutral",
  control_failed: "warn",
};

export function SelfLabFlow() {
  return (
    <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {STEPS.map((s) => (
        <li key={s.n} className="flex flex-col rounded-[var(--ns-radius)] border border-ns-border bg-ns-panel p-5">
          <span className="font-mono text-[12px] text-ns-accent">{s.n}</span>
          <h3 className="mt-3 text-[17px] font-medium leading-tight text-ns-text">{s.title}</h3>
          <p className="mt-2 text-[13.5px] leading-snug text-ns-text-muted">{s.body}</p>
        </li>
      ))}
    </ol>
  );
}

const NOT_CLAIMED = [
  "A delivery receipt proves the document reached the model. It does not prove the model used it.",
  "A model's report about its own answer is a report, not a measurement of its internals.",
  "A status holds only within its scope: one case is one case.",
  "No sentience, consciousness or 'true thought' meters. Nothing here measures them.",
];

function PendingRecord() {
  return (
    <div className="ns-hatch rounded-[var(--ns-radius)] border border-dashed border-ns-border-strong p-6" data-testid="selflab-example-pending">
      <StatusChip tone="future">Pilot record pending</StatusChip>
      <p className="mt-4 max-w-[60ch] text-[15px] leading-relaxed text-ns-text">
        No recorded SelfLab result is published here yet.
      </p>
      <p className="mt-2 max-w-[60ch] text-[14px] leading-relaxed text-ns-text-muted">
        When the pilot run is exported, its record appears in this space: model, committed spec hash, results and evidence, copied from the export file.
        Until then there is nothing to show, so nothing is shown. You can run the workflow yourself in the workspace.
      </p>
      <div className="mt-5">
        <ButtonLink href="/app/selflab" variant="secondary" size="md" caps>
          Open SelfLab
          <ArrowRight aria-hidden="true" />
        </ButtonLink>
      </div>
    </div>
  );
}

function RecordedExample({ ex }: { ex: SelfLabRecordedExample }) {
  return (
    <figure className="rounded-[var(--ns-radius)] border border-ns-border bg-ns-panel" data-testid="selflab-example">
      <div className="flex flex-wrap items-center gap-2 border-b border-ns-border px-5 py-3">
        <StatusChip tone="paper" dot>
          Recorded run
        </StatusChip>
        {ex.is_demo ? <StatusChip tone="warn">Demo / fixture: no scientific meaning</StatusChip> : null}
        <span className="font-mono text-[11px] text-ns-text-muted">
          {ex.model_id} · {ex.template} · {ex.split} · {ex.recorded_at}
        </span>
      </div>
      <div className="grid gap-6 p-5 lg:grid-cols-2">
        <div className="min-w-0">
          {ex.task ? (
            <div className="space-y-2 font-mono text-[12.5px] leading-relaxed">
              {ex.task.document ? (
                <p>
                  <span className="text-ns-text-faint">document </span>
                  <span className="text-ns-text-muted">{ex.task.document}</span>
                </p>
              ) : null}
              {ex.task.hint ? (
                <p>
                  <span className="text-ns-text-faint">hint </span>
                  <span className="text-ns-text-muted">{ex.task.hint}</span>
                </p>
              ) : null}
              <p>
                <span className="text-ns-text-faint">question </span>
                <span className="text-ns-text">{ex.task.question}</span>
              </p>
              <p>
                <span className="text-ns-text-faint">response </span>
                <span className="text-ns-text">{ex.task.response}</span>
              </p>
              {ex.task.self_report ? (
                <p>
                  <span className="text-ns-text-faint">self-report </span>
                  <span className="text-ns-accent-2">{ex.task.self_report}</span>
                </p>
              ) : null}
            </div>
          ) : null}
          {ex.results.length ? (
            <dl className="mt-5 grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1.5 border-t border-ns-border pt-4 font-mono text-[12px]">
              {ex.results.map((r) => (
                <div key={r.label} className="contents">
                  <dt className="text-ns-text-muted">
                    {r.label}
                    {r.detail ? <span className="block text-[11px] text-ns-text-faint">{r.detail}</span> : null}
                  </dt>
                  <dd className="text-right text-ns-text">{r.value}</dd>
                </div>
              ))}
            </dl>
          ) : null}
        </div>
        <div className="min-w-0 space-y-3">
          {ex.evidence.map((e) => (
            <article key={e.title} className="rounded-[var(--ns-radius)] border border-ns-border bg-ns-bg p-3">
              <h4 className="text-[13.5px] text-ns-text">{e.title}</h4>
              <dl className="mt-2 grid grid-cols-[64px_minmax(0,1fr)] gap-x-2 gap-y-1 text-[12px]">
                <dt className="ns-label text-[9.5px]">Kind</dt>
                <dd className="text-ns-text-muted">{KIND_LABEL[e.kind]}</dd>
                <dt className="ns-label text-[9.5px]">Status</dt>
                <dd>
                  <StatusChip tone={STATUS_TONE[e.status]}>{e.status.replace("_", " ")}</StatusChip>
                </dd>
                <dt className="ns-label text-[9.5px]">Scope</dt>
                <dd className="text-ns-text-muted">{e.scope}</dd>
              </dl>
              <p className="mt-2 text-[12.5px] leading-snug text-ns-text">{e.statement}</p>
            </article>
          ))}
          {ex.limitations.length ? (
            <ul className="list-disc space-y-1 pl-5 text-[12.5px] leading-snug text-ns-text-muted">
              {ex.limitations.map((l) => (
                <li key={l}>{l}</li>
              ))}
            </ul>
          ) : null}
        </div>
      </div>
      <figcaption className="border-t border-ns-border px-5 py-3 font-mono text-[10.5px] text-ns-text-faint">
        experiment {ex.experiment_id} · spec {ex.spec_sha256.slice(0, 12)} · plan {ex.plan_sha256.slice(0, 12)}
        {ex.code_commit ? ` · code ${ex.code_commit.slice(0, 10)}` : ""}
        {ex.export_path ? ` · ${ex.export_path}` : ""}
      </figcaption>
    </figure>
  );
}

export function SelfLabExample({ example }: { example: SelfLabRecordedExample | null }) {
  return example && example.source === "recorded" ? <RecordedExample ex={example} /> : <PendingRecord />;
}

export function SelfLabNotClaimed({ className }: { className?: string }) {
  return (
    <ul className={cn("grid gap-x-8 gap-y-3 sm:grid-cols-2", className)}>
      {NOT_CLAIMED.map((t) => (
        <li key={t} className="flex gap-3 text-[14px] leading-snug text-ns-text-muted">
          <span aria-hidden="true" className="mt-[8px] h-px w-3 shrink-0 bg-ns-accent-2" />
          {t}
        </li>
      ))}
    </ul>
  );
}
