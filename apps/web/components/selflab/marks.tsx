import type { ReactNode } from "react";
import { AlertTriangle } from "lucide-react";
import { StatusChip, type StatusChipTone } from "@/components/ui/StatusChip";
import type {
  Alternative,
  EvidenceKind,
  FeatureStatus,
  ParticipantRole,
  SelfLabJobStatus,
  TestStatus,
  TrialStatus,
} from "@/lib/api/selflab";
import { cn } from "@/lib/cn";

/* ------------------------------------------------------------------ feature status */

export const FEATURE_LABEL: Record<FeatureStatus, string> = {
  implemented: "Available",
  research_prototype: "Research prototype",
  fixture: "Fixture",
  api_only_view: "API-only view",
  unsupported: "Unsupported",
  roadmap: "Roadmap",
};

const FEATURE_TONE: Record<FeatureStatus, StatusChipTone> = {
  implemented: "ok",
  research_prototype: "paper",
  fixture: "warn",
  api_only_view: "neutral",
  unsupported: "danger",
  roadmap: "future",
};

const FEATURE_TITLE: Record<FeatureStatus, string> = {
  implemented: "Implemented and covered by tests.",
  research_prototype: "Research prototype: runs end to end, but methods and outputs may change; read the limitations.",
  fixture: "Fixture data (demo model or synthetic): carries no scientific meaning.",
  api_only_view: "API-only view: no activations; only black-box comparisons are possible.",
  unsupported: "Not supported in this build.",
  roadmap: "Planned, not built.",
};

export function FeatureBadge({ status, className }: { status: FeatureStatus | string | null | undefined; className?: string }) {
  const s = (status && status in FEATURE_LABEL ? status : null) as FeatureStatus | null;
  if (!s) {
    return (
      <StatusChip tone="neutral" className={className} title="The backend did not report a feature status.">
        status unknown
      </StatusChip>
    );
  }
  return (
    <StatusChip tone={FEATURE_TONE[s]} className={className} title={FEATURE_TITLE[s]} data-feature-status={s}>
      {FEATURE_LABEL[s]}
    </StatusChip>
  );
}

/** Loud marker for demo-model / fixture data. Rendered wherever `is_fixture` / `is_demo` is true. */
export function FixtureMark({ show, className, what }: { show: boolean | null | undefined; className?: string; what?: string }) {
  if (!show) return null;
  return (
    <StatusChip
      tone="warn"
      icon={<AlertTriangle aria-hidden="true" />}
      className={className}
      data-testid="fixture-mark"
      title={`${what ?? "This record"} comes from the DEMO fixture model: outputs carry no scientific meaning.`}
    >
      Demo / fixture
    </StatusChip>
  );
}

/* ------------------------------------------------------------------ evidence */

export const KIND_LABEL: Record<EvidenceKind, string> = {
  direct_observation: "Direct observation",
  deterministic_derivation: "Deterministic derivation",
  model_report: "Model report",
  interpretive_hypothesis: "Interpretive hypothesis",
};

const KIND_TITLE: Record<EvidenceKind, string> = {
  direct_observation: "Recorded directly from the target (outputs, tokens, receipts).",
  deterministic_derivation: "Computed by versioned analysis code from recorded measurements.",
  model_report: "What a model said about itself. A report, not a measurement of its internals.",
  interpretive_hypothesis: "An interpretation awaiting a test.",
};

// Kind uses the matching epistemic hue only where the meaning coincides (observed / derived / inferred);
// a model report gets a neutral paper tone so it is never mistaken for a measurement.
const KIND_CLASS: Record<EvidenceKind, string> = {
  direct_observation: "border-ns-observed/35 text-ns-observed bg-ns-observed/8",
  deterministic_derivation: "border-ns-derived/35 text-ns-derived bg-ns-derived/8",
  model_report: "border-ns-accent-2/35 text-ns-accent-2 bg-ns-accent-2/8",
  interpretive_hypothesis: "border-dashed border-ns-inferred/50 text-ns-inferred bg-transparent",
};

export function KindChip({ kind, className }: { kind: EvidenceKind | string; className?: string }) {
  const k = (kind in KIND_LABEL ? kind : null) as EvidenceKind | null;
  return (
    <span
      data-kind={kind}
      title={k ? KIND_TITLE[k] : undefined}
      className={cn(
        "inline-flex h-5 shrink-0 items-center whitespace-nowrap rounded-[3px] border px-1.5 font-mono text-[10px] font-medium uppercase leading-none tracking-[0.1em]",
        k ? KIND_CLASS[k] : "border-ns-border-strong text-ns-text-muted",
        className,
      )}
    >
      {k ? KIND_LABEL[k] : String(kind)}
    </span>
  );
}

export const STATUS_LABEL: Record<TestStatus, string> = {
  untested: "Untested",
  supported: "Supported",
  contradicted: "Contradicted",
  inconclusive: "Inconclusive",
  control_failed: "Control failed",
};

const STATUS_TONE: Record<TestStatus, StatusChipTone> = {
  untested: "future",
  supported: "ok",
  contradicted: "danger",
  inconclusive: "neutral",
  control_failed: "warn",
};

const STATUS_TITLE: Record<TestStatus, string> = {
  untested: "No test has been run.",
  supported: "The pre-registered test came out in the predicted direction, within the stated scope only.",
  contradicted: "The pre-registered test came out against the prediction, within the stated scope.",
  inconclusive: "The test ran but could not decide; see limitations for the reason.",
  control_failed: "A control (replay, delivery receipt, parse) failed, so the test cannot be read.",
};

export function TestStatusChip({ status, className }: { status: TestStatus | string; className?: string }) {
  const s = (status in STATUS_LABEL ? status : null) as TestStatus | null;
  return (
    <StatusChip tone={s ? STATUS_TONE[s] : "neutral"} title={s ? STATUS_TITLE[s] : undefined} className={className} data-test-status={status}>
      {s ? STATUS_LABEL[s] : String(status)}
    </StatusChip>
  );
}

const ALT_TONE: Record<Alternative["state"], StatusChipTone> = { ruled_out: "ok", not_ruled_out: "warn", untested: "future" };
const ALT_LABEL: Record<Alternative["state"], string> = { ruled_out: "Ruled out", not_ruled_out: "Not ruled out", untested: "Untested" };

export function AlternativeStateChip({ state }: { state: Alternative["state"] | string }) {
  const s = (state in ALT_LABEL ? state : null) as Alternative["state"] | null;
  return <StatusChip tone={s ? ALT_TONE[s] : "neutral"}>{s ? ALT_LABEL[s] : String(state)}</StatusChip>;
}

/* ------------------------------------------------------------------ trials / jobs */

const TRIAL_TONE: Record<TrialStatus, StatusChipTone> = {
  ok: "ok",
  malformed: "warn",
  refusal: "neutral",
  abstention: "neutral",
  truncated: "warn",
  error: "danger",
};

export function TrialStatusChip({ status }: { status: TrialStatus | string }) {
  return <StatusChip tone={TRIAL_TONE[status as TrialStatus] ?? "neutral"}>{status}</StatusChip>;
}

const JOB_TONE: Record<SelfLabJobStatus, StatusChipTone> = {
  committed: "paper",
  queued: "accent",
  running: "accent",
  cancelling: "warn",
  cancelled: "neutral",
  done: "ok",
  error: "danger",
};

export function JobStatusChip({ status }: { status: SelfLabJobStatus | string }) {
  const active = status === "queued" || status === "running" || status === "cancelling";
  return (
    <StatusChip tone={JOB_TONE[status as SelfLabJobStatus] ?? "neutral"} dot={active ? "pulse" : true}>
      {status}
    </StatusChip>
  );
}

/* ------------------------------------------------------------------ participants */

export const ROLE_LABEL: Record<ParticipantRole, string> = {
  target: "Target",
  investigator: "Investigator",
  activation_decoder: "Activation decoder",
  statistical_scorer: "Statistical scorer",
  observer: "Observer baseline",
};

export const ROLE_MEANING: Record<ParticipantRole, string> = {
  target: "The model whose claim is audited. White-box: loaded locally, so its activations can be recorded.",
  investigator: "Proposes experiments. Its text is a hypothesis, never evidence.",
  activation_decoder: "An external reader of activations (lens, probe). Its output is an instrument reading about the target, not evidence that the target introspects.",
  statistical_scorer: "Deterministic, versioned analysis code. It turns stored trials into numbers.",
  observer: "A baseline that sees only the prompt or the transcript, used to ask whether a self-report needs privileged access.",
};

/* ------------------------------------------------------------------ small helpers */

/** Short hash with the full value in the title and copy-friendly mono text. */
export function Hash({ value, label, className }: { value: string | null | undefined; label?: string; className?: string }) {
  if (!value) return <span className="font-mono text-ns-text-faint">—</span>;
  return (
    <span className={cn("font-mono text-ns-text", className)} title={`${label ? `${label}: ` : ""}${value}`}>
      {value.length > 16 ? `${value.slice(0, 12)}…${value.slice(-4)}` : value}
    </span>
  );
}

/** A labelled field line used in cards: small caps label + content. */
export function FieldRow({ label, children, className }: { label: ReactNode; children: ReactNode; className?: string }) {
  return (
    <div className={cn("grid grid-cols-[92px_minmax(0,1fr)] items-baseline gap-2 py-0.5", className)}>
      <span className="ns-label text-[9px]">{label}</span>
      <div className="min-w-0 text-[11.5px] leading-snug text-ns-text">{children}</div>
    </div>
  );
}
