"use client";

import { Ban, GitCompare, Lock, Play, RotateCw, Square, TestTube } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Panel } from "@/components/ui/Panel";
import { StatusChip } from "@/components/ui/StatusChip";
import { Caveat, EmptyState, ErrorBox, Field, KV, Loading, NumberInput, ProgressBar, SelectInput } from "@/components/workspace/ui";
import { useNow } from "@/components/workspace/useResource";
import {
  isActiveJob,
  type Case,
  type ExperimentRecord,
  type ExperimentSpec,
  type FeatureStatus,
  type Split,
  type TemplateInfo,
} from "@/lib/api/selflab";
import { cn } from "@/lib/cn";
import { fmtAgo, fmtPct } from "@/lib/format";
import { CompareContexts } from "./CompareContexts";
import { HypothesisCard } from "./EvidenceCard";
import { ParamsEditor } from "./ParamsEditor";
import { FeatureBadge, FixtureMark, Hash, JobStatusChip } from "./marks";
import { useSelfLab } from "./SelfLabProvider";
import { showBlank } from "./spanSelection";
import { SuggestPanel } from "./SuggestPanel";
import { ValueTable } from "./values";

export const CASE_TEMPLATE = "source_influence_case";
export const PREDICTION_TEMPLATE = "csp1_carry";

const NOT_RUNNABLE: FeatureStatus[] = ["unsupported", "roadmap"];

/** Whether a template can be committed in the current context, with the reason when it cannot. */
export function templateAvailability(t: TemplateInfo, c: Case | null | undefined): { ok: boolean; reason?: string } {
  if (NOT_RUNNABLE.includes(t.feature_status)) return { ok: false, reason: `marked ${t.feature_status} by the backend` };
  if (t.requires_case && !c) return { ok: false, reason: "needs a case: select a claim and press INVESTIGATE CLAIM first" };
  if (c && t.access_levels && t.access_levels.length && !t.access_levels.includes(c.access_level))
    return { ok: false, reason: `not available for ${c.access_level === "api_only" ? "API-only" : "white-box"} cases` };
  if (t.id === CASE_TEMPLATE && c && !(c.segments ?? []).some((s) => s.kind === "document"))
    return { ok: false, reason: "needs a case whose context contains a document segment" };
  if (t.id === CASE_TEMPLATE && c?.access_level === "api_only")
    return { ok: false, reason: "replays the target locally, which an API-only case cannot do" };
  if (!t.default_spec && t.default_spec_error) return { ok: false, reason: `the backend gave no default spec (${t.default_spec_error})` };
  return { ok: true };
}

/** TEST PREDICTION availability: only when the csp1_carry template is listed and runnable. */
export function predictionAvailability(templates: TemplateInfo[] | undefined): { ok: boolean; reason: string; template?: TemplateInfo } {
  if (!templates) return { ok: false, reason: "Loading templates…" };
  const t = templates.find((x) => x.id === PREDICTION_TEMPLATE);
  if (!t) return { ok: false, reason: "The csp1_carry template is not listed by this backend, so there is no prediction test to launch." };
  if (NOT_RUNNABLE.includes(t.feature_status)) return { ok: false, reason: `csp1_carry is marked ${t.feature_status}.`, template: t };
  return { ok: true, reason: "Opens the csp1_carry template (self-prediction vs measured carry behaviour) as a draft to commit.", template: t };
}

export function PlanColumn() {
  return (
    <div className="flex flex-col gap-2">
      <ClaimHeader />
      <ActionsBar />
      <SuggestPanel />
      <TemplatePicker />
      <SpecPanel />
      <ExperimentList />
      <CompareContexts />
    </div>
  );
}

/* ------------------------------------------------------------------ claim header */

function ClaimHeader() {
  const { caseId, caseDetail } = useSelfLab();
  const detail = caseDetail.data?.case.id === caseId ? caseDetail.data : undefined;
  if (!caseId) {
    return (
      <Panel label="Investigation" className="ns-dot-bg">
        <EmptyState title="No claim under investigation">
          1. Send a question to the target (left). 2. Select the span of the answer you want to audit and press INVESTIGATE CLAIM. 3. Pick a
          template here, COMMIT its pre-registered spec, then RUN it. Prediction templates that need no case can be opened with TEST PREDICTION.
        </EmptyState>
      </Panel>
    );
  }
  if (caseDetail.error && !detail) return <ErrorBox error={caseDetail.error} onRetry={caseDetail.reload} />;
  if (!detail) return <Loading label="Loading case" />;
  const c = detail.case;
  return (
    <Panel
      label="Claim under audit"
      right={
        <>
          <FixtureMark show={c.is_fixture} what="This case" />
          <StatusChip tone={c.access_level === "api_only" ? "neutral" : "accent"}>{c.access_level === "api_only" ? "API-only view" : "white-box"}</StatusChip>
        </>
      }
      bodyClassName="p-2.5"
    >
      <blockquote className="whitespace-pre-wrap break-words border-l-2 border-ns-accent-2/70 pl-2.5 font-mono text-[13px] leading-relaxed text-ns-text">
        “{showBlank(c.claim.text)}”
      </blockquote>
      {!c.claim.text.trim() ? <p className="mt-1 text-[10.5px] text-ns-warn">The claim span is whitespace only (shown as ↵ / ·).</p> : null}
      {c.notes ? <p className="mt-1 text-[10.5px] text-ns-text-faint">{c.notes}</p> : null}
      <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10px] text-ns-text-muted">
        <span>{c.title}</span>
        <span>type {c.claim_type.replace(/_/g, " ")}</span>
        <span>
          chars [{c.claim.char_start}, {c.claim.char_end})
        </span>
        {c.claim.token_start !== null && c.claim.token_start !== undefined ? (
          <span>
            tokens [{c.claim.token_start}, {c.claim.token_end})
          </span>
        ) : null}
        <span>case {c.id}</span>
      </div>
      <Caveat className="mt-1.5">This is what the target said. Experiments below test it; the claim itself is not evidence.</Caveat>
    </Panel>
  );
}

/* ------------------------------------------------------------------ actions */

function ActionsBar() {
  const { caseId, caseDetail, templates, selectTemplate, experiments, selectExperiment } = useSelfLab();
  const detail = caseDetail.data?.case.id === caseId ? caseDetail.data : undefined;
  const caseTpl = templates.data?.find((t) => t.id === CASE_TEMPLATE);
  const caseAvail = caseTpl
    ? templateAvailability(caseTpl, detail?.case)
    : { ok: false, reason: templates.data ? "source_influence_case is not listed by this backend" : "loading templates" };
  const pred = predictionAvailability(templates.data);
  const existing = (experiments.data ?? []).find((e) => e.committed.spec.template === CASE_TEMPLATE && e.case_id === caseId);

  return (
    <div className="flex flex-wrap items-start gap-2 rounded-[var(--ns-radius)] border border-ns-border bg-ns-panel px-2.5 py-2">
      <div className="flex min-w-[220px] flex-1 flex-col gap-1">
        <Button
          variant="secondary"
          caps
          size="sm"
          disabled={!caseAvail.ok}
          onClick={() => (existing ? selectExperiment(existing.id) : selectTemplate(CASE_TEMPLATE))}
          aria-describedby="compare-contexts-help"
        >
          <GitCompare aria-hidden="true" /> Compare contexts
        </Button>
        <p id="compare-contexts-help" className="text-[10.5px] leading-snug text-ns-text-faint">
          {caseAvail.ok
            ? existing
              ? `Opens ${existing.id}: original vs document removed / replaced / hint added.`
              : "Drafts the case-local counterfactual test: original vs document removed / replaced / hint added."
            : `Unavailable: ${caseAvail.reason}.`}
        </p>
      </div>
      <div className="flex min-w-[220px] flex-1 flex-col gap-1">
        <Button variant="secondary" caps size="sm" disabled={!pred.ok} onClick={() => selectTemplate(PREDICTION_TEMPLATE)} aria-describedby="test-prediction-help">
          <TestTube aria-hidden="true" /> Test prediction
        </Button>
        <p id="test-prediction-help" className="text-[10.5px] leading-snug text-ns-text-faint" data-testid="test-prediction-help">
          {pred.ok ? pred.reason : `Unavailable: ${pred.reason}`}
        </p>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ template picker */

function TemplatePicker() {
  const { templates, templateId, selectTemplate, caseId, caseDetail } = useSelfLab();
  const detail = caseDetail.data?.case.id === caseId ? caseDetail.data : undefined;
  return (
    <Panel label="Experiment templates" right={<span className="font-mono text-[10px] text-ns-text-faint">registered templates only</span>} bodyClassName="p-2">
      {templates.error && !templates.data ? <ErrorBox error={templates.error} onRetry={templates.reload} /> : null}
      {!templates.data && !templates.error ? <Loading compact label="Loading templates" /> : null}
      {templates.data && templates.data.length === 0 ? (
        <p className="p-1 text-[11px] text-ns-text-faint">The backend lists no experiment templates.</p>
      ) : null}
      <ul className="grid gap-1.5 md:grid-cols-2" aria-label="Templates">
        {(templates.data ?? []).map((t) => {
          const av = templateAvailability(t, detail?.case);
          const on = t.id === templateId;
          return (
            <li key={t.id}>
              <button
                type="button"
                onClick={() => selectTemplate(t.id)}
                disabled={!av.ok}
                aria-pressed={on}
                className={cn(
                  "flex h-full w-full flex-col gap-1 rounded-[3px] border px-2 py-1.5 text-left transition-colors disabled:cursor-not-allowed",
                  on ? "border-ns-accent bg-ns-accent/8" : "border-ns-border bg-ns-bg hover:border-ns-border-strong",
                  !av.ok && "opacity-60",
                )}
              >
                <span className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[12px] text-ns-text">{t.title}</span>
                  <FeatureBadge status={t.feature_status} />
                  {t.requires_case ? <StatusChip tone="neutral">needs case</StatusChip> : null}
                </span>
                <span className="font-mono text-[10px] text-ns-text-faint">{t.id}</span>
                <span className="text-[11px] leading-snug text-ns-text-muted">{t.description}</span>
                {!av.ok ? <span className="text-[10.5px] text-ns-warn">Unavailable: {av.reason}</span> : null}
              </button>
            </li>
          );
        })}
      </ul>
    </Panel>
  );
}

/* ------------------------------------------------------------------ spec (draft / committed) + run controls */

function SpecFields({
  spec,
  editable,
  onEdit,
  schema,
  editorKey,
}: {
  spec: ExperimentSpec;
  editable: boolean;
  onEdit?: (s: ExperimentSpec) => void;
  schema?: Record<string, unknown> | null;
  /** Remounts the params editor for a fresh draft so per-field text state never leaks between drafts. */
  editorKey?: string | number;
}) {
  const budget = spec.budget ?? {};
  return (
    <div className="flex flex-col gap-2">
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 md:grid-cols-4">
        <KV k="template" v={spec.template} />
        <KV k="version" v={spec.version ?? "—"} />
        {editable && onEdit ? (
          <>
            <Field label="seed">
              <NumberInput aria-label="Seed" value={spec.seed ?? 0} min={0} onValue={(v) => onEdit({ ...spec, seed: Math.max(0, Math.round(v)) })} />
            </Field>
            <Field label="split">
              <SelectInput aria-label="Split" value={spec.split ?? "debug_pilot"} onChange={(e) => onEdit({ ...spec, split: e.target.value as Split })}>
                <option value="debug_pilot">debug_pilot</option>
                <option value="discovery">discovery</option>
                <option value="heldout_confirmation">heldout_confirmation</option>
              </SelectInput>
            </Field>
          </>
        ) : (
          <>
            <KV k="seed" v={spec.seed ?? 0} />
            <KV k="split" v={spec.split ?? "debug_pilot"} />
          </>
        )}
      </div>
      <div>
        <h4 className="ns-label mb-1 text-[9.5px]">Pre-registered hypotheses</h4>
        {(spec.hypotheses ?? []).length === 0 ? (
          <p className="text-[11px] text-ns-text-faint">This spec registers no hypotheses; the run is descriptive only.</p>
        ) : (
          <div className="flex flex-col gap-1">
            {(spec.hypotheses ?? []).map((h) => (
              <HypothesisCard key={h.id} h={h} compact />
            ))}
          </div>
        )}
      </div>
      <div className="grid gap-x-4 gap-y-1 md:grid-cols-2">
        <KV k="primary metric" v={spec.primary_metric || "—"} />
        <KV k="case" v={spec.case_id ?? "none"} />
      </div>
      <div className="grid gap-2 md:grid-cols-2">
        <div>
          <h4 className="ns-label mb-1 text-[9.5px]">Controls</h4>
          {(spec.controls ?? []).length ? (
            <ul className="list-disc pl-4 text-[11px] leading-snug text-ns-text-muted">
              {(spec.controls ?? []).map((c) => (
                <li key={c}>{c}</li>
              ))}
            </ul>
          ) : (
            <p className="text-[11px] text-ns-text-faint">none listed</p>
          )}
        </div>
        <div>
          <h4 className="ns-label mb-1 text-[9.5px]">Exclusions</h4>
          {(spec.exclusions ?? []).length ? (
            <ul className="list-disc pl-4 text-[11px] leading-snug text-ns-text-muted">
              {(spec.exclusions ?? []).map((c) => (
                <li key={c}>{c}</li>
              ))}
            </ul>
          ) : (
            <p className="text-[11px] text-ns-text-faint">none listed</p>
          )}
        </div>
      </div>
      <div>
        <h4 className="ns-label mb-1 text-[9.5px]">Budget</h4>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 md:grid-cols-4">
          {editable && onEdit ? (
            <Field label="max trials">
              <NumberInput
                aria-label="Max trials"
                value={budget.max_trials ?? 400}
                min={1}
                max={5000}
                onValue={(v) => onEdit({ ...spec, budget: { ...budget, max_trials: Math.min(5000, Math.max(1, Math.round(v))) } })}
              />
            </Field>
          ) : (
            <KV k="max trials" v={budget.max_trials ?? "—"} />
          )}
          <KV k="max new tokens" v={budget.max_new_tokens ?? "—"} />
          <KV k="max seconds" v={budget.max_seconds ?? "—"} />
          <KV k="max prompt tokens" v={budget.max_prompt_tokens ?? "—"} />
        </div>
      </div>
      {editable && onEdit ? (
        <div>
          <h4 className="ns-label mb-1 text-[9.5px]">Template params</h4>
          <ParamsEditor key={editorKey} schema={schema} params={spec.params ?? {}} onChange={(params) => onEdit({ ...spec, params })} />
        </div>
      ) : spec.params && Object.keys(spec.params).length > 0 ? (
        <div>
          <h4 className="ns-label mb-1 text-[9.5px]">Template params</h4>
          <ValueTable data={spec.params} />
        </div>
      ) : null}
    </div>
  );
}

function SpecPanel() {
  const { templateId, templates, draft, setDraft, draftKey, draftPrefill, commit, busy, record, experimentId, run, cancel, resume, caseId, caseDetail } =
    useSelfLab();
  const detail = caseDetail.data?.case.id === caseId ? caseDetail.data : undefined;
  const now = useNow(1000, Boolean(record && isActiveJob(record.status)));

  if (experimentId) {
    if (!record) return <Loading label={`Loading experiment ${experimentId}`} />;
    return <CommittedView rec={record} busy={busy} now={now} onRun={run} onCancel={cancel} onResume={resume} />;
  }
  if (!templateId) return null;
  const t = templates.data?.find((x) => x.id === templateId);
  if (!t) return null;
  const av = templateAvailability(t, detail?.case);
  return (
    <Panel
      label="Spec · draft"
      right={
        <>
          <FeatureBadge status={t.feature_status} />
          <StatusChip tone="future">not committed</StatusChip>
        </>
      }
      bodyClassName="flex flex-col gap-2.5 p-2.5"
    >
      {!draft ? (
        <p className="text-[11px] text-ns-text-faint">
          The backend did not return a default spec for {t.id}
          {t.default_spec_error ? `: ${t.default_spec_error}` : t.requires_case ? " (it needs a case)" : ""}. Nothing can be committed.
        </p>
      ) : (
        <>
          {draftPrefill.length ? (
            <p role="note" className="rounded-[3px] border border-ns-accent-2/40 px-2 py-1 text-[10.5px] leading-snug text-ns-accent-2" data-testid="prefill-note">
              Pre-filled from the loaded example (inputs, not results): {draftPrefill.join(", ")}. Edit them below before committing.
            </p>
          ) : null}
          <SpecFields spec={draft} editable onEdit={setDraft} schema={t.params_schema} editorKey={`${t.id}:${draftKey}`} />
          {t.id === CASE_TEMPLATE && !draft.params?.replacement_text && (draft.params?.preset_seed ?? null) === null ? (
            <p role="note" className="rounded-[3px] border border-ns-warn/40 bg-ns-warn/8 px-2 py-1 text-[10.5px] leading-snug text-ns-warn" data-testid="replacement-warning">
              Fill <code>replacement_text</code>: the document the &ldquo;document replaced&rdquo; branch uses (and, to test whether it is followed,{" "}
              <code>replacement_value</code> / <code>original_value</code>). Without it the backend refuses the commit unless the case is a corpus preset.
            </p>
          ) : null}
          <Caveat>
            COMMIT fixes the spec and the trial plan by hash before any trial runs. Hypotheses, metrics and controls cannot change afterwards; a
            different choice is a new experiment.
          </Caveat>
          <Button variant="primary" caps onClick={() => void commit()} disabled={!av.ok || busy === "commit"} title={av.ok ? undefined : av.reason}>
            <Lock aria-hidden="true" /> {busy === "commit" ? "Committing…" : "Commit spec"}
          </Button>
          {!av.ok ? <p className="text-[10.5px] text-ns-warn">Unavailable: {av.reason}.</p> : null}
        </>
      )}
    </Panel>
  );
}

function CommittedView({
  rec,
  busy,
  now,
  onRun,
  onCancel,
  onResume,
}: {
  rec: ExperimentRecord;
  busy: string | null;
  now: number;
  onRun: () => void;
  onCancel: () => void;
  onResume: () => void;
}) {
  const c = rec.committed;
  const active = isActiveJob(rec.status);
  const prov = c.provenance;
  return (
    <Panel
      label={`Experiment ${rec.id}`}
      right={
        <>
          <FixtureMark show={rec.is_fixture || prov?.model.is_demo} what="This experiment" />
          <JobStatusChip status={rec.status} />
        </>
      }
      bodyClassName="flex flex-col gap-2.5 p-2.5"
      data-testid="committed-spec"
    >
      <div className="grid gap-x-4 gap-y-1 rounded-[3px] border border-ns-border bg-ns-bg px-2 py-1.5 md:grid-cols-2">
        <KV k="spec_sha256" v={<Hash value={c.spec_sha256} label="spec_sha256" />} />
        <KV k="plan_sha256" v={<Hash value={c.plan_sha256} label="plan_sha256" />} />
        <KV k="committed_at" v={c.committed_at} />
        <KV k="planned trials" v={c.n_planned} />
        {prov ? (
          <>
            <KV k="target" v={prov.model.model_id} />
            <KV k="code" v={prov.code_commit ? `${prov.code_commit.slice(0, 10)}${prov.code_dirty ? " (dirty)" : ""}` : "—"} />
            <KV k="analysis" v={prov.analysis_version} />
            <KV k="device" v={`${prov.model.device} · ${prov.model.dtype}`} />
          </>
        ) : null}
      </div>
      <details>
        <summary className="cursor-pointer font-mono text-[10.5px] uppercase tracking-[0.08em] text-ns-text-muted hover:text-ns-text">committed spec</summary>
        <div className="mt-2">
          <SpecFields spec={c.spec} editable={false} />
        </div>
      </details>
      <div className="flex flex-col gap-1">
        <ProgressBar value={rec.progress} />
        <div className="flex flex-wrap items-center gap-x-3 font-mono text-[10.5px] text-ns-text-muted" aria-live="polite">
          <span>
            {rec.n_done}/{c.n_planned} trials · {fmtPct(rec.progress)}
          </span>
          {rec.n_errors ? <span className="text-ns-danger">{rec.n_errors} errors</span> : null}
          {rec.message ? <span className="truncate">{rec.message}</span> : null}
          {rec.started_at && now ? <span className="text-ns-text-faint">started {fmtAgo(rec.started_at, now)}</span> : null}
        </div>
      </div>
      {rec.error ? <ErrorBox error={rec.error} /> : null}
      <div className="flex flex-wrap gap-2">
        {rec.status === "committed" ? (
          <Button variant="primary" caps onClick={onRun} disabled={busy === "run"}>
            <Play aria-hidden="true" /> {busy === "run" ? "Queuing…" : "Run"}
          </Button>
        ) : null}
        {active ? (
          <Button variant="danger" caps onClick={onCancel} disabled={busy === "cancel" || rec.status === "cancelling"}>
            <Square aria-hidden="true" /> {rec.status === "cancelling" ? "Cancelling…" : "Cancel"}
          </Button>
        ) : null}
        {rec.status === "cancelled" || rec.status === "error" ? (
          <Button variant="secondary" caps onClick={onResume} disabled={busy === "resume"}>
            <RotateCw aria-hidden="true" /> Resume
          </Button>
        ) : null}
      </div>
      {rec.status === "cancelled" || rec.status === "error" ? (
        <Caveat>RESUME re-derives the plan and is refused if plan_sha256 differs. Completed trials are never re-run or re-randomised.</Caveat>
      ) : null}
      {active ? <Caveat>Cancellation is cooperative: the current trial finishes and is stored, then the run stops.</Caveat> : null}
      {rec.cost && Object.keys(rec.cost).length > 0 ? (
        <div>
          <h4 className="ns-label mb-1 text-[9.5px]">Cost</h4>
          <ValueTable data={rec.cost} className="max-w-md" />
        </div>
      ) : null}
    </Panel>
  );
}

/* ------------------------------------------------------------------ experiment list */

function ExperimentList() {
  const { experiments, experimentId, selectExperiment, caseId } = useSelfLab();
  const list = experiments.data ?? [];
  const now = useNow(30000);
  if (list.length === 0) return null;
  return (
    <Panel label={caseId ? "Experiments on this case" : "Recent experiments"} bodyClassName="p-1.5">
      <ol className="flex flex-col gap-1">
        {list.slice(0, 20).map((e) => (
          <li key={e.id}>
            <button
              type="button"
              onClick={() => selectExperiment(e.id)}
              aria-current={e.id === experimentId ? "true" : undefined}
              className={cn(
                "flex w-full flex-wrap items-center gap-x-2 gap-y-0.5 rounded-[3px] border px-2 py-1 text-left",
                e.id === experimentId ? "border-ns-accent bg-ns-accent/8" : "border-ns-border bg-ns-bg hover:border-ns-border-strong",
              )}
            >
              <span className="font-mono text-[11px] text-ns-text">{e.id}</span>
              <span className="font-mono text-[10px] text-ns-text-muted">{e.committed.spec.template}</span>
              <JobStatusChip status={e.status} />
              <FixtureMark show={e.is_fixture} what="This experiment" />
              <span className="ml-auto font-mono text-[10px] text-ns-text-faint">
                {e.n_done}/{e.committed.n_planned} · {now ? fmtAgo(e.created_at, now) : ""}
              </span>
            </button>
          </li>
        ))}
      </ol>
      {experimentId ? (
        <button
          type="button"
          onClick={() => selectExperiment(null)}
          className="mt-1 inline-flex items-center gap-1 px-1 font-mono text-[10px] uppercase tracking-[0.08em] text-ns-text-faint hover:text-ns-text"
        >
          <Ban aria-hidden="true" className="size-3" /> deselect
        </button>
      ) : null}
    </Panel>
  );
}
