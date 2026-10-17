"use client";

import { ExternalLink } from "lucide-react";
import { shortId } from "@/lib/format";
import type { EvidenceItem, Hypothesis } from "@/lib/api/selflab";
import { cn } from "@/lib/cn";
import { AlternativeStateChip, FieldRow, FixtureMark, KindChip, TestStatusChip } from "./marks";
import { ValueTable } from "./values";

/**
 * A pre-registered hypothesis. Drawn hatched and dashed so it can never be mistaken for evidence:
 * it states what a test will check, not what was found.
 */
export function HypothesisCard({ h, compact }: { h: Hypothesis; compact?: boolean }) {
  return (
    <article
      data-testid="hypothesis-card"
      aria-label={`Hypothesis ${h.id}`}
      className={cn("ns-hatch rounded-[3px] border border-dashed border-ns-inferred/45 px-2 py-1.5", compact && "py-1")}
    >
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="font-mono text-[10px] font-medium uppercase tracking-[0.1em] text-ns-inferred">Hypothesis · not evidence</span>
        <span className="font-mono text-[11px] text-ns-text">{h.id}</span>
        {h.primary ? <span className="font-mono text-[9.5px] uppercase tracking-[0.1em] text-ns-text-muted">primary</span> : null}
      </div>
      <p className="mt-0.5 text-[11.5px] leading-snug text-ns-text">{h.statement}</p>
      <p className="mt-0.5 font-mono text-[10px] text-ns-text-muted">
        metric {h.metric}
        {h.direction ? ` · ${h.direction.replace("_", "-")}` : ""}
        {h.threshold !== null && h.threshold !== undefined ? ` · threshold ${h.threshold}` : ""}
      </p>
    </article>
  );
}

export interface EvidenceCardProps {
  item: EvidenceItem;
  /** The case / experiment this item came from is fixture data. */
  fixture?: boolean;
  onSelectTrial?: (trialId: string) => void;
  onOpenRun?: (runId: string) => void;
}

/**
 * One evidence item. KIND, STATUS and SCOPE are rendered as three separate fields; there is no
 * combined verdict badge, and the status is always read together with the scope it holds in.
 */
export function EvidenceCard({ item, fixture, onSelectTrial, onOpenRun }: EvidenceCardProps) {
  const scope = item.scope;
  const trials = item.trial_ids ?? [];
  const runs = item.run_ids ?? [];
  return (
    <article
      data-testid="evidence-card"
      data-evidence-id={item.id}
      aria-label={`Evidence: ${item.title}`}
      className="rounded-[3px] border border-ns-border bg-ns-bg px-2.5 py-2"
    >
      <h4 className="text-[12.5px] font-medium leading-snug text-ns-text">{item.title}</h4>
      <div className="mt-1.5 flex flex-col">
        <FieldRow label="Kind">
          <KindChip kind={item.kind} />
        </FieldRow>
        <FieldRow label="Status">
          <span className="inline-flex flex-wrap items-center gap-1.5">
            <TestStatusChip status={item.status} />
            {fixture ? <FixtureMark show what="This evidence" /> : null}
          </span>
        </FieldRow>
        <FieldRow label="Scope">
          {scope ? (
            <span className="block text-[11px] leading-snug text-ns-text-muted" data-testid="evidence-scope">
              <span className="text-ns-text">
                n = {scope.n_units} {scope.unit}
                {scope.n_units === 1 ? "" : "s"}
              </span>{" "}
              · {scope.n_trials} trials · split {scope.split ?? "—"}
              <br />
              checkpoint <span className="font-mono text-ns-text">{scope.checkpoint}</span>
              <br />
              tasks: {scope.task_distribution}
              <br />
              intervention: {scope.intervention}
              {scope.controls && scope.controls.length ? (
                <>
                  <br />
                  controls: {scope.controls.join("; ")}
                </>
              ) : null}
              <br />
              tested generalisation: {scope.tested_generalization ?? "none"}
            </span>
          ) : (
            <span className="text-[11px] text-ns-warn">no scope recorded: read as unscoped</span>
          )}
        </FieldRow>
      </div>
      <p className="mt-1.5 text-[11.5px] leading-relaxed text-ns-text">{item.statement}</p>
      {item.measurement && Object.keys(item.measurement).length > 0 ? (
        <div className="mt-1.5 rounded-[3px] border border-ns-border px-2 py-1">
          <h5 className="ns-label mb-0.5 text-[9px]">Measurement</h5>
          <ValueTable data={item.measurement} />
        </div>
      ) : null}
      {(item.supports?.length || item.contradicts?.length) ? (
        <p className="mt-1.5 flex flex-wrap gap-x-3 font-mono text-[10.5px] text-ns-text-muted">
          {item.supports?.length ? <span>bears for: {item.supports.join(", ")}</span> : null}
          {item.contradicts?.length ? <span>bears against: {item.contradicts.join(", ")}</span> : null}
        </p>
      ) : null}
      {item.alternatives && item.alternatives.length > 0 ? (
        <div className="mt-1.5">
          <h5 className="ns-label mb-0.5 text-[9px]">Alternative explanations</h5>
          <ul className="flex flex-col gap-1">
            {item.alternatives.map((a, i) => (
              <li key={i} className="rounded-[3px] border border-ns-border/70 px-1.5 py-1 text-[11px] leading-snug">
                <div className="flex items-start gap-1.5">
                  <span className="min-w-0 flex-1 text-ns-text">{a.explanation}</span>
                  <AlternativeStateChip state={a.state} />
                </div>
                <div className="text-[10.5px] text-ns-text-faint">test: {a.discriminating_test}</div>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {item.limitations && item.limitations.length > 0 ? (
        <div className="mt-1.5">
          <h5 className="ns-label mb-0.5 text-[9px]">Limitations</h5>
          <ul className="list-disc pl-4 text-[10.5px] leading-snug text-ns-text-muted">
            {item.limitations.map((l, i) => (
              <li key={i}>{l}</li>
            ))}
          </ul>
        </div>
      ) : null}
      <div className="mt-1.5 flex flex-wrap items-center gap-1 font-mono text-[10px] text-ns-text-faint">
        {item.analysis ? (
          <span className="basis-full break-all" title="analysis code that produced this item">
            {item.analysis}
          </span>
        ) : null}
        {trials.length ? <span className="ml-1">trials:</span> : null}
        {trials.slice(0, 12).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => onSelectTrial?.(t)}
            className="rounded-[2px] border border-ns-border px-1 text-ns-text-muted hover:border-ns-accent hover:text-ns-text"
            title={`Inspect trial ${t}`}
          >
            {t.includes(":") ? `#${t.split(":").pop()}` : t}
          </button>
        ))}
        {trials.length > 12 ? <span>+{trials.length - 12} more</span> : null}
        {runs.map((r) => (
          <button
            key={r}
            type="button"
            onClick={() => onOpenRun?.(r)}
            className="inline-flex items-center gap-0.5 rounded-[2px] border border-ns-observed/30 px-1 text-ns-observed hover:border-ns-observed"
            title={`Open Lab run ${r} in the debugger`}
          >
            <ExternalLink aria-hidden="true" className="size-2.5" /> open run {shortId(r)} in lab
          </button>
        ))}
      </div>
    </article>
  );
}
