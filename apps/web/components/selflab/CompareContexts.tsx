"use client";

import { ExternalLink, Search } from "lucide-react";
import { Panel } from "@/components/ui/Panel";
import { StatusChip } from "@/components/ui/StatusChip";
import { Caveat, ErrorBox, Loading } from "@/components/workspace/ui";
import { useResource } from "@/components/workspace/useResource";
import { selflab, type Trial } from "@/lib/api/selflab";
import { cn } from "@/lib/cn";
import { shortId } from "@/lib/format";
import { FixtureMark, TrialStatusChip } from "./marks";
import { useSelfLab } from "./SelfLabProvider";
import { showBlank } from "./spanSelection";
import { asScore, fmtCell } from "./values";

export const BRANCH_ORDER = ["original", "document_removed", "document_replaced", "hint_added"] as const;

export const BRANCH_LABEL: Record<string, { title: string; meaning: string }> = {
  original: { title: "Original", meaning: "Replay of the case context. Must reproduce the case response, else the control fails." },
  document_removed: { title: "Document removed", meaning: "Same message without the document segment." },
  document_replaced: { title: "Document replaced", meaning: "Document swapped for a controlled replacement." },
  hint_added: { title: "Hint added", meaning: "Original context plus a user suggestion." },
};

/** Group trials into ordered branches (known conditions first, others after, in first-seen order). */
export function groupBranches(trials: Trial[]): { condition: string; trials: Trial[] }[] {
  const map = new Map<string, Trial[]>();
  for (const t of [...trials].sort((a, b) => a.index - b.index)) {
    const list = map.get(t.condition) ?? [];
    list.push(t);
    map.set(t.condition, list);
  }
  const known = BRANCH_ORDER.filter((c) => map.has(c)).map((c) => ({ condition: c as string, trials: map.get(c) as Trial[] }));
  const rest = [...map.entries()].filter(([c]) => !(BRANCH_ORDER as readonly string[]).includes(c)).map(([condition, ts]) => ({ condition, trials: ts }));
  return [...known, ...rest];
}

function scalarEntries(o: Record<string, unknown> | undefined, max = 6): [string, string][] {
  return Object.entries(o ?? {})
    .filter(([, v]) => v === null || typeof v !== "object")
    .slice(0, max)
    .map(([k, v]) => [k, fmtCell(v)]);
}

interface ReportReadout {
  order: string;
  letter: string | null;
  mass: number | null;
  freeText: string | null;
}

/** The constrained source report readouts (letter log-probs and parsed free text) for each label order. */
export function sourceReports(measurements: Record<string, unknown> | undefined): ReportReadout[] {
  const raw = measurements?.source_report;
  if (!Array.isArray(raw)) return [];
  return raw
    .filter((r): r is Record<string, unknown> => typeof r === "object" && r !== null)
    .map((r, i) => {
      const parse = (r.free_text_parse ?? null) as Record<string, unknown> | null;
      return {
        order: Array.isArray(r.label_order) ? `order ${i === 0 ? "A" : "B"}` : `readout ${i + 1}`,
        letter: typeof r.argmax_category === "string" ? r.argmax_category : null,
        mass: typeof r.mass === "number" ? r.mass : null,
        freeText: parse && typeof parse.category === "string" ? parse.category : parse && typeof parse.status === "string" ? parse.status : null,
      };
    });
}

function scoreEntries(o: Record<string, unknown> | undefined): [string, string][] {
  const out: [string, string][] = [];
  for (const [k, v] of Object.entries(o ?? {})) {
    const s = asScore(v);
    if (s) out.push([k, s]);
    else if (v && typeof v === "object" && !Array.isArray(v)) {
      for (const [k2, v2] of Object.entries(v as Record<string, unknown>)) {
        const s2 = asScore(v2);
        if (s2) out.push([`${k}.${k2}`, s2]);
      }
    }
  }
  return out.slice(0, 8);
}

export function BranchColumn({
  condition,
  trials,
  caseResponse,
  onInspect,
  onOpenRun,
}: {
  condition: string;
  trials: Trial[];
  caseResponse?: string;
  onInspect: (id: string) => void;
  onOpenRun: (id: string) => void;
}) {
  const info = BRANCH_LABEL[condition] ?? { title: condition.replace(/_/g, " "), meaning: "" };
  return (
    <div className="flex min-w-[220px] flex-1 flex-col gap-1.5 rounded-[3px] border border-ns-border bg-ns-bg p-2" data-testid={`branch-${condition}`}>
      <div>
        <h4 className="ns-label text-[9.5px] text-ns-text">{info.title}</h4>
        {info.meaning ? <p className="mt-0.5 text-[10.5px] leading-snug text-ns-text-faint">{info.meaning}</p> : null}
      </div>
      {trials.map((t) => {
        const backendReplay = typeof t.measurements?.replay_equal === "boolean" ? (t.measurements.replay_equal as boolean) : null;
        const matches =
          backendReplay === null && caseResponse !== undefined && condition === "original" ? t.output_text === caseResponse : null;
        const reports = sourceReports(t.measurements);
        return (
          <div key={t.id} className="flex flex-col gap-1 border-t border-ns-border pt-1.5">
            <div className="flex flex-wrap items-center gap-1">
              <TrialStatusChip status={t.status} />
              <span className="font-mono text-[10px] text-ns-text-faint">#{t.index}</span>
              <button
                type="button"
                onClick={() => onInspect(t.id)}
                className="ml-auto inline-flex items-center gap-0.5 font-mono text-[10px] text-ns-accent hover:underline"
              >
                <Search aria-hidden="true" className="size-3" /> inspect
              </button>
            </div>
            <pre className="max-h-28 overflow-auto whitespace-pre-wrap break-words font-mono text-[11.5px] leading-snug text-ns-text">
              {t.output_text.trim() ? t.output_text : <span className="text-ns-text-faint">{showBlank(t.output_text) || "∅"} (blank)</span>}
            </pre>
            {backendReplay !== null ? (
              <p className={cn("font-mono text-[10px]", backendReplay ? "text-ns-ok" : "text-ns-warn")} data-testid="replay-check">
                replay {backendReplay ? "reproduces" : "does NOT reproduce"} the case response (backend control)
              </p>
            ) : matches !== null ? (
              <p className={cn("font-mono text-[10px]", matches ? "text-ns-ok" : "text-ns-warn")} title="Exact string comparison done in the browser; no backend replay flag was recorded.">
                {matches ? "identical to the case response" : "differs from the case response"} (exact string compare)
              </p>
            ) : null}
            {scalarEntries(t.ground_truth, 8).length ? (
              <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-2 font-mono text-[10px]" aria-label="ground truth fields">
                {scalarEntries(t.ground_truth, 8).map(([k, v]) => (
                  <div key={k} className="contents">
                    <dt className="text-ns-text-muted">{k}</dt>
                    <dd className="truncate text-right text-ns-text" title={v}>
                      {v}
                    </dd>
                  </div>
                ))}
              </dl>
            ) : null}
            {reports.length ? (
              <div className="font-mono text-[10px]" aria-label="constrained source report">
                <span className="text-ns-text-muted">source report (model report):</span>
                {reports.map((r) => (
                  <div key={r.order} className="flex flex-wrap gap-x-2 text-ns-text">
                    <span className="text-ns-text-faint">{r.order}</span>
                    <span>letter → {r.letter ?? "—"}</span>
                    {r.mass !== null ? <span className="text-ns-text-faint">mass {fmtCell(r.mass)}</span> : null}
                    <span>text → {r.freeText ?? "—"}</span>
                  </div>
                ))}
              </div>
            ) : null}
            {scoreEntries(t.measurements).length ? (
              <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-2 font-mono text-[10px]" aria-label="log-likelihoods">
                {scoreEntries(t.measurements).map(([k, v]) => (
                  <div key={k} className="contents">
                    <dt className="truncate text-ns-text-muted" title={k}>
                      {k}
                    </dt>
                    <dd className="text-right text-ns-text">{v}</dd>
                  </div>
                ))}
              </dl>
            ) : null}
            {(t.run_ids ?? []).length ? (
              <div className="flex flex-wrap gap-1">
                {(t.run_ids ?? []).map((r) => (
                  <button
                    key={r}
                    type="button"
                    onClick={() => onOpenRun(r)}
                    className="inline-flex items-center gap-0.5 rounded-[2px] border border-ns-observed/30 px-1 font-mono text-[10px] text-ns-observed hover:border-ns-observed"
                  >
                    <ExternalLink aria-hidden="true" className="size-2.5" /> run {shortId(r)}
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

export function CompareContexts() {
  const { record, experimentId, dataVersion, caseId, caseDetail, selectTrial, openRunInLab } = useSelfLab();
  const isCaseTemplate = record?.committed.spec.template === "source_influence_case";
  const trials = useResource<Trial[]>(
    isCaseTemplate && experimentId && record && record.n_done > 0 ? `selflab:compare:${experimentId}:${dataVersion}` : null,
    (s) => selflab.trials(experimentId as string, { offset: 0, limit: 200 }, s),
    { cache: false },
  );
  if (!isCaseTemplate || !record) return null;
  const detail = caseDetail.data?.case.id === caseId ? caseDetail.data : undefined;
  if (record.n_done === 0) {
    return (
      <Panel label="Compare contexts">
        <p className="text-[11px] text-ns-text-faint">
          No branch has run yet. RUN the committed experiment; each branch appears here as soon as its trial is stored.
        </p>
      </Panel>
    );
  }
  const branches = groupBranches(trials.data ?? []);
  return (
    <Panel label="Compare contexts" right={<FixtureMark show={record.is_fixture} what="These branches" />} bodyClassName="flex flex-col gap-2 p-2.5">
      {trials.error && !trials.data ? <ErrorBox error={trials.error} onRetry={trials.reload} /> : null}
      {!trials.data && !trials.error ? <Loading compact label="Loading branches" /> : null}
      {trials.data ? (
        <>
          <div className="flex gap-2 overflow-x-auto pb-1">
            {branches.map((b) => (
              <BranchColumn
                key={b.condition}
                condition={b.condition}
                trials={b.trials}
                caseResponse={detail?.case.response_text}
                onInspect={selectTrial}
                onOpenRun={openRunInLab}
              />
            ))}
          </div>
          <Caveat>
            Outputs are greedy and verbatim. Whether the claim is supported, contradicted or inconclusive is decided by the template&apos;s versioned rules
            and appears as evidence on the right, with n = 1 case as its scope.
          </Caveat>
          {record.status !== "done" ? <StatusChip tone="accent" dot="pulse">{record.status}: more branches may follow</StatusChip> : null}
        </>
      ) : null}
    </Panel>
  );
}
