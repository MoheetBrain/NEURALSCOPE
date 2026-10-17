"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp, Download, FileJson } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { StatusChip } from "@/components/ui/StatusChip";
import { Tabs, type TabItem } from "@/components/ui/Tabs";
import { Caveat, ErrorBox, Loading } from "@/components/workspace/ui";
import { useResource } from "@/components/workspace/useResource";
import { errorMessage } from "@/lib/api/client";
import { looksLikeFreeText, selflab, splitAnalysis, type ExperimentRecord, type ExportPreview, type Trial } from "@/lib/api/selflab";
import { cn } from "@/lib/cn";
import { shortId, truncate } from "@/lib/format";
import { FixtureMark, TestStatusChip, TrialStatusChip } from "./marks";
import { useSelfLab } from "./SelfLabProvider";
import { SummaryView, fmtBytes, fmtCell, fmtScalar } from "./values";

type BottomTab = "stats" | "trials" | "export";

/** Tabs that have content for this record — no empty tabs. */
export function availableTabs(rec: ExperimentRecord | null): BottomTab[] {
  if (!rec) return [];
  const out: BottomTab[] = [];
  if (rec.analysis && Object.keys(rec.analysis).length > 0) out.push("stats");
  if (rec.n_done > 0) out.push("trials");
  out.push("export");
  return out;
}

const TAB_LABEL: Record<BottomTab, string> = { stats: "Statistics", trials: "Trials", export: "Export evidence" };

export function BottomPanel() {
  const { record } = useSelfLab();
  const [tab, setTab] = useState<BottomTab>("stats");
  const [collapsed, setCollapsed] = useState(false);
  const tabs = availableTabs(record);
  const active = tabs.includes(tab) ? tab : tabs[0];
  const items: TabItem<BottomTab>[] = tabs.map((t) => ({ value: t, label: TAB_LABEL[t] }));

  return (
    <section
      aria-label="Comparisons, statistics and export"
      className={cn(
        "flex shrink-0 flex-col border-t border-ns-border bg-ns-bg px-2 pb-2 pt-1.5",
        collapsed ? "xl:h-[44px]" : record ? "xl:h-[clamp(220px,32vh,360px)]" : "xl:h-[96px]",
      )}
    >
      <div className="flex min-h-0 flex-1 flex-col rounded-[var(--ns-radius)] border border-ns-border bg-ns-panel">
        <div className="flex h-8 shrink-0 items-center gap-2 border-b border-ns-border px-3">
          <button
            type="button"
            onClick={() => setCollapsed((c) => !c)}
            aria-expanded={!collapsed}
            className="hidden text-ns-text-faint hover:text-ns-text xl:inline-flex"
            aria-label={collapsed ? "Expand bottom panel" : "Collapse bottom panel"}
          >
            {collapsed ? <ChevronUp aria-hidden="true" className="size-3.5" /> : <ChevronDown aria-hidden="true" className="size-3.5" />}
          </button>
          {items.length ? (
            <Tabs items={items} value={active as BottomTab} onValueChange={setTab} aria-label="Experiment results" idBase="selflab-bottom" size="sm" className="min-w-0 gap-3" />
          ) : (
            <span className="ns-label">Comparisons · statistics · export</span>
          )}
          {record ? (
            <span className="ml-auto flex items-center gap-1.5 font-mono text-[10px] text-ns-text-faint">
              <FixtureMark show={record.is_fixture} what="These results" />
              {record.id}
            </span>
          ) : null}
        </div>
        {!collapsed ? (
          <div
            role={items.length ? "tabpanel" : undefined}
            id={items.length ? `selflab-bottom-panel-${active}` : undefined}
            aria-labelledby={items.length ? `selflab-bottom-tab-${active}` : undefined}
            className="ns-scroll min-h-0 flex-1 px-3 py-2"
          >
            {!record ? (
              <p className="py-1 text-[11.5px] leading-snug text-ns-text-faint">
                <span className="font-mono uppercase tracking-[0.1em] text-ns-text-muted">Nothing to compare yet.</span> Select or commit an experiment:
                statistics appear once its analysis has run, the trial table once a trial is stored, and EXPORT EVIDENCE as soon as the spec is committed.
              </p>
            ) : active === "stats" ? (
              <StatsView rec={record} />
            ) : active === "trials" ? (
              <TrialTable rec={record} />
            ) : (
              <ExportView rec={record} />
            )}
          </div>
        ) : null}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ statistics */

function StatsView({ rec }: { rec: ExperimentRecord }) {
  const { meta, summary } = splitAnalysis(rec.analysis);
  const version =
    (typeof meta.analysis_version === "string" && meta.analysis_version) ||
    (typeof summary.analysis_version === "string" && summary.analysis_version) ||
    rec.committed.provenance?.analysis_version;
  const incomplete = meta.complete === false;
  return (
    <div className="flex flex-col gap-2">
      <p className="flex flex-wrap items-center gap-2 text-[10.5px] text-ns-text-muted">
        <StatusChip tone="neutral">statistical scorer</StatusChip>
        Numbers exactly as computed by the backend analysis{version ? ` (${version})` : ""}. Intervals are the ones the analysis reports.
        <span className="font-mono">
          {typeof meta.n_trials_analyzed === "number" ? `${meta.n_trials_analyzed} trials analysed` : `${rec.n_done} trials stored`} ·{" "}
          {typeof meta.n_errors === "number" ? meta.n_errors : rec.n_errors} errors
          {typeof meta.stopped_by === "string" ? ` · stopped by ${meta.stopped_by}` : ""}
        </span>
      </p>
      {incomplete ? (
        <p role="note" className="rounded-[3px] border border-ns-warn/40 bg-ns-warn/8 px-2 py-1 text-[11px] text-ns-warn">
          Incomplete analysis: the run stopped before its plan finished, so these numbers cover the completed trials only.
        </p>
      ) : null}
      <SummaryView summary={summary} renderStatus={(st) => <TestStatusChip status={st} />} />
    </div>
  );
}

/* ------------------------------------------------------------------ trials */

const PAGE = 25;

function answerOf(t: Trial): string {
  const p = t.parsed ?? {};
  const g = t.ground_truth ?? {};
  for (const k of ["answer_category", "answer_normalized", "answer", "source_report", "letter"]) {
    const v = (g as Record<string, unknown>)[k] ?? (p as Record<string, unknown>)[k];
    if (v !== undefined && (v === null || typeof v !== "object")) return `${k}: ${fmtScalar(v)}`;
  }
  return "";
}

function TrialTable({ rec }: { rec: ExperimentRecord }) {
  const { dataVersion, selectTrial, selectedTrialId } = useSelfLab();
  const [offset, setOffset] = useState(0);
  const trials = useResource<Trial[]>(`selflab:trials:${rec.id}:${offset}:${dataVersion}`, (s) => selflab.trials(rec.id, { offset, limit: PAGE }, s), {
    cache: false,
  });
  const rows = trials.data ?? [];
  const total = rec.n_done;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2 font-mono text-[10.5px] text-ns-text-muted">
        <span>
          {total === 0 ? "no trials" : `${offset + 1}–${Math.min(offset + rows.length, offset + PAGE)} of ${total} stored`} · every status is kept
        </span>
        <span className="ml-auto flex gap-1">
          <Button size="xs" variant="ghost" caps disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>
            prev
          </Button>
          <Button size="xs" variant="ghost" caps disabled={rows.length < PAGE} onClick={() => setOffset(offset + PAGE)}>
            next
          </Button>
        </span>
      </div>
      {trials.error && !trials.data ? <ErrorBox error={trials.error} onRetry={trials.reload} /> : null}
      {!trials.data && !trials.error ? <Loading compact label="Loading trials" /> : null}
      {rows.length ? (
        <div className={cn("overflow-x-auto", trials.stale && "ns-stale")}>
          <table className="w-full border-collapse text-left font-mono text-[10.5px]" aria-label="Trials">
            <thead>
              <tr className="border-b border-ns-border text-ns-text-faint">
                {["#", "condition", "family", "item", "status", "output", "answer", "latency", "runs"].map((h) => (
                  <th key={h} scope="col" className="whitespace-nowrap py-1 pr-3 font-normal">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((t) => (
                <tr
                  key={t.id}
                  onClick={() => selectTrial(t.id)}
                  aria-selected={t.id === selectedTrialId}
                  className={cn("cursor-pointer border-b border-ns-border/60 hover:bg-ns-panel-2", t.id === selectedTrialId && "bg-ns-accent/8")}
                >
                  <td className="py-1 pr-3 text-ns-text">
                    <button type="button" className="hover:underline" onClick={() => selectTrial(t.id)} aria-label={`Inspect trial ${t.index}`}>
                      {t.index}
                    </button>
                  </td>
                  <td className="whitespace-nowrap py-1 pr-3 text-ns-text">{t.condition}</td>
                  <td className="whitespace-nowrap py-1 pr-3 text-ns-text-muted">{t.family}</td>
                  <td className="whitespace-nowrap py-1 pr-3 text-ns-text-muted">{t.item_id}</td>
                  <td className="py-1 pr-3">
                    <TrialStatusChip status={t.status} />
                  </td>
                  <td className="max-w-[280px] truncate py-1 pr-3 text-ns-text" title={t.output_text}>
                    {truncate(t.output_text.replace(/\n/g, "↵"), 60) || "∅"}
                  </td>
                  <td className="whitespace-nowrap py-1 pr-3 text-ns-text-muted">{answerOf(t)}</td>
                  <td className="ns-tabular whitespace-nowrap py-1 pr-3 text-ns-text-muted">{t.latency_s !== undefined ? `${fmtCell(t.latency_s)} s` : "—"}</td>
                  <td className="whitespace-nowrap py-1 text-ns-text-faint">{(t.run_ids ?? []).map((r) => shortId(r)).join(" ") || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : trials.data ? (
        <p className="text-[11px] text-ns-text-faint">No trials on this page.</p>
      ) : null}
    </div>
  );
}

/* ------------------------------------------------------------------ export */

export function ExportPreviewView({ preview }: { preview: ExportPreview }) {
  return (
    <div className="flex flex-col gap-1.5" data-testid="export-preview">
      <table className="w-full max-w-2xl border-collapse text-left font-mono text-[10.5px]" aria-label="Fields included in the export">
        <thead>
          <tr className="border-b border-ns-border text-ns-text-faint">
            <th scope="col" className="py-1 pr-3 font-normal">
              field
            </th>
            <th scope="col" className="py-1 pr-3 font-normal">
              size
            </th>
            <th scope="col" className="py-1 pr-3 font-normal">
              items
            </th>
            <th scope="col" className="py-1 font-normal">
              free text
            </th>
          </tr>
        </thead>
        <tbody>
          {preview.fields.map((f) => (
            <tr key={f.name} className="border-b border-ns-border/60" data-free-text={f.free_text}>
              <td className="py-1 pr-3 text-ns-text">
                {f.name}
                {f.note ? <span className="ml-1.5 text-ns-text-faint">{f.note}</span> : null}
              </td>
              <td className="ns-tabular py-1 pr-3 text-ns-text-muted">{fmtBytes(f.bytes)}</td>
              <td className="ns-tabular py-1 pr-3 text-ns-text-muted">{f.count ?? "—"}</td>
              <td className="py-1">
                {f.free_text ? (
                  <StatusChip tone="warn" title="Contains free text (model output, document or claim text). Review before sharing.">
                    free text{f.free_text_paths?.length ? ` · ${f.free_text_paths.length} path${f.free_text_paths.length === 1 ? "" : "s"}` : ""}
                  </StatusChip>
                ) : looksLikeFreeText(f.name) ? (
                  <span className="text-ns-text-faint" title="Not flagged by the backend, but the name suggests text content">
                    not flagged
                  </span>
                ) : (
                  <span className="text-ns-text-faint">no</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="font-mono text-[10.5px] text-ns-text-muted">
        total {fmtBytes(preview.total_bytes)} · {preview.free_text_fields.length} field{preview.free_text_fields.length === 1 ? "" : "s"} with free text
        {preview.free_text_bytes !== null ? ` (${fmtBytes(preview.free_text_bytes)})` : ""}
        {preview.redacted_bytes !== null ? ` · ${fmtBytes(preview.redacted_bytes)} with free text redacted` : ""}
        {preview.raw_tensors_included === false ? " · raw tensors not included" : ""}
        {preview.n_tensor_references !== null && preview.n_tensor_references !== undefined ? ` · ${preview.n_tensor_references} tensor references` : ""}
      </p>
      {preview.trial_status_counts && Object.keys(preview.trial_status_counts).length ? (
        <p className="font-mono text-[10.5px] text-ns-text-muted" data-testid="export-status-counts">
          trial statuses:{" "}
          {Object.entries(preview.trial_status_counts)
            .map(([k, n]) => `${k} ${n}`)
            .join(" · ")}
        </p>
      ) : null}
      {preview.free_text_paths.length ? (
        <details>
          <summary className="cursor-pointer font-mono text-[10.5px] text-ns-warn">{preview.free_text_paths.length} free-text paths</summary>
          <ul className="mt-1 max-h-32 overflow-auto font-mono text-[10px] text-ns-text-muted">
            {preview.free_text_paths.map((p) => (
              <li key={p.path}>
                {p.path}
                {p.bytes !== null ? <span className="text-ns-text-faint"> · {fmtBytes(p.bytes)}</span> : null}
              </li>
            ))}
          </ul>
        </details>
      ) : null}
      {preview.redaction ? <Caveat>Redaction: {preview.redaction}</Caveat> : null}
      {preview.notes.map((n, i) => (
        <Caveat key={i}>{n}</Caveat>
      ))}
    </div>
  );
}

function ExportView({ rec }: { rec: ExperimentRecord }) {
  const { notify } = useSelfLab();
  const [preview, setPreview] = useState<{ id: string; data: ExportPreview } | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState<"preview" | "download" | null>(null);
  const [redact, setRedact] = useState(false);
  const current = preview?.id === rec.id ? preview.data : null;

  const loadPreview = async () => {
    setBusy("preview");
    setErr(null);
    try {
      setPreview({ id: rec.id, data: await selflab.exportPreview(rec.id) });
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(null);
    }
  };

  const download = async () => {
    setBusy("download");
    try {
      const data = await selflab.exportRecord(rec.id, { excludeFreeText: redact });
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `selflab-${rec.id}${redact ? "-redacted" : ""}.json`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      notify("ok", `Exported reproduction record for ${rec.id}.`);
    } catch (e) {
      notify("error", `Export failed: ${errorMessage(e)}`);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <p className="text-[11px] leading-snug text-ns-text-muted">
        The reproduction record holds the committed spec and hashes, provenance, the trial table, the analysis and the evidence. Tensors are referenced by
        Lab run id, never embedded. Preview what is included before downloading.
      </p>
      <div className="flex flex-wrap gap-2">
        <Button size="sm" variant="secondary" caps onClick={() => void loadPreview()} disabled={busy !== null}>
          <FileJson aria-hidden="true" /> {busy === "preview" ? "Loading preview…" : current ? "Refresh preview" : "Export evidence: preview"}
        </Button>
        <Button size="sm" variant="primary" caps onClick={() => void download()} disabled={!current || busy !== null} title={current ? undefined : "Preview first"}>
          <Download aria-hidden="true" /> {busy === "download" ? "Downloading…" : "Download JSON"}
        </Button>
        <label className="inline-flex items-center gap-1.5 text-[11px] text-ns-text-muted">
          <input type="checkbox" checked={redact} onChange={(e) => setRedact(e.target.checked)} className="accent-[var(--ns-accent)]" />
          redact free text (replaced by sha256 + length)
        </label>
      </div>
      {err ? <ErrorBox error={err} onRetry={() => void loadPreview()} /> : null}
      {current ? <ExportPreviewView preview={current} /> : null}
    </div>
  );
}
