"use client";

import { useMemo, useState } from "react";
import { Button } from "@/components/ui/Button";
import { api } from "@/lib/api/client";
import type { PatchSweepResult, RunDetail } from "@/lib/api/types";
import { divergingNormalize } from "@/lib/colormap";
import { fmtNum, fmtSeconds, shortId, truncate, visibleToken } from "@/lib/format";
import { Colorbar } from "../charts/Colorbar";
import { HeatmapCanvas, colorsFromGrid, type HeatCell } from "../charts/HeatmapCanvas";
import { useElementSize } from "../charts/useElementSize";
import { useWorkspace } from "../WorkspaceProvider";
import { Caveat, ErrorBox, Field, ProgressBar, SelectInput, StatusBadge } from "../ui";

/**
 * Residual activation-patching sweep: this run is the CORRUPT run; for every (depth, position)
 * from the first differing token onward, h_corrupt ← h_clean and the restoration of
 * logit(τ_clean) − logit(τ_corrupt) is measured (CAUSALLY TESTED, one counterfactual pair).
 */
export function PatchSweepPanel({ run }: { run: RunDetail }) {
  const { history, jobs, activeJob, startJob, bumpRun, notify, update } = useWorkspace();
  const [clean, setClean] = useState("");
  const candidates = (history.data ?? []).filter((r) => r.kind === "baseline" && r.id !== run.id && r.n_tokens === run.tokens.length);
  const stored = (run.metrics.sweeps ?? []).filter((s): s is PatchSweepResult => s.kind === "resid_patch");
  const job = jobs.patch_sweep;
  const running = job && job.scopeKey === run.id && (job.status === "queued" || job.status === "running") ? job : null;
  const fresh = job && job.scopeKey === run.id && job.status === "done" ? (job.result as PatchSweepResult | undefined) : undefined;
  const result = stored[stored.length - 1] ?? fresh;
  const isIntervention = run.kind === "intervention";

  return (
    <div className="flex flex-col gap-2">
      <p className="text-[10.5px] leading-snug text-ns-text-muted">
        This run is the <span className="text-ns-text">corrupt</span> run. Pick a <span className="text-ns-text">clean</span> run with the same token length (e.g. clean “37 + 58” vs corrupt “27 + 58”: same length, different answer token); each residual state is patched from clean into corrupt and the restored logit difference is measured.
      </p>
      <div className="flex flex-wrap items-end gap-2">
        <Field label={`clean run (${candidates.length} with ${run.tokens.length} tokens)`} className="min-w-0 flex-1">
          <SelectInput value={clean} onChange={(e) => setClean(e.target.value)} aria-label="Clean run for patching sweep">
            <option value="">— pick a clean run —</option>
            {candidates.map((r) => (
              <option key={r.id} value={r.id}>
                {shortId(r.id)} · {truncate(r.prompt, 44)}
              </option>
            ))}
          </SelectInput>
        </Field>
        <Button
          size="sm"
          caps
          variant="secondary"
          disabled={!clean || Boolean(activeJob) || isIntervention}
          title={isIntervention ? "Patch sweeps run between un-intervened runs" : activeJob ? `Busy: ${activeJob.label}` : undefined}
          onClick={() =>
            void startJob<PatchSweepResult>("patch_sweep", "Patch sweep", run.id, () => api.patchSweep({ clean_run_id: clean, corrupt_run_id: run.id }), () => {
              bumpRun();
              notify("ok", "Patching sweep finished (CAUSALLY TESTED, one counterfactual pair).");
            })
          }
        >
          Run patch sweep
        </Button>
      </div>
      <span className="text-[10.5px] text-ns-text-faint">≈100 s on CPU (29 depths × patched positions).</span>
      {running ? (
        <div className="flex flex-col gap-1">
          <ProgressBar value={running.progress} tone="causal" />
          <span className="font-mono text-[10px] text-ns-text-muted">
            {running.status} · {running.message}
          </span>
        </div>
      ) : null}
      {job?.status === "error" && job.scopeKey === run.id ? <ErrorBox error={job.error} /> : null}
      {result ? (
        <PatchHeatmap
          result={result}
          onPick={(depth, position) =>
            update((s) => ({
              draft: { ...s.draft, type: "resid_patch", depth, position, sourceRunId: result.clean_run_id, sourcePosition: "", alpha: 1 },
            }))
          }
        />
      ) : !running ? (
        <p className="text-[11px] text-ns-text-faint">No patching sweep for this run yet.</p>
      ) : null}
    </div>
  );
}

function PatchHeatmap({ result, onPick }: { result: PatchSweepResult; onPick: (depth: number, position: number) => void }) {
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const [hover, setHover] = useState<HeatCell | null>(null);
  const D = result.depth_labels.length;
  const cols = result.positions;
  const labelW = 34;
  const cellW = Math.max(8, Math.min(30, Math.floor((size.width - labelW - 4) / Math.max(1, cols.length))));
  const cellH = 11;
  const absMax = useMemo(() => {
    let m = 1;
    for (const row of result.restoration) for (const v of row) if (v !== null && Math.abs(v) > m) m = Math.abs(v);
    return m;
  }, [result]);
  const grid = useMemo(() => Array.from({ length: D }, (_, r) => cols.map((p) => result.restoration[D - 1 - r]?.[p] ?? null)), [result, D, cols]);
  const colors = useMemo(() => colorsFromGrid(grid, (v) => divergingNormalize(v, absMax), "diverging"), [grid, absMax]);
  const hv = hover ? { depth: D - 1 - hover.row, position: cols[hover.col] } : null;

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex flex-wrap items-center gap-2">
        <StatusBadge status={result.status} note={result.scope} />
        <span className="font-mono text-[10px] text-ns-text-muted">
          {result.metric} · clean {fmtNum(result.metric_clean, 3)} · corrupt {fmtNum(result.metric_corrupt, 3)} · {fmtSeconds(result.seconds)}
        </span>
      </div>
      <div ref={wrapRef} className="relative">
        {size.width > 0 ? (
          <div className="flex">
            <div className="shrink-0 pr-1" style={{ width: labelW }} aria-hidden="true">
              {Array.from({ length: D }, (_, r) => {
                const d = D - 1 - r;
                return (
                  <div key={r} className="text-right font-mono text-[8.5px] text-ns-text-faint" style={{ height: cellH, lineHeight: `${cellH}px` }}>
                    {d % 4 === 0 || d === D - 1 ? result.depth_labels[d] : ""}
                  </div>
                );
              })}
            </div>
            <div>
              <HeatmapCanvas
                rows={D}
                cols={cols.length}
                colors={colors}
                cellW={cellW}
                cellH={cellH}
                gap={1}
                hovered={hover}
                onHoverCell={setHover}
                onClickCell={(c) => onPick(D - 1 - c.row, cols[c.col])}
                ariaLabel={`Activation patching restoration by depth and position (${cols.length} positions). 1 = clean behaviour fully restored.`}
              />
              <div className="flex" aria-hidden="true">
                {cols.map((p) => (
                  <div key={p} className="overflow-hidden whitespace-pre text-center font-mono text-[8.5px] text-ns-text-faint" style={{ width: cellW }} title={`t${p}`}>
                    {truncate(visibleToken(result.tokens[p] ?? ""), 3)}
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div style={{ height: D * cellH }} />
        )}
        {hv ? (
          <div className="pointer-events-none absolute right-0 top-0 z-20 rounded-[3px] border border-ns-border-strong bg-ns-bg/95 px-2 py-1 font-mono text-[10.5px] text-ns-text">
            {result.depth_labels[hv.depth]} · t{hv.position} {visibleToken(result.tokens[hv.position] ?? "")}
            <br />
            restoration {fmtNum(result.restoration[hv.depth]?.[hv.position] ?? null, 3)}
            <br />
            <span className="text-ns-text-faint">click → prefill residual patch</span>
          </div>
        ) : null}
      </div>
      <Colorbar
        cmap="diverging"
        width={170}
        label="restoration"
        ticks={[
          { t: 0, label: fmtNum(-absMax, 1) },
          { t: 0.5, label: "0" },
          { t: 1, label: `${fmtNum(absMax, 1)} restored` },
        ]}
      />
      <Caveat>Restoration = (LD_patched − LD_corrupt) / (LD_clean − LD_corrupt). Single counterfactual pair; it localises where clean information is sufficient, not a full circuit.</Caveat>
    </div>
  );
}
