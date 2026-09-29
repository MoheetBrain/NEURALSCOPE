"use client";

import { useMemo, useState, type KeyboardEvent } from "react";
import { ArrowUp } from "lucide-react";
import { Panel } from "@/components/ui/Panel";
import { api } from "@/lib/api/client";
import type { MatrixMetric, MatrixResponse, RunDetail } from "@/lib/api/types";
import { cn } from "@/lib/cn";
import { COLORMAP_NAMES, linearTicks, logTicks, normalize, type SequentialName } from "@/lib/colormap";
import { depthMeaning, truncate, visibleToken } from "@/lib/format";
import { columnOf, depthToRow, isNavKey, moveSelection, rowToDepth } from "@/lib/matrix";
import { MATRIX_METRICS, formatMatrixValue, formatTick, metricInfo, scaleForMatrix } from "@/lib/metrics";
import { Colorbar, type ColorbarTick } from "./charts/Colorbar";
import { HeatmapCanvas, colorsFromGrid, type HeatCell } from "./charts/HeatmapCanvas";
import { useElementSize } from "./charts/useElementSize";
import { targetId, useColumns, useSelection } from "./hooks";
import { useResource } from "./useResource";
import { useWorkspace } from "./WorkspaceProvider";
import { EmptyState, ErrorBox, Loading, SelectInput, StatusBadge, Switch, Tok } from "./ui";

const ROW_LABEL_W = 40;
const COL_LABEL_H = 58;

export function MatrixPanel() {
  const { run, state } = useWorkspace();
  const r = run.data;
  return (
    <Panel
      label="Token × depth matrix"
      className="h-full min-h-0"
      bodyClassName="flex min-h-0 flex-col p-0"
      right={r ? <MatrixControls run={r} /> : null}
    >
      {r ? (
        <MatrixBody run={r} key={r.id} />
      ) : run.loading || (state.runId && !run.error) ? (
        <div className="p-4">
          <Loading label="Loading run" />
        </div>
      ) : run.error ? (
        <div className="p-3">
          <ErrorBox error={run.error} onRetry={run.reload} />
        </div>
      ) : (
        <div className="ns-dot-bg grid flex-1 place-items-center">
          <EmptyState title="No run open">
            Type a prompt and press <span className="font-mono text-ns-text">RUN</span>, start the guided demo, or open a run from the history below.
          </EmptyState>
        </div>
      )}
    </Panel>
  );
}

function MatrixControls({ run }: { run: RunDetail }) {
  const { state, update } = useWorkspace();
  const isChild = Boolean(run.parent_run_id);
  return (
    <div className="flex items-center gap-2">
      {isChild ? (
        <button
          type="button"
          onClick={() => update({ metric: "delta_resid" })}
          aria-pressed={state.metric === "delta_resid"}
          className={cn(
            "h-6 whitespace-nowrap rounded-[3px] border px-2 font-mono text-[10px] uppercase tracking-[0.08em]",
            state.metric === "delta_resid"
              ? "border-ns-causal/60 bg-ns-causal/15 text-ns-causal"
              : "border-ns-causal/35 text-ns-causal hover:bg-ns-causal/10",
          )}
          title="‖h_int − h_base‖/‖h_base‖ — where the intervention changed the residual stream"
        >
          Δ vs parent
        </button>
      ) : null}
      <label className="flex items-center gap-1.5">
        <span className="ns-label hidden text-[9.5px] 2xl:inline">metric</span>
        <SelectInput
          aria-label="Matrix metric"
          className="h-6 w-[150px] text-[11px]"
          value={state.metric}
          onChange={(e) => update({ metric: e.target.value as MatrixMetric })}
        >
          {MATRIX_METRICS.map((m) => (
            <option key={m.value} value={m.value} disabled={(m.needsParent && !isChild) || (m.needsTarget && run.target?.token_id == null)}>
              {m.short}
              {m.needsParent && !isChild ? " (intervention runs)" : ""}
            </option>
          ))}
        </SelectInput>
      </label>
      <SelectInput
        aria-label="Colormap"
        className="h-6 w-[84px] text-[11px]"
        value={state.colormap}
        onChange={(e) => update({ colormap: e.target.value as SequentialName })}
      >
        {COLORMAP_NAMES.map((c) => (
          <option key={c} value={c}>
            {c}
          </option>
        ))}
      </SelectInput>
      <span className="hidden 2xl:inline">
        <Switch checked={state.hideTemplate} onChange={(v) => update({ hideTemplate: v })} label="hide template" />
      </span>
    </div>
  );
}

function colorbarTicks(m: MatrixResponse): ColorbarTick[] {
  const scale = scaleForMatrix(m);
  const info = metricInfo(m.metric);
  let values: number[];
  if (info.log10Values) {
    values = [];
    for (let e = Math.ceil(scale.vmin); e <= Math.floor(scale.vmax); e++) values.push(e);
    if (values.length > 6) values = values.filter((_, i) => i % 2 === 0);
  } else if (scale.kind === "log") values = logTicks(scale.vmin, scale.vmax);
  else values = linearTicks(scale.vmin, scale.vmax, 4);
  return values
    .map((v) => ({ t: normalize(v, { ...scale, reverse: false }) ?? 0, label: formatTick(m, v) }))
    .filter((tk) => tk.t >= 0 && tk.t <= 1);
}

function MatrixBody({ run }: { run: RunDetail }) {
  const { state, select, update } = useWorkspace();
  const sel = useSelection(run);
  const columns = useColumns(run);
  const tid = targetId(run);
  const metric: MatrixMetric = state.metric === "delta_resid" && !run.parent_run_id ? "lens_top1_prob" : state.metric;
  const info = metricInfo(metric);
  const key = `matrix:${run.id}:${metric}:${info.needsTarget ? tid : "-"}`;
  const res = useResource<MatrixResponse>(key, (signal) => api.matrix(run.id, metric, info.needsTarget ? tid : null, signal));
  const m = res.data && res.data.metric === metric ? res.data : res.stale ? res.data : undefined;
  const [areaRef, area] = useElementSize<HTMLDivElement>();
  const [hover, setHover] = useState<{ cell: HeatCell; x: number; y: number } | null>(null);

  const D = run.depth_labels.length;
  const cols = columns.length;
  const cellW = Math.max(6, Math.min(34, Math.floor((area.width - ROW_LABEL_W - 8) / Math.max(1, cols))));
  const cellH = Math.max(7, Math.min(24, Math.floor((area.height - COL_LABEL_H - 14) / Math.max(1, D))));

  const scale = useMemo(() => (m ? scaleForMatrix(m) : null), [m]);
  const colors = useMemo(() => {
    if (!m || !scale) return new Uint8ClampedArray(0);
    const grid = Array.from({ length: D }, (_, r) => {
      const row = m.values[rowToDepth(r, D)] ?? [];
      return columns.map((p) => row[p] ?? null);
    });
    return colorsFromGrid(grid, (v) => normalize(v, scale), state.colormap);
  }, [m, scale, D, columns, state.colormap]);

  const selCell: HeatCell | null = sel ? { row: depthToRow(sel.depth, D), col: columnOf(sel.position, columns) } : null;
  const targetCol = run.target ? columns.indexOf(run.target.position) : -1;
  const markCols = useMemo(() => (targetCol >= 0 ? [targetCol] : []), [targetCol]);

  const onKey = (e: KeyboardEvent<HTMLCanvasElement>) => {
    if (!sel || !isNavKey(e.key)) return;
    e.preventDefault();
    select(moveSelection(sel, e.key, D, columns));
  };

  const hovered = hover ? { depth: rowToDepth(hover.cell.row, D), position: columns[hover.cell.col] } : null;
  const labelEvery = cellW >= 11 ? 1 : Math.ceil(11 / cellW);
  const rowLabelEvery = cellH >= 11 ? 1 : 2;

  const selValue = m && sel ? formatMatrixValue(m, sel.depth, sel.position) : null;
  const selTop1 = m?.top1_tokens && sel ? m.top1_tokens[sel.depth]?.[sel.position] : undefined;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex min-h-9 shrink-0 flex-wrap items-center gap-x-3 gap-y-1 border-b border-ns-border px-3 py-1.5">
        {m ? <StatusBadge status={m.status} /> : <StatusBadge status={metric === "delta_resid" ? "causal" : "derived"} />}
        <span className="text-[11.5px] text-ns-text">{m?.label ?? info.short}</span>
        {info.needsTarget && run.target ? (
          <span className="font-mono text-[10.5px] text-ns-text-muted">
            target <Tok text={run.target.token ?? ""} className="text-ns-accent-2" /> (id {run.target.token_id})
          </span>
        ) : null}
        {metric === "delta_resid" ? <span className="font-mono text-[10.5px] text-ns-text-muted">vs parent {run.parent_run_id}</span> : null}
        <span className="ml-auto text-[10.5px] text-ns-text-faint">{info.description}</span>
      </div>

      <div ref={areaRef} className="relative min-h-0 flex-1 overflow-auto px-2 pt-2">
        {res.error && !m ? (
          <ErrorBox error={res.error} onRetry={res.reload} className="m-2" />
        ) : !m ? (
          <Loading
            className="mx-2"
            label={`Computing ${info.short}`}
            hint={
              metric === "target_rank" || metric === "target_prob" || metric === "next_token_rank"
                ? "First request projects every residual state through the unembedding (a few seconds on CPU); later requests are cached."
                : run.kind === "math"
                  ? "Math Lab runs are stored at trace level minimal — the Logit Lens is computed lazily on first view."
                  : undefined
            }
          />
        ) : area.width > 0 ? (
          <div className={cn("relative inline-block", res.stale && "ns-stale")}>
            <div className="flex">
              {/* depth labels */}
              <div className="shrink-0 pr-1" style={{ width: ROW_LABEL_W }} aria-hidden="true">
                {Array.from({ length: D }, (_, r) => {
                  const d = rowToDepth(r, D);
                  const on = sel?.depth === d;
                  return (
                    <div
                      key={r}
                      className={cn("truncate text-right font-mono text-[9.5px]", on ? "text-ns-accent" : "text-ns-text-faint")}
                      style={{ height: cellH, lineHeight: `${cellH}px` }}
                    >
                      {r % rowLabelEvery === 0 || on ? run.depth_labels[d] : ""}
                    </div>
                  );
                })}
              </div>
              <div className="relative">
                <HeatmapCanvas
                  rows={D}
                  cols={cols}
                  colors={colors}
                  cellW={cellW}
                  cellH={cellH}
                  gap={cellW >= 8 && cellH >= 8 ? 1 : 0}
                  selected={selCell}
                  hovered={hover?.cell ?? null}
                  markCols={markCols}
                  crosshair
                  focusable
                  onKeyDown={onKey}
                  onHoverCell={(c, e) => {
                    if (!c || !e) return setHover(null);
                    const box = e.currentTarget.parentElement?.getBoundingClientRect();
                    setHover({ cell: c, x: e.clientX - (box?.left ?? 0), y: e.clientY - (box?.top ?? 0) });
                  }}
                  onClickCell={(c) => {
                    select({ depth: rowToDepth(c.row, D), position: columns[c.col] });
                    update({ attnLayer: null });
                  }}
                  ariaLabel={`Token by depth matrix: ${m.label}. ${D} depths (embeddings at the bottom, ${run.depth_labels[D - 1]} at the top) by ${cols} tokens. ${
                    sel ? `Selected ${run.depth_labels[sel.depth]} at t${sel.position}: ${selValue}.` : ""
                  } Use arrow keys to move the selection.`}
                />
                {/* token labels */}
                <div className="relative" style={{ height: COL_LABEL_H, width: cols * cellW }} aria-hidden="true">
                  {columns.map((p, c) => {
                    const t = run.tokens[p];
                    const on = sel?.position === p;
                    const isTarget = run.target?.position === p;
                    if (!(c % labelEvery === 0 || on || isTarget)) return null;
                    return (
                      <div
                        key={p}
                        className={cn(
                          "absolute top-1 overflow-hidden whitespace-pre font-mono text-[10px] leading-none",
                          t.role === "template" && "text-ns-text-faint",
                          t.role === "user" && "text-ns-text-muted",
                          t.role === "generated" && "text-ns-accent",
                          on && "text-ns-text",
                          isTarget && "text-ns-accent-2",
                        )}
                        style={{
                          left: c * cellW + cellW / 2 - 6,
                          width: 12,
                          height: COL_LABEL_H - 4,
                          writingMode: "vertical-rl",
                          transform: "rotate(180deg)",
                          textAlign: "right",
                        }}
                        title={`t${p} ${visibleToken(t.text)}`}
                      >
                        {truncate(visibleToken(t.text), 9)}
                      </div>
                    );
                  })}
                </div>
                {targetCol >= 0 ? (
                  <div
                    className="pointer-events-none absolute -top-2 font-mono text-[8px] leading-none text-ns-accent-2"
                    style={{ left: targetCol * cellW + cellW / 2 - 3 }}
                    aria-hidden="true"
                  >
                    ▼
                  </div>
                ) : null}
                {hover && hovered ? (
                  <MatrixTooltip run={run} m={m} depth={hovered.depth} position={hovered.position} x={hover.x} y={hover.y} width={cols * cellW} />
                ) : null}
              </div>
            </div>
          </div>
        ) : null}
      </div>

      <div className="flex shrink-0 flex-wrap items-end gap-x-5 gap-y-2 border-t border-ns-border px-3 py-2">
        {m && scale ? (
          <Colorbar
            cmap={state.colormap}
            reverse={scale.reverse}
            ticks={colorbarTicks(m)}
            width={200}
            label={m.label}
            noDataLabel={m.values.some((row) => row.some((v) => v === null)) ? "n/a" : undefined}
          />
        ) : null}
        <div className="flex items-center gap-1.5 pb-3 font-mono text-[10px] text-ns-text-faint">
          <ArrowUp aria-hidden="true" className="size-3" /> deeper · emb at bottom · ▼ target t{run.target?.position ?? "—"} · ←↑↓→ move
        </div>
        <div className="ml-auto min-w-0 pb-2 text-right" aria-live="polite">
          {sel ? (
            <div className="font-mono text-[11px] text-ns-text">
              <span className="text-ns-accent">{run.depth_labels[sel.depth]}</span> · t{sel.position}{" "}
              <Tok text={run.tokens[sel.position]?.text ?? ""} className="text-ns-text" /> · {selValue ?? "…"}
              {selTop1 !== undefined ? (
                <>
                  {" "}· top-1 <Tok text={selTop1} className="text-ns-derived" />
                </>
              ) : null}
            </div>
          ) : null}
          {sel ? <div className="text-[10.5px] text-ns-text-faint">{depthMeaning(sel.depth)}</div> : null}
        </div>
      </div>
    </div>
  );
}

function MatrixTooltip({
  run,
  m,
  depth,
  position,
  x,
  y,
  width,
}: {
  run: RunDetail;
  m: MatrixResponse;
  depth: number;
  position: number;
  x: number;
  y: number;
  width: number;
}) {
  const t = run.tokens[position];
  const top1 = m.top1_tokens?.[depth]?.[position];
  const left = x + 14 + 200 > width ? Math.max(0, x - 214) : x + 14;
  return (
    <div
      role="tooltip"
      className="pointer-events-none absolute z-30 w-[200px] rounded-[3px] border border-ns-border-strong bg-ns-bg/95 px-2 py-1.5 shadow-xl"
      style={{ left, top: Math.max(0, y - 10) }}
    >
      <div className="flex items-center justify-between font-mono text-[10.5px]">
        <span className="text-ns-accent">{run.depth_labels[depth]}</span>
        <span className="text-ns-text-faint">depth {depth}</span>
      </div>
      <div className="text-[10px] text-ns-text-faint">{depthMeaning(depth)}</div>
      <div className="mt-1 flex items-center gap-1.5 font-mono text-[10.5px] text-ns-text-muted">
        t{position} <Tok text={t?.text ?? ""} className="text-ns-text" /> <span className="text-ns-text-faint">{t?.role}</span>
      </div>
      <div className="mt-1 font-mono text-[11px] text-ns-text">{formatMatrixValue(m, depth, position)}</div>
      {top1 !== undefined ? (
        <div className="font-mono text-[10.5px] text-ns-text-muted">
          top-1 <Tok text={top1} className="text-ns-derived" /> <span className="text-ns-text-faint">(decoded by Logit Lens)</span>
        </div>
      ) : null}
    </div>
  );
}
