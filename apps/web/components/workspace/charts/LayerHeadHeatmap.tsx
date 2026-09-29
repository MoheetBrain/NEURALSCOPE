"use client";

import { useMemo, useState, type ReactNode } from "react";
import { absMax2d, divergingNormalize } from "@/lib/colormap";
import { fmtNum } from "@/lib/format";
import { Colorbar } from "./Colorbar";
import { HeatmapCanvas, colorsFromGrid, type HeatCell } from "./HeatmapCanvas";
import { useElementSize } from "./useElementSize";

/**
 * Layer × head grid with a diverging colormap centred on 0 (dark = no effect).
 * Rows are drawn deepest-first (last layer at the top) to match the token × depth matrix.
 */
export function LayerHeadHeatmap({
  values,
  onCellClick,
  selected,
  tooltip,
  ariaLabel,
  colorbarLabel,
  negLabel = "−",
  posLabel = "+",
  maxCell = 22,
  formatTick = (v: number) => fmtNum(v, 2),
}: {
  values: readonly (readonly (number | null)[])[];
  onCellClick?: (layer: number, head: number) => void;
  selected?: { layer: number; head: number } | null;
  tooltip: (layer: number, head: number) => ReactNode;
  ariaLabel: string;
  colorbarLabel: string;
  negLabel?: string;
  posLabel?: string;
  maxCell?: number;
  formatTick?: (v: number) => string;
}) {
  const nL = values.length;
  const nH = nL ? values[0].length : 0;
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const [hover, setHover] = useState<HeatCell | null>(null);
  const labelW = 30;
  const cellW = Math.max(8, Math.min(maxCell * 1.6, Math.floor((size.width - labelW - 4) / Math.max(1, nH))));
  const cellH = Math.max(7, Math.min(maxCell * 0.62, 13));
  const m = useMemo(() => absMax2d(values), [values]);
  const grid = useMemo(() => values.map((_, r) => values[nL - 1 - r]), [values, nL]);
  const colors = useMemo(() => colorsFromGrid(grid, (v) => divergingNormalize(v, m), "diverging"), [grid, m]);
  const sel = selected ? { row: nL - 1 - selected.layer, col: selected.head } : null;
  const hoverLH = hover ? { layer: nL - 1 - hover.row, head: hover.col } : null;

  return (
    <div ref={wrapRef} className="relative w-full">
      {size.width > 0 && nL > 0 ? (
        <>
          <div className="flex">
            <div className="shrink-0 pr-1" style={{ width: labelW }} aria-hidden="true">
              {Array.from({ length: nL }, (_, r) => {
                const l = nL - 1 - r;
                return (
                  <div key={r} className="text-right font-mono text-[8.5px] text-ns-text-faint" style={{ height: cellH, lineHeight: `${cellH}px` }}>
                    {l % 4 === 3 || l === 0 || l === nL - 1 ? `L${l}` : ""}
                  </div>
                );
              })}
            </div>
            <div>
              <HeatmapCanvas
                rows={nL}
                cols={nH}
                colors={colors}
                cellW={cellW}
                cellH={cellH}
                gap={1}
                selected={sel}
                hovered={hover}
                onHoverCell={(c) => setHover(c)}
                onClickCell={onCellClick ? (c) => onCellClick(nL - 1 - c.row, c.col) : undefined}
                ariaLabel={ariaLabel}
              />
              <div className="flex" aria-hidden="true">
                {Array.from({ length: nH }, (_, h) => (
                  <div key={h} className="text-center font-mono text-[8.5px] text-ns-text-faint" style={{ width: cellW }}>
                    {h}
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="mt-1 flex flex-wrap items-end justify-between gap-2">
            <Colorbar
              cmap="diverging"
              width={170}
              label={colorbarLabel}
              ticks={[
                { t: 0, label: `${formatTick(-m)} ${negLabel}` },
                { t: 0.5, label: "0" },
                { t: 1, label: `${formatTick(m)} ${posLabel}` },
              ]}
            />
            <span className="pb-3 font-mono text-[9.5px] text-ns-text-faint">rows: layer (deepest on top) · cols: head</span>
          </div>
          {hoverLH ? (
            <div className="pointer-events-none absolute right-0 top-0 z-20 rounded-[3px] border border-ns-border-strong bg-ns-bg/95 px-2 py-1 font-mono text-[10.5px] text-ns-text shadow-lg">
              {tooltip(hoverLH.layer, hoverLH.head)}
            </div>
          ) : null}
        </>
      ) : (
        <div style={{ height: 120 }} />
      )}
    </div>
  );
}
