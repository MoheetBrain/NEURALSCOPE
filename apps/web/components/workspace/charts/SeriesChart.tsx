"use client";

import { useState, type ReactNode } from "react";
import { linearTicks, logTicks } from "@/lib/colormap";
import { cn } from "@/lib/cn";
import { fmtNum } from "@/lib/format";
import { useElementSize } from "./useElementSize";

export interface ChartSeries {
  id: string;
  label: string;
  color: string;
  values: readonly (number | null)[];
}

export interface ChartMarker {
  index: number;
  label: string;
  color?: string;
}

export interface SeriesChartProps {
  /** One label per x index (e.g. depth labels). */
  x: readonly string[];
  series: readonly ChartSeries[];
  kind?: "line" | "bar";
  yScale?: "linear" | "log";
  yDomain?: [number, number];
  /** Tick values on y (defaults computed). */
  yTicks?: readonly number[];
  formatY?: (v: number) => string;
  /** Tooltip value formatter (defaults to formatY). */
  formatValue?: (v: number, s: ChartSeries, i: number) => string;
  height?: number;
  markers?: readonly ChartMarker[];
  highlightIndex?: number | null;
  onIndexClick?: (i: number) => void;
  /** Show every n-th x label (auto when omitted). */
  xTickEvery?: number;
  ariaLabel: string;
  /** Extra tooltip rows for index i. */
  tooltipExtra?: (i: number) => ReactNode;
  yLabel?: string;
  className?: string;
  zeroLine?: boolean;
  /** Upper bound for the auto domain (e.g. vocabulary size for ranks). */
  yCap?: number;
}

const PAD = { top: 10, right: 12, bottom: 22, left: 42 };

/**
 * Minimal SVG line / bar chart with one y axis, hairline grid, crosshair + tooltip,
 * a legend for ≥2 series and optional index markers (e.g. "first top-1 depth").
 */
export function SeriesChart({
  x,
  series,
  kind = "line",
  yScale = "linear",
  yDomain,
  yTicks,
  formatY = (v) => fmtNum(v, 2),
  formatValue,
  height = 160,
  markers,
  highlightIndex,
  onIndexClick,
  xTickEvery,
  ariaLabel,
  tooltipExtra,
  yLabel,
  className,
  zeroLine,
  yCap,
}: SeriesChartProps) {
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const width = Math.max(120, size.width);
  const n = x.length;
  const plotW = width - PAD.left - PAD.right;
  const plotH = height - PAD.top - PAD.bottom;

  // domain
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of series)
    for (const v of s.values) {
      if (v === null || !Number.isFinite(v) || (yScale === "log" && v <= 0)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  if (!(lo <= hi)) {
    lo = yScale === "log" ? 1 : 0;
    hi = 1;
  }
  let [y0, y1] =
    yDomain ?? (yScale === "log" ? [10 ** Math.floor(Math.log10(lo)), 10 ** Math.ceil(Math.log10(hi))] : [Math.min(0, lo), hi]);
  if (kind === "bar" && !yDomain) y0 = Math.min(0, lo);
  if (yCap !== undefined && !yDomain) y1 = Math.min(y1, Math.max(yCap, hi));
  if (y0 === y1) y1 = y0 + (yScale === "log" ? y0 * 9 || 1 : 1);
  const ly = (v: number) => (yScale === "log" ? Math.log10(Math.max(v, 1e-12)) : v);
  const Y0 = ly(y0);
  const Y1 = ly(y1);
  const yPx = (v: number) => PAD.top + plotH - ((ly(v) - Y0) / (Y1 - Y0 || 1)) * plotH;
  const band = n > 0 ? plotW / n : plotW;
  const xPx = (i: number) => PAD.left + band * (i + 0.5);
  const ticks = yTicks ?? (yScale === "log" ? logTicks(y0, y1) : linearTicks(y0, y1, 4));
  const every = xTickEvery ?? Math.max(1, Math.ceil(n / Math.max(1, Math.floor(plotW / 34))));

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * width;
    const i = Math.floor((px - PAD.left) / band);
    setHover(i >= 0 && i < n ? i : null);
  };

  const fmtV = formatValue ?? ((v: number) => formatY(v));
  const barW = Math.max(1, Math.min(18, (band - 2) / Math.max(1, series.length)));
  const groupW = barW * series.length;

  return (
    <div className={cn("relative w-full", className)} ref={wrapRef}>
      {series.length >= 2 ? (
        <div className="mb-1 flex flex-wrap items-center gap-x-3 gap-y-1" aria-hidden="true">
          {series.map((s) => (
            <span key={s.id} className="inline-flex items-center gap-1.5 font-mono text-[10px] text-ns-text-muted">
              <span className="inline-block h-0.5 w-3 rounded-full" style={{ background: s.color }} />
              {s.label}
            </span>
          ))}
        </div>
      ) : null}
      {size.width > 0 ? (
        <svg
          width={width}
          height={height}
          role="img"
          aria-label={ariaLabel}
          className={cn("block overflow-visible", onIndexClick && "cursor-pointer")}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
          onClick={() => {
            if (hover !== null && onIndexClick) onIndexClick(hover);
          }}
        >
          {/* grid + y ticks */}
          {ticks.map((t) => {
            const yy = yPx(t);
            if (yy < PAD.top - 1 || yy > PAD.top + plotH + 1) return null;
            return (
              <g key={t}>
                <line x1={PAD.left} x2={PAD.left + plotW} y1={yy} y2={yy} stroke="var(--ns-border)" strokeWidth={1} />
                <text x={PAD.left - 5} y={yy + 3} textAnchor="end" className="fill-ns-text-faint font-mono text-[9.5px]">
                  {formatY(t)}
                </text>
              </g>
            );
          })}
          {zeroLine && y0 < 0 && y1 > 0 ? (
            <line x1={PAD.left} x2={PAD.left + plotW} y1={yPx(0)} y2={yPx(0)} stroke="var(--ns-border-strong)" strokeWidth={1} />
          ) : null}
          {yLabel ? (
            <text
              x={10}
              y={PAD.top + plotH / 2}
              transform={`rotate(-90 10 ${PAD.top + plotH / 2})`}
              textAnchor="middle"
              className="fill-ns-text-faint font-mono text-[9px] uppercase tracking-[0.1em]"
            >
              {yLabel}
            </text>
          ) : null}
          {/* x labels */}
          {x.map((lab, i) =>
            i === n - 1 || (i % every === 0 && n - 1 - i >= Math.ceil(every * 0.75)) || i === highlightIndex ? (
              <text
                key={i}
                x={xPx(i)}
                y={height - 6}
                textAnchor="middle"
                className={cn("font-mono text-[9.5px]", highlightIndex === i ? "fill-ns-accent" : "fill-ns-text-faint")}
              >
                {lab}
              </text>
            ) : null,
          )}
          {/* highlight band */}
          {highlightIndex !== null && highlightIndex !== undefined && highlightIndex >= 0 && highlightIndex < n ? (
            <rect x={xPx(highlightIndex) - band / 2} y={PAD.top} width={band} height={plotH} fill="rgba(143,179,255,0.08)" />
          ) : null}
          {/* series */}
          {kind === "bar"
            ? series.map((s, si) =>
                s.values.map((v, i) => {
                  if (v === null || !Number.isFinite(v)) return null;
                  const base = yPx(yScale === "log" ? y0 : Math.max(y0, Math.min(0, y1)));
                  const top = yPx(v);
                  const xx = xPx(i) - groupW / 2 + si * barW;
                  return (
                    <rect
                      key={`${s.id}-${i}`}
                      x={xx}
                      y={Math.min(base, top)}
                      width={Math.max(1, barW - (series.length > 1 ? 1 : 0))}
                      height={Math.max(1, Math.abs(base - top))}
                      rx={Math.min(2, barW / 3)}
                      fill={s.color}
                      opacity={hover === null || hover === i ? 1 : 0.55}
                    />
                  );
                }),
              )
            : series.map((s) => {
                let d = "";
                let pen = false;
                s.values.forEach((v, i) => {
                  if (v === null || !Number.isFinite(v) || (yScale === "log" && v <= 0)) {
                    pen = false;
                    return;
                  }
                  d += `${pen ? "L" : "M"}${xPx(i).toFixed(1)},${yPx(v).toFixed(1)}`;
                  pen = true;
                });
                return (
                  <g key={s.id}>
                    <path d={d} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
                    {n <= 40
                      ? s.values.map((v, i) =>
                          v === null || !Number.isFinite(v) || (yScale === "log" && v <= 0) ? null : (
                            <circle
                              key={i}
                              cx={xPx(i)}
                              cy={yPx(v)}
                              r={hover === i || highlightIndex === i ? 3.5 : 2}
                              fill={s.color}
                              stroke="var(--ns-panel)"
                              strokeWidth={1.5}
                            />
                          ),
                        )
                      : null}
                  </g>
                );
              })}
          {/* markers */}
          {markers?.map((m) =>
            m.index >= 0 && m.index < n ? (
              <g key={`${m.index}-${m.label}`}>
                <line
                  x1={xPx(m.index)}
                  x2={xPx(m.index)}
                  y1={PAD.top}
                  y2={PAD.top + plotH}
                  stroke={m.color ?? "var(--ns-accent-2)"}
                  strokeWidth={1}
                />
                <text
                  x={xPx(m.index) + (m.index > n * 0.7 ? -4 : 4)}
                  y={PAD.top + 9}
                  textAnchor={m.index > n * 0.7 ? "end" : "start"}
                  className="font-mono text-[9.5px]"
                  fill={m.color ?? "var(--ns-accent-2)"}
                  stroke="var(--ns-panel)"
                  strokeWidth={3}
                  paintOrder="stroke"
                >
                  {m.label}
                </text>
              </g>
            ) : null,
          )}
          {/* crosshair */}
          {hover !== null ? (
            <line x1={xPx(hover)} x2={xPx(hover)} y1={PAD.top} y2={PAD.top + plotH} stroke="rgba(231,233,236,0.35)" strokeWidth={1} />
          ) : null}
          <line x1={PAD.left} x2={PAD.left + plotW} y1={PAD.top + plotH} y2={PAD.top + plotH} stroke="var(--ns-border-strong)" />
        </svg>
      ) : (
        <div style={{ height }} />
      )}
      {hover !== null && size.width > 0 ? (
        <div
          className="pointer-events-none absolute z-20 min-w-32 rounded-[3px] border border-ns-border-strong bg-ns-bg/95 px-2 py-1.5 shadow-lg"
          style={{
            left: Math.min(Math.max(0, xPx(hover) + 10), width - 150),
            top: (series.length >= 2 ? 18 : 0) + 4,
          }}
        >
          <div className="font-mono text-[10px] text-ns-text-muted">{x[hover]}</div>
          {series.map((s) => {
            const v = s.values[hover];
            return (
              <div key={s.id} className="flex items-center justify-between gap-3 font-mono text-[10.5px]">
                <span className="inline-flex items-center gap-1 text-ns-text-muted">
                  <span className="inline-block size-1.5 rounded-full" style={{ background: s.color }} />
                  {s.label}
                </span>
                <span className="ns-tabular text-ns-text">{v === null || v === undefined || !Number.isFinite(v) ? "—" : fmtV(v, s, hover)}</span>
              </div>
            );
          })}
          {tooltipExtra ? tooltipExtra(hover) : null}
        </div>
      ) : null}
    </div>
  );
}
