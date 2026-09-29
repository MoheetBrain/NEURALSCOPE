"use client";

import { useEffect, useRef, useState } from "react";
import { fmtNum } from "@/lib/format";
import { useElementSize } from "./useElementSize";

/**
 * All d_model coordinates of a residual vector as a compact bar strip (canvas).
 * Bars grow up (positive) / down (negative) from a centre line; the y range is
 * symmetric (±max|v|). Hover reads out the dimension and its value.
 */
export function VectorStrip({
  values,
  height = 72,
  color = "#7cc7f5",
  highlight,
  ariaLabel,
}: {
  values: readonly (number | null)[];
  height?: number;
  color?: string;
  highlight?: readonly number[];
  ariaLabel: string;
}) {
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<{ dim: number; x: number } | null>(null);
  const width = Math.max(100, Math.floor(size.width));
  const n = values.length;

  useEffect(() => {
    const c = canvasRef.current;
    if (!c || size.width === 0) return;
    const dpr = Math.min(3, window.devicePixelRatio || 1);
    c.width = Math.round(width * dpr);
    c.height = Math.round(height * dpr);
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    let m = 0;
    for (const v of values) if (v !== null && Math.abs(v) > m) m = Math.abs(v);
    const mid = height / 2;
    ctx.fillStyle = "#22262d";
    ctx.fillRect(0, mid - 0.5, width, 1);
    const bw = width / Math.max(1, n);
    const hl = new Set(highlight ?? []);
    for (let i = 0; i < n; i++) {
      const v = values[i];
      if (v === null || m === 0) continue;
      const h = (Math.abs(v) / m) * (mid - 2);
      ctx.fillStyle = hl.has(i) ? "#d7c9a6" : color;
      ctx.globalAlpha = hl.has(i) ? 1 : 0.8;
      ctx.fillRect(i * bw, v >= 0 ? mid - h : mid, Math.max(bw, 0.75), Math.max(h, 0.5));
    }
    ctx.globalAlpha = 1;
    if (hover) {
      ctx.fillStyle = "rgba(231,233,236,0.5)";
      ctx.fillRect(hover.dim * bw, 0, Math.max(1, bw), height);
    }
  }, [values, width, height, color, highlight, hover, n, size.width]);

  const hv = hover ? values[hover.dim] : null;
  return (
    <div ref={wrapRef} className="relative w-full">
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={ariaLabel}
        style={{ width, height }}
        className="block cursor-crosshair rounded-[2px] bg-ns-bg"
        onMouseMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          const x = e.clientX - r.left;
          const dim = Math.min(n - 1, Math.max(0, Math.floor((x / r.width) * n)));
          setHover({ dim, x });
        }}
        onMouseLeave={() => setHover(null)}
      />
      {hover ? (
        <div
          className="pointer-events-none absolute -top-6 z-10 whitespace-nowrap rounded-[3px] border border-ns-border-strong bg-ns-bg/95 px-1.5 py-0.5 font-mono text-[10px] text-ns-text"
          style={{ left: Math.min(Math.max(0, hover.x - 40), width - 110) }}
        >
          dim {hover.dim} · {fmtNum(hv, 4)}
        </div>
      ) : null}
    </div>
  );
}

/** Histogram of vector values (SVG bars). */
export function Histogram({
  counts,
  min,
  max,
  height = 56,
  color = "#7cc7f5",
}: {
  counts: readonly number[];
  min: number | null;
  max: number | null;
  height?: number;
  color?: string;
}) {
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const width = Math.max(100, size.width);
  const peak = Math.max(1, ...counts);
  const bw = width / Math.max(1, counts.length);
  const zeroX = min !== null && max !== null && min < 0 && max > 0 ? ((0 - min) / (max - min)) * width : null;
  return (
    <div ref={wrapRef} className="w-full">
      {size.width > 0 ? (
        <svg width={width} height={height + 14} role="img" aria-label={`Histogram of ${counts.reduce((a, b) => a + b, 0)} values from ${fmtNum(min, 3)} to ${fmtNum(max, 3)}`}>
          {counts.map((c, i) => {
            const h = Math.sqrt(c / peak) * height; // sqrt: heavy-tailed residual values stay visible
            return <rect key={i} x={i * bw + 0.5} y={height - h} width={Math.max(1, bw - 1)} height={h} rx={1} fill={color} opacity={0.85} />;
          })}
          {zeroX !== null ? <line x1={zeroX} x2={zeroX} y1={0} y2={height} stroke="var(--ns-border-strong)" /> : null}
          <line x1={0} x2={width} y1={height + 0.5} y2={height + 0.5} stroke="var(--ns-border-strong)" />
          <text x={0} y={height + 11} className="fill-ns-text-faint font-mono text-[9.5px]">
            {fmtNum(min, 2)}
          </text>
          {zeroX !== null ? (
            <text x={zeroX} y={height + 11} textAnchor="middle" className="fill-ns-text-faint font-mono text-[9.5px]">
              0
            </text>
          ) : null}
          <text x={width} y={height + 11} textAnchor="end" className="fill-ns-text-faint font-mono text-[9.5px]">
            {fmtNum(max, 2)}
          </text>
        </svg>
      ) : (
        <div style={{ height: height + 14 }} />
      )}
    </div>
  );
}

/** Horizontal probability bar with token label (lens tables, query rows, top-k). */
export function ProbBar({
  value,
  max = 1,
  color = "var(--ns-derived)",
  className,
}: {
  value: number | null;
  max?: number;
  color?: string;
  className?: string;
}) {
  const w = value === null || !Number.isFinite(value) || max <= 0 ? 0 : Math.min(1, Math.max(0, value / max));
  return (
    <span className={`relative inline-block h-1.5 overflow-hidden rounded-full bg-ns-border ${className ?? ""}`}>
      <span className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${w * 100}%`, background: color }} />
    </span>
  );
}
