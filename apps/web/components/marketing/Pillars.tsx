import type { ReactNode } from "react";
import { EpistemicBadge } from "@/components/ui/EpistemicBadge";
import type { EpistemicStatus } from "@/lib/epistemic";
import { cn } from "@/lib/cn";

interface Pillar {
  verb: string;
  body: string;
  glyph: ReactNode;
  yields: { prefix: string; status: EpistemicStatus };
}

const stroke = { stroke: "currentColor", strokeWidth: 1.25, fill: "none" } as const;

const PILLARS: Pillar[] = [
  {
    verb: "Observe",
    body: "Capture the actual internal states produced during inference.",
    yields: { prefix: "Yields", status: "observed" },
    glyph: (
      <svg viewBox="0 0 96 48" className="h-12 w-24" aria-hidden="true">
        <path d="M0 24h96" stroke="var(--ns-border-strong)" strokeWidth="1" />
        <path
          d="M2 30C10 30 12 12 20 12s10 26 18 26 10-20 18-20 10 14 18 14 10-8 18-8"
          {...stroke}
          className="text-ns-text-muted"
        />
        {[20, 38, 56, 74].map((x, i) => (
          <circle key={x} cx={x} cy={[12, 38, 18, 32][i]} r="2.25" fill="var(--ns-observed)" />
        ))}
        <path d="M56 2v44" stroke="var(--ns-observed)" strokeWidth="1" strokeDasharray="2 2" />
      </svg>
    ),
  },
  {
    verb: "Decode",
    body: "Translate high-dimensional representations into experimentally useful views.",
    yields: { prefix: "Yields", status: "derived" },
    glyph: (
      <svg viewBox="0 0 96 48" className="h-12 w-24" aria-hidden="true">
        {Array.from({ length: 16 }, (_, i) => (
          <rect
            key={i}
            x={2 + (i % 4) * 9}
            y={6 + Math.floor(i / 4) * 9}
            width="7"
            height="7"
            rx="1"
            fill="var(--ns-text-muted)"
            opacity={[0.25, 0.6, 0.35, 0.9, 0.5, 0.2, 0.75, 0.3, 0.4, 0.85, 0.25, 0.55, 0.7, 0.35, 0.2, 0.6][i]}
          />
        ))}
        <path d="M44 24h14M54 20.5 58 24l-4 3.5" {...stroke} className="text-ns-text-faint" />
        {[34, 22, 14, 8].map((w, i) => (
          <rect key={i} x="64" y={7 + i * 9} width={w} height="5" rx="1" fill="var(--ns-derived)" opacity={1 - i * 0.2} />
        ))}
      </svg>
    ),
  },
  {
    verb: "Intervene",
    body: "Ablate, patch and manipulate components of the computation.",
    yields: { prefix: "Acts on", status: "inferred" },
    glyph: (
      <svg viewBox="0 0 96 48" className="h-12 w-24" aria-hidden="true">
        <g className="text-ns-text-muted" {...stroke}>
          <path d="M10 24 40 10M10 24l30 14M40 10l32 0M40 38l32 0M40 10l32 28" />
        </g>
        <path d="M40 38 72 10" stroke="var(--ns-danger)" strokeWidth="1.25" strokeDasharray="3 2.5" fill="none" />
        <path d="M52 20l8 8M60 20l-8 8" stroke="var(--ns-danger)" strokeWidth="1.5" />
        {[
          [10, 24],
          [40, 10],
          [40, 38],
          [72, 10],
          [72, 38],
        ].map(([x, y]) => (
          <circle key={`${x}-${y}`} cx={x} cy={y} r="3" fill="var(--ns-panel)" stroke="var(--ns-text)" strokeWidth="1.25" />
        ))}
      </svg>
    ),
  },
  {
    verb: "Verify",
    body: "Rerun the model and measure whether the proposed mechanism actually mattered.",
    yields: { prefix: "Yields", status: "causal" },
    glyph: (
      <svg viewBox="0 0 96 48" className="h-12 w-24" aria-hidden="true">
        <path d="M2 44h92" stroke="var(--ns-border-strong)" strokeWidth="1" />
        <rect x="14" y="8" width="16" height="36" rx="1" fill="var(--ns-text-muted)" opacity="0.7" />
        <rect x="40" y="31" width="16" height="13" rx="1" fill="var(--ns-causal)" />
        <path d="M66 8v23M62 12l4-4 4 4M62 27l4 4 4-4" stroke="var(--ns-text-faint)" strokeWidth="1" fill="none" />
        <text x="74" y="23" fontSize="9" fontFamily="var(--font-mono)" fill="var(--ns-text-muted)">
          Δp
        </text>
      </svg>
    ),
  },
];

export function Pillars() {
  return (
    <ol className="grid gap-px border-y border-ns-border bg-ns-border sm:grid-cols-2 lg:grid-cols-4">
      {PILLARS.map((p, i) => (
        <li
          key={p.verb}
          className={cn(
            "flex flex-col bg-ns-bg py-8 sm:pr-6",
            i % 2 === 0 ? "sm:pl-0" : "sm:pl-6",
            i === 0 ? "lg:pl-0" : "lg:pl-6",
          )}
        >
          <div className="text-ns-text">{p.glyph}</div>
          <p className="mt-8 flex items-baseline gap-3">
            <span className="font-mono text-[11px] text-ns-text-faint">{String(i + 1).padStart(2, "0")}</span>
            <span className="font-mono text-[15px] font-medium uppercase tracking-[0.14em] text-ns-text">{p.verb}</span>
          </p>
          <p className="mt-3 max-w-[34ch] text-[15px] leading-relaxed text-ns-text-muted">{p.body}</p>
          <p className="mt-auto flex items-center gap-2 pt-8 font-mono text-[10px] uppercase tracking-[0.12em] text-ns-text-faint">
            {p.yields.prefix} <EpistemicBadge status={p.yields.status} size="xs" />
          </p>
        </li>
      ))}
    </ol>
  );
}
