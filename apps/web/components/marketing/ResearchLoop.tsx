import { EpistemicBadge } from "@/components/ui/EpistemicBadge";
import type { EpistemicStatus } from "@/lib/epistemic";
import { cn } from "@/lib/cn";

interface LoopStep {
  verb: string;
  body: string;
  status?: EpistemicStatus;
  color: string;
  tag?: string;
}

export const LOOP: LoopStep[] = [
  {
    verb: "Observe",
    body: "Run the model on a prompt and capture residual stream, attention and MLP states with forward hooks.",
    status: "observed",
    color: "var(--ns-observed)",
  },
  {
    verb: "Interpret",
    body: "Decode captured states into readable views: Logit Lens, attention patterns, direct logit attribution.",
    status: "derived",
    color: "var(--ns-derived)",
  },
  {
    verb: "Hypothesise",
    body: "Propose what a component computes, e.g. “L17.H5 moves carry information to the answer position”.",
    status: "inferred",
    color: "var(--ns-inferred)",
  },
  {
    verb: "Intervene",
    body: "Ablate, scale or patch exactly that component in a rerun. The spec is stored with the run so it can be replayed.",
    color: "var(--ns-danger)",
    tag: "edit",
  },
  {
    verb: "Measure",
    body: "Compare against the clean run: target probability and rank, KL divergence, and the generated output.",
    status: "observed",
    color: "var(--ns-observed)",
  },
  {
    verb: "Causally validate",
    body: "Repeat over controlled examples and against baselines. Only effects that survive are labelled causally tested.",
    status: "causal",
    color: "var(--ns-causal)",
  },
  {
    verb: "Improve",
    body: "Refine the hypothesis or the method, record negative results, and go around again.",
    color: "var(--ns-accent-2)",
    tag: "iterate",
  },
];

function LoopRing() {
  const cx = 160;
  const cy = 160;
  const r = 118;
  const n = LOOP.length;
  const pts = LOOP.map((_, i) => {
    const a = -Math.PI / 2 + (i * 2 * Math.PI) / n;
    return { x: cx + r * Math.cos(a), y: cy + r * Math.sin(a), a };
  });
  return (
    <svg viewBox="0 0 320 320" className="mx-auto h-auto w-full max-w-[320px]" aria-hidden="true">
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--ns-border-strong)" strokeWidth="1" />
      <circle cx={cx} cy={cy} r={r - 26} fill="none" stroke="var(--ns-border)" strokeWidth="1" strokeDasharray="2 4" />
      {/* arrowheads at arc midpoints (clockwise) */}
      {pts.map((p, i) => {
        const a = p.a + Math.PI / n;
        const x = cx + r * Math.cos(a);
        const y = cy + r * Math.sin(a);
        const deg = (a * 180) / Math.PI + 90;
        return (
          <path
            key={`arrow-${i}`}
            d="M-3.5 -3 0 0.5-3.5 4"
            transform={`translate(${x} ${y}) rotate(${deg})`}
            fill="none"
            stroke="var(--ns-text-muted)"
            strokeWidth="1.25"
          />
        );
      })}
      {pts.map((p, i) => (
        <g key={LOOP[i].verb} transform={`translate(${p.x} ${p.y})`}>
          <circle r="15" fill="var(--ns-bg)" stroke={LOOP[i].color} strokeWidth="1.25" />
          <text
            textAnchor="middle"
            dominantBaseline="central"
            fontFamily="var(--font-mono)"
            fontSize="10.5"
            fill="var(--ns-text)"
          >
            {String(i + 1).padStart(2, "0")}
          </text>
        </g>
      ))}
      <text x={cx} y={cy - 8} textAnchor="middle" fontFamily="var(--font-mono)" fontSize="10" letterSpacing="2" fill="var(--ns-text-muted)">
        RESEARCH LOOP
      </text>
      <text x={cx} y={cy + 12} textAnchor="middle" fontFamily="var(--font-mono)" fontSize="9.5" fill="var(--ns-text-faint)">
        measure → claim → test
      </text>
    </svg>
  );
}

export function ResearchLoop() {
  return (
    <div className="grid items-start gap-10 lg:grid-cols-[320px_minmax(0,1fr)] lg:gap-16">
      <div className="lg:sticky lg:top-24">
        <LoopRing />
      </div>
      <ol className="divide-y divide-ns-border border-y border-ns-border">
        {LOOP.map((s, i) => (
          <li key={s.verb} className="grid grid-cols-[32px_minmax(0,1fr)] gap-x-3 py-5 sm:grid-cols-[32px_180px_minmax(0,1fr)]">
            <span className="font-mono text-[11px] leading-6 text-ns-text-faint">{String(i + 1).padStart(2, "0")}</span>
            <span className="flex flex-col items-start gap-2">
              <span className="font-mono text-[13px] uppercase leading-6 tracking-[0.12em] text-ns-text">{s.verb}</span>
              {s.status ? (
                <EpistemicBadge status={s.status} size="xs" />
              ) : (
                <span
                  className={cn(
                    "inline-flex h-4 items-center rounded-[3px] border border-dashed border-ns-border-strong px-1 font-mono text-[9px] uppercase tracking-[0.1em] text-ns-text-faint",
                  )}
                >
                  {s.tag}
                </span>
              )}
            </span>
            <p className="col-start-2 mt-2 text-[15px] leading-relaxed text-ns-text-muted sm:col-start-3 sm:mt-0">{s.body}</p>
          </li>
        ))}
      </ol>
    </div>
  );
}
