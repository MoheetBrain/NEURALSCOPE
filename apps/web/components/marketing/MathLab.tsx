import type { ReactNode } from "react";
import { Check } from "lucide-react";
import { EpistemicBadge } from "@/components/ui/EpistemicBadge";
import { StatusChip } from "@/components/ui/StatusChip";
import { cn } from "@/lib/cn";

function Step({
  n,
  label,
  children,
  last,
  className,
}: {
  n: number;
  label: string;
  children: ReactNode;
  last?: boolean;
  className?: string;
}) {
  return (
    <li className={cn("relative flex min-w-0 flex-col rounded-[var(--ns-radius)] border border-ns-border bg-ns-panel", className)}>
      {!last ? (
        <svg
          aria-hidden="true"
          viewBox="0 0 12 10"
          className="absolute -right-[13px] top-[52px] z-10 hidden h-2.5 w-3 text-ns-text-faint lg:block"
          fill="none"
        >
          <path d="M0 5h10M7 1.5 10.5 5 7 8.5" stroke="currentColor" strokeWidth="1" />
        </svg>
      ) : null}
      <div className="flex h-8 items-center gap-2 border-b border-ns-border px-3">
        <span className="font-mono text-[10px] text-ns-text-faint">{String(n).padStart(2, "0")}</span>
        <span className="ns-label truncate">{label}</span>
      </div>
      <div className="flex flex-1 flex-col gap-2 p-3 font-mono text-[12px] leading-relaxed">{children}</div>
    </li>
  );
}

const MINI = [
  [0.1, 0.12, 0.08, 0.1, 0.14, 0.1, 0.09],
  [0.12, 0.1, 0.15, 0.1, 0.1, 0.12, 0.1],
  [0.1, 0.18, 0.12, 0.2, 0.15, 0.1, 0.16],
  [0.2, 0.3, 0.2, 0.35, 0.25, 0.2, 0.3],
  [0.35, 0.5, 0.3, 0.5, 0.45, 0.3, 0.55],
  [0.5, 0.7, 0.45, 0.6, 0.65, 0.4, 0.8],
];

export function MathLabPipeline() {
  return (
    <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4" aria-label="Math Lab pipeline">
      <Step n={1} label="Problem">
        <span className="text-ns-text">Compute 37 + 48.</span>
        <span className="text-ns-text-faint">answer is machine-checkable</span>
      </Step>
      <Step n={2} label="Model solution">
        <span className="text-ns-text">37 + 48 = 85</span>
        <span className="text-ns-text-faint">generated under instrumentation</span>
      </Step>
      <Step n={3} label="Verification">
        <span className="text-ns-text-muted">
          sympy: <span className="text-ns-text">Eq(85, 37 + 48)</span>
        </span>
        <span className="flex flex-wrap gap-1.5">
          <StatusChip tone="ok" icon={<Check aria-hidden="true" />}>
            Correct
          </StatusChip>
          <StatusChip tone="neutral">SymPy oracle</StatusChip>
        </span>
      </Step>
      <Step n={4} label="Neural trace" last>
        <span aria-hidden="true" className="grid w-fit grid-cols-7 gap-px">
          {MINI.flatMap((row, d) =>
            row.map((v, t) => (
              <span
                key={`${d}-${t}`}
                className="h-1.5 w-3.5 rounded-[1px]"
                style={{ background: `color-mix(in oklab, var(--ns-accent) ${Math.round(v * 100)}%, var(--ns-panel-2))` }}
              />
            )),
          )}
        </span>
        <span className="flex flex-wrap items-center gap-1.5 text-ns-text-faint">
          linked to the run <EpistemicBadge status="observed" size="xs" />
        </span>
      </Step>
    </ol>
  );
}

const METHOD = [
  { k: "Compare", v: "Run matched problems the model solves and fails. Align the two trajectories token by token, layer by layer." },
  { k: "Locate", v: "Find where their internal states first diverge: residual stream, attention, Logit Lens readouts." },
  { k: "Intervene", v: "Ablate or patch the diverging state, e.g. copy it from the successful run into the failed one." },
  { k: "Test rescue", v: "Rerun and verify with the oracle. Did the failed computation recover? How often, across controls?" },
];

export function MathLabMethod() {
  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)] lg:gap-12">
      <ol className="divide-y divide-ns-border border-y border-ns-border">
        {METHOD.map((m, i) => (
          <li key={m.k} className="grid grid-cols-[28px_110px_minmax(0,1fr)] gap-3 py-4 max-sm:grid-cols-[28px_minmax(0,1fr)]">
            <span className="font-mono text-[11px] text-ns-text-faint">{String(i + 1).padStart(2, "0")}</span>
            <span className="font-mono text-[12px] uppercase tracking-[0.12em] text-ns-text">{m.k}</span>
            <span className="text-[14px] leading-relaxed text-ns-text-muted max-sm:col-start-2">{m.v}</span>
          </li>
        ))}
      </ol>
      <TrajectorySchematic />
    </div>
  );
}

function TrajectorySchematic() {
  // Two trajectories through depth that coincide, then diverge at d*.
  const correct = "M8 70 C 60 68, 110 74, 160 70 S 230 66, 250 64 S 330 40, 392 30";
  const failed = "M8 72 C 60 70, 110 76, 160 72 S 230 70, 250 70 S 330 104, 392 118";
  const ticks = Array.from({ length: 29 }, (_, i) => 8 + (i * 384) / 28);
  return (
    <figure className="rounded-[var(--ns-radius)] border border-ns-border bg-ns-panel">
      <div className="flex h-8 items-center gap-2 border-b border-ns-border px-3">
        <span className="ns-label">Trajectory comparison</span>
        <StatusChip tone="neutral" className="ml-auto">
          Schematic
        </StatusChip>
      </div>
      <div className="p-4">
        <svg viewBox="0 0 400 150" className="h-auto w-full" role="img" aria-label="Schematic: a correct and a failed run share internal states until a divergence depth, after which a patch from the correct run is tested on the failed run.">
          {ticks.map((x, i) => (
            <path key={i} d={`M${x} 138v${i % 4 === 0 ? 6 : 3}`} stroke="var(--ns-border-strong)" strokeWidth="1" />
          ))}
          <path d="M8 138h384" stroke="var(--ns-border-strong)" strokeWidth="1" />
          <path d="M250 10v128" stroke="var(--ns-text-faint)" strokeWidth="1" strokeDasharray="3 3" />
          <path d={correct} stroke="var(--ns-ok)" strokeWidth="2" fill="none" />
          <path d={failed} stroke="var(--ns-danger)" strokeWidth="2" fill="none" />
          <path
            d="M300 52 C 318 70, 318 84, 304 98"
            stroke="var(--ns-text)"
            strokeWidth="1.25"
            strokeDasharray="3 2.5"
            fill="none"
          />
          <path d="M300 92.5 304 99l6.5-3" stroke="var(--ns-text)" strokeWidth="1.25" fill="none" />
          <g transform="translate(250 67)">
            <circle r="7" stroke="var(--ns-text)" strokeWidth="1.25" fill="var(--ns-panel)" />
            <path d="M0 -12v5M0 7v5M-12 0h5M7 0h5" stroke="var(--ns-text)" strokeWidth="1.25" />
          </g>
        </svg>
        <ul className="mt-3 flex flex-wrap gap-x-5 gap-y-2 font-mono text-[11px] text-ns-text-muted">
          <li className="flex items-center gap-2">
            <span aria-hidden="true" className="h-0.5 w-4 bg-ns-ok" /> correct run
          </li>
          <li className="flex items-center gap-2">
            <span aria-hidden="true" className="h-0.5 w-4 bg-ns-danger" /> failed run
          </li>
          <li className="flex items-center gap-2">
            <span aria-hidden="true" className="h-3 w-0 border-l border-dashed border-ns-text-faint" /> first divergence
          </li>
          <li className="flex items-center gap-2">
            <span aria-hidden="true" className="h-0 w-4 border-t border-dashed border-ns-text" /> patch → rerun
          </li>
          <li className="text-ns-text-faint">x-axis: depth (emb → final layer)</li>
        </ul>
      </div>
    </figure>
  );
}

export function ProofFuture() {
  return (
    <div className="ns-hatch relative rounded-[var(--ns-radius)] border border-dashed border-ns-border-strong p-5 sm:p-8">
      <div className="grid gap-8 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] md:items-center">
        <div>
          <StatusChip tone="future">Future — not in V1</StatusChip>
          <p className="mt-5 text-[20px] font-light leading-snug text-ns-text sm:text-[24px]">
            Eventually: formal proof state <span className="text-ns-text-faint">↕</span> neural state
          </p>
          <p className="mt-4 max-w-[52ch] text-[14px] leading-relaxed text-ns-text-muted">
            A research direction, not a shipped capability. V1 checks final answers with a SymPy oracle on problems with
            computable answers. It does not prove theorems or read proof states.
          </p>
        </div>
        <div aria-hidden="true" className="mx-auto grid w-full max-w-sm justify-items-center gap-2 opacity-80">
          <div className="w-full rounded-[var(--ns-radius)] border border-dashed border-ns-border-strong bg-ns-bg/85 px-4 py-3">
            <p className="ns-label">Formal proof state</p>
            <p className="mt-2 font-mono text-[12px] text-ns-text-faint">goals · hypotheses · tactic steps</p>
          </div>
          <span className="font-mono text-lg leading-none text-ns-text-faint">↕</span>
          <div className="w-full rounded-[var(--ns-radius)] border border-dashed border-ns-border-strong bg-ns-bg/85 px-4 py-3">
            <p className="ns-label">Neural state</p>
            <p className="mt-2 font-mono text-[12px] text-ns-text-faint">activations · heads · features</p>
          </div>
        </div>
      </div>
    </div>
  );
}
