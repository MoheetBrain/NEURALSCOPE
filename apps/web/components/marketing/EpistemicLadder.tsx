import { EpistemicBadge } from "@/components/ui/EpistemicBadge";
import { Panel } from "@/components/ui/Panel";
import { EPISTEMIC, EPISTEMIC_ORDER } from "@/lib/epistemic";
import { cn } from "@/lib/cn";

const BAR: Record<string, string> = {
  observed: "bg-ns-observed",
  derived: "bg-ns-derived",
  inferred: "bg-ns-inferred",
  causal: "bg-ns-causal",
};

// Staircase offsets (lg): each rung sits higher than the last.
const OFFSET = ["lg:mt-24", "lg:mt-16", "lg:mt-8", "lg:mt-0"];

export function EpistemicLadder() {
  return (
    <div>
      <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 lg:items-start lg:gap-4">
        {EPISTEMIC_ORDER.map((s, i) => {
          const info = EPISTEMIC[s];
          return (
            <li key={s} className={cn("relative", OFFSET[i])}>
              <article className="relative h-full overflow-hidden rounded-[var(--ns-radius)] border border-ns-border bg-ns-panel">
                <span aria-hidden="true" className={cn("absolute inset-x-0 top-0 h-[2px]", BAR[s])} />
                <div className="flex items-center justify-between gap-3 px-4 pt-5">
                  <EpistemicBadge status={s} />
                  <span className="font-mono text-[10px] text-ns-text-faint">{String(i + 1).padStart(2, "0")}</span>
                </div>
                <p className="px-4 pt-4 text-[14px] leading-relaxed text-ns-text">{info.definition}</p>
                <div className="mx-4 mb-4 mt-5 border-t border-dashed border-ns-border pt-3">
                  <p className="ns-label text-ns-text-faint">Example</p>
                  <p className="mt-2 font-mono text-[12px] leading-relaxed text-ns-text-muted">{info.example}</p>
                </div>
              </article>
            </li>
          );
        })}
      </ol>

      {/* axis */}
      <div aria-hidden="true" className="mt-6 hidden items-center gap-3 font-mono text-[10px] uppercase tracking-[0.14em] text-ns-text-faint lg:flex">
        <span>Measurement</span>
        <span className="relative h-px flex-1 bg-ns-border-strong">
          <span className="absolute -right-px -top-[3.5px] size-2 rotate-45 border-r border-t border-ns-border-strong" />
        </span>
        <span>Mechanism</span>
      </div>
    </div>
  );
}

/** A mock view header showing how labels appear in the debugger. */
export function LabelledViewExample() {
  return (
    <Panel
      label="Residual stream"
      right={
        <span className="truncate font-mono text-[10.5px] text-ns-text-faint">
          <span className="text-ns-text-muted">resid[L14, t7]</span> · illustrative values
        </span>
      }
      padded={false}
      className="text-[13px]"
    >
      <dl className="divide-y divide-ns-border">
        {[
          { k: "h[L14, t7]", v: "float32[1536]", status: "observed" as const, note: "captured by forward hook" },
          { k: "Logit Lens top-1", v: "“8”  p=0.14", status: "derived" as const, note: "W_U · norm(h)" },
          {
            k: "Auto-description",
            v: "“may track the tens-digit carry”",
            status: "inferred" as const,
            note: "LLM-generated · hypothesis",
          },
          { k: "Ablate L17.H5", v: "p(“8”) 0.71 → 0.18", status: "causal" as const, note: "n=1 prompt" },
        ].map((row) => (
          <div key={row.k} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-1 px-3 py-2.5 sm:grid-cols-[150px_minmax(0,1fr)_auto]">
            <dt className="font-mono text-[11px] text-ns-text-muted">{row.k}</dt>
            <dd className="order-3 col-span-2 min-w-0 truncate font-mono text-[12px] text-ns-text sm:order-none sm:col-span-1">
              {row.v} <span className="text-ns-text-faint">· {row.note}</span>
            </dd>
            <dd className="justify-self-end">
              <EpistemicBadge status={row.status} size="xs" />
            </dd>
          </div>
        ))}
      </dl>
    </Panel>
  );
}
