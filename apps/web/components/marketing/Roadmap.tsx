import { StatusChip } from "@/components/ui/StatusChip";
import { cn } from "@/lib/cn";

const PHASES = [
  {
    v: "V1",
    name: "Neural debugger",
    now: true,
    items: ["Activation capture", "Token × layer explorer", "Logit Lens", "Attention", "Causal ablation", "Math Lab"],
  },
  {
    v: "V2",
    name: "Feature debugger",
    items: ["Sparse autoencoders (SAEs)", "Transcoders", "Probes", "Automated activation patching", "Representation geometry"],
  },
  {
    v: "V3",
    name: "Circuit compiler",
    items: ["Attribution graphs", "Automated circuit discovery", "Causal validation", "Model diffing"],
  },
  {
    v: "V4",
    name: "Agent + training control",
    items: [
      "Autonomous interpretability scientist",
      "Agent flight recorder",
      "Pre-action Neural Interlock",
      "Training-time feedback",
    ],
  },
];

export function Roadmap() {
  return (
    <div>
      {/* rail */}
      <div aria-hidden="true" className="relative mb-6 hidden h-3 lg:block">
        <span className="absolute inset-x-0 top-1/2 h-px bg-ns-border-strong" />
        <span className="absolute left-0 top-1/2 h-px w-[calc(25%-12px)] bg-ns-accent" />
        {PHASES.map((p, i) => (
          <span
            key={p.v}
            className={cn(
              "absolute top-1/2 size-2.5 -translate-y-1/2 rounded-full border",
              p.now ? "border-ns-accent bg-ns-accent" : "border-ns-border-strong bg-ns-bg",
            )}
            style={{ left: `calc(${i * 25}% + ${i * 4 + 20}px)` }}
          />
        ))}
      </div>

      <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4 lg:gap-4">
        {PHASES.map((p) => (
          <li
            key={p.v}
            className={cn(
              "flex flex-col rounded-[var(--ns-radius)] border p-5",
              p.now ? "border-ns-accent/45 bg-ns-panel shadow-[inset_0_1px_0_0_var(--ns-accent)]" : "border-dashed border-ns-border-strong bg-transparent",
            )}
          >
            <div className="flex flex-wrap items-center gap-2">
              <span className={cn("font-mono text-[13px] font-medium", p.now ? "text-ns-accent" : "text-ns-text-muted")}>{p.v}</span>
              {p.now ? (
                <StatusChip tone="accent" dot="pulse">
                  Building now / V1
                </StatusChip>
              ) : (
                <StatusChip tone="future">Future</StatusChip>
              )}
            </div>
            <h3 className={cn("mt-4 text-[19px] font-medium leading-tight", p.now ? "text-ns-text" : "text-ns-text-muted")}>
              {p.name}
            </h3>
            <ul className="mt-5 space-y-2 border-t border-ns-border pt-4">
              {p.items.map((it) => (
                <li
                  key={it}
                  className={cn("flex gap-2.5 text-[13.5px] leading-snug", p.now ? "text-ns-text-muted" : "text-ns-text-faint")}
                >
                  <span aria-hidden="true" className={cn("mt-[7px] size-1 shrink-0 rounded-full", p.now ? "bg-ns-accent" : "bg-ns-text-faint")} />
                  {it}
                </li>
              ))}
            </ul>
          </li>
        ))}
      </ol>
    </div>
  );
}
