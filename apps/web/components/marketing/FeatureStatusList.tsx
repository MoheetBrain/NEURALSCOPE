import { StatusChip, type StatusChipTone } from "@/components/ui/StatusChip";
import { cn } from "@/lib/cn";

interface Group {
  key: string;
  label: string;
  tone: StatusChipTone;
  meaning: string;
  items: { name: string; note: string }[];
}

/**
 * What runs today and what does not. Every entry carries its marking; nothing in the
 * "available" group is a plan, and nothing in the roadmap group is implied to work.
 */
export const FEATURE_GROUPS: Group[] = [
  {
    key: "available",
    label: "Available now",
    tone: "ok",
    meaning: "Implemented, tested, running on a local model.",
    items: [
      { name: "Activation capture", note: "PyTorch forward hooks on a locally loaded Qwen2.5 model; every run is stored and deep-linkable." },
      { name: "Token × layer explorer", note: "Logit Lens, attention patterns, direct logit attribution, per-cell inspection." },
      { name: "Interventions", note: "Head and MLP ablation, residual patching and steering, head and patch sweeps, with before/after diffs." },
      { name: "Math Lab", note: "Generated arithmetic benchmarks checked by a SymPy oracle; compare correct and failed runs; evaluate interventions." },
    ],
  },
  {
    key: "prototype",
    label: "Research prototypes",
    tone: "paper",
    meaning: "Runs end to end; methods and outputs may still change. Each template reports its own status in the workspace.",
    items: [
      { name: "SelfLab claim audits", note: "Select a span of a response, commit a pre-registered experiment, and record evidence with kind, status and scope kept apart." },
      { name: "Counterfactual context branches", note: "Rerun one case with the document removed, replaced or a hint added; compare outputs side by side." },
      { name: "Constrained source reports", note: "Ask the target where its answer came from; read the report two ways and compare it with observers that see only the prompt or transcript." },
      { name: "Reproduction export", note: "Spec and plan hashes, provenance, trial table, analysis and evidence in one JSON record; tensors by reference." },
    ],
  },
  {
    key: "fixture",
    label: "Fixtures",
    tone: "warn",
    meaning: "For exercising the interface only.",
    items: [{ name: "DEMO model", note: "A tiny random-weight model so the app runs without weights. Every workspace screen then shows a DEMO banner; its outputs mean nothing." }],
  },
  {
    key: "api_only",
    label: "API-only views",
    tone: "neutral",
    meaning: "Black-box only.",
    items: [{ name: "Transcripts from API-only models", note: "No activations, so only black-box comparisons. Disabled unless the backend is configured for it." }],
  },
  {
    key: "unsupported",
    label: "Unsupported",
    tone: "danger",
    meaning: "Deliberately not offered.",
    items: [
      { name: "LLM investigator", note: "A language model proposing experiments. Unsupported until approved and configured; suggestions today are deterministic rules." },
      { name: "Sentience or 'true thought' readouts", note: "None exist here. No instrument in NeuralScope measures them and none is shown." },
    ],
  },
  {
    key: "roadmap",
    label: "Roadmap",
    tone: "future",
    meaning: "Planned, not built.",
    items: [
      { name: "Feature debugger", note: "Sparse autoencoders, transcoders, probes." },
      { name: "Circuit compiler", note: "Attribution graphs and automated circuit discovery with causal validation." },
      { name: "Agent flight recorder", note: "Recording and auditing multi-step agent runs." },
    ],
  },
];

export function FeatureStatusList({ className }: { className?: string }) {
  return (
    <div className={cn("grid gap-3 md:grid-cols-2 lg:grid-cols-3", className)}>
      {FEATURE_GROUPS.map((g) => (
        <section
          key={g.key}
          aria-labelledby={`fs-${g.key}`}
          data-testid={`feature-group-${g.key}`}
          className={cn(
            "flex flex-col rounded-[var(--ns-radius)] border p-5",
            g.key === "roadmap" || g.key === "unsupported" ? "border-dashed border-ns-border-strong" : "border-ns-border bg-ns-panel",
          )}
        >
          <div className="flex flex-wrap items-center gap-2">
            <StatusChip tone={g.tone} dot={g.key === "available"}>
              {g.label}
            </StatusChip>
          </div>
          <h3 id={`fs-${g.key}`} className="sr-only">
            {g.label}
          </h3>
          <p className="mt-3 text-[13px] leading-snug text-ns-text-faint">{g.meaning}</p>
          <ul className="mt-4 space-y-3 border-t border-ns-border pt-4">
            {g.items.map((it) => (
              <li key={it.name}>
                <p className="text-[14px] text-ns-text">{it.name}</p>
                <p className="mt-0.5 text-[13px] leading-snug text-ns-text-muted">{it.note}</p>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}
