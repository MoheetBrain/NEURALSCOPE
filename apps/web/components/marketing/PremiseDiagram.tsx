import type { ReactNode } from "react";
import { LogoMark } from "@/components/brand/Logo";
import { cn } from "@/lib/cn";

type NodeKind = "plain" | "known" | "unknown";

interface LaneNode {
  actor: string;
  verb: string;
  object: string;
  kind: NodeKind;
}

const LANES: { label: string; sub: string; nodes: LaneNode[]; tail?: ReactNode }[] = [
  {
    label: "Classical software",
    sub: "the algorithm is written down",
    nodes: [
      { actor: "Human", verb: "writes", object: "algorithm", kind: "plain" },
      { actor: "Computer", verb: "executes", object: "algorithm", kind: "known" },
    ],
  },
  {
    label: "Learned systems",
    sub: "the algorithm is discovered",
    nodes: [
      { actor: "Human", verb: "specifies", object: "objective", kind: "plain" },
      { actor: "Training", verb: "discovers", object: "algorithm", kind: "plain" },
      { actor: "Humans", verb: "do not completely understand", object: "the learned computation", kind: "unknown" },
    ],
  },
];

function Arrow() {
  return (
    <span aria-hidden="true" className="flex items-center justify-center text-ns-text-faint md:px-1">
      {/* horizontal on md+, vertical on mobile */}
      <svg viewBox="0 0 40 10" className="hidden h-2.5 w-10 md:block" fill="none">
        <path d="M0 5h37M33 1.5 37.5 5 33 8.5" stroke="currentColor" strokeWidth="1" />
      </svg>
      <svg viewBox="0 0 10 24" className="h-6 w-2.5 md:hidden" fill="none">
        <path d="M5 0v21M1.5 17.5 5 22l3.5-4.5" stroke="currentColor" strokeWidth="1" />
      </svg>
    </span>
  );
}

function Node({ node }: { node: LaneNode }) {
  return (
    <div
      className={cn(
        "relative flex min-h-[92px] flex-col justify-between rounded-[var(--ns-radius)] border px-4 py-3",
        node.kind === "plain" && "border-ns-border-strong bg-ns-panel",
        node.kind === "known" && "border-ns-border-strong bg-ns-panel",
        node.kind === "unknown" && "ns-hatch border-dashed border-ns-accent/50 bg-ns-bg",
      )}
    >
      <span className="ns-label">{node.actor}</span>
      <span className="mt-3 text-[15px] leading-snug text-ns-text">
        <span className="text-ns-text-muted">{node.verb} </span>
        {node.object}
      </span>
      {node.kind === "unknown" ? (
        <span className="absolute -right-2 -top-2 grid size-7 place-items-center rounded-full border border-ns-accent/50 bg-ns-bg text-ns-accent">
          <LogoMark size={16} />
        </span>
      ) : null}
    </div>
  );
}

/** "human writes algorithm → computer executes" vs "objective → training discovers → not fully understood". */
export function PremiseDiagram() {
  return (
    <figure className="rounded-[6px] border border-ns-border bg-ns-panel/40">
      <div className="divide-y divide-ns-border">
        {LANES.map((lane) => (
          <div key={lane.label} className="grid gap-5 p-5 sm:p-6 md:grid-cols-[180px_minmax(0,1fr)] md:gap-8">
            <div>
              <p className="ns-label text-ns-text">{lane.label}</p>
              <p className="mt-2 font-mono text-[11px] text-ns-text-faint">{lane.sub}</p>
            </div>
            <ol className="grid items-stretch md:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)_auto_minmax(0,1fr)]">
              {lane.nodes.map((node, i) => (
                <li key={node.actor + node.verb} className="contents">
                  {i > 0 ? <Arrow /> : null}
                  <Node node={node} />
                </li>
              ))}
              {lane.nodes.length === 2 ? (
                <li aria-hidden="true" className="hidden md:contents">
                  <span className="w-12" />
                  <div className="flex items-center px-4 font-mono text-[11px] leading-relaxed text-ns-text-faint">
                    Every step is legible: it was written by a person.
                  </div>
                </li>
              ) : null}
            </ol>
          </div>
        ))}
      </div>
      <figcaption className="flex flex-wrap items-center gap-2 border-t border-ns-border px-5 py-3 font-mono text-[11px] text-ns-text-muted sm:px-6">
        <span className="inline-flex items-center gap-1.5 text-ns-accent">
          <LogoMark size={14} /> NeuralScope
        </span>
        instruments the hatched box: the computation that training produced but no one wrote.
      </figcaption>
    </figure>
  );
}
