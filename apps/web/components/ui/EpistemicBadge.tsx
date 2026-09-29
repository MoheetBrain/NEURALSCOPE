import { EPISTEMIC, type EpistemicStatus } from "@/lib/epistemic";
import { cn } from "@/lib/cn";

export type EpistemicBadgeSize = "sm" | "xs";

export interface EpistemicBadgeProps {
  status: EpistemicStatus;
  size?: EpistemicBadgeSize;
  /** Optional trailing qualifier, e.g. "n=1" or "L14". Rendered after the label. */
  note?: string;
  className?: string;
}

// Static class strings so Tailwind can see them.
const STATUS_CLASSES: Record<EpistemicStatus, string> = {
  observed: "text-ns-observed border-ns-observed/30 bg-ns-observed/8",
  derived: "text-ns-derived border-ns-derived/30 bg-ns-derived/8",
  inferred: "text-ns-inferred border-ns-inferred/30 bg-ns-inferred/8",
  causal: "text-ns-causal border-ns-causal/30 bg-ns-causal/8",
};

const DOT_CLASSES: Record<EpistemicStatus, string> = {
  observed: "bg-ns-observed",
  derived: "bg-ns-derived",
  inferred: "bg-ns-inferred",
  causal: "bg-ns-causal",
};

const SIZE_CLASSES: Record<EpistemicBadgeSize, string> = {
  sm: "h-5 gap-1.5 px-1.5 text-[10px]",
  xs: "h-4 gap-1 px-1 text-[9px]",
};

/**
 * Small mono pill declaring the epistemic status of a view or claim.
 * `title` carries the definition; screen readers get "Epistemic status: ...".
 */
export function EpistemicBadge({ status, size = "sm", note, className }: EpistemicBadgeProps) {
  const info = EPISTEMIC[status];
  return (
    <span
      data-status={status}
      title={`${info.label.toUpperCase()} — ${info.definition}`}
      className={cn(
        "ns-epistemic-badge",
        `ns-epistemic-${status}`,
        "inline-flex shrink-0 items-center whitespace-nowrap rounded-[3px] border font-mono font-medium uppercase leading-none tracking-[0.1em]",
        STATUS_CLASSES[status],
        SIZE_CLASSES[size],
        className,
      )}
    >
      <span
        aria-hidden="true"
        className={cn("shrink-0 rounded-full", DOT_CLASSES[status], size === "xs" ? "size-1" : "size-1.5")}
      />
      <span className="sr-only">Epistemic status: </span>
      {info.label}
      {note ? <span className="text-ns-text-muted normal-case tracking-normal">· {note}</span> : null}
    </span>
  );
}
