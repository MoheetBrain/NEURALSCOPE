import type { HTMLAttributes, ReactNode } from "react";
import type { EpistemicStatus } from "@/lib/epistemic";
import { cn } from "@/lib/cn";
import { EpistemicBadge } from "./EpistemicBadge";

export interface PanelProps extends Omit<HTMLAttributes<HTMLElement>, "title"> {
  /** Small-caps header label. */
  label?: ReactNode;
  /** Epistemic status of the panel's content, shown as a badge next to the label. */
  status?: EpistemicStatus;
  /** Optional qualifier for the badge, e.g. "n=1". */
  statusNote?: string;
  /** Right-aligned header slot (controls, readouts). */
  right?: ReactNode;
  /** Element to render as. Default "section". */
  as?: "section" | "div" | "article" | "aside" | "figure";
  /** Padding for the body. Default true. */
  padded?: boolean;
  bodyClassName?: string;
  headerClassName?: string;
  children?: ReactNode;
}

/**
 * Instrument panel: hairline frame, 32px header row with a small-caps label,
 * optional epistemic badge and a right-hand slot.
 */
export function Panel({
  label,
  status,
  statusNote,
  right,
  as: Tag = "section",
  padded = true,
  className,
  bodyClassName,
  headerClassName,
  children,
  ...rest
}: PanelProps) {
  const hasHeader = label != null || status != null || right != null;
  return (
    <Tag
      aria-label={typeof label === "string" ? label : undefined}
      className={cn("flex min-w-0 flex-col rounded-[var(--ns-radius)] border border-ns-border bg-ns-panel", className)}
      {...rest}
    >
      {hasHeader ? (
        <div
          className={cn(
            "flex h-8 shrink-0 items-center gap-2 border-b border-ns-border px-3",
            headerClassName,
          )}
        >
          {label != null ? <span className="ns-label truncate">{label}</span> : null}
          {status ? <EpistemicBadge status={status} size="xs" note={statusNote} /> : null}
          {right != null ? <div className="ml-auto flex min-w-0 items-center gap-2">{right}</div> : null}
        </div>
      ) : null}
      <div className={cn("min-h-0 min-w-0 flex-1", padded && "p-3", bodyClassName)}>{children}</div>
    </Tag>
  );
}
