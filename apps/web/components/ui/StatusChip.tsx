import type { HTMLAttributes, ReactNode } from "react";
import { cn } from "@/lib/cn";

export type StatusChipTone = "neutral" | "accent" | "paper" | "ok" | "warn" | "danger" | "future";

export interface StatusChipProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: StatusChipTone;
  /** Show a leading dot. `"pulse"` blinks it (respects reduced motion). */
  dot?: boolean | "pulse";
  icon?: ReactNode;
}

const TONES: Record<StatusChipTone, string> = {
  neutral: "border-ns-border-strong text-ns-text-muted bg-ns-panel",
  accent: "border-ns-accent/40 text-ns-accent bg-ns-accent/8",
  paper: "border-ns-accent-2/35 text-ns-accent-2 bg-ns-accent-2/8",
  ok: "border-ns-ok/35 text-ns-ok bg-ns-ok/8",
  warn: "border-ns-warn/45 text-ns-warn bg-ns-warn/10",
  danger: "border-ns-danger/45 text-ns-danger bg-ns-danger/10",
  future: "border-dashed border-ns-text-faint/70 text-ns-text-muted bg-transparent",
};

const DOTS: Record<StatusChipTone, string> = {
  neutral: "bg-ns-text-muted",
  accent: "bg-ns-accent",
  paper: "bg-ns-accent-2",
  ok: "bg-ns-ok",
  warn: "bg-ns-warn",
  danger: "bg-ns-danger",
  future: "bg-ns-text-faint",
};

/**
 * Non-epistemic state chip: "BUILDING NOW / V1", "FUTURE — NOT IN V1",
 * "PLANNED", "PLACEHOLDER DATA", ... Mono, uppercase, hairline.
 */
export function StatusChip({ tone = "neutral", dot, icon, className, children, ...rest }: StatusChipProps) {
  return (
    <span
      className={cn(
        "inline-flex h-5 shrink-0 items-center gap-1.5 whitespace-nowrap rounded-[3px] border px-1.5",
        "font-mono text-[10px] font-medium uppercase leading-none tracking-[0.1em] [&_svg]:size-3",
        TONES[tone],
        className,
      )}
      {...rest}
    >
      {dot ? (
        <span
          aria-hidden="true"
          className={cn("size-1.5 rounded-full", DOTS[tone], dot === "pulse" && "animate-[ns-blink_1.6s_ease-in-out_infinite]")}
        />
      ) : null}
      {icon}
      {children}
    </span>
  );
}
