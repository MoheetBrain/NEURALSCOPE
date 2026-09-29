import { AlertTriangle } from "lucide-react";
import { cn } from "@/lib/cn";

export const DEMO_BANNER_TEXT = "DEMO — RANDOM-WEIGHT FIXTURE MODEL · outputs carry no scientific meaning";

/**
 * Permanent banner shown whenever the backend (health/model) or the open run is a
 * demo fixture. It is deliberately loud (warn colour + hatch) and cannot be dismissed.
 */
export function DemoBanner({ visible, reason, className }: { visible: boolean; reason?: string; className?: string }) {
  if (!visible) return null;
  return (
    <div
      role="alert"
      aria-live="assertive"
      data-testid="demo-banner"
      className={cn(
        "ns-demo-banner flex shrink-0 items-center justify-center gap-2 border-y border-ns-warn/60 bg-ns-warn/15 px-4 py-1.5",
        "font-mono text-[11px] font-medium uppercase tracking-[0.12em] text-ns-warn",
        className,
      )}
    >
      <AlertTriangle aria-hidden="true" className="size-3.5 shrink-0" />
      <span>{DEMO_BANNER_TEXT}</span>
      {reason ? <span className="normal-case tracking-normal text-ns-warn/80">({reason})</span> : null}
    </div>
  );
}

/** Reasons the workspace is in demo mode (empty = not demo). */
export function demoReasons(input: {
  healthDemo?: boolean | null;
  modelDemo?: boolean | null;
  runDemo?: boolean | null;
}): string[] {
  const out: string[] = [];
  if (input.healthDemo) out.push("backend reports is_demo");
  else if (input.modelDemo) out.push("model reports is_demo");
  if (input.runDemo) out.push("this run was recorded on the demo model");
  return out;
}
