import { cn } from "@/lib/cn";

export interface LogoMarkProps {
  /** Rendered size in px (square). Default 20. */
  size?: number;
  className?: string;
  /** Accessible title. Omit when the mark sits next to the wordmark. */
  title?: string;
}

/**
 * Mark: a microscope targeting reticle (circle + crosshair ticks) locked onto a
 * node of a small three-node graph. Uses currentColor for the reticle and
 * --ns-accent for the targeted node.
 */
export function LogoMark({ size = 20, className, title }: LogoMarkProps) {
  const labelled = Boolean(title);
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      role={labelled ? "img" : undefined}
      aria-hidden={labelled ? undefined : true}
      className={cn("shrink-0", className)}
    >
      {labelled ? <title>{title}</title> : null}
      {/* reticle */}
      <circle cx="16" cy="16" r="11" stroke="currentColor" strokeWidth="1.5" />
      <path
        d="M16 1.5v5M16 25.5v5M1.5 16h5M25.5 16h5"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="square"
      />
      {/* graph */}
      <path d="M10.5 19.5 16 13.5l5.5 5M10.5 19.5l11-1" stroke="currentColor" strokeOpacity="0.55" strokeWidth="1" />
      <circle cx="10.5" cy="19.5" r="1.6" fill="currentColor" />
      <circle cx="21.5" cy="18.5" r="1.6" fill="currentColor" />
      <circle cx="16" cy="13.5" r="2.6" fill="var(--ns-accent)" />
    </svg>
  );
}

export interface LogoProps {
  /** Mark size in px. Wordmark scales with it. Default 20. */
  size?: number;
  /** Show the "NeuralScope" wordmark. Default true. */
  wordmark?: boolean;
  className?: string;
}

export function Logo({ size = 20, wordmark = true, className }: LogoProps) {
  return (
    <span className={cn("inline-flex items-center gap-2 text-ns-text", className)}>
      <LogoMark size={size} title={wordmark ? undefined : "NeuralScope"} />
      {wordmark ? (
        <span className="font-sans font-medium tracking-[-0.01em]" style={{ fontSize: Math.round(size * 0.78) }}>
          Neural<span className="text-ns-text-muted">Scope</span>
        </span>
      ) : null}
    </span>
  );
}
