import { cn } from "@/lib/cn";

/** Large hairline instrument reticle for backgrounds. Purely decorative. */
export function ReticleBackdrop({ className }: { className?: string }) {
  const ticks = Array.from({ length: 72 }, (_, i) => i * 5);
  return (
    <svg viewBox="-300 -300 600 600" fill="none" aria-hidden="true" className={cn("pointer-events-none", className)}>
      <g stroke="currentColor" vectorEffect="non-scaling-stroke">
        <circle r="280" strokeWidth="1" vectorEffect="non-scaling-stroke" />
        <circle r="200" strokeWidth="1" strokeDasharray="2 6" vectorEffect="non-scaling-stroke" />
        <circle r="120" strokeWidth="1" vectorEffect="non-scaling-stroke" />
        <circle r="6" strokeWidth="1" vectorEffect="non-scaling-stroke" />
        {ticks.map((deg) => (
          <line
            key={deg}
            x1="0"
            y1={-280}
            x2="0"
            y2={deg % 45 === 0 ? -262 : -272}
            strokeWidth="1"
            vectorEffect="non-scaling-stroke"
            transform={`rotate(${deg})`}
          />
        ))}
        <path d="M0 -300v160M0 140v160M-300 0h160M140 0h160" strokeWidth="1" vectorEffect="non-scaling-stroke" />
      </g>
    </svg>
  );
}
