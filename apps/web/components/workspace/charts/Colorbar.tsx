"use client";

import { colormapGradient, type ColormapName } from "@/lib/colormap";
import { cn } from "@/lib/cn";

export interface ColorbarTick {
  /** Position along the bar, 0..1 (left → right). */
  t: number;
  label: string;
}

/** Horizontal colorbar with tick labels. `reverse` flips the colormap (e.g. ranks: small = bright). */
export function Colorbar({
  cmap,
  reverse,
  ticks,
  width = 200,
  label,
  className,
  noDataLabel,
}: {
  cmap: ColormapName;
  reverse?: boolean;
  ticks: readonly ColorbarTick[];
  width?: number;
  label?: string;
  className?: string;
  noDataLabel?: string;
}) {
  return (
    <div className={cn("flex items-end gap-3", className)}>
      <figure className="m-0" aria-label={label ? `Colour scale: ${label}` : "Colour scale"}>
        <div
          className="h-2 rounded-[2px] shadow-[inset_0_0_0_1px_rgba(255,255,255,0.06)]"
          style={{ width, background: colormapGradient(cmap, reverse) }}
        />
        <div className="relative mt-1 h-3" style={{ width }}>
          {ticks.map((tk, i) => (
            <span
              key={`${tk.t}-${i}`}
              className="ns-tabular absolute top-0 whitespace-nowrap font-mono text-[9.5px] leading-3 text-ns-text-muted"
              style={{
                left: `${tk.t * 100}%`,
                transform: tk.t <= 0.02 ? "none" : tk.t >= 0.98 ? "translateX(-100%)" : "translateX(-50%)",
              }}
            >
              {tk.label}
            </span>
          ))}
        </div>
      </figure>
      {noDataLabel ? (
        <span className="mb-3.5 inline-flex items-center gap-1 font-mono text-[9.5px] text-ns-text-faint">
          <span aria-hidden="true" className="ns-hatch inline-block size-2.5 rounded-[1px] border border-ns-border-strong bg-[#15181d]" />
          {noDataLabel}
        </span>
      ) : null}
    </div>
  );
}
