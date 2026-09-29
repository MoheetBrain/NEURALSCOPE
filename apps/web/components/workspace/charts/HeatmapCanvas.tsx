"use client";

import { useEffect, useRef, type KeyboardEvent, type MouseEvent } from "react";
import { buildLut, type ColormapName } from "@/lib/colormap";
import { cn } from "@/lib/cn";

export interface HeatCell {
  row: number;
  col: number;
}

const LUTS = new Map<ColormapName, Uint8ClampedArray>();
function lutFor(name: ColormapName): Uint8ClampedArray {
  let l = LUTS.get(name);
  if (!l) {
    l = buildLut(name, 256);
    LUTS.set(name, l);
  }
  return l;
}

/**
 * RGBA bytes (row-major, rows × cols) from a value grid. `toT` maps a value to
 * t ∈ [0,1] or null (no data → transparent). Optional `reverse` flips the LUT.
 */
export function colorsFromGrid(
  grid: readonly (readonly (number | null)[])[],
  toT: (v: number | null, row: number, col: number) => number | null,
  cmap: ColormapName,
): Uint8ClampedArray {
  const rows = grid.length;
  const cols = rows ? grid[0].length : 0;
  const lut = lutFor(cmap);
  const out = new Uint8ClampedArray(rows * cols * 4);
  for (let r = 0; r < rows; r++) {
    const row = grid[r];
    for (let c = 0; c < cols; c++) {
      const t = toT(row[c] ?? null, r, c);
      const o = (r * cols + c) * 4;
      if (t === null) continue; // alpha 0
      const i = Math.round(Math.min(1, Math.max(0, t)) * 255) * 4;
      out[o] = lut[i];
      out[o + 1] = lut[i + 1];
      out[o + 2] = lut[i + 2];
      out[o + 3] = 255;
    }
  }
  return out;
}

export interface HeatmapCanvasProps {
  rows: number;
  cols: number;
  /** RGBA bytes, rows × cols × 4 (alpha 0 = no data). */
  colors: Uint8ClampedArray;
  cellW: number;
  cellH: number;
  /** Hairline gap between cells (CSS px). 0 = continuous image. */
  gap?: number;
  selected?: HeatCell | null;
  hovered?: HeatCell | null;
  /** Columns outlined as markers (e.g. the target position). */
  markCols?: readonly number[];
  markRows?: readonly number[];
  /** Draw a faint band along the selected row/column. */
  crosshair?: boolean;
  onHoverCell?: (cell: HeatCell | null, e?: MouseEvent<HTMLCanvasElement>) => void;
  onClickCell?: (cell: HeatCell) => void;
  onKeyDown?: (e: KeyboardEvent<HTMLCanvasElement>) => void;
  ariaLabel: string;
  focusable?: boolean;
  className?: string;
  background?: string;
  markColor?: string;
  selectColor?: string;
}

/**
 * Canvas heatmap: cells are rasterised into an rows×cols ImageData and scaled up with
 * nearest-neighbour sampling (fast for 29×200 or 12×(T×T) grids), then overlays
 * (gaps, markers, selection, hover) are stroked on top at device resolution.
 */
export function HeatmapCanvas({
  rows,
  cols,
  colors,
  cellW,
  cellH,
  gap = 0,
  selected,
  hovered,
  markCols,
  markRows,
  crosshair,
  onHoverCell,
  onClickCell,
  onKeyDown,
  ariaLabel,
  focusable,
  className,
  background = "#15181d",
  markColor = "#d7c9a6",
  selectColor = "#8fb3ff",
}: HeatmapCanvasProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const width = Math.max(1, Math.round(cols * cellW));
  const height = Math.max(1, Math.round(rows * cellH));

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || rows === 0 || cols === 0) return;
    const dpr = Math.min(3, window.devicePixelRatio || 1);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = background;
    ctx.fillRect(0, 0, width, height);

    if (colors.length === rows * cols * 4) {
      const img = new ImageData(new Uint8ClampedArray(colors), cols, rows);
      const off = document.createElement("canvas");
      off.width = cols;
      off.height = rows;
      off.getContext("2d")?.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off, 0, 0, cols, rows, 0, 0, cols * cellW, rows * cellH);
    }

    if (gap > 0) {
      ctx.fillStyle = "#0a0b0d";
      for (let c = 1; c < cols; c++) ctx.fillRect(c * cellW - gap / 2, 0, gap, height);
      for (let r = 1; r < rows; r++) ctx.fillRect(0, r * cellH - gap / 2, width, gap);
    }

    if (crosshair && selected) {
      ctx.fillStyle = "rgba(143,179,255,0.10)";
      ctx.fillRect(selected.col * cellW, 0, cellW, height);
      ctx.fillRect(0, selected.row * cellH, width, cellH);
    }

    ctx.lineWidth = 1;
    if (markCols?.length) {
      ctx.strokeStyle = markColor;
      for (const c of markCols) {
        if (c < 0 || c >= cols) continue;
        ctx.strokeRect(c * cellW + 0.5, 0.5, Math.max(1, cellW - 1), height - 1);
      }
    }
    if (markRows?.length) {
      ctx.strokeStyle = markColor;
      for (const r of markRows) {
        if (r < 0 || r >= rows) continue;
        ctx.strokeRect(0.5, r * cellH + 0.5, width - 1, Math.max(1, cellH - 1));
      }
    }
    if (hovered && (!selected || hovered.row !== selected.row || hovered.col !== selected.col)) {
      ctx.strokeStyle = "rgba(231,233,236,0.85)";
      ctx.lineWidth = 1;
      ctx.strokeRect(hovered.col * cellW + 0.5, hovered.row * cellH + 0.5, Math.max(1, cellW - 1), Math.max(1, cellH - 1));
    }
    if (selected && selected.row >= 0 && selected.col >= 0) {
      ctx.strokeStyle = "#0a0b0d";
      ctx.lineWidth = 3;
      ctx.strokeRect(selected.col * cellW + 1, selected.row * cellH + 1, Math.max(1, cellW - 2), Math.max(1, cellH - 2));
      ctx.strokeStyle = selectColor;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(selected.col * cellW + 1, selected.row * cellH + 1, Math.max(1, cellW - 2), Math.max(1, cellH - 2));
    }
  }, [rows, cols, colors, cellW, cellH, gap, width, height, selected, hovered, markCols, markRows, crosshair, background, markColor, selectColor]);

  const cellFromEvent = (e: MouseEvent<HTMLCanvasElement>): HeatCell | null => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * width;
    const y = ((e.clientY - rect.top) / rect.height) * height;
    const col = Math.floor(x / cellW);
    const row = Math.floor(y / cellH);
    if (col < 0 || row < 0 || col >= cols || row >= rows) return null;
    return { row, col };
  };

  return (
    <canvas
      ref={ref}
      role="img"
      aria-label={ariaLabel}
      tabIndex={focusable ? 0 : undefined}
      className={cn("block select-none", (onClickCell || onHoverCell) && "cursor-crosshair", className)}
      style={{ width, height }}
      onMouseMove={onHoverCell ? (e) => onHoverCell(cellFromEvent(e), e) : undefined}
      onMouseLeave={onHoverCell ? () => onHoverCell(null) : undefined}
      onClick={
        onClickCell
          ? (e) => {
              const c = cellFromEvent(e);
              if (c) onClickCell(c);
            }
          : undefined
      }
      onKeyDown={onKeyDown}
    />
  );
}
