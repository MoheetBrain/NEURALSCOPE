/**
 * Colormaps and value scales for heatmaps (canvas) and legends.
 *
 * Sequential maps are perceptually-uniform matplotlib maps (viridis / magma / cividis),
 * stored as anchor stops and linearly interpolated in sRGB (visually indistinguishable
 * from the 256-entry LUTs at heatmap cell sizes). Lightness increases monotonically
 * from t=0 to t=1. The diverging map is built for a dark UI: a dark neutral midpoint
 * ("no effect" reads as absence) and two opposed, bright poles (negative = warm red,
 * positive = blue), so magnitude reads as brightness in both directions.
 */

export type Rgb = readonly [number, number, number];
export type SequentialName = "viridis" | "magma" | "cividis";
export type ColormapName = SequentialName | "diverging";

function hex(h: string): Rgb {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

type Stops = readonly (readonly [number, Rgb])[];

function evenStops(hexes: string[]): Stops {
  return hexes.map((h, i) => [i / (hexes.length - 1), hex(h)] as const);
}

const STOPS: Record<ColormapName, Stops> = {
  viridis: evenStops([
    "#440154", "#482475", "#414487", "#355f8d", "#2a788e", "#21918c",
    "#22a884", "#44bf70", "#7ad151", "#bddf26", "#fde725",
  ]),
  magma: evenStops([
    "#000004", "#140e36", "#3b0f70", "#641a80", "#8c2981", "#b73779",
    "#de4968", "#f7705c", "#fe9f6d", "#fecf92", "#fcfdbf",
  ]),
  cividis: evenStops(["#00204d", "#233a6d", "#414d6b", "#5d626e", "#7c7b78", "#9b9477", "#bcaf6f", "#dfcb5d", "#fee838"]),
  diverging: [
    [0, hex("#ff8f6b")],
    [0.22, hex("#c9573f")],
    [0.5, hex("#24272d")],
    [0.78, hex("#3f6fbf")],
    [1, hex("#8fbcff")],
  ],
};

export const COLORMAP_NAMES: readonly SequentialName[] = ["viridis", "magma", "cividis"] as const;

export function clamp01(t: number): number {
  if (!Number.isFinite(t)) return t > 0 ? 1 : 0;
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

/** Colour at t ∈ [0, 1] (clamped). */
export function sampleColormap(name: ColormapName, t: number): Rgb {
  const stops = STOPS[name];
  const x = clamp01(t);
  for (let i = 1; i < stops.length; i++) {
    const [t1, c1] = stops[i];
    if (x <= t1) {
      const [t0, c0] = stops[i - 1];
      const f = t1 === t0 ? 0 : (x - t0) / (t1 - t0);
      return [
        Math.round(c0[0] + (c1[0] - c0[0]) * f),
        Math.round(c0[1] + (c1[1] - c0[1]) * f),
        Math.round(c0[2] + (c1[2] - c0[2]) * f),
      ];
    }
  }
  return stops[stops.length - 1][1];
}

export function rgbCss([r, g, b]: Rgb): string {
  return `rgb(${r} ${g} ${b})`;
}

export function colormapCss(name: ColormapName, t: number): string {
  return rgbCss(sampleColormap(name, t));
}

/** Precomputed 256-step LUT, RGBA bytes. Index with Math.round(t * 255). */
export function buildLut(name: ColormapName, n = 256): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
    const [r, g, b] = sampleColormap(name, i / (n - 1));
    lut[i * 4] = r;
    lut[i * 4 + 1] = g;
    lut[i * 4 + 2] = b;
    lut[i * 4 + 3] = 255;
  }
  return lut;
}

/** CSS linear-gradient for a colorbar (left→right = t 0→1). */
export function colormapGradient(name: ColormapName, reverse = false, direction = "to right", steps = 12): string {
  const parts: string[] = [];
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    parts.push(`${colormapCss(name, reverse ? 1 - t : t)} ${(t * 100).toFixed(1)}%`);
  }
  return `linear-gradient(${direction}, ${parts.join(", ")})`;
}

/** WCAG relative luminance of an sRGB colour (0..1). */
export function relativeLuminance([r, g, b]: Rgb): number {
  const lin = (c: number) => {
    const s = c / 255;
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

/** Pick legible text ink over a heatmap cell. */
export function inkFor(rgb: Rgb): string {
  return relativeLuminance(rgb) > 0.32 ? "#0a0b0d" : "#e7e9ec";
}

// ---------------------------------------------------------------- scales

export interface ValueScale {
  /** "linear": map [vmin, vmax] linearly. "log": map log10(v) (v > 0). */
  kind: "linear" | "log";
  vmin: number;
  vmax: number;
  /** Flip the mapping so vmin → 1 (bright) — e.g. ranks and entropy, where small is notable. */
  reverse?: boolean;
}

const LOG_FLOOR = 1e-12;

/** Map a value to t ∈ [0, 1] (clamped). `null`/NaN → null (rendered as "no data"). */
export function normalize(v: number | null | undefined, scale: ValueScale): number | null {
  if (v === null || v === undefined || !Number.isFinite(v)) return null;
  let a = scale.vmin;
  let b = scale.vmax;
  let x = v;
  if (scale.kind === "log") {
    a = Math.log10(Math.max(a, LOG_FLOOR));
    b = Math.log10(Math.max(b, LOG_FLOOR));
    x = Math.log10(Math.max(x, LOG_FLOOR));
  }
  const t = b === a ? 0.5 : clamp01((x - a) / (b - a));
  return scale.reverse ? 1 - t : t;
}

/** Diverging normalisation: 0 → 0.5, ±absMax → 0 / 1 (clamped). */
export function divergingNormalize(v: number | null | undefined, absMax: number): number | null {
  if (v === null || v === undefined || !Number.isFinite(v)) return null;
  if (!(absMax > 0)) return 0.5;
  return clamp01(0.5 + v / (2 * absMax));
}

/** Largest |v| in a 2D array (ignores null / non-finite). */
export function absMax2d(values: readonly (readonly (number | null)[])[]): number {
  let m = 0;
  for (const row of values) for (const v of row) if (v !== null && Number.isFinite(v) && Math.abs(v) > m) m = Math.abs(v);
  return m;
}

export function minMax2d(values: readonly (readonly (number | null)[])[]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const row of values)
    for (const v of row) {
      if (v === null || !Number.isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  return lo <= hi ? [lo, hi] : [0, 1];
}

/** "Nice" linear ticks (≤ count+1) within [a, b]. */
export function linearTicks(a: number, b: number, count = 4): number[] {
  if (!Number.isFinite(a) || !Number.isFinite(b)) return [];
  if (a === b) return [a];
  const lo = Math.min(a, b);
  const hi = Math.max(a, b);
  const raw = (hi - lo) / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const norm = raw / mag;
  // d3-style thresholds: pick the nice step (1, 2, 5, 10 × 10^k) closest to the raw step
  const step = (norm >= Math.SQRT2 * 5 ? 10 : norm >= Math.sqrt(10) ? 5 : norm >= Math.SQRT2 ? 2 : 1) * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

/** Decade ticks (1, 10, 100, ...) within [a, b] for a log axis over raw values. */
export function logTicks(a: number, b: number): number[] {
  const lo = Math.max(Math.min(a, b), LOG_FLOOR);
  const hi = Math.max(a, b, LOG_FLOOR);
  const out: number[] = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) {
    const v = 10 ** e;
    if (v >= lo * (1 - 1e-9) && v <= hi * (1 + 1e-9)) out.push(v);
  }
  if (out.length < 2) return [lo, hi];
  return out;
}
