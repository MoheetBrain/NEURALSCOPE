import { describe, expect, it } from "vitest";
import {
  COLORMAP_NAMES,
  absMax2d,
  buildLut,
  clamp01,
  colormapGradient,
  divergingNormalize,
  linearTicks,
  logTicks,
  minMax2d,
  normalize,
  relativeLuminance,
  sampleColormap,
} from "@/lib/colormap";
import { scaleForMatrix } from "@/lib/metrics";
import type { MatrixResponse } from "@/lib/api/types";

describe("sequential colormaps", () => {
  it.each(COLORMAP_NAMES)("%s is monotonic in luminance (perceptually ordered)", (name) => {
    let prev = -1;
    for (let i = 0; i <= 64; i++) {
      const L = relativeLuminance(sampleColormap(name, i / 64));
      expect(L).toBeGreaterThanOrEqual(prev - 1e-9);
      prev = L;
    }
    // spans a real lightness range
    expect(relativeLuminance(sampleColormap(name, 1)) - relativeLuminance(sampleColormap(name, 0))).toBeGreaterThan(0.5);
  });

  it("hits the matplotlib endpoints for viridis", () => {
    expect(sampleColormap("viridis", 0)).toEqual([0x44, 0x01, 0x54]);
    expect(sampleColormap("viridis", 1)).toEqual([0xfd, 0xe7, 0x25]);
  });

  it("clamps t outside [0, 1] and non-finite t", () => {
    expect(sampleColormap("magma", -3)).toEqual(sampleColormap("magma", 0));
    expect(sampleColormap("magma", 7)).toEqual(sampleColormap("magma", 1));
    expect(sampleColormap("cividis", Number.NaN)).toEqual(sampleColormap("cividis", 0));
    expect(clamp01(Infinity)).toBe(1);
    expect(clamp01(-Infinity)).toBe(0);
  });

  it("builds a 256-entry RGBA LUT and a CSS gradient", () => {
    const lut = buildLut("viridis");
    expect(lut).toHaveLength(256 * 4);
    expect([lut[0], lut[1], lut[2], lut[3]]).toEqual([0x44, 0x01, 0x54, 255]);
    expect(colormapGradient("viridis")).toMatch(/^linear-gradient\(to right, rgb\(68 1 84\) 0\.0%/);
    expect(colormapGradient("viridis", true)).toMatch(/rgb\(253 231 37\) 0\.0%/);
  });
});

describe("diverging colormap", () => {
  it("has a dark neutral midpoint and bright opposite poles", () => {
    const mid = relativeLuminance(sampleColormap("diverging", 0.5));
    expect(mid).toBeLessThan(relativeLuminance(sampleColormap("diverging", 0)));
    expect(mid).toBeLessThan(relativeLuminance(sampleColormap("diverging", 1)));
    const [r0, , b0] = sampleColormap("diverging", 0);
    const [r1, , b1] = sampleColormap("diverging", 1);
    expect(r0).toBeGreaterThan(b0); // warm negative pole
    expect(b1).toBeGreaterThan(r1); // cool positive pole
  });

  it("maps 0 → 0.5 and clamps beyond ±absMax", () => {
    expect(divergingNormalize(0, 2)).toBe(0.5);
    expect(divergingNormalize(-2, 2)).toBe(0);
    expect(divergingNormalize(1, 2)).toBe(0.75);
    expect(divergingNormalize(10, 2)).toBe(1);
    expect(divergingNormalize(-10, 2)).toBe(0);
    expect(divergingNormalize(null, 2)).toBeNull();
    expect(divergingNormalize(1, 0)).toBe(0.5);
  });
});

describe("normalize (value scales)", () => {
  it("is monotonic and clamps on a linear scale", () => {
    const s = { kind: "linear" as const, vmin: -1, vmax: 3 };
    expect(normalize(-1, s)).toBe(0);
    expect(normalize(1, s)).toBe(0.5);
    expect(normalize(3, s)).toBe(1);
    expect(normalize(-50, s)).toBe(0);
    expect(normalize(50, s)).toBe(1);
    const xs = [-1, -0.5, 0, 0.7, 2, 3].map((v) => normalize(v, s) as number);
    expect([...xs].sort((a, b) => a - b)).toEqual(xs);
  });

  it("maps decades evenly on a log scale", () => {
    const s = { kind: "log" as const, vmin: 1, vmax: 1e4 };
    expect(normalize(1, s)).toBeCloseTo(0);
    expect(normalize(10, s)).toBeCloseTo(0.25);
    expect(normalize(100, s)).toBeCloseTo(0.5);
    expect(normalize(1e4, s)).toBeCloseTo(1);
    expect(normalize(1e9, s)).toBe(1);
    // non-positive values clamp to the bottom instead of producing NaN
    expect(normalize(0, s)).toBe(0);
    expect(normalize(-5, s)).toBe(0);
  });

  it("reverses when requested and returns null for missing values", () => {
    const s = { kind: "linear" as const, vmin: 0, vmax: 10, reverse: true };
    expect(normalize(0, s)).toBe(1);
    expect(normalize(10, s)).toBe(0);
    expect(normalize(null, s)).toBeNull();
    expect(normalize(undefined, s)).toBeNull();
    expect(normalize(Number.NaN, s)).toBeNull();
  });

  it("returns the midpoint for a degenerate domain", () => {
    expect(normalize(5, { kind: "linear", vmin: 5, vmax: 5 })).toBe(0.5);
  });
});

describe("ticks and extents", () => {
  it("produces decade ticks for log axes", () => {
    expect(logTicks(1, 150381)).toEqual([1, 10, 100, 1000, 10000, 100000]);
    expect(logTicks(3, 7)).toEqual([3, 7]);
  });

  it("produces nice linear ticks", () => {
    expect(linearTicks(0, 1, 4)).toEqual([0, 0.2, 0.4, 0.6, 0.8, 1]);
    expect(linearTicks(0, 5.18, 4)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(linearTicks(-0.5, 2.3, 4)).toEqual([-0.5, 0, 0.5, 1, 1.5, 2]);
  });

  it("computes absMax / min-max ignoring nulls", () => {
    const g = [
      [1, null, -3],
      [2, Number.NaN, 0.5],
    ];
    expect(absMax2d(g)).toBe(3);
    expect(minMax2d(g)).toEqual([-3, 2]);
    expect(minMax2d([[null]])).toEqual([0, 1]);
  });
});

describe("matrix colour scales from API responses", () => {
  const base: Omit<MatrixResponse, "metric" | "scale" | "colormap" | "vmin" | "vmax"> = {
    run_id: "run_x",
    label: "",
    status: "derived",
    depth_labels: ["emb", "L0"],
    n_depths: 2,
    n_tokens: 1,
    values: [[0], [1]],
  };

  it("treats rank metrics as log10 values mapped linearly, reversed (rank 1 = bright)", () => {
    const s = scaleForMatrix({ ...base, metric: "target_rank", scale: "log", colormap: "sequential_r", vmin: 0, vmax: 5.18 });
    expect(s.kind).toBe("linear");
    expect(s.reverse).toBe(true);
    expect(normalize(0, s)).toBe(1); // log10(rank 1) = 0 → brightest
    expect(normalize(5.18, s)).toBe(0);
  });

  it("uses a true log scale for residual norms", () => {
    const s = scaleForMatrix({ ...base, metric: "resid_norm", scale: "log", colormap: "sequential", vmin: 1, vmax: 300 });
    expect(s.kind).toBe("log");
    expect(normalize(Math.sqrt(300), s)).toBeCloseTo(0.5);
  });

  it("pins probabilities to [0, 1]", () => {
    const s = scaleForMatrix({ ...base, metric: "lens_top1_prob", scale: "linear", colormap: "sequential", vmin: 0.2, vmax: 0.9 });
    expect([s.vmin, s.vmax]).toEqual([0, 1]);
  });
});
