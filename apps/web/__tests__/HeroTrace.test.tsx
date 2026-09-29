import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { HeroTrace } from "@/components/marketing/HeroTrace";
import { HeroTraceError, interventionLabel, parseHeroTrace, visibleToken, type HeroTraceSnapshot } from "@/lib/hero-trace";
import snapshotJson from "@/public/snapshots/hero-trace.json";

function makeSnapshot(overrides: Partial<HeroTraceSnapshot> = {}): HeroTraceSnapshot {
  const depth_labels = ["emb", "L0", "L1", "L2"];
  const tokens = ["2", " +", " 2", " ="];
  return {
    source: "REAL",
    model_id: "Qwen/Qwen2.5-1.5B-Instruct",
    run_id: "run_abc123",
    captured_at: "2026-09-29T10:00:00Z",
    prompt: "2 + 2 =",
    tokens,
    depth_labels,
    matrix: { metric: "lens_top1_prob", label: "Logit Lens · top-1 probability", values: depth_labels.map(() => tokens.map(() => 0.5)) },
    target: { token: " 4", column: 3, rank_by_depth: [900, 40, 2, 1], prob_by_depth: [0, 0.01, 0.2, 0.8] },
    intervention: {
      type: "head_ablation",
      layer: 1,
      head: 3,
      description: "Zero-ablate head L1.H3",
      target_prob_before: 0.8,
      target_prob_after: 0.3,
      kl: 0.5,
      output_before: " 4",
      output_after: " 5",
    },
    ...overrides,
  };
}

describe("parseHeroTrace", () => {
  it("accepts the committed snapshot file", () => {
    const snap = parseHeroTrace(snapshotJson);
    expect(["REAL", "PLACEHOLDER"]).toContain(snap.source);
    expect(snap.matrix.values).toHaveLength(snap.depth_labels.length);
    expect(snap.matrix.values[0]).toHaveLength(snap.tokens.length);
  });

  it("rejects shape mismatches", () => {
    const bad = { ...makeSnapshot(), matrix: { metric: "m", label: "l", values: [[0.1]] } };
    expect(() => parseHeroTrace(bad)).toThrow(HeroTraceError);
    expect(() => parseHeroTrace({ ...makeSnapshot(), source: "MAYBE" })).toThrow(/source/);
  });

  it("clamps matrix values into [0, 1]", () => {
    const snap = makeSnapshot();
    snap.matrix.values[0][0] = 3;
    snap.matrix.values[0][1] = -1;
    const parsed = parseHeroTrace(snap);
    expect(parsed.matrix.values[0][0]).toBe(1);
    expect(parsed.matrix.values[0][1]).toBe(0);
  });

  it("formats helper labels", () => {
    expect(visibleToken(" Answer")).toBe("␣Answer");
    expect(interventionLabel(makeSnapshot().intervention)).toBe("ABLATE L1.H3");
  });
});

describe("HeroTrace caption", () => {
  it("shows the recorded-trace caption for REAL snapshots", () => {
    render(<HeroTrace snapshot={makeSnapshot()} initialPlaying={false} />);
    expect(screen.getByText(/Recorded trace · Qwen\/Qwen2\.5-1\.5B-Instruct · run run_abc123/)).toBeInTheDocument();
    expect(screen.queryByText(/Placeholder data/i)).toBeNull();
  });

  it("shows a warning chip for PLACEHOLDER snapshots", () => {
    render(<HeroTrace snapshot={makeSnapshot({ source: "PLACEHOLDER" })} initialPlaying={false} />);
    expect(screen.getByText(/Placeholder data — not a model trace/i)).toBeInTheDocument();
    expect(screen.queryByText(/Recorded trace/)).toBeNull();
  });

  it("renders the full pipeline on first paint (final stage)", () => {
    const { container } = render(<HeroTrace snapshot={makeSnapshot()} initialPlaying={false} />);
    expect(container.querySelector("figure")?.getAttribute("data-stage")).toBe("output");
    expect(screen.getAllByText("ABLATE L1.H3").length).toBeGreaterThan(0);
    expect(screen.getByText("0.30")).toBeInTheDocument();
  });
});
