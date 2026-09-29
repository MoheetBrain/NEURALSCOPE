import { describe, expect, it } from "vitest";
import { describeIntervention, draftToIntervention, parsePositions } from "@/components/workspace/actions";
import { orderWithChildren } from "@/components/workspace/Timeline";
import { INITIAL_DRAFT } from "@/components/workspace/WorkspaceProvider";
import type { RunSummary } from "@/lib/api/types";
import { depthLabel, diffMiddle, fmtDelta, fmtProb, fmtRank, splitSpecials, visibleToken } from "@/lib/format";
import { LruCache } from "@/lib/api/cache";

describe("token display", () => {
  it("makes whitespace and chat specials visible", () => {
    expect(visibleToken(" is")).toBe("·is");
    expect(visibleToken("\n")).toBe("↵");
    expect(visibleToken("<|im_end|>")).toBe("⟨im_end⟩");
    expect(visibleToken("")).toBe("∅");
  });

  it("splits generated text around special tokens", () => {
    expect(splitSpecials("95<|im_end|>")).toEqual([
      { text: "95", special: false },
      { text: "<|im_end|>", special: true },
    ]);
  });

  it("diffs outputs to the changed middle", () => {
    expect(diffMiddle("95<|im_end|>", "85<|im_end|>")).toEqual({ prefix: "", a: "9", b: "8", suffix: "5<|im_end|>" });
    expect(diffMiddle("abc", "abc")).toEqual({ prefix: "abc", a: "", b: "", suffix: "" });
  });
});

describe("number formatting", () => {
  it("formats probabilities, ranks and signed deltas", () => {
    expect(fmtProb(0.999849)).toBe("0.9998");
    expect(fmtProb(0.000348)).toBe("3.5e−4");
    expect(fmtProb(null)).toBe("—");
    expect(fmtRank(1)).toBe("1");
    expect(fmtRank(6515)).toBe("6.5k");
    expect(fmtRank(150381)).toBe("150k");
    expect(fmtDelta(-0.999501)).toBe("−0.9995");
    expect(fmtDelta(0.25, 2)).toBe("+0.25");
    expect(fmtDelta(0)).toBe("±0");
  });

  it("labels depths (0 = emb, d = L{d−1})", () => {
    expect(depthLabel(0)).toBe("emb");
    expect(depthLabel(23)).toBe("L22");
  });
});

describe("intervention builder", () => {
  it("parses position lists and ranges", () => {
    expect(parsePositions("", 50)).toBeNull();
    expect(parsePositions("3, 5-7 43", 50)).toEqual([3, 5, 6, 7, 43]);
    expect(() => parsePositions("3, x", 50)).toThrow(/cannot parse/);
    expect(() => parsePositions("60", 50)).toThrow(/beyond/);
  });

  it("builds a zero head ablation at all positions (positions = null)", () => {
    const spec = draftToIntervention({ ...INITIAL_DRAFT, layer: 22, head: 10 }, 43, 47);
    expect(spec).toEqual({ type: "head_ablation", layer: 22, head: 10, factor: 0, positions: null });
    expect(describeIntervention(spec)).toBe("zero-ablate head L22.H10 (all positions)");
  });

  it("uses the selected position when asked", () => {
    const spec = draftToIntervention({ ...INITIAL_DRAFT, type: "mlp_ablation", layer: 3, factor: 0.5, positionsMode: "selected" }, 43, 47);
    expect(spec).toEqual({ type: "mlp_ablation", layer: 3, factor: 0.5, positions: [43] });
    expect(describeIntervention(spec)).toBe("scale ×0.5 MLP L3 @ t43");
  });

  it("requires a source run for residual patches", () => {
    expect(() => draftToIntervention({ ...INITIAL_DRAFT, type: "resid_patch" }, 0, 10)).toThrow(/source run/);
    const spec = draftToIntervention({ ...INITIAL_DRAFT, type: "resid_patch", sourceRunId: "run_abc", depth: 5, position: 7, sourcePosition: "" }, 0, 10);
    expect(spec).toEqual({ type: "resid_patch", depth: 5, position: 7, source_run_id: "run_abc", source_position: null, alpha: 1 });
  });
});

describe("run history ordering", () => {
  const run = (id: string, parent: string | null = null): RunSummary =>
    ({ id, parent_run_id: parent, kind: parent ? "intervention" : "baseline" }) as RunSummary;

  it("places intervention children right after their baseline", () => {
    const out = orderWithChildren([run("c2", "b1"), run("b2"), run("c1", "b1"), run("b1"), run("orphan", "gone")]);
    expect(out.map((x) => `${x.run.id}:${x.depth}`)).toEqual(["b2:0", "b1:0", "c2:1", "c1:1", "orphan:0"]);
  });
});

describe("view cache", () => {
  it("evicts least-recently-used entries", () => {
    const c = new LruCache<number>(2);
    c.set("a", 1);
    c.set("b", 2);
    expect(c.get("a")).toBe(1); // a is now most recent
    c.set("c", 3);
    expect(c.has("b")).toBe(false);
    expect(c.has("a")).toBe(true);
    c.invalidatePrefix("c");
    expect(c.size).toBe(1);
  });
});
