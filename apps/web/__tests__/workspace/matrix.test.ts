import { describe, expect, it } from "vitest";
import { cellAt, columnOf, depthToRow, isNavKey, layerForDepth, moveSelection, rowToDepth, visibleColumns, type ColumnToken } from "@/lib/matrix";

const D = 29; // emb + L0..L27

function chatTokens(): ColumnToken[] {
  // 0..23 template (system prompt + "user\n"), 24..38 user, 39..43 template (assistant header), 44..46 generated
  return Array.from({ length: 47 }, (_, i) => ({
    index: i,
    role: i < 24 ? "template" : i < 39 ? "user" : i < 44 ? "template" : "generated",
  }));
}

describe("row ↔ depth mapping (emb at the bottom)", () => {
  it("puts the last depth on row 0 and emb on the last row", () => {
    expect(rowToDepth(0, D)).toBe(28);
    expect(rowToDepth(D - 1, D)).toBe(0);
    expect(depthToRow(0, D)).toBe(D - 1);
    expect(depthToRow(28, D)).toBe(0);
  });

  it("is an involution", () => {
    for (let d = 0; d < D; d++) expect(rowToDepth(depthToRow(d, D), D)).toBe(d);
  });

  it("links depth d to attention block d−1 (emb → block 0)", () => {
    expect(layerForDepth(0, 28)).toBe(0);
    expect(layerForDepth(1, 28)).toBe(0);
    expect(layerForDepth(23, 28)).toBe(22);
    expect(layerForDepth(28, 28)).toBe(27);
    expect(layerForDepth(99, 28)).toBe(27);
  });
});

describe("visible columns", () => {
  it("shows every position when not hiding the template", () => {
    expect(visibleColumns(chatTokens(), false)).toHaveLength(47);
  });

  it("collapses only the template before the first user token", () => {
    const cols = visibleColumns(chatTokens(), true);
    expect(cols[0]).toBe(24);
    expect(cols).toContain(43); // assistant header (prediction position) stays
    expect(cols).toHaveLength(47 - 24);
  });

  it("never hides kept positions (e.g. the target)", () => {
    const cols = visibleColumns(chatTokens(), true, [5]);
    expect(cols[0]).toBe(5);
    expect(cols[1]).toBe(24);
  });

  it("does nothing for non-chat prompts", () => {
    const toks: ColumnToken[] = [0, 1, 2].map((i) => ({ index: i, role: "user" }));
    expect(visibleColumns(toks, true)).toEqual([0, 1, 2]);
  });

  it("finds the column of a position or the nearest visible one", () => {
    const cols = [24, 25, 26, 43];
    expect(columnOf(26, cols)).toBe(2);
    expect(columnOf(40, cols)).toBe(3);
    expect(columnOf(3, cols)).toBe(0);
    expect(columnOf(3, [])).toBe(-1);
  });
});

describe("keyboard navigation", () => {
  const cols = [24, 25, 26, 43, 44];

  it("moves up (deeper) / down (shallower) and clamps", () => {
    expect(moveSelection({ depth: 10, position: 25 }, "ArrowUp", D, cols)).toEqual({ depth: 11, position: 25 });
    expect(moveSelection({ depth: 10, position: 25 }, "ArrowDown", D, cols)).toEqual({ depth: 9, position: 25 });
    expect(moveSelection({ depth: 28, position: 25 }, "ArrowUp", D, cols)).toEqual({ depth: 28, position: 25 });
    expect(moveSelection({ depth: 0, position: 25 }, "ArrowDown", D, cols)).toEqual({ depth: 0, position: 25 });
  });

  it("steps through visible columns only", () => {
    expect(moveSelection({ depth: 5, position: 26 }, "ArrowRight", D, cols)).toEqual({ depth: 5, position: 43 });
    expect(moveSelection({ depth: 5, position: 43 }, "ArrowLeft", D, cols)).toEqual({ depth: 5, position: 26 });
    expect(moveSelection({ depth: 5, position: 44 }, "ArrowRight", D, cols)).toEqual({ depth: 5, position: 44 });
    expect(moveSelection({ depth: 5, position: 24 }, "ArrowLeft", D, cols)).toEqual({ depth: 5, position: 24 });
  });

  it("snaps a hidden position to the nearest column", () => {
    expect(moveSelection({ depth: 5, position: 3 }, "ArrowUp", D, cols)).toEqual({ depth: 6, position: 24 });
  });

  it("supports Home / End / PageUp / PageDown", () => {
    const s = { depth: 12, position: 26 };
    expect(moveSelection(s, "Home", D, cols)).toEqual({ depth: 12, position: 24 });
    expect(moveSelection(s, "End", D, cols)).toEqual({ depth: 12, position: 44 });
    expect(moveSelection(s, "PageUp", D, cols)).toEqual({ depth: 28, position: 26 });
    expect(moveSelection(s, "PageDown", D, cols)).toEqual({ depth: 0, position: 26 });
  });

  it("recognises navigation keys", () => {
    expect(isNavKey("ArrowUp")).toBe(true);
    expect(isNavKey("Enter")).toBe(false);
  });
});

describe("hit testing", () => {
  const cols = [24, 25, 26];
  it("maps canvas pixels to (depth, position)", () => {
    // cell 10×20 px: x=15 → column 1 (t25); y=5 → row 0 → deepest depth
    expect(cellAt(15, 5, 10, 20, D, cols)).toEqual({ depth: 28, position: 25 });
    expect(cellAt(0, (D - 1) * 20 + 1, 10, 20, D, cols)).toEqual({ depth: 0, position: 24 });
  });
  it("returns null outside the grid", () => {
    expect(cellAt(-1, 5, 10, 20, D, cols)).toBeNull();
    expect(cellAt(30, 5, 10, 20, D, cols)).toBeNull();
    expect(cellAt(5, D * 20, 10, 20, D, cols)).toBeNull();
  });
});
