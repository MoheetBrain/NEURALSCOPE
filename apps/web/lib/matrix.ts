/**
 * Token × depth matrix geometry.
 *
 * Orientation (documented in the UI): rows are drawn as a stack, deepest residual
 * state at the TOP and the embeddings ("emb") at the BOTTOM, so the residual stream
 * "rises" from the input tokens (labelled along the bottom) to the output.
 *   row 0         = depth D−1 (after the last block)
 *   row D−1       = depth 0 (embeddings)
 * Columns are token positions, optionally with the system-prompt template collapsed.
 */

export interface MatrixSelection {
  depth: number;
  position: number;
}

export interface ColumnToken {
  index: number;
  role: "template" | "user" | "generated";
}

export function rowToDepth(row: number, nDepths: number): number {
  return nDepths - 1 - row;
}

export function depthToRow(depth: number, nDepths: number): number {
  return nDepths - 1 - depth;
}

/**
 * Positions shown as columns. With `hideTemplate`, template tokens that precede the
 * first user token (system prompt + chat header) are collapsed; template tokens after
 * the user message (e.g. the assistant header, whose last token is usually the
 * prediction position) stay visible. `keep` positions are never hidden.
 */
export function visibleColumns(tokens: readonly ColumnToken[], hideTemplate: boolean, keep: readonly number[] = []): number[] {
  if (!hideTemplate) return tokens.map((t) => t.index);
  const firstUser = tokens.findIndex((t) => t.role !== "template");
  if (firstUser <= 0) return tokens.map((t) => t.index);
  const keepSet = new Set(keep);
  return tokens.filter((t) => t.index >= tokens[firstUser].index || keepSet.has(t.index)).map((t) => t.index);
}

/** Column index of `position` in `columns`, or the nearest visible column. */
export function columnOf(position: number, columns: readonly number[]): number {
  if (columns.length === 0) return -1;
  const exact = columns.indexOf(position);
  if (exact >= 0) return exact;
  let best = 0;
  for (let i = 1; i < columns.length; i++) {
    if (Math.abs(columns[i] - position) < Math.abs(columns[best] - position)) best = i;
  }
  return best;
}

export type NavKey = "ArrowUp" | "ArrowDown" | "ArrowLeft" | "ArrowRight" | "Home" | "End" | "PageUp" | "PageDown";

export const NAV_KEYS: readonly NavKey[] = ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Home", "End", "PageUp", "PageDown"];

export function isNavKey(key: string): key is NavKey {
  return (NAV_KEYS as readonly string[]).includes(key);
}

/**
 * Keyboard navigation. Up = deeper (towards the output, visually up), Down = shallower.
 * Left/Right step through visible columns; Home/End jump to the first/last column;
 * PageUp/PageDown jump to the last depth / embeddings. Always clamps.
 */
export function moveSelection(sel: MatrixSelection, key: NavKey, nDepths: number, columns: readonly number[]): MatrixSelection {
  if (columns.length === 0 || nDepths <= 0) return sel;
  const col = columnOf(sel.position, columns);
  const depth = Math.min(Math.max(sel.depth, 0), nDepths - 1);
  switch (key) {
    case "ArrowUp":
      return { depth: Math.min(depth + 1, nDepths - 1), position: columns[col] };
    case "ArrowDown":
      return { depth: Math.max(depth - 1, 0), position: columns[col] };
    case "ArrowLeft":
      return { depth, position: columns[Math.max(col - 1, 0)] };
    case "ArrowRight":
      return { depth, position: columns[Math.min(col + 1, columns.length - 1)] };
    case "Home":
      return { depth, position: columns[0] };
    case "End":
      return { depth, position: columns[columns.length - 1] };
    case "PageUp":
      return { depth: nDepths - 1, position: columns[col] };
    case "PageDown":
      return { depth: 0, position: columns[col] };
  }
}

/** Hit-test canvas coordinates → (depth, position) or null (outside / in a gap-free sense). */
export function cellAt(
  x: number,
  y: number,
  cellW: number,
  cellH: number,
  nDepths: number,
  columns: readonly number[],
): MatrixSelection | null {
  if (x < 0 || y < 0) return null;
  const c = Math.floor(x / cellW);
  const r = Math.floor(y / cellH);
  if (c >= columns.length || r >= nDepths) return null;
  return { depth: rowToDepth(r, nDepths), position: columns[c] };
}

/** Attention layer linked to a residual depth: depth d is written by block d−1 (emb → block 0). */
export function layerForDepth(depth: number, nLayers: number): number {
  return Math.min(Math.max(depth - 1, 0), Math.max(nLayers - 1, 0));
}
