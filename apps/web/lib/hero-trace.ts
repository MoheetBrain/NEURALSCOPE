/**
 * Schema + validation for the landing-page hero trace snapshot
 * (public/snapshots/hero-trace.json).
 *
 * The file is produced by the backend capture script (source: "REAL") or is a
 * hand-shaped placeholder (source: "PLACEHOLDER"). The UI must say which.
 */

export type HeroTraceSource = "REAL" | "PLACEHOLDER";

export interface HeroTraceSnapshot {
  source: HeroTraceSource;
  model_id: string;
  run_id: string;
  captured_at: string;
  prompt: string;
  /** Displayed token strings (matrix columns). */
  tokens: string[];
  /** Row labels, e.g. ["emb", "L0", ..., "L27"]. */
  depth_labels: string[];
  /** values[depth][token], each in [0, 1]. */
  matrix: { metric: string; label: string; values: number[][] };
  target: { token: string; column: number; rank_by_depth: number[]; prob_by_depth: number[] };
  intervention: {
    type: string;
    layer: number;
    head: number | null;
    description: string;
    target_prob_before: number;
    target_prob_after: number;
    kl: number;
    output_before: string;
    output_after: string;
  };
}

export class HeroTraceError extends Error {}

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isStr = (v: unknown): v is string => typeof v === "string";
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

function fail(msg: string): never {
  throw new HeroTraceError(`hero-trace.json: ${msg}`);
}

/**
 * Validate an unknown JSON value against the snapshot schema. Throws
 * HeroTraceError on structural problems; clamps matrix/probabilities to [0, 1].
 */
export function parseHeroTrace(raw: unknown): HeroTraceSnapshot {
  if (!isObj(raw)) fail("root must be an object");
  const { source, model_id, run_id, captured_at, prompt, tokens, depth_labels, matrix, target, intervention } = raw;

  if (source !== "REAL" && source !== "PLACEHOLDER") fail(`source must be "REAL" or "PLACEHOLDER"`);
  for (const [k, v] of Object.entries({ model_id, run_id, captured_at, prompt })) {
    if (!isStr(v)) fail(`${k} must be a string`);
  }
  if (!Array.isArray(tokens) || tokens.length === 0 || !tokens.every(isStr)) fail("tokens must be a non-empty string[]");
  if (!Array.isArray(depth_labels) || depth_labels.length === 0 || !depth_labels.every(isStr)) {
    fail("depth_labels must be a non-empty string[]");
  }
  const T = tokens.length;
  const D = depth_labels.length;

  if (!isObj(matrix) || !isStr(matrix.metric) || !isStr(matrix.label) || !Array.isArray(matrix.values)) {
    fail("matrix must have metric, label and values");
  }
  if (matrix.values.length !== D) fail(`matrix.values must have ${D} rows (one per depth label)`);
  const values = matrix.values.map((row, d) => {
    if (!Array.isArray(row) || row.length !== T || !row.every(isNum)) fail(`matrix.values[${d}] must be number[${T}]`);
    return row.map(clamp01);
  });

  if (!isObj(target) || !isStr(target.token) || !isNum(target.column)) fail("target must have token and column");
  if (target.column < 0 || target.column >= T || !Number.isInteger(target.column)) fail("target.column out of range");
  const rank = target.rank_by_depth;
  const prob = target.prob_by_depth;
  if (!Array.isArray(rank) || rank.length !== D || !rank.every(isNum)) fail(`target.rank_by_depth must be number[${D}]`);
  if (!Array.isArray(prob) || prob.length !== D || !prob.every(isNum)) fail(`target.prob_by_depth must be number[${D}]`);

  if (!isObj(intervention)) fail("intervention must be an object");
  const iv = intervention;
  if (!isStr(iv.type) || !isNum(iv.layer) || !(iv.head === null || isNum(iv.head)) || !isStr(iv.description)) {
    fail("intervention must have type, layer, head (number|null) and description");
  }
  for (const k of ["target_prob_before", "target_prob_after", "kl"] as const) {
    if (!isNum(iv[k])) fail(`intervention.${k} must be a number`);
  }
  if (!isStr(iv.output_before) || !isStr(iv.output_after)) fail("intervention.output_before/after must be strings");

  return {
    source,
    model_id: model_id as string,
    run_id: run_id as string,
    captured_at: captured_at as string,
    prompt: prompt as string,
    tokens: tokens as string[],
    depth_labels: depth_labels as string[],
    matrix: { metric: matrix.metric, label: matrix.label, values },
    target: {
      token: target.token,
      column: target.column,
      rank_by_depth: rank as number[],
      prob_by_depth: (prob as number[]).map(clamp01),
    },
    intervention: {
      type: iv.type as string,
      layer: iv.layer as number,
      head: iv.head as number | null,
      description: iv.description as string,
      target_prob_before: clamp01(iv.target_prob_before as number),
      target_prob_after: clamp01(iv.target_prob_after as number),
      kl: iv.kl as number,
      output_before: iv.output_before as string,
      output_after: iv.output_after as string,
    },
  };
}

/** Row index for a transformer block's output ("L{layer}"), falling back to layer + 1 (row 0 = embeddings). */
export function depthRowForLayer(depthLabels: readonly string[], layer: number): number {
  const i = depthLabels.indexOf(`L${layer}`);
  if (i >= 0) return i;
  return Math.min(depthLabels.length - 1, Math.max(0, layer + 1));
}

/** Make whitespace visible in token strings: spaces → "␣", newlines → "↵", tabs → "⇥". */
export function visibleToken(tok: string): string {
  if (tok.length === 0) return "∅";
  return tok.replace(/ /g, "␣").replace(/\n/g, "↵").replace(/\t/g, "⇥");
}

/** Short human label for an intervention, e.g. "ABLATE L17.H5". */
export function interventionLabel(iv: HeroTraceSnapshot["intervention"]): string {
  const verb = iv.type.includes("ablation") ? "ABLATE" : iv.type.includes("patch") ? "PATCH" : iv.type.toUpperCase();
  const where = iv.head === null ? `L${iv.layer}` : `L${iv.layer}.H${iv.head}`;
  return `${verb} ${where}`;
}
