import type { MatrixMetric, MatrixResponse } from "@/lib/api/types";
import type { ValueScale } from "@/lib/colormap";
import { fmtNum, fmtProb, fmtRank } from "@/lib/format";

export interface MetricInfo {
  value: MatrixMetric;
  short: string;
  /** Unit / meaning line under the selector. */
  description: string;
  /** Values arrive as log10(rank) from the API. */
  log10Values?: boolean;
  needsTarget?: boolean;
  needsParent?: boolean;
}

export const MATRIX_METRICS: readonly MetricInfo[] = [
  { value: "lens_top1_prob", short: "Lens top-1 P", description: "Probability of the Logit Lens top-1 token at each residual state." },
  { value: "target_rank", short: "Target rank", description: "Logit Lens rank of the target token (log scale; 1 = top-1).", log10Values: true, needsTarget: true },
  { value: "target_prob", short: "P(target)", description: "Logit Lens probability of the target token.", needsTarget: true },
  { value: "next_token_rank", short: "Next-token rank", description: "Logit Lens rank of the token that actually comes next (log scale).", log10Values: true },
  { value: "lens_entropy", short: "Lens entropy", description: "Entropy (nats) of the Logit Lens distribution; low = peaked." },
  { value: "resid_norm", short: "‖h‖", description: "L2 norm of the residual-stream vector (log colour scale)." },
  { value: "cos_delta", short: "1 − cos Δ", description: "1 − cos(h_d, h_{d−1}): how much each block rotates the residual state." },
  { value: "delta_resid", short: "Δ vs parent", description: "‖h_int − h_base‖ / ‖h_base‖: relative change of the residual state caused by the intervention.", needsParent: true },
] as const;

export function metricInfo(m: MatrixMetric): MetricInfo {
  return MATRIX_METRICS.find((x) => x.value === m) ?? MATRIX_METRICS[0];
}

/** Colour scale for a matrix response. Rank metrics arrive log10-transformed → linear mapping over log values. */
export function scaleForMatrix(m: MatrixResponse): ValueScale {
  const info = metricInfo(m.metric);
  const vmin = m.vmin ?? 0;
  const vmax = m.vmax ?? 1;
  const reverse = m.colormap === "sequential_r";
  if (info.log10Values) return { kind: "linear", vmin: Math.max(0, vmin), vmax: Math.max(vmax, 0.001), reverse };
  if (m.scale === "log") return { kind: "log", vmin: Math.max(vmin, 1e-6), vmax: Math.max(vmax, vmin * 1.0001, 1e-6), reverse };
  // probabilities: use the full [0, 1] range so colours are comparable across runs
  if (m.metric === "lens_top1_prob" || m.metric === "target_prob") return { kind: "linear", vmin: 0, vmax: 1, reverse };
  return { kind: "linear", vmin, vmax: vmax === vmin ? vmin + 1 : vmax, reverse };
}

/** Human value for a cell (tooltips, readouts). */
export function formatMatrixValue(m: MatrixResponse, depth: number, position: number): string {
  const v = m.values[depth]?.[position];
  if (v === null || v === undefined) return "—";
  switch (m.metric) {
    case "target_rank": {
      const r = m.ranks?.[depth]?.[position];
      return `rank ${fmtRank(r ?? 10 ** v)}`;
    }
    case "next_token_rank":
      return `rank ${fmtRank(10 ** v)}`;
    case "lens_top1_prob":
    case "target_prob":
      return `p = ${fmtProb(v)}`;
    case "lens_entropy":
      return `${fmtNum(v, 3)} nats`;
    case "resid_norm":
      return `‖h‖ = ${fmtNum(v, 3)}`;
    case "cos_delta":
      return `1 − cos = ${fmtNum(v, 4)}`;
    case "delta_resid":
      return `Δ = ${fmtNum(v, 4)}`;
  }
}

/** Tick label for the colorbar (value in API units). */
export function formatTick(m: MatrixResponse, v: number): string {
  const info = metricInfo(m.metric);
  if (info.log10Values) return fmtRank(10 ** v);
  if (m.metric === "lens_top1_prob" || m.metric === "target_prob") return v === 0 ? "0" : v === 1 ? "1" : v.toFixed(2);
  return fmtNum(v, 2);
}
