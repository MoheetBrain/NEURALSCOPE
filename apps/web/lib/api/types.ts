/**
 * Response / request types for the NeuralScope FastAPI backend.
 *
 * Source of truth: packages/neuralscope/neuralscope/experiments/service.py (views),
 * apps/api/neuralscope_api/main.py (routes) and schemas/{api,interventions}.py (requests).
 * Numbers that the backend rounds with `_r()` become `null` when they are NaN / ±inf.
 */

/** Epistemic status strings as sent by the API ("causal" = CAUSALLY TESTED). */
export type ApiStatus = "observed" | "derived" | "inferred" | "causal";

export type TraceLevel = "minimal" | "standard" | "full";
export type TokenRole = "template" | "user" | "generated";
export type RunKind = "baseline" | "intervention" | "math" | (string & {});
export type MathStatus = "correct" | "incorrect" | "truncated" | "unparseable" | "ambiguous";

// ---------------------------------------------------------------- model / health

export interface Health {
  status: "loading" | "ready" | "error";
  model_id: string;
  is_demo: boolean;
  error: string | null;
  trace_level_default: TraceLevel;
  version: string;
}

export interface ModelCapabilities {
  residual_stream?: boolean;
  attention_patterns?: boolean;
  head_outputs?: boolean;
  mlp_outputs?: boolean;
  logit_lens?: boolean;
  direct_logit_attribution?: boolean;
  head_ablation?: boolean;
  mlp_ablation?: boolean;
  activation_patching?: boolean;
  steering?: boolean;
  qkv?: boolean;
  mlp_preactivations?: boolean;
  gradients?: boolean;
  sae_features?: boolean;
  transcoders?: boolean;
  kv_cache_inspection?: boolean;
  [key: string]: boolean | undefined;
}

export interface ModelInfo {
  model_id: string;
  architecture: string;
  n_layers: number;
  n_heads: number;
  n_kv_heads: number;
  d_model: number;
  d_head: number;
  d_mlp: number;
  vocab_size: number;
  dtype: string;
  device: string;
  n_params: number;
  tied_embeddings: boolean;
  norm_type: string;
  is_demo: boolean;
  capabilities: ModelCapabilities;
  n_depths: number;
  load_seconds: number | null;
}

// ---------------------------------------------------------------- tokens / runs

export interface TopToken {
  id: number;
  text: string;
  logprob: number | null;
}

export interface TokenInfo {
  index: number;
  id: number;
  text: string;
  special: boolean;
  role: TokenRole;
  /** log P(token[index+1] | tokens[..index]) under the run's own final logits. */
  next_logprob?: number | null;
  next_rank?: number | null;
  entropy?: number | null;
  /** Top-5 final-layer predictions at this position. */
  top?: TopToken[];
}

export interface TokenizeResponse {
  text: string;
  user_span: [number, number] | null;
  tokens: TokenInfo[];
  n_tokens: number;
  limit: number;
  status: ApiStatus;
}

export interface RunTarget {
  position: number;
  token_id: number | null;
  token: string | null;
  contrast_id: number | null;
  contrast: string | null;
  source: string;
  note: string | null;
}

export interface RunSummary {
  id: string;
  created_at: string;
  kind: RunKind;
  model_id: string;
  is_demo: boolean;
  prompt: string;
  generated_text: string;
  free_generated_text: string | null;
  parent_run_id: string | null;
  interventions_label: string;
  n_tokens: number;
  trace_level: TraceLevel;
  correctness: MathStatus | null;
  benchmark_id: string | null;
  notes: string | null;
  problem_kind: string | null;
}

export interface MathProblem {
  id: string;
  kind: string;
  question: string;
  expression?: string;
  operands?: Record<string, unknown>;
  oracle_answer: string;
  answer_type?: string;
  difficulty?: string;
}

export interface MathVerification {
  correct: boolean;
  status: MathStatus;
  parsed_answer: string | null;
  oracle_answer: string;
  note?: string;
}

export interface RunMath {
  problem: MathProblem;
  verification: MathVerification;
  verification_under_intervention?: MathVerification;
  divergence?: { index: number; model_char: string; oracle_char: string; common_prefix: string } | null;
  oracle_answer_logprob?: { tokens: string[]; token_ids: number[]; per_token: (number | null)[]; total: number | null };
  decision_point?: { position: number; k: number; how: string };
  truncated?: boolean;
  label?: MathStatus;
}

export interface RunDetail extends RunSummary {
  prompt_text: string;
  chat: boolean;
  system_prompt: string | null;
  generation: {
    decoding: string;
    max_new_tokens: number;
    stopped_by?: string;
    stopped_by_free?: string;
    kv_cache?: boolean;
    attn_implementation?: string;
    dtype?: string;
    seconds_generate?: number;
    seconds_capture?: number;
    teacher_forced_on?: string;
  };
  input_ids: number[];
  prompt_len: number;
  user_span: [number, number] | null;
  generated_ids: number[];
  free_generated_ids?: number[] | null;
  target: RunTarget | null;
  interventions: Intervention[];
  metrics: RunMetrics;
  math: RunMath | null;
  tags: string[];
  trace_consistent: boolean | null;
  tokens: TokenInfo[];
  depth_labels: string[];
  children: RunSummary[];
  status: Record<string, ApiStatus>;
}

export interface RunMetrics {
  comparison?: InterventionComparison;
  /** Head-ablation and residual-patching sweeps run on this run (appended by the backend). */
  sweeps?: (HeadSweepResult | PatchSweepResult)[];
  lazy_recapture?: { level: TraceLevel; max_abs_resid_drift: number; at: string };
  [key: string]: unknown;
}

// ---------------------------------------------------------------- views

export type MatrixMetric =
  | "lens_top1_prob"
  | "lens_entropy"
  | "target_prob"
  | "target_rank"
  | "next_token_rank"
  | "resid_norm"
  | "cos_delta"
  | "delta_resid";

export interface MatrixResponse {
  run_id: string;
  metric: MatrixMetric;
  label: string;
  status: ApiStatus;
  scale: "linear" | "log";
  colormap: "sequential" | "sequential_r" | "diverging" | (string & {});
  depth_labels: string[];
  n_depths: number;
  n_tokens: number;
  /** values[depth][position]; `null` where undefined (e.g. last position of next_token_rank). */
  values: (number | null)[][];
  vmin: number | null;
  vmax: number | null;
  top1_tokens?: string[][];
  ranks?: number[][];
  target?: { token_id: number; token: string };
  parent_run_id?: string;
}

export interface LensEntry {
  id: number;
  text: string;
  logit: number | null;
  prob: number | null;
}

export interface CellResponse {
  run_id: string;
  depth: number;
  depth_label: string;
  position: number;
  token: string;
  token_id: number;
  layer_note: string;
  vector: {
    status: ApiStatus;
    d_model: number;
    values: (number | null)[];
    norm: number | null;
    mean: number | null;
    std: number | null;
    max_abs: number | null;
    top_dims: { dim: number; value: number | null }[];
    histogram: { counts: number[]; min: number | null; max: number | null };
  };
  geometry: {
    status: ApiStatus;
    cos_prev_depth: number | null;
    cos_next_depth: number | null;
    cos_final_depth: number | null;
    norm_ratio_prev: number | null;
  };
  lens: { status: ApiStatus; top: LensEntry[]; entropy: number | null };
  next_token?: { id: number; text: string; lens_rank: number; lens_prob: number | null };
  target?: { id: number; text: string; lens_rank: number; lens_prob: number | null; lens_logit: number | null };
}

export interface LogitLensDepth {
  depth: number;
  label: string;
  top: LensEntry[];
  entropy: number | null;
  target?: { rank: number; prob: number | null; logit: number | null; logprob: number | null };
}

export interface LogitLensResponse {
  run_id: string;
  position: number;
  token: string;
  predicts_next: string | null;
  target: { id: number; text: string } | null;
  depths: LogitLensDepth[];
  first_depth_target_top1: number | null;
  status: ApiStatus;
  method: string;
}

export interface AttentionHeadSummary {
  head: number;
  entropy: number | null;
  attn_to_first: number | null;
  attn_to_self: number | null;
  top_sources: { position: number; weight: number | null }[];
  output_norm_at_query: number | null;
  kv_group: number;
}

export interface AttentionResponse {
  run_id: string;
  layer: number;
  query_position: number;
  n_heads: number;
  n_tokens: number;
  /** patterns[head][query][key] */
  patterns: (number | null)[][][];
  /** query_rows[head][key] at the query position */
  query_rows: (number | null)[][];
  head_output_norms: (number | null)[][];
  heads: AttentionHeadSummary[];
  status: Record<string, ApiStatus>;
  caveat: string;
}

export interface HeadDetailResponse {
  run_id: string;
  layer: number;
  head: number;
  position: number;
  pattern: (number | null)[][];
  output_norm_by_position: (number | null)[];
  promoted: { id: number; text: string; value: number | null }[];
  suppressed: { id: number; text: string; value: number | null }[];
  dla_to_target?: { token: string; value: number | null };
  status: Record<string, ApiStatus>;
}

export interface DlaResponse {
  run_id: string;
  position: number;
  target: { id: number; text: string };
  contrast: { id: number; text: string } | null;
  embed: number | null;
  heads: (number | null)[][];
  mlps: (number | null)[];
  sum_of_components: number | null;
  actual_logit: number | null;
  decomposition_error: number | null;
  final_norm_scale: number | null;
  top_heads: { layer: number; head: number; value: number | null }[];
  status: ApiStatus;
  caveat: string;
}

// ---------------------------------------------------------------- interventions

export interface HeadAblation {
  type: "head_ablation";
  layer: number;
  head: number;
  factor?: number;
  positions?: number[] | null;
}

export interface MLPAblation {
  type: "mlp_ablation";
  layer: number;
  factor?: number;
  positions?: number[] | null;
}

export interface ResidualPatch {
  type: "resid_patch";
  depth: number;
  position: number;
  source_run_id: string;
  source_position?: number | null;
  alpha?: number;
}

export interface ResidualSteer {
  type: "resid_steer";
  depth: number;
  positions?: number[] | null;
  plus_run_id: string;
  plus_position: number;
  minus_run_id: string;
  minus_position: number;
  scale?: number;
}

export type Intervention = HeadAblation | MLPAblation | ResidualPatch | ResidualSteer;
export type InterventionType = Intervention["type"];

export interface TargetStats {
  logit: number | null;
  prob: number | null;
  logprob: number | null;
  rank: number;
  logit_diff?: number | null;
}

export interface PerPositionEffect {
  position: number;
  next_token: string | null;
  kl: number | null;
  base_logprob: number | null;
  int_logprob: number | null;
  base_top1: string;
  int_top1: string;
  top1_changed: boolean;
}

export interface InterventionComparison {
  target: RunTarget | null;
  target_position: number;
  before: TargetStats | null;
  after: TargetStats | null;
  delta: Partial<Record<"logit" | "prob" | "logprob" | "rank" | "logit_diff", number | null>> | null;
  kl_at_target: number | null;
  max_kl: number | null;
  top_before: { id: number; text: string; prob: number | null; logit: number | null }[];
  top_after: { id: number; text: string; prob: number | null; logit: number | null }[];
  per_position: PerPositionEffect[];
  output_before: string;
  output_after: string;
  output_changed: boolean;
  changed_token_indices: number[];
  first_changed_step?: number | null;
  resid_delta_by_depth: (number | null)[];
  earliest_affected_depth: number | null;
  status: ApiStatus;
  scope: string;
  correctness_before?: MathStatus | null;
  correctness_after?: MathStatus | null;
}

export interface InterventionResponse {
  run: RunSummary;
  comparison: InterventionComparison;
}

// ---------------------------------------------------------------- comparison

export interface CompareSeries {
  resid_norm: (number | null)[];
  cos_change: (number | null)[];
  lens_entropy: (number | null)[];
  lens_top1: string[];
  lens_top1_prob: (number | null)[];
  target_rank?: number[];
  target_prob?: (number | null)[];
}

export interface CrossRunComparison {
  mode: "cross_run";
  run_a: RunSummary;
  run_b: RunSummary;
  position_a: number;
  position_b: number;
  token_a: string;
  token_b: string;
  target_a: Partial<RunTarget>;
  target_b: Partial<RunTarget>;
  depth_labels: string[];
  series_a: CompareSeries;
  series_b: CompareSeries;
  cross_cosine: (number | null)[];
  final_kl_a_b: number | null;
  status: ApiStatus;
  caveat: string;
  attn_entropy_a?: (number | null)[][];
  attn_entropy_b?: (number | null)[][];
  attn_entropy_diff?: (number | null)[][];
  attn_error?: string;
  same_length: boolean;
  token_diff_positions?: number[];
  resid_rel_diff?: (number | null)[][];
}

export interface InterventionModeComparison {
  mode: "intervention";
  run_a: RunSummary;
  run_b: RunSummary;
  comparison: InterventionComparison;
}

export type CompareResponse = CrossRunComparison | InterventionModeComparison;

// ---------------------------------------------------------------- jobs & sweeps

export type JobStatus = "queued" | "running" | "done" | "error";

export interface Job<R = unknown> {
  id: string;
  kind: string;
  status: JobStatus;
  progress: number;
  message: string;
  result: R | null;
  error: string | null;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  meta: Record<string, unknown>;
}

export interface HeadSweepResult {
  id: string;
  kind: "head_ablation";
  run_id: string;
  position: number;
  target: { id: number; text: string };
  contrast: { id: number; text: string } | null;
  metric: "logit_diff" | "target_logprob";
  baseline_metric: number | null;
  baseline_prob: number | null;
  factor: number;
  delta_metric: (number | null)[][];
  delta_prob: (number | null)[][];
  kl: (number | null)[][];
  most_harmful: { layer: number; head: number; delta: number | null }[];
  most_helpful: { layer: number; head: number; delta: number | null }[];
  seconds: number;
  created_at: string;
  status: ApiStatus;
  scope: string;
}

export interface PatchSweepResult {
  id: string;
  kind: "resid_patch";
  clean_run_id: string;
  corrupt_run_id: string;
  position: number;
  positions: number[];
  diff_positions: number[];
  clean_target: string;
  corrupt_target: string;
  metric: string;
  metric_clean: number | null;
  metric_corrupt: number | null;
  restoration: (number | null)[][];
  raw_metric: (number | null)[][];
  tokens: string[];
  depth_labels: string[];
  seconds: number;
  status: ApiStatus;
  scope: string;
}

// ---------------------------------------------------------------- Math Lab

export interface BenchmarkResultRow {
  problem_id: string;
  kind: string;
  difficulty?: string;
  question: string;
  oracle_answer: string;
  model_output: string;
  parsed_answer: string | null;
  status: MathStatus;
  correct: boolean;
  run_id: string;
  oracle_logprob: number | null;
}

export interface BenchmarkEvalSummary {
  id: string;
  benchmark_id: string;
  interventions: Intervention[];
  interventions_label: string;
  n: number;
  accuracy_before: number;
  accuracy_after: number;
  flips_correct_to_incorrect: number;
  flips_incorrect_to_correct: number;
  mean_delta_oracle_logprob: number | null;
  sd_delta_oracle_logprob: number | null;
  sem_delta_oracle_logprob: number | null;
  seconds: number;
  created_at: string;
  status: ApiStatus;
  scope: string;
}

export interface BenchmarkEvalRow {
  problem_id: string;
  question: string;
  run_id: string;
  correct_before: boolean;
  correct_after: boolean;
  status_after: MathStatus;
  output_after: string;
  oracle_logprob_before: number | null;
  oracle_logprob_after: number | null;
  delta_oracle_logprob: number | null;
}

export interface BenchmarkEvalResult extends BenchmarkEvalSummary {
  rows: BenchmarkEvalRow[];
}

export interface BenchmarkSummary {
  n: number;
  correct: number;
  incorrect: number;
  unparseable: number;
  truncated: number;
  accuracy: number | null;
  by_kind: Record<string, { n: number; correct: number; accuracy: number }>;
  status: ApiStatus;
  seconds?: number;
  intervention_evals?: BenchmarkEvalSummary[];
}

export interface Benchmark {
  id: string;
  created_at: string;
  model_id: string;
  is_demo: boolean | number;
  name: string;
  results: BenchmarkResultRow[];
  summary: BenchmarkSummary | null;
  status: string;
}

export interface MathVerifyResponse extends MathVerification {
  status_label: ApiStatus;
  oracle: string;
}

// ---------------------------------------------------------------- experiments

export interface ExperimentPayload {
  run_id?: string;
  diff_run_id?: string | null;
  selection?: { depth: number; position: number } | null;
  metric?: MatrixMetric;
  tab?: string;
  attention?: { layer: number | null; head: number | null };
  [key: string]: unknown;
}

export interface Experiment {
  id: string;
  created_at: string;
  updated_at: string;
  name: string;
  notes: string;
  run_ids: string[];
  payload: ExperimentPayload;
  runs?: RunSummary[];
}

// ---------------------------------------------------------------- requests

export interface CreateRunRequest {
  prompt: string;
  chat?: boolean;
  system_prompt?: string | null;
  max_new_tokens?: number;
  trace_level?: TraceLevel | null;
  target_text?: string | null;
  target_token_id?: number | null;
  target_position?: number | null;
  contrast_text?: string | null;
  notes?: string | null;
}

export interface SetTargetRequest {
  target_text?: string | null;
  target_token_id?: number | null;
  target_position?: number | null;
  contrast_text?: string | null;
  contrast_token_id?: number | null;
}
