/**
 * SelfLab API types + typed client (`/api/selflab/*`, interface `selflab-1`).
 *
 * Source of truth: packages/neuralscope/neuralscope/selflab/schemas.py (persisted objects) and
 * docs/selflab/INTERFACES.md §5 (routes). The persisted objects below mirror the pydantic models
 * field-for-field. A few route envelopes (status, case detail, export preview, suggestions) are not
 * frozen in the contract, so they are typed loosely and read through the `normalize*` helpers,
 * which never invent values: a missing field stays missing and the UI says so.
 */

import { ApiError, isAbortError, request } from "./client";

export const SCHEMA_VERSION = "selflab-1";

// ---------------------------------------------------------------- literals (schemas.py)

export type AccessLevel = "white_box_local" | "api_only";
export type ParticipantRole = "target" | "investigator" | "activation_decoder" | "statistical_scorer" | "observer";
export type EvidenceKind = "direct_observation" | "deterministic_derivation" | "model_report" | "interpretive_hypothesis";
export type TestStatus = "untested" | "supported" | "contradicted" | "inconclusive" | "control_failed";
export type Split = "discovery" | "debug_pilot" | "heldout_confirmation";
export type TrialStatus = "ok" | "malformed" | "refusal" | "abstention" | "truncated" | "error";
export type SelfLabJobStatus = "committed" | "queued" | "running" | "cancelling" | "cancelled" | "done" | "error";
export type ClaimType = "source_attribution" | "verification" | "process_description" | "self_prediction" | "other";
export type TemplateId = "source_influence_case" | "source_influence_battery" | "csp1_carry" | "blinded_state";
export type FeatureStatus = "implemented" | "research_prototype" | "fixture" | "api_only_view" | "unsupported" | "roadmap";
export type SourceCategory = "document" | "user_suggestion" | "other_or_unknown" | "insufficient_evidence";
export type SegmentKind = "document" | "user_hint" | "question" | "system" | "other";
export type ReceiptKind = "context_delivery" | "tool_call" | "retrieval" | "api_usage";

export const SOURCE_CATEGORIES: readonly SourceCategory[] = ["document", "user_suggestion", "other_or_unknown", "insufficient_evidence"];
export const EVIDENCE_KINDS: readonly EvidenceKind[] = ["direct_observation", "deterministic_derivation", "model_report", "interpretive_hypothesis"];
export const TEST_STATUSES: readonly TestStatus[] = ["untested", "supported", "contradicted", "inconclusive", "control_failed"];
export const CLAIM_TYPES: readonly ClaimType[] = ["source_attribution", "verification", "process_description", "self_prediction", "other"];
export const PARTICIPANT_ROLES: readonly ParticipantRole[] = ["target", "investigator", "activation_decoder", "statistical_scorer", "observer"];
/** Statuses during which the experiment record keeps changing (poll it). */
export const ACTIVE_JOB_STATUSES: readonly SelfLabJobStatus[] = ["queued", "running", "cancelling"];

// ---------------------------------------------------------------- provenance

export interface ModelProvenance {
  model_id: string;
  revision?: string | null;
  tokenizer_id: string;
  tokenizer_revision?: string | null;
  chat_template_sha256?: string | null;
  dtype: string;
  device: string;
  backend?: string;
  attn_implementation?: string;
  library_versions?: Record<string, string>;
  is_demo?: boolean;
}

export interface RunProvenance {
  model: ModelProvenance;
  code_commit?: string | null;
  code_dirty?: boolean | null;
  analysis_version: string;
  decoding?: Record<string, unknown>;
  seed?: number | null;
  platform?: string | null;
  created_at: string;
}

// ---------------------------------------------------------------- participants / context

export interface Participant {
  role: ParticipantRole;
  id: string;
  /** Absent on the backend's `disabled_participants` entries, which carry `status` + `reason` instead. */
  description?: string;
  /** Access manifest: names of the fields/inputs visible to this participant. */
  sees?: string[];
  enabled?: boolean;
  status?: FeatureStatus | string;
  reason?: string;
}

export interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

/** A labelled character span inside one message, with its token span in the target input. */
export interface ContextSegment {
  kind: SegmentKind;
  message_index: number;
  char_start: number;
  char_end: number;
  token_start?: number | null;
  token_end?: number | null;
  sha256: string;
}

/** An independently checkable record. A delivery receipt proves access, never causal use. */
export interface Receipt {
  id: string;
  kind: ReceiptKind;
  statement: string;
  proves: string;
  does_not_prove: string;
  data?: Record<string, unknown>;
}

export interface ClaimSpan {
  text: string;
  char_start: number;
  char_end: number;
  token_start?: number | null;
  token_end?: number | null;
}

export interface Case {
  id: string;
  created_at: string;
  title: string;
  access_level: AccessLevel;
  run_id?: string | null;
  messages: Message[];
  response_text: string;
  claim: ClaimSpan;
  claim_type: ClaimType;
  segments?: ContextSegment[];
  receipts?: Receipt[];
  participants?: Participant[];
  provenance?: RunProvenance | null;
  experiment_ids?: string[];
  is_fixture?: boolean;
  notes?: string | null;
  schema_version?: string;
}

// ---------------------------------------------------------------- experiments

export interface Budget {
  max_trials: number;
  max_new_tokens: number;
  max_seconds: number;
  max_prompt_tokens: number;
}

export interface Hypothesis {
  id: string;
  statement: string;
  metric: string;
  direction?: "greater" | "less" | "two_sided" | "equivalence";
  threshold?: number | null;
  primary?: boolean;
}

export interface ExperimentSpec {
  template: TemplateId;
  version?: string;
  case_id?: string | null;
  split?: Split;
  seed?: number;
  params?: Record<string, unknown>;
  hypotheses?: Hypothesis[];
  primary_metric?: string;
  exclusions?: string[];
  controls?: string[];
  budget?: Partial<Budget>;
  notes?: string | null;
}

export interface TrialPlanItem {
  index: number;
  item_id: string;
  group_id: string;
  family: string;
  condition: string;
  pair_key?: string | null;
  label_order?: string[] | null;
  params?: Record<string, unknown>;
}

export interface CommittedSpec {
  spec: ExperimentSpec;
  spec_sha256: string;
  plan_sha256: string;
  n_planned: number;
  committed_at: string;
  provenance?: RunProvenance | null;
}

export interface Trial {
  id: string;
  experiment_id: string;
  index: number;
  item_id: string;
  group_id: string;
  family: string;
  condition: string;
  pair_key?: string | null;
  status: TrialStatus;
  prompt_sha256?: string;
  input_ids_sha256?: string;
  n_prompt_tokens?: number;
  output_text: string;
  output_ids?: number[];
  stopped_by?: string | null;
  parsed?: Record<string, unknown>;
  /** Separate ground-truth fields, never one label. */
  ground_truth?: Record<string, unknown>;
  /** Log-likelihoods with explicit token counts. */
  measurements?: Record<string, unknown>;
  run_ids?: string[];
  latency_s?: number;
  n_generated_tokens?: number;
  error?: string | null;
  created_at: string;
  /** `GET /trials/{id}` only: Lab runs linked to this trial (e.g. open-in-lab replays). */
  lab_runs?: LabRunLink[];
}

export interface LabRunLink {
  run_id: string;
  created_at?: string;
  kind?: string;
  reproduced?: boolean;
  compared?: string;
  [key: string]: unknown;
}

export interface Alternative {
  explanation: string;
  discriminating_test: string;
  state: "ruled_out" | "not_ruled_out" | "untested";
  evidence_ids?: string[];
}

export interface Scope {
  checkpoint: string;
  task_distribution: string;
  intervention: string;
  unit: string;
  n_units: number;
  n_trials: number;
  controls?: string[];
  tested_generalization?: string;
  split?: Split;
}

/** Evidence keeps KIND, STATUS and SCOPE as separate fields; no combined verdict exists. */
export interface EvidenceItem {
  id: string;
  case_id?: string | null;
  experiment_id?: string | null;
  kind: EvidenceKind;
  status: TestStatus;
  title: string;
  statement: string;
  measurement?: Record<string, unknown>;
  scope?: Scope | null;
  trial_ids?: string[];
  receipt_ids?: string[];
  run_ids?: string[];
  analysis?: string;
  supports?: string[];
  contradicts?: string[];
  alternatives?: Alternative[];
  limitations?: string[];
  created_at: string;
}

export interface ExperimentRecord {
  id: string;
  case_id?: string | null;
  committed: CommittedSpec;
  status: SelfLabJobStatus;
  progress: number;
  message?: string;
  n_done: number;
  n_errors: number;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  cost?: Record<string, unknown>;
  analysis?: Record<string, unknown> | null;
  evidence_ids?: string[];
  error?: string | null;
  is_fixture?: boolean;
}

// ---------------------------------------------------------------- route envelopes (§5)

export interface ChatRequest {
  messages?: Message[];
  question: string;
  document?: string | null;
  hint?: string | null;
  system?: string | null;
  max_new_tokens?: number;
}

export interface ChatResponse {
  run_id: string;
  messages: Message[];
  response_text: string;
  segments: ContextSegment[];
  receipts: Receipt[];
  provenance?: RunProvenance | null;
  is_fixture: boolean;
  stopped_by?: string | null;
  max_new_tokens?: number;
  /** False when re-tokenising the composed messages did not reproduce the run's input ids (token spans withheld). */
  tokenization_consistent?: boolean;
  trace_consistent?: boolean | null;
  n_prompt_tokens?: number;
  n_generated_tokens?: number;
  lab_url?: string;
}

export interface CreateCaseWhiteBox {
  run_id: string;
  char_start: number;
  char_end: number;
  claim_type?: ClaimType;
  title?: string;
}

export interface CreateCaseApiOnly {
  access_level: "api_only";
  messages: Message[];
  response_text: string;
  char_start: number;
  char_end: number;
  receipts?: Receipt[];
  claim_type?: ClaimType;
  title?: string;
}

export type CreateCaseRequest = CreateCaseWhiteBox | CreateCaseApiOnly;

/** `GET /cases/{id}`: the case plus its experiments and evidence (normalised). */
export interface CaseDetail {
  case: Case;
  experiments: ExperimentRecord[];
  evidence: EvidenceItem[];
}

export interface TemplateInfo {
  id: TemplateId | string;
  title: string;
  description: string;
  feature_status: FeatureStatus;
  requires_case: boolean;
  params_schema?: Record<string, unknown> | null;
  /** Present when requested with `?case_id=` (or for templates that need no case). */
  default_spec?: ExperimentSpec | null;
  /** Why `default_spec` is null (e.g. "requires ?case_id=", or the case lacks a document segment). */
  default_spec_error?: string;
  access_levels?: AccessLevel[];
}

export interface OpenInLabResponse {
  run_id: string;
  reproduced: boolean;
  diff?: unknown;
  trial_id?: string;
  compared?: string;
  lab_url?: string;
  existing_run_ids?: string[];
  is_fixture?: boolean;
}

/** A preset invented-entity task the backend offers for the composer. */
export interface ChatExample {
  title?: string;
  document?: string | null;
  hint?: string | null;
  question: string;
  note?: string;
  /** Controlled counterfactual inputs for the case-local template (all optional). */
  replacement_text?: string | null;
  hint_text?: string | null;
  original_value?: string | null;
  replacement_value?: string | null;
  value_set?: readonly string[] | null;
}

export interface AccessLevelInfo {
  level: AccessLevel;
  enabled: boolean;
  description?: string;
  reason?: string;
}

/** `GET /status` — loosely typed (route envelope not frozen); read it through the helpers below. */
export interface SelfLabStatus {
  schema_version?: string;
  service_version?: string;
  capabilities?: Record<string, unknown>;
  access_levels?: unknown;
  participants?: Participant[];
  /** Participants the backend lists as present-but-disabled (LLM investigator, activation decoders). */
  disabled_participants?: Participant[];
  /** Backend shape: `{status, error, provenance}`; a bare ModelProvenance is also accepted. */
  model?: { status?: string; error?: string | null; provenance?: ModelProvenance | null } | ModelProvenance | null;
  provenance?: ModelProvenance | RunProvenance | null;
  /** Feature id → FeatureStatus. Backend key: `features` (`feature_statuses` also accepted). */
  features?: unknown;
  feature_statuses?: unknown;
  templates?: unknown;
  limits?: Record<string, number>;
  is_demo?: boolean;
  is_fixture?: boolean;
  example?: ChatExample | null;
  examples?: ChatExample[];
  [key: string]: unknown;
}

export interface ExportField {
  name: string;
  bytes: number | null;
  count?: number | null;
  free_text: boolean;
  /** Free-text sub-paths inside this section, as flagged by the backend. */
  free_text_paths?: string[];
  note?: string;
}

export interface ExportPreview {
  experiment_id?: string;
  /** Top-level sections of the reproduction record. */
  fields: ExportField[];
  total_bytes: number | null;
  /** Names of sections that contain free text. */
  free_text_fields: string[];
  /** Every flagged free-text path with its size (backend `free_text_fields`). */
  free_text_paths: { path: string; bytes: number | null }[];
  free_text_bytes: number | null;
  /** Size of the record with free text redacted (`exclude_free_text=1`). */
  redacted_bytes: number | null;
  redaction?: string;
  raw_tensors_included?: boolean;
  n_tensor_references?: number | null;
  trial_status_counts?: Record<string, number> | null;
  is_fixture?: boolean;
  notes: string[];
  raw: unknown;
}

export interface Suggestion {
  template: string;
  title: string;
  rationale: string;
  feature_status?: FeatureStatus;
  rule?: string;
  /** The rules consider this template applicable to the case (backend `applicable`). */
  applicable?: boolean;
  /** The template is registered on this server (backend `available`). */
  available?: boolean;
  /** What the case lacks for this template to apply. */
  missing?: string[];
  note?: string;
}

export interface SuggestResponse {
  participant: string;
  suggestions: Suggestion[];
  llm_investigator?: { status?: FeatureStatus | string; reason?: string } | null;
  disclaimer?: string;
  raw: unknown;
}

export interface TrialsQuery {
  offset?: number;
  limit?: number;
}

// ---------------------------------------------------------------- normalisers (pure)

const isObj = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const asNum = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

/** Accept `{case, experiments, evidence}` or a bare `Case` (optionally with those arrays inlined). */
export function normalizeCaseDetail(raw: unknown): CaseDetail {
  if (isObj(raw) && isObj(raw.case)) {
    return {
      case: raw.case as unknown as Case,
      experiments: Array.isArray(raw.experiments) ? (raw.experiments as ExperimentRecord[]) : [],
      evidence: Array.isArray(raw.evidence) ? (raw.evidence as EvidenceItem[]) : [],
    };
  }
  if (isObj(raw) && typeof raw.id === "string" && isObj(raw.claim)) {
    const { experiments, evidence, ...rest } = raw;
    return {
      case: rest as unknown as Case,
      experiments: Array.isArray(experiments) ? (experiments as ExperimentRecord[]) : [],
      evidence: Array.isArray(evidence) ? (evidence as EvidenceItem[]) : [],
    };
  }
  throw new ApiError(0, "Unexpected case payload from /api/selflab/cases/{id}", raw);
}

function listOf<T>(raw: unknown, key: string): T[] {
  if (Array.isArray(raw)) return raw as T[];
  if (isObj(raw) && Array.isArray(raw[key])) return raw[key] as T[];
  if (isObj(raw) && Array.isArray(raw.items)) return raw.items as T[];
  return [];
}

/** Access-level availability from `/status`. API-only is disabled unless the backend says enabled. */
export function accessLevels(status: SelfLabStatus | null | undefined): AccessLevelInfo[] {
  const out: AccessLevelInfo[] = [];
  const raw = status?.access_levels;
  const read = (level: AccessLevel, v: unknown): AccessLevelInfo => {
    if (typeof v === "boolean") return { level, enabled: v };
    if (typeof v === "string") return { level, enabled: v === "enabled" || v === "implemented" };
    if (isObj(v)) {
      const enabled = v.enabled === true || v.status === "enabled" || v.status === "implemented";
      return {
        level,
        enabled,
        description: typeof v.description === "string" ? v.description : enabled && typeof v.note === "string" ? v.note : undefined,
        reason: typeof v.reason === "string" ? v.reason : !enabled && typeof v.note === "string" ? v.note : undefined,
      };
    }
    return { level, enabled: false };
  };
  if (Array.isArray(raw)) {
    for (const v of raw) {
      if (isObj(v) && (v.level === "white_box_local" || v.level === "api_only")) out.push(read(v.level, v));
      else if (isObj(v) && (v.id === "white_box_local" || v.id === "api_only")) out.push(read(v.id, v));
    }
  } else if (isObj(raw)) {
    for (const level of ["white_box_local", "api_only"] as const) if (level in raw) out.push(read(level, raw[level]));
  }
  if (!out.some((a) => a.level === "white_box_local")) out.unshift({ level: "white_box_local", enabled: true });
  if (!out.some((a) => a.level === "api_only"))
    out.push({ level: "api_only", enabled: false, reason: "not reported as enabled by /api/selflab/status" });
  return out;
}

export function isApiOnlyEnabled(status: SelfLabStatus | null | undefined): boolean {
  return accessLevels(status).find((a) => a.level === "api_only")?.enabled === true;
}

const FEATURE_STATUSES: readonly FeatureStatus[] = ["implemented", "research_prototype", "fixture", "api_only_view", "unsupported", "roadmap"];
export function isFeatureStatus(v: unknown): v is FeatureStatus {
  return typeof v === "string" && (FEATURE_STATUSES as readonly string[]).includes(v);
}

/** Feature status for `key` from `/status.feature_statuses` (dict or list); `null` if not reported. */
export function featureStatusOf(status: SelfLabStatus | null | undefined, key: string): { status: FeatureStatus; note?: string } | null {
  const raw = status?.feature_statuses ?? status?.features;
  const read = (v: unknown) => {
    if (isFeatureStatus(v)) return { status: v };
    if (isObj(v) && isFeatureStatus(v.status))
      return { status: v.status, note: typeof v.note === "string" ? v.note : typeof v.reason === "string" ? v.reason : undefined };
    return null;
  };
  if (isObj(raw) && key in raw) return read(raw[key]);
  if (Array.isArray(raw)) {
    const hit = raw.find((v) => isObj(v) && (v.id === key || v.key === key || v.name === key));
    return hit ? read(hit) : null;
  }
  return null;
}

/** Composer presets offered by the backend (`example`, `examples`, or `presets`). Empty when none. */
export function chatExamples(status: SelfLabStatus | null | undefined): ChatExample[] {
  if (!status) return [];
  const cands: unknown[] = [];
  if (status.example) cands.push(status.example);
  for (const key of ["examples", "presets", "chat_presets"]) {
    const v = status[key];
    if (Array.isArray(v)) cands.push(...v);
  }
  return cands.filter((c): c is ChatExample => isObj(c) && typeof c.question === "string" && c.question.trim() !== "");
}

/** Model provenance reported by `/status` (either `model` or `provenance[.model]`). */
export function statusModel(status: SelfLabStatus | null | undefined): ModelProvenance | null {
  if (!status) return null;
  const m = status.model;
  if (isObj(m) && typeof (m as ModelProvenance).model_id === "string") return m as ModelProvenance;
  if (isObj(m) && isObj((m as { provenance?: unknown }).provenance)) return (m as { provenance: ModelProvenance }).provenance;
  const p = status.provenance;
  if (isObj(p) && isObj((p as RunProvenance).model)) return (p as RunProvenance).model;
  if (isObj(p) && typeof (p as ModelProvenance).model_id === "string") return p as ModelProvenance;
  return null;
}

export function statusIsFixture(status: SelfLabStatus | null | undefined): boolean {
  return Boolean(status?.is_demo || status?.is_fixture || statusModel(status)?.is_demo);
}

/** Active participants (case's own list when it has one, else the status list) plus the disabled ones. */
export function allParticipants(status: SelfLabStatus | null | undefined, caseParticipants?: Participant[] | null): Participant[] {
  const base = caseParticipants && caseParticipants.length ? caseParticipants : (status?.participants ?? []);
  const disabled = (status?.disabled_participants ?? []).filter(
    (d) => isObj(d) && typeof d.role === "string" && !base.some((b) => b.id === d.id && b.role === d.role),
  );
  return [...base, ...disabled.map((d) => ({ ...d, enabled: false }))];
}

/** The experiment's analysis: engine wrapper fields (`meta`) and the template's `summary`. */
export function splitAnalysis(analysis: Record<string, unknown> | null | undefined): {
  meta: Record<string, unknown>;
  summary: Record<string, unknown>;
} {
  if (!analysis) return { meta: {}, summary: {} };
  if (isObj(analysis.summary)) {
    const { summary, ...meta } = analysis;
    return { meta, summary: summary as Record<string, unknown> };
  }
  return { meta: {}, summary: analysis };
}

const FREE_TEXT_HINT = /free[_ -]?text|output_text|response_text|statement|messages|notes|claim/i;

/** Export preview: accept `fields` as a list or a `{name: info}` dict; sizes are reported, never estimated. */
export function normalizeExportPreview(raw: unknown): ExportPreview {
  const fields: ExportField[] = [];
  const pushField = (name: string, v: unknown) => {
    if (isObj(v)) {
      const free = v.free_text ?? v.contains_free_text ?? v.has_free_text;
      const sub = Array.isArray(v.free_text_fields) ? v.free_text_fields.filter((x): x is string => typeof x === "string") : undefined;
      fields.push({
        name,
        bytes: asNum(v.bytes ?? v.size_bytes ?? v.size),
        count: asNum(v.count ?? v.n ?? v.n_items),
        free_text: free === true || Boolean(sub && sub.length),
        free_text_paths: sub,
        note: typeof v.note === "string" ? v.note : typeof v.description === "string" ? v.description : undefined,
      });
    } else if (typeof v === "number") {
      fields.push({ name, bytes: asNum(v), free_text: false });
    } else {
      fields.push({ name, bytes: null, free_text: false });
    }
  };
  const paths: { path: string; bytes: number | null }[] = [];
  const notes: string[] = [];
  let out: Partial<ExportPreview> = {};
  if (isObj(raw)) {
    const f = raw.fields ?? raw.included_fields ?? raw.included ?? raw.sections;
    if (Array.isArray(f)) {
      for (const v of f) {
        if (typeof v === "string") pushField(v, null);
        else if (isObj(v)) pushField(String(v.name ?? v.path ?? v.field ?? v.key ?? "?"), v);
      }
    } else if (isObj(f)) {
      for (const [k, v] of Object.entries(f)) pushField(k, v);
    }
    const ft = raw.free_text_fields ?? raw.free_text;
    if (Array.isArray(ft)) {
      for (const x of ft) {
        if (typeof x === "string") paths.push({ path: x, bytes: null });
        else if (isObj(x) && typeof (x.path ?? x.name) === "string") paths.push({ path: String(x.path ?? x.name), bytes: asNum(x.bytes) });
      }
    }
    if (Array.isArray(raw.notes)) notes.push(...raw.notes.filter((x): x is string => typeof x === "string"));
    else if (typeof raw.note === "string") notes.push(raw.note);
    const counts = isObj(raw.trial_status_counts)
      ? (Object.fromEntries(Object.entries(raw.trial_status_counts).filter(([, n]) => typeof n === "number")) as Record<string, number>)
      : null;
    out = {
      experiment_id: typeof raw.experiment_id === "string" ? raw.experiment_id : undefined,
      total_bytes: asNum(raw.total_bytes ?? raw.size_bytes ?? raw.bytes),
      free_text_bytes: asNum(raw.free_text_bytes),
      redacted_bytes: asNum(raw.exclude_free_text_bytes ?? raw.redacted_bytes),
      redaction: typeof raw.redaction === "string" ? raw.redaction : undefined,
      raw_tensors_included: typeof raw.raw_tensors_included === "boolean" ? raw.raw_tensors_included : undefined,
      n_tensor_references: asNum(raw.n_tensor_references),
      trial_status_counts: counts,
      is_fixture: typeof raw.is_fixture === "boolean" ? raw.is_fixture : undefined,
    };
  }
  // A flagged path marks its top-level section ("trials[].output_text" → "trials").
  for (const { path } of paths) {
    const top = path.split(/[.[]/)[0];
    const hit = fields.find((f) => f.name === top || f.name === path);
    if (hit) hit.free_text = true;
    else if (!path.includes(".") && !path.includes("[")) fields.push({ name: path, bytes: null, free_text: true });
  }
  return {
    fields,
    total_bytes: null,
    free_text_bytes: null,
    redacted_bytes: null,
    ...out,
    free_text_fields: fields.filter((f) => f.free_text).map((f) => f.name),
    free_text_paths: paths,
    notes,
    raw,
  };
}

/** True when a field name suggests free text but the preview did not flag it (shown as a caution only). */
export function looksLikeFreeText(name: string): boolean {
  return FREE_TEXT_HINT.test(name);
}

export function normalizeSuggestions(raw: unknown): SuggestResponse {
  const list = listOf<unknown>(raw, "suggestions");
  const suggestions: Suggestion[] = [];
  for (const s of list) {
    if (!isObj(s)) continue;
    const template = String(s.template ?? s.template_id ?? s.id ?? "");
    if (!template) continue;
    suggestions.push({
      template,
      title: String(s.title ?? s.label ?? template),
      rationale: String(s.rationale ?? s.reason ?? s.why ?? ""),
      feature_status: isFeatureStatus(s.feature_status) ? s.feature_status : undefined,
      rule: typeof s.rule === "string" ? s.rule : typeof s.rule_id === "string" ? s.rule_id : undefined,
      applicable: typeof s.applicable === "boolean" ? s.applicable : undefined,
      available: typeof s.available === "boolean" ? s.available : undefined,
      missing: Array.isArray(s.missing) ? s.missing.filter((x): x is string => typeof x === "string") : undefined,
      note: typeof s.note === "string" ? s.note : undefined,
    });
  }
  const participant =
    isObj(raw) && typeof raw.participant === "string"
      ? raw.participant
      : isObj(raw) && isObj(raw.participant) && typeof raw.participant.id === "string"
        ? raw.participant.id
        : "rules:template-suggester@1";
  const llm = isObj(raw) && isObj(raw.llm_investigator) ? (raw.llm_investigator as SuggestResponse["llm_investigator"]) : null;
  const disclaimer = isObj(raw) && typeof raw.disclaimer === "string" ? raw.disclaimer : undefined;
  return { participant, suggestions, llm_investigator: llm, disclaimer, raw };
}

// ---------------------------------------------------------------- client

const enc = encodeURIComponent;
const P = "/selflab";

export const selflab = {
  status: (signal?: AbortSignal) => request<SelfLabStatus>(`${P}/status`, { signal }),
  chat: (body: ChatRequest, signal?: AbortSignal) => request<ChatResponse>(`${P}/chat`, { body, signal }),

  createCase: (body: CreateCaseRequest, signal?: AbortSignal) => request<Case>(`${P}/cases`, { body, signal }),
  listCases: (signal?: AbortSignal) => request<unknown>(`${P}/cases`, { signal }).then((r) => listOf<Case>(r, "cases")),
  getCase: (id: string, signal?: AbortSignal) => request<unknown>(`${P}/cases/${enc(id)}`, { signal }).then(normalizeCaseDetail),

  templates: (caseId?: string | null, signal?: AbortSignal) =>
    request<unknown>(`${P}/templates`, { query: { case_id: caseId ?? undefined }, signal }).then((r) => listOf<TemplateInfo>(r, "templates")),

  /** Validates via the template, builds the plan and COMMITS it (hashes + timestamp). Does not run. */
  commitExperiment: (spec: ExperimentSpec, signal?: AbortSignal) => request<ExperimentRecord>(`${P}/experiments`, { body: { spec }, signal }),
  runExperiment: (id: string) => request<ExperimentRecord>(`${P}/experiments/${enc(id)}/run`, { method: "POST" }),
  cancelExperiment: (id: string) => request<ExperimentRecord>(`${P}/experiments/${enc(id)}/cancel`, { method: "POST" }),
  resumeExperiment: (id: string) => request<ExperimentRecord>(`${P}/experiments/${enc(id)}/resume`, { method: "POST" }),
  listExperiments: (caseId?: string | null, signal?: AbortSignal) =>
    request<unknown>(`${P}/experiments`, { query: { case_id: caseId ?? undefined }, signal }).then((r) =>
      listOf<ExperimentRecord>(r, "experiments"),
    ),
  getExperiment: (id: string, signal?: AbortSignal) => request<ExperimentRecord>(`${P}/experiments/${enc(id)}`, { signal }),
  trials: (id: string, q: TrialsQuery = {}, signal?: AbortSignal) =>
    request<unknown>(`${P}/experiments/${enc(id)}/trials`, { query: { offset: q.offset, limit: q.limit }, signal }).then((r) =>
      listOf<Trial>(r, "trials"),
    ),
  trial: (id: string, signal?: AbortSignal) => request<Trial>(`${P}/trials/${enc(id)}`, { signal }),
  openInLab: (trialId: string) => request<OpenInLabResponse>(`${P}/trials/${enc(trialId)}/open-in-lab`, { method: "POST" }),

  evidence: (q: { case_id?: string | null; experiment_id?: string | null }, signal?: AbortSignal) =>
    request<unknown>(`${P}/evidence`, { query: { case_id: q.case_id ?? undefined, experiment_id: q.experiment_id ?? undefined }, signal }).then((r) =>
      listOf<EvidenceItem>(r, "evidence"),
    ),

  exportPreview: (id: string, signal?: AbortSignal) =>
    request<unknown>(`${P}/experiments/${enc(id)}/export`, { query: { preview: 1 }, signal }).then(normalizeExportPreview),
  /** The full reproduction record (spec, hashes, provenance, trials, analysis, evidence; tensor references only). */
  exportRecord: (id: string, opts: { excludeFreeText?: boolean } = {}, signal?: AbortSignal) =>
    request<unknown>(`${P}/experiments/${enc(id)}/export`, { query: { exclude_free_text: opts.excludeFreeText ? 1 : undefined }, signal }),

  suggest: (caseId: string, signal?: AbortSignal) =>
    request<unknown>(`${P}/investigate/suggest`, { body: { case_id: caseId }, signal }).then(normalizeSuggestions),
};

// ---------------------------------------------------------------- polling

export function isActiveJob(status: SelfLabJobStatus | string | null | undefined): boolean {
  return (ACTIVE_JOB_STATUSES as readonly string[]).includes(String(status));
}

export interface PollExperimentOptions {
  signal?: AbortSignal;
  intervalMs?: number;
  onUpdate?: (rec: ExperimentRecord) => void;
  /** Injected for tests. */
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
  fetchRecord?: (id: string, signal?: AbortSignal) => Promise<ExperimentRecord>;
}

function sleepMs(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("Aborted", "AbortError"));
    const t = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(t);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });
}

/**
 * Poll `GET /experiments/{id}` while it is queued / running / cancelling. Resolves with the first
 * record in a settled state (committed, cancelled, done, error); tolerates a few transient failures.
 */
export async function pollExperiment(id: string, opts: PollExperimentOptions = {}): Promise<ExperimentRecord> {
  const { signal, intervalMs = 1200, onUpdate } = opts;
  const sleep = opts.sleep ?? sleepMs;
  const fetchRecord = opts.fetchRecord ?? ((x: string, s?: AbortSignal) => selflab.getExperiment(x, s));
  let failures = 0;
  for (;;) {
    let rec: ExperimentRecord;
    try {
      rec = await fetchRecord(id, signal);
      failures = 0;
    } catch (e) {
      if (isAbortError(e)) throw e;
      if (e instanceof ApiError && e.status !== 404 && failures < 5) {
        failures += 1;
        await sleep(intervalMs * 2, signal);
        continue;
      }
      throw e;
    }
    onUpdate?.(rec);
    if (!isActiveJob(rec.status)) return rec;
    await sleep(intervalMs, signal);
  }
}
