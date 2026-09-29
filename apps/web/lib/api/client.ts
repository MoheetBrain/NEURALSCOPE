/**
 * Typed fetch wrappers for the NeuralScope API.
 *
 * The browser always calls same-origin `/api/*` (Next rewrites proxy it to FastAPI).
 * Errors surface the backend's `detail` message (string or pydantic validation list)
 * as an `ApiError`; aborted requests reject with the original AbortError so callers
 * can ignore superseded requests via `isAbortError`.
 */

import type {
  AttentionResponse,
  Benchmark,
  BenchmarkEvalResult,
  CellResponse,
  CompareResponse,
  CreateRunRequest,
  DlaResponse,
  Experiment,
  ExperimentPayload,
  HeadDetailResponse,
  HeadSweepResult,
  Health,
  Intervention,
  InterventionResponse,
  Job,
  LogitLensResponse,
  MathProblem,
  MathVerifyResponse,
  MatrixMetric,
  MatrixResponse,
  ModelInfo,
  PatchSweepResult,
  RunDetail,
  RunSummary,
  RunTarget,
  SetTargetRequest,
  TokenizeResponse,
  TraceLevel,
} from "./types";

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;
  readonly body: unknown;

  constructor(status: number, detail: string, body?: unknown) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.body = body;
  }
}

export function isAbortError(e: unknown): boolean {
  return (
    (e instanceof DOMException && e.name === "AbortError") ||
    (typeof e === "object" && e !== null && (e as { name?: string }).name === "AbortError")
  );
}

/** Human-readable message for anything thrown by the client. */
export function errorMessage(e: unknown): string {
  if (e instanceof ApiError) return e.detail;
  if (e instanceof Error) return e.message;
  return String(e);
}

interface ValidationIssue {
  loc?: (string | number)[];
  msg?: string;
}

/** Turn FastAPI's `detail` (string | validation issue list | object) into one message. */
export function formatDetail(detail: unknown, fallback: string): string {
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail) && detail.length > 0) {
    return detail
      .map((d: ValidationIssue) => {
        const loc = Array.isArray(d?.loc) ? d.loc.filter((p) => p !== "body").join(".") : "";
        return loc ? `${loc}: ${d?.msg ?? "invalid"}` : (d?.msg ?? JSON.stringify(d));
      })
      .join("; ");
  }
  if (detail && typeof detail === "object") return JSON.stringify(detail);
  return fallback;
}

type Query = Record<string, string | number | boolean | null | undefined>;

export interface RequestOptions {
  method?: "GET" | "POST" | "PATCH" | "DELETE";
  body?: unknown;
  query?: Query;
  signal?: AbortSignal;
}

let BASE = "/api";

/** Override the API base (tests). */
export function setApiBase(base: string): void {
  BASE = base.replace(/\/+$/, "");
}

export function buildUrl(path: string, query?: Query): string {
  let url = `${BASE}${path.startsWith("/") ? path : `/${path}`}`;
  if (query) {
    const qs = Object.entries(query)
      .filter(([, v]) => v !== undefined && v !== null && v !== "")
      .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
      .join("&");
    if (qs) url += `?${qs}`;
  }
  return url;
}

export async function request<T>(path: string, opts: RequestOptions = {}): Promise<T> {
  const { method = opts.body !== undefined ? "POST" : "GET", body, query, signal } = opts;
  let res: Response;
  try {
    res = await fetch(buildUrl(path, query), {
      method,
      signal,
      headers: body !== undefined ? { "content-type": "application/json", accept: "application/json" } : { accept: "application/json" },
      body: body !== undefined ? JSON.stringify(body) : undefined,
      cache: "no-store",
    });
  } catch (e) {
    if (isAbortError(e)) throw e;
    throw new ApiError(0, "Backend unreachable — is the NeuralScope API running (default http://127.0.0.1:8000)?", e);
  }
  const text = await res.text();
  let data: unknown = undefined;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }
  if (!res.ok) {
    const fallback = `${res.status} ${res.statusText || "request failed"}`.trim();
    const detail =
      data && typeof data === "object" && "detail" in (data as Record<string, unknown>)
        ? formatDetail((data as Record<string, unknown>).detail, fallback)
        : typeof data === "string" && data.trim() && data.length < 300
          ? data
          : fallback;
    throw new ApiError(res.status, detail, data);
  }
  return data as T;
}

// ---------------------------------------------------------------- endpoints

const enc = encodeURIComponent;

export const api = {
  health: (signal?: AbortSignal) => request<Health>("/health", { signal }),
  model: (signal?: AbortSignal) => request<ModelInfo>("/model", { signal }),
  tokenize: (prompt: string, chat = true, signal?: AbortSignal) =>
    request<TokenizeResponse>("/tokenize", { body: { prompt, chat }, signal }),

  createRun: (body: CreateRunRequest, signal?: AbortSignal) => request<RunDetail>("/runs", { body, signal }),
  listRuns: (params: { limit?: number; kind?: string; benchmark_id?: string } = {}, signal?: AbortSignal) =>
    request<RunSummary[]>("/runs", { query: params, signal }),
  getRun: (id: string, signal?: AbortSignal) => request<RunDetail>(`/runs/${enc(id)}`, { signal }),
  patchRun: (id: string, body: { notes?: string | null; tags?: string[] }) =>
    request<RunDetail>(`/runs/${enc(id)}`, { method: "PATCH", body }),
  deleteRun: (id: string) => request<{ deleted: string }>(`/runs/${enc(id)}`, { method: "DELETE" }),
  setTarget: (id: string, body: SetTargetRequest) => request<RunTarget>(`/runs/${enc(id)}/target`, { body }),

  matrix: (id: string, metric: MatrixMetric, targetTokenId?: number | null, signal?: AbortSignal) =>
    request<MatrixResponse>(`/runs/${enc(id)}/matrix`, { query: { metric, target_token_id: targetTokenId }, signal }),
  cell: (id: string, depth: number, position: number, targetTokenId?: number | null, signal?: AbortSignal) =>
    request<CellResponse>(`/runs/${enc(id)}/cell/${depth}/${position}`, { query: { target_token_id: targetTokenId }, signal }),
  logitLens: (id: string, position: number | null, targetTokenId?: number | null, signal?: AbortSignal) =>
    request<LogitLensResponse>(`/runs/${enc(id)}/logit-lens`, {
      query: { position, target_token_id: targetTokenId },
      signal,
    }),
  attention: (id: string, layer: number, queryPosition?: number | null, signal?: AbortSignal) =>
    request<AttentionResponse>(`/runs/${enc(id)}/attention/${layer}`, { query: { query_position: queryPosition }, signal }),
  head: (id: string, layer: number, head: number, position?: number | null, targetTokenId?: number | null, signal?: AbortSignal) =>
    request<HeadDetailResponse>(`/runs/${enc(id)}/heads/${layer}/${head}`, {
      query: { position, target_token_id: targetTokenId },
      signal,
    }),
  dla: (
    id: string,
    params: { position?: number | null; target_token_id?: number | null; contrast_token_id?: number | null } = {},
    signal?: AbortSignal,
  ) => request<DlaResponse>(`/runs/${enc(id)}/dla`, { query: params, signal }),

  intervene: (body: { base_run_id: string; interventions: Intervention[]; max_new_tokens?: number | null; notes?: string | null }) =>
    request<InterventionResponse>("/interventions", { body }),
  compare: (body: { run_a: string; run_b: string; position_a?: number | null; position_b?: number | null }, signal?: AbortSignal) =>
    request<CompareResponse>("/compare", { body, signal }),

  headSweep: (body: { run_id: string; position?: number | null; target_token_id?: number | null; contrast_token_id?: number | null; factor?: number }) =>
    request<Job<HeadSweepResult>>("/sweeps/heads", { body }),
  patchSweep: (body: { clean_run_id: string; corrupt_run_id: string; positions?: number[] | null }) =>
    request<Job<PatchSweepResult>>("/sweeps/patch", { body }),
  job: <R = unknown>(id: string, signal?: AbortSignal) => request<Job<R>>(`/jobs/${enc(id)}`, { signal }),

  mathVerify: (body: { expression?: string; answer: string; problem?: MathProblem }) =>
    request<MathVerifyResponse>("/math/verify", { body }),
  mathProblems: (body: { n: number; seed: number }) => request<MathProblem[]>("/math/problems", { body }),
  mathRun: (body: { problem?: MathProblem; kind?: string; operands?: Record<string, unknown>; trace_level?: TraceLevel }) =>
    request<RunDetail>("/math/run", { body }),
  runBenchmark: (body: { n: number; seed: number; name?: string }) => request<Job<Benchmark>>("/math/benchmarks", { body }),
  listBenchmarks: (signal?: AbortSignal) => request<Benchmark[]>("/math/benchmarks", { signal }),
  getBenchmark: (id: string, signal?: AbortSignal) => request<Benchmark>(`/math/benchmarks/${enc(id)}`, { signal }),
  benchmarkEval: (id: string, interventions: Intervention[]) =>
    request<Job<BenchmarkEvalResult>>(`/math/benchmarks/${enc(id)}/intervention-eval`, { body: { interventions } }),

  createExperiment: (body: { name: string; run_ids: string[]; notes: string; payload: ExperimentPayload }) =>
    request<Experiment>("/experiments", { body }),
  listExperiments: (signal?: AbortSignal) => request<Experiment[]>("/experiments", { signal }),
  getExperiment: (id: string, signal?: AbortSignal) => request<Experiment>(`/experiments/${enc(id)}`, { signal }),
  patchExperiment: (id: string, body: Partial<Pick<Experiment, "name" | "notes" | "run_ids" | "payload">>) =>
    request<Experiment>(`/experiments/${enc(id)}`, { method: "PATCH", body }),
  deleteExperiment: (id: string) => request<{ deleted: string }>(`/experiments/${enc(id)}`, { method: "DELETE" }),
};

// ---------------------------------------------------------------- jobs

export interface PollOptions<R> {
  signal?: AbortSignal;
  intervalMs?: number;
  onUpdate?: (job: Job<R>) => void;
  /** Injected for tests. */
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
  fetchJob?: (id: string, signal?: AbortSignal) => Promise<Job<R>>;
}

function defaultSleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
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
 * Poll /api/jobs/{id} until it is done (resolves with the job) or errors
 * (rejects with an ApiError carrying the job's error text).
 */
export async function pollJob<R>(jobId: string, opts: PollOptions<R> = {}): Promise<Job<R>> {
  const { signal, intervalMs = 1000, onUpdate } = opts;
  const sleep = opts.sleep ?? defaultSleep;
  const fetchJob = opts.fetchJob ?? ((id: string, s?: AbortSignal) => api.job<R>(id, s));
  let transientFailures = 0;
  for (;;) {
    let job: Job<R>;
    try {
      job = await fetchJob(jobId, signal);
      transientFailures = 0;
    } catch (e) {
      if (isAbortError(e)) throw e;
      // tolerate brief proxy hiccups while the backend is busy with the job itself
      if (e instanceof ApiError && e.status !== 404 && transientFailures < 5) {
        transientFailures += 1;
        await sleep(intervalMs * 2, signal);
        continue;
      }
      throw e;
    }
    onUpdate?.(job);
    if (job.status === "done") return job;
    if (job.status === "error") throw new ApiError(500, job.error || "job failed", job);
    await sleep(intervalMs, signal);
  }
}
