"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, errorMessage, isAbortError, pollJob } from "@/lib/api/client";
import { viewCache } from "@/lib/api/cache";
import type {
  Health,
  Intervention,
  InterventionType,
  Job,
  JobStatus,
  MatrixMetric,
  ModelInfo,
  RunDetail,
  RunSummary,
  TraceLevel,
} from "@/lib/api/types";
import type { SequentialName } from "@/lib/colormap";
import type { MatrixSelection } from "@/lib/matrix";
import { useResource, type Resource } from "./useResource";

export type InspectorTab = "trace" | "lens" | "attention" | "interventions" | "diff";
export type PositionsMode = "all" | "selected" | "custom";

export const FLAGSHIP_PROMPT = "What is 37 + 58? Give only the answer.";

export interface InterventionDraft {
  type: InterventionType;
  layer: number;
  head: number;
  factor: number;
  positionsMode: PositionsMode;
  customPositions: string;
  depth: number;
  position: number;
  sourceRunId: string;
  sourcePosition: string;
  alpha: number;
  plusRunId: string;
  plusPosition: number;
  minusRunId: string;
  minusPosition: number;
  scale: number;
}

export interface WorkspaceState {
  runId: string | null;
  runVersion: number;
  historyVersion: number;
  selection: MatrixSelection | null;
  metric: MatrixMetric;
  colormap: SequentialName;
  hideTemplate: boolean;
  tab: InspectorTab;
  /** Attention layer override; null = follow the matrix selection (depth − 1). */
  attnLayer: number | null;
  attnHead: number | null;
  draft: InterventionDraft;
  stack: Intervention[];
  diffRunId: string | null;
  sweepView: "sweep" | "patch" | "dla";
  prompt: string;
  chat: boolean;
  maxNewTokens: number;
  traceLevel: TraceLevel;
  guideOpen: boolean;
  guideStep: number;
  benchmarkId: string | null;
  compareA: string | null;
  compareB: string | null;
}

export type JobKind = "head_sweep" | "benchmark" | "benchmark_eval" | "patch_sweep";

export interface JobState {
  id: string;
  kind: JobKind;
  label: string;
  /** What the job belongs to (run id / benchmark id) — views filter on it. */
  scopeKey: string;
  status: JobStatus;
  progress: number;
  message: string;
  error?: string;
  result?: unknown;
  startedAt: number;
}

export interface Toast {
  id: number;
  tone: "error" | "ok" | "info";
  text: string;
}

export interface Pending {
  kind: "run" | "intervention" | "target" | "math";
  label: string;
  startedAt: number;
}

interface WorkspaceContextValue {
  state: WorkspaceState;
  update: (patch: Partial<WorkspaceState> | ((s: WorkspaceState) => Partial<WorkspaceState>)) => void;
  updateDraft: (patch: Partial<InterventionDraft>) => void;
  select: (sel: MatrixSelection) => void;
  openRun: (id: string, opts?: { selection?: MatrixSelection | null; keepTab?: boolean; diffRunId?: string | null }) => void;
  bumpRun: () => void;
  bumpHistory: () => void;
  health: Health | null;
  healthError: string | null;
  model: ModelInfo | null;
  run: Resource<RunDetail>;
  history: Resource<RunSummary[]>;
  isDemo: boolean;
  jobs: Partial<Record<JobKind, JobState>>;
  activeJob: JobState | null;
  startJob: <R>(
    kind: JobKind,
    label: string,
    scopeKey: string,
    submit: () => Promise<Job<R>>,
    onDone?: (result: R) => void,
  ) => Promise<R | null>;
  pending: Pending | null;
  withPending: <T>(p: Omit<Pending, "startedAt">, fn: () => Promise<T>) => Promise<T | null>;
  toasts: Toast[];
  notify: (tone: Toast["tone"], text: string) => void;
  dismissToast: (id: number) => void;
  submitRun: (overrides?: { prompt?: string }) => Promise<RunDetail | null>;
}

const WorkspaceContext = createContext<WorkspaceContextValue | null>(null);

export function useWorkspace(): WorkspaceContextValue {
  const ctx = useContext(WorkspaceContext);
  if (!ctx) throw new Error("useWorkspace must be used inside <WorkspaceProvider>");
  return ctx;
}

/** Optional variant (components that can render outside the provider, e.g. in tests). */
export function useOptionalWorkspace(): WorkspaceContextValue | null {
  return useContext(WorkspaceContext);
}

export const INITIAL_DRAFT: InterventionDraft = {
  type: "head_ablation",
  layer: 0,
  head: 0,
  factor: 0,
  positionsMode: "all",
  customPositions: "",
  depth: 0,
  position: 0,
  sourceRunId: "",
  sourcePosition: "",
  alpha: 1,
  plusRunId: "",
  plusPosition: 0,
  minusRunId: "",
  minusPosition: 0,
  scale: 4,
};

export const INITIAL_STATE: WorkspaceState = {
  runId: null,
  runVersion: 0,
  historyVersion: 0,
  selection: null,
  metric: "lens_top1_prob",
  colormap: "viridis",
  hideTemplate: true,
  tab: "trace",
  attnLayer: null,
  attnHead: null,
  draft: INITIAL_DRAFT,
  stack: [],
  diffRunId: null,
  sweepView: "sweep",
  prompt: FLAGSHIP_PROMPT,
  chat: true,
  maxNewTokens: 16,
  traceLevel: "standard",
  guideOpen: true,
  guideStep: 0,
  benchmarkId: null,
  compareA: null,
  compareB: null,
};

function runIdFromUrl(): string | null {
  if (typeof window === "undefined") return null;
  const id = new URLSearchParams(window.location.search).get("run");
  return id && /^[\w-]+$/.test(id) ? id : null;
}

export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<WorkspaceState>(INITIAL_STATE);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [jobs, setJobs] = useState<Partial<Record<JobKind, JobState>>>({});
  const [pending, setPending] = useState<Pending | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const toastSeq = useRef(0);
  const initialised = useRef(false);

  const update = useCallback<WorkspaceContextValue["update"]>((patch) => {
    setState((s) => ({ ...s, ...(typeof patch === "function" ? patch(s) : patch) }));
  }, []);

  const updateDraft = useCallback((patch: Partial<InterventionDraft>) => {
    setState((s) => ({ ...s, draft: { ...s.draft, ...patch } }));
  }, []);

  const notify = useCallback((tone: Toast["tone"], text: string) => {
    toastSeq.current += 1;
    const id = toastSeq.current;
    setToasts((ts) => [...ts.slice(-3), { id, tone, text }]);
    setTimeout(() => setToasts((ts) => ts.filter((t) => t.id !== id)), tone === "error" ? 9000 : 4500);
  }, []);

  const dismissToast = useCallback((id: number) => setToasts((ts) => ts.filter((t) => t.id !== id)), []);

  // ------------------------------------------------------------ health / model
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const ctl = new AbortController();
    const poll = async () => {
      let delay = 2000;
      try {
        const h = await api.health(ctl.signal);
        if (cancelled) return;
        setHealth(h);
        setHealthError(null);
        if (h.status === "ready") {
          delay = 20000;
          try {
            const m = await api.model(ctl.signal);
            if (!cancelled) setModel(m);
          } catch (e) {
            if (!cancelled && !isAbortError(e)) setHealthError(errorMessage(e));
          }
        } else if (h.status === "error") {
          delay = 10000;
        }
      } catch (e) {
        if (cancelled || isAbortError(e)) return;
        setHealthError(errorMessage(e));
        delay = 3000;
      }
      if (!cancelled) timer = setTimeout(poll, delay);
    };
    void poll();
    return () => {
      cancelled = true;
      ctl.abort();
      if (timer) clearTimeout(timer);
    };
  }, []);

  const ready = health?.status === "ready" && model !== null;

  // ------------------------------------------------------------ run + history
  const run = useResource<RunDetail>(
    ready && state.runId ? `run:${state.runId}:${state.runVersion}` : null,
    (signal) => api.getRun(state.runId as string, signal),
  );
  const history = useResource<RunSummary[]>(
    ready ? `runs:${state.historyVersion}` : null,
    (signal) => api.listRuns({ limit: 150 }, signal),
  );

  // pick the initial run once the history is known (URL ?run= wins, else the latest workspace run)
  useEffect(() => {
    if (initialised.current || !history.data) return;
    initialised.current = true;
    const fromUrl = runIdFromUrl();
    const latest = history.data.find((r) => r.kind === "baseline") ?? history.data.find((r) => r.kind !== "math");
    const id = fromUrl ?? latest?.id ?? null;
    if (id) queueMicrotask(() => setState((s) => (s.runId ? s : { ...s, runId: id })));
  }, [history.data]);

  // keep ?run= in the URL (shareable / reload-safe)
  useEffect(() => {
    if (typeof window === "undefined" || !state.runId || window.location.pathname !== "/app") return;
    const url = new URL(window.location.href);
    if (url.searchParams.get("run") === state.runId) return;
    url.searchParams.set("run", state.runId);
    window.history.replaceState(window.history.state, "", url.toString());
  }, [state.runId]);

  const historyRef = useRef<RunSummary[] | undefined>(undefined);
  useEffect(() => {
    historyRef.current = history.data;
  }, [history.data]);

  const openRun = useCallback<WorkspaceContextValue["openRun"]>((id, opts = {}) => {
    const summary = historyRef.current?.find((r) => r.id === id);
    setState((s) => ({
      ...s,
      // the prompt box follows the opened run so RUN re-runs what is on screen
      prompt: summary?.prompt ?? s.prompt,
      runId: id,
      selection: opts.selection === undefined ? null : opts.selection,
      diffRunId: opts.diffRunId === undefined ? (s.runId === id ? s.diffRunId : null) : opts.diffRunId,
      attnLayer: null,
      stack: [],
      metric: s.metric === "delta_resid" ? "lens_top1_prob" : s.metric,
      tab: opts.keepTab ? s.tab : s.tab === "diff" ? "trace" : s.tab,
    }));
  }, []);

  const select = useCallback((sel: MatrixSelection) => {
    setState((s) => ({ ...s, selection: sel }));
  }, []);

  const bumpRun = useCallback(() => {
    setState((s) => ({ ...s, runVersion: s.runVersion + 1 }));
  }, []);
  const bumpHistory = useCallback(() => {
    setState((s) => ({ ...s, historyVersion: s.historyVersion + 1 }));
  }, []);

  // ------------------------------------------------------------ pending ops & jobs
  const withPending = useCallback<WorkspaceContextValue["withPending"]>(
    async (p, fn) => {
      setPending({ ...p, startedAt: Date.now() });
      try {
        return await fn();
      } catch (e) {
        notify("error", `${p.label} failed: ${errorMessage(e)}`);
        return null;
      } finally {
        setPending(null);
      }
    },
    [notify],
  );

  const jobsRef = useRef(jobs);
  useEffect(() => {
    jobsRef.current = jobs;
  }, [jobs]);

  const startJob = useCallback<WorkspaceContextValue["startJob"]>(
    async (kind, label, scopeKey, submit, onDone) => {
      const busy = Object.values(jobsRef.current).find((j) => j && (j.status === "queued" || j.status === "running"));
      if (busy) {
        notify("info", `Another job is running (${busy.label}). The backend runs heavy jobs one at a time.`);
        return null;
      }
      const startedAt = Date.now();
      const setJob = (patch: Partial<JobState>) =>
        setJobs((js) => ({ ...js, [kind]: { ...(js[kind] as JobState), ...patch } }));
      setJobs((js) => ({
        ...js,
        [kind]: { id: "", kind, label, scopeKey, status: "queued", progress: 0, message: "submitting…", startedAt },
      }));
      try {
        const job = await submit();
        setJob({ id: job.id, status: job.status, message: job.message });
        const done = await pollJob(job.id, {
          intervalMs: 1200,
          onUpdate: (j) => setJob({ status: j.status, progress: j.progress, message: j.message }),
        });
        const result = done.result as Parameters<NonNullable<typeof onDone>>[0];
        setJob({ status: "done", progress: 1, message: "done", result });
        onDone?.(result);
        return result;
      } catch (e) {
        setJob({ status: "error", error: errorMessage(e), message: "error" });
        notify("error", `${label} failed: ${errorMessage(e)}`);
        return null;
      }
    },
    [notify],
  );

  const activeJob = useMemo(
    () => Object.values(jobs).find((j) => j && (j.status === "queued" || j.status === "running")) ?? null,
    [jobs],
  );

  // ------------------------------------------------------------ run submission
  const stateRef = useRef(state);
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const submitRun = useCallback<WorkspaceContextValue["submitRun"]>(
    async (overrides = {}) => {
      const s = stateRef.current;
      const prompt = (overrides.prompt ?? s.prompt).trim();
      if (!prompt) {
        notify("error", "Prompt is empty.");
        return null;
      }
      const detail = await withPending({ kind: "run", label: "Forward pass + capture" }, () =>
        api.createRun({
          prompt,
          chat: s.chat,
          max_new_tokens: s.maxNewTokens,
          trace_level: s.traceLevel,
        }),
      );
      if (detail) {
        viewCache.set(`run:${detail.id}:0`, detail);
        setState((st) => ({
          ...st,
          prompt,
          runId: detail.id,
          runVersion: 0,
          selection: null,
          diffRunId: null,
          stack: [],
          attnLayer: null,
          metric: st.metric === "delta_resid" ? "lens_top1_prob" : st.metric,
          historyVersion: st.historyVersion + 1,
          tab: st.tab === "diff" ? "trace" : st.tab,
        }));
      }
      return detail;
    },
    [notify, withPending],
  );

  const isDemo = Boolean(health?.is_demo || model?.is_demo || run.data?.is_demo);

  const value: WorkspaceContextValue = {
    state,
    update,
    updateDraft,
    select,
    openRun,
    bumpRun,
    bumpHistory,
    health,
    healthError,
    model,
    run,
    history,
    isDemo,
    jobs,
    activeJob,
    startJob,
    pending,
    withPending,
    toasts,
    notify,
    dismissToast,
    submitRun,
  };

  return <WorkspaceContext.Provider value={value}>{children}</WorkspaceContext.Provider>;
}
