"use client";

import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { errorMessage, isAbortError } from "@/lib/api/client";
import {
  isActiveJob,
  pollExperiment,
  selflab,
  type AccessLevel,
  type Case,
  type CaseDetail,
  type ChatExample,
  type ChatRequest,
  type ChatResponse,
  type CreateCaseRequest,
  type EvidenceItem,
  type ExperimentRecord,
  type ExperimentSpec,
  type SelfLabStatus,
  type TemplateInfo,
} from "@/lib/api/selflab";
import { useResource, type Resource } from "@/components/workspace/useResource";
import { useOptionalWorkspace } from "@/components/workspace/WorkspaceProvider";
import { prefillFromExample } from "./example";

export type Busy = "chat" | "case" | "commit" | "run" | "cancel" | "resume" | null;

export interface SelfLabContextValue {
  status: Resource<SelfLabStatus>;
  mode: AccessLevel;
  setMode: (m: AccessLevel) => void;

  chat: ChatResponse | null;
  sendChat: (req: ChatRequest) => Promise<ChatResponse | null>;
  clearConversation: () => void;

  cases: Resource<Case[]>;
  caseId: string | null;
  openCase: (id: string | null) => void;
  caseDetail: Resource<CaseDetail>;
  createCase: (req: CreateCaseRequest) => Promise<Case | null>;

  templates: Resource<TemplateInfo[]>;
  templateId: string | null;
  selectTemplate: (id: string | null) => void;
  draft: ExperimentSpec | null;
  setDraft: (spec: ExperimentSpec | null) => void;
  /** Changes whenever a fresh draft is created (resets per-field editor state). */
  draftKey: number;
  /** Params of the current draft that were pre-filled from the loaded example. */
  draftPrefill: string[];
  /** The example the composer last loaded (its counterfactual inputs pre-fill the case template). */
  rememberExample: (ex: ChatExample | null) => void;

  experiments: Resource<ExperimentRecord[]>;
  experimentId: string | null;
  selectExperiment: (id: string | null) => void;
  record: ExperimentRecord | null;
  commit: () => Promise<ExperimentRecord | null>;
  run: () => Promise<void>;
  cancel: () => Promise<void>;
  resume: () => Promise<void>;

  evidence: Resource<EvidenceItem[]>;
  /** Bumped whenever trials/evidence may have changed (progress, completion). */
  dataVersion: number;

  selectedTrialId: string | null;
  selectTrial: (id: string | null) => void;
  openRunInLab: (runId: string) => void;

  busy: Busy;
  notify: (tone: "error" | "ok" | "info", text: string) => void;
}

const Ctx = createContext<SelfLabContextValue | null>(null);

export function useSelfLab(): SelfLabContextValue {
  const v = useContext(Ctx);
  if (!v) throw new Error("useSelfLab must be used inside <SelfLabProvider>");
  return v;
}

const SAFE_ID = /^[\w:.-]+$/;

function readUrlParam(name: string): string | null {
  if (typeof window === "undefined") return null;
  const v = new URLSearchParams(window.location.search).get(name);
  return v && SAFE_ID.test(v) ? v : null;
}

/** Deep copy so edits to the draft never mutate the cached template payload. */
function cloneSpec(spec: ExperimentSpec): ExperimentSpec {
  return JSON.parse(JSON.stringify(spec)) as ExperimentSpec;
}

export function SelfLabProvider({ children }: { children: ReactNode }) {
  const ws = useOptionalWorkspace();
  const router = useRouter();
  const notifyWs = ws?.notify;
  const notify = useCallback<SelfLabContextValue["notify"]>((tone, text) => notifyWs?.(tone, text), [notifyWs]);

  const [mode, setMode] = useState<AccessLevel>("white_box_local");
  const [chat, setChat] = useState<ChatResponse | null>(null);
  const [caseId, setCaseId] = useState<string | null>(null);
  const [caseVersion, setCaseVersion] = useState(0);
  const [casesVersion, setCasesVersion] = useState(0);
  const [templateId, setTemplateId] = useState<string | null>(null);
  const [draft, setDraft] = useState<ExperimentSpec | null>(null);
  const [draftKey, setDraftKey] = useState(0);
  const [draftPrefill, setDraftPrefill] = useState<string[]>([]);
  const [example, setExample] = useState<ChatExample | null>(null);
  const [experimentId, setExperimentId] = useState<string | null>(null);
  const [record, setRecord] = useState<ExperimentRecord | null>(null);
  const [pollNonce, setPollNonce] = useState(0);
  const [dataVersion, setDataVersion] = useState(0);
  const [experimentsVersion, setExperimentsVersion] = useState(0);
  const [selectedTrialId, setSelectedTrialId] = useState<string | null>(null);
  const [busy, setBusy] = useState<Busy>(null);

  // ------------------------------------------------------------ resources
  const status = useResource<SelfLabStatus>("selflab:status", (s) => selflab.status(s), { cache: false });
  const cases = useResource<Case[]>(`selflab:cases:${casesVersion}`, (s) => selflab.listCases(s), { cache: false });
  const caseDetail = useResource<CaseDetail>(
    caseId ? `selflab:case:${caseId}:${caseVersion}` : null,
    (s) => selflab.getCase(caseId as string, s),
    { cache: false, keepPrevious: false },
  );
  const templates = useResource<TemplateInfo[]>(`selflab:templates:${caseId ?? "-"}`, (s) => selflab.templates(caseId, s), {
    cache: false,
  });
  const experiments = useResource<ExperimentRecord[]>(
    `selflab:experiments:${caseId ?? "-"}:${experimentsVersion}`,
    (s) => selflab.listExperiments(caseId, s),
    { cache: false },
  );
  // With an experiment selected, filter by experiment alone: a case-less experiment (e.g. a battery opened while a
  // case is selected) has no case_id, and filtering by both would hide its evidence.
  const evidenceQuery = experimentId ? { experiment_id: experimentId } : caseId ? { case_id: caseId } : null;
  const evidence = useResource<EvidenceItem[]>(
    evidenceQuery ? `selflab:evidence:${experimentId ? `exp:${experimentId}` : `case:${caseId}`}:${dataVersion}` : null,
    (s) => selflab.evidence(evidenceQuery ?? {}, s),
    { cache: false },
  );

  // ------------------------------------------------------------ URL state (?case= & ?exp=)
  const initialised = useRef(false);
  useEffect(() => {
    if (initialised.current) return;
    initialised.current = true;
    const c = readUrlParam("case");
    const e = readUrlParam("exp");
    if (c || e)
      queueMicrotask(() => {
        if (c) setCaseId(c);
        if (e) setExperimentId(e);
      });
  }, []);

  useEffect(() => {
    if (typeof window === "undefined" || !initialised.current) return;
    if (!window.location.pathname.startsWith("/app/selflab")) return;
    const url = new URL(window.location.href);
    if (caseId) url.searchParams.set("case", caseId);
    else url.searchParams.delete("case");
    if (experimentId) url.searchParams.set("exp", experimentId);
    else url.searchParams.delete("exp");
    if (url.toString() !== window.location.href) window.history.replaceState(window.history.state, "", url.toString());
  }, [caseId, experimentId]);

  // ------------------------------------------------------------ experiment polling
  const lastDone = useRef<number>(-1);
  useEffect(() => {
    if (!experimentId) return;
    const ctl = new AbortController();
    pollExperiment(experimentId, {
      signal: ctl.signal,
      onUpdate: (rec) => {
        setRecord(rec);
        if (rec.n_done !== lastDone.current) {
          lastDone.current = rec.n_done;
          setDataVersion((v) => v + 1);
        }
      },
    }).then(
      (rec) => {
        setRecord(rec);
        setDataVersion((v) => v + 1);
        setExperimentsVersion((v) => v + 1);
        setCaseVersion((v) => v + 1);
      },
      (e) => {
        if (!isAbortError(e)) notify("error", `Experiment ${experimentId}: ${errorMessage(e)}`);
      },
    );
    return () => ctl.abort();
  }, [experimentId, pollNonce, notify]);

  // ------------------------------------------------------------ actions
  const sendChat = useCallback<SelfLabContextValue["sendChat"]>(
    async (req) => {
      setBusy("chat");
      try {
        const res = await selflab.chat(req);
        setChat(res);
        setCaseId(null);
        setExperimentId(null);
        setRecord(null);
        setSelectedTrialId(null);
        ws?.bumpHistory();
        return res;
      } catch (e) {
        notify("error", `Chat failed: ${errorMessage(e)}`);
        return null;
      } finally {
        setBusy(null);
      }
    },
    [notify, ws],
  );

  const clearConversation = useCallback(() => {
    setChat(null);
    setCaseId(null);
    setExperimentId(null);
    setRecord(null);
    setTemplateId(null);
    setDraft(null);
    setSelectedTrialId(null);
  }, []);

  const openCase = useCallback((id: string | null) => {
    setCaseId(id);
    setExperimentId(null);
    setRecord(null);
    setTemplateId(null);
    setDraft(null);
    setSelectedTrialId(null);
  }, []);

  const createCase = useCallback<SelfLabContextValue["createCase"]>(
    async (req) => {
      setBusy("case");
      try {
        const c = await selflab.createCase(req);
        setCasesVersion((v) => v + 1);
        openCase(c.id);
        notify("ok", `Case ${c.id} created. Pick a template to test the claim.`);
        return c;
      } catch (e) {
        notify("error", `Could not create the case: ${errorMessage(e)}`);
        return null;
      } finally {
        setBusy(null);
      }
    },
    [notify, openCase],
  );

  const selectTemplate = useCallback(
    (id: string | null) => {
      setTemplateId(id);
      const t = templates.data?.find((x) => x.id === id);
      const c = caseDetail.data?.case.id === caseId ? caseDetail.data?.case : null;
      if (t?.default_spec) {
        let spec = cloneSpec(t.default_spec);
        if (t.requires_case && caseId) spec.case_id = caseId;
        const pre = prefillFromExample(spec, c, example);
        spec = pre.spec;
        setDraft(spec);
        setDraftPrefill(pre.applied);
      } else {
        setDraft(null);
        setDraftPrefill([]);
      }
      setDraftKey((k) => k + 1);
      // a new draft replaces the committed view until it is committed
      setExperimentId(null);
      setRecord(null);
      setSelectedTrialId(null);
    },
    [templates.data, caseId, caseDetail.data, example],
  );

  const selectExperiment = useCallback((id: string | null) => {
    setExperimentId(id);
    setRecord(null);
    setSelectedTrialId(null);
    lastDone.current = -1;
  }, []);

  const commit = useCallback(async () => {
    if (!draft) return null;
    setBusy("commit");
    try {
      const rec = await selflab.commitExperiment(draft);
      setRecord(rec);
      setExperimentId(rec.id);
      setDraft(null);
      setExperimentsVersion((v) => v + 1);
      setCaseVersion((v) => v + 1);
      notify("ok", `Committed ${rec.id}: spec and plan hashes fixed before any trial runs.`);
      return rec;
    } catch (e) {
      notify("error", `Commit refused: ${errorMessage(e)}`);
      return null;
    } finally {
      setBusy(null);
    }
  }, [draft, notify]);

  const mutate = useCallback(
    async (kind: "run" | "cancel" | "resume") => {
      if (!experimentId) return;
      setBusy(kind);
      try {
        const fn = kind === "run" ? selflab.runExperiment : kind === "cancel" ? selflab.cancelExperiment : selflab.resumeExperiment;
        const rec = await fn(experimentId);
        setRecord(rec);
        if (isActiveJob(rec.status)) setPollNonce((n) => n + 1);
      } catch (e) {
        notify("error", `${kind} failed: ${errorMessage(e)}`);
      } finally {
        setBusy(null);
      }
    },
    [experimentId, notify],
  );

  const openRunInLab = useCallback(
    (runId: string) => {
      if (!SAFE_ID.test(runId)) {
        notify("error", `Refusing to open an unexpected run id: ${runId}`);
        return;
      }
      ws?.openRun(runId, { selection: null });
      ws?.bumpHistory();
      router.push(`/app?run=${encodeURIComponent(runId)}`);
    },
    [ws, router, notify],
  );

  // The record shown is the polled one when it matches the selected experiment.
  const shownRecord = record && record.id === experimentId ? record : null;

  const value: SelfLabContextValue = {
    status,
    mode,
    setMode,
    chat,
    sendChat,
    clearConversation,
    cases,
    caseId,
    openCase,
    caseDetail,
    createCase,
    templates,
    templateId,
    selectTemplate,
    draft,
    setDraft,
    draftKey,
    draftPrefill,
    rememberExample: setExample,
    experiments,
    experimentId,
    selectExperiment,
    record: shownRecord,
    commit,
    run: () => mutate("run"),
    cancel: () => mutate("cancel"),
    resume: () => mutate("resume"),
    evidence,
    dataVersion,
    selectedTrialId,
    selectTrial: setSelectedTrialId,
    openRunInLab,
    busy,
    notify,
  };

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
