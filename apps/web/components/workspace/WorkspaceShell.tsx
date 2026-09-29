"use client";

import type { ReactNode } from "react";
import { X } from "lucide-react";
import { StatusChip } from "@/components/ui/StatusChip";
import { cn } from "@/lib/cn";
import { DemoBanner, demoReasons } from "./DemoBanner";
import { WorkspaceHeader } from "./Header";
import { WorkspaceProvider, useWorkspace } from "./WorkspaceProvider";
import { Loading } from "./ui";

function ModelGate({ children }: { children: ReactNode }) {
  const { health, healthError, model } = useWorkspace();

  if (health?.status === "ready" && model) return <>{children}</>;

  let body: ReactNode;
  if (!health && healthError) {
    body = (
      <>
        <StatusChip tone="danger" dot>
          API unreachable
        </StatusChip>
        <h1 className="mt-4 text-lg font-medium text-ns-text">Cannot reach the NeuralScope backend.</h1>
        <p className="mt-2 font-mono text-xs leading-relaxed text-ns-danger">{healthError}</p>
        <p className="mt-3 text-sm leading-relaxed text-ns-text-muted">
          Start it with <code className="text-ns-text">uvicorn neuralscope_api.server:app --port 8000</code>. Retrying every few seconds.
        </p>
      </>
    );
  } else if (health?.status === "error") {
    body = (
      <>
        <StatusChip tone="danger" dot>
          Model failed to load
        </StatusChip>
        <h1 className="mt-4 text-lg font-medium text-ns-text">{health.model_id}</h1>
        <p className="mt-2 whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-ns-danger">{health.error ?? "unknown error"}</p>
      </>
    );
  } else {
    body = (
      <>
        <StatusChip tone="warn" dot="pulse">
          {health ? "Model loading" : "Connecting"}
        </StatusChip>
        <h1 className="mt-4 text-lg font-medium text-ns-text">{health ? health.model_id : "Connecting to the backend"}</h1>
        <p className="mt-2 text-sm leading-relaxed text-ns-text-muted">
          {health
            ? "Weights are loading on the backend. The workspace opens as soon as /api/health reports ready (polling every 2 s)."
            : "Contacting /api/health…"}
        </p>
        <Loading label={health ? "loading model" : "connecting"} className="mt-2" />
      </>
    );
  }

  return (
    <div className="ns-dot-bg grid min-h-0 flex-1 place-items-center px-4 py-16">
      <div className="w-full max-w-lg rounded-[var(--ns-radius)] border border-ns-border bg-ns-panel p-6">{body}</div>
    </div>
  );
}

function Toasts() {
  const { toasts, dismissToast } = useWorkspace();
  if (toasts.length === 0) return null;
  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-[380px] max-w-[calc(100vw-2rem)] flex-col gap-2" aria-live="polite">
      {toasts.map((t) => (
        <div
          key={t.id}
          role={t.tone === "error" ? "alert" : "status"}
          className={cn(
            "pointer-events-auto flex items-start gap-2 rounded-[var(--ns-radius)] border bg-ns-panel px-3 py-2 text-xs shadow-xl",
            t.tone === "error" && "border-ns-danger/50 text-ns-danger",
            t.tone === "ok" && "border-ns-ok/40 text-ns-text",
            t.tone === "info" && "border-ns-border-strong text-ns-text",
          )}
        >
          <span className="min-w-0 flex-1 break-words leading-relaxed">{t.text}</span>
          <button type="button" aria-label="Dismiss" onClick={() => dismissToast(t.id)} className="text-ns-text-faint hover:text-ns-text">
            <X className="size-3.5" aria-hidden="true" />
          </button>
        </div>
      ))}
    </div>
  );
}

function ShellFrame({ children }: { children: ReactNode }) {
  const { health, model, run } = useWorkspace();
  const reasons = demoReasons({ healthDemo: health?.is_demo, modelDemo: model?.is_demo, runDemo: run.data?.is_demo });
  return (
    <div className="ns-workspace flex h-dvh min-h-0 flex-col overflow-hidden bg-ns-bg text-ns-text">
      <WorkspaceHeader />
      <DemoBanner visible={reasons.length > 0} reason={reasons.join("; ")} />
      <ModelGate>{children}</ModelGate>
      <Toasts />
    </div>
  );
}

export function WorkspaceShell({ children }: { children: ReactNode }) {
  return (
    <WorkspaceProvider>
      <ShellFrame>{children}</ShellFrame>
    </WorkspaceProvider>
  );
}
