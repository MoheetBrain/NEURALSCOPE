"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Cpu, FlaskConical, Microscope } from "lucide-react";
import { Logo } from "@/components/brand/Logo";
import { StatusChip } from "@/components/ui/StatusChip";
import { cn } from "@/lib/cn";
import { fmtParams, fmtPct, shortId, truncate } from "@/lib/format";
import { useWorkspace } from "./WorkspaceProvider";
import { useNow } from "./useResource";

function NavTab({ href, active, children }: { href: "/app" | "/app/math"; active: boolean; children: React.ReactNode }) {
  return (
    <Link
      href={href}
      aria-current={active ? "page" : undefined}
      className={cn(
        "inline-flex h-7 items-center gap-1.5 rounded-[3px] px-2.5 font-mono text-[10.5px] uppercase tracking-[0.1em] transition-colors [&_svg]:size-3.5",
        active ? "bg-ns-panel-2 text-ns-text shadow-[inset_0_0_0_1px_var(--ns-border-strong)]" : "text-ns-text-muted hover:text-ns-text",
      )}
    >
      {children}
    </Link>
  );
}

export function RunSelector({ className }: { className?: string }) {
  const { state, history, openRun } = useWorkspace();
  const runs = history.data ?? [];
  const current = state.runId;
  const inList = runs.some((r) => r.id === current);
  return (
    <label className={cn("flex min-w-0 items-center gap-2", className)}>
      <span className="ns-label shrink-0 text-[9.5px]">Run</span>
      <select
        aria-label="Open run from history"
        className="h-7 min-w-0 max-w-[320px] flex-1 cursor-pointer rounded-[3px] border border-ns-border-strong bg-ns-bg px-2 font-mono text-[11px] text-ns-text"
        value={current ?? ""}
        onChange={(e) => e.target.value && openRun(e.target.value)}
      >
        {!current ? <option value="">— no run —</option> : null}
        {current && !inList ? <option value={current}>{current}</option> : null}
        {runs.slice(0, 80).map((r) => (
          <option key={r.id} value={r.id}>
            {r.is_demo ? "[DEMO] " : ""}
            {r.kind === "intervention" ? "↳ " : r.kind === "math" ? "∑ " : "● "}
            {shortId(r.id)} · {truncate(r.prompt, 34)}
            {r.kind === "intervention" ? ` · ${truncate(r.interventions_label, 26)}` : ""}
          </option>
        ))}
      </select>
    </label>
  );
}

export function WorkspaceHeader() {
  const pathname = usePathname();
  const { health, model, healthError, pending, activeJob } = useWorkspace();
  const now = useNow(500, Boolean(pending));
  const onMath = pathname?.startsWith("/app/math") ?? false;

  let healthChip: React.ReactNode;
  if (healthError && !health) healthChip = <StatusChip tone="danger" dot>API unreachable</StatusChip>;
  else if (!health) healthChip = <StatusChip tone="neutral" dot="pulse">Connecting</StatusChip>;
  else if (health.status === "loading") healthChip = <StatusChip tone="warn" dot="pulse">Model loading</StatusChip>;
  else if (health.status === "error") healthChip = <StatusChip tone="danger" dot>Model error</StatusChip>;
  else healthChip = <StatusChip tone="ok" dot>Model ready</StatusChip>;

  return (
    <header className="flex h-11 shrink-0 items-center gap-3 border-b border-ns-border bg-ns-bg px-3">
      <Link href="/" aria-label="NeuralScope home" className="shrink-0">
        <Logo size={18} />
      </Link>
      <span className="ns-label hidden shrink-0 text-[9.5px] xl:inline">Neural Debugger</span>
      <nav aria-label="Workspace views" className="flex shrink-0 items-center gap-0.5 rounded-[var(--ns-radius)] border border-ns-border bg-ns-panel p-0.5">
        <NavTab href="/app" active={!onMath}>
          <Microscope aria-hidden="true" /> Workspace
        </NavTab>
        <NavTab href="/app/math" active={onMath}>
          <FlaskConical aria-hidden="true" /> Math Lab
        </NavTab>
      </nav>

      {model ? (
        <div
          className="hidden min-w-0 items-center gap-2 rounded-[3px] border border-ns-border bg-ns-panel px-2 py-1 lg:flex"
          title={`${model.architecture} · ${model.n_params.toLocaleString()} params · ${model.norm_type}${model.tied_embeddings ? " · tied embeddings" : ""}`}
        >
          <Cpu aria-hidden="true" className="size-3.5 shrink-0 text-ns-text-faint" />
          <span className="truncate font-mono text-[11px] text-ns-text">{model.model_id}</span>
          <span className="ns-tabular shrink-0 whitespace-nowrap font-mono text-[10.5px] text-ns-text-muted">
            {model.n_layers}L×{model.n_heads}H · {model.n_kv_heads} KV · d{model.d_model} · {fmtParams(model.n_params)} · {model.device} · {model.dtype}
          </span>
        </div>
      ) : health ? (
        <span className="hidden truncate font-mono text-[11px] text-ns-text-muted lg:inline">{health.model_id}</span>
      ) : null}

      <div className="ml-auto flex min-w-0 items-center gap-2">
        {!onMath ? <RunSelector className="hidden md:flex" /> : null}
        {pending ? (
          <StatusChip tone="accent" dot="pulse" title={pending.label}>
            {truncate(pending.label, 26)} {now > 0 ? `${Math.max(0, (now - pending.startedAt) / 1000).toFixed(1)}s` : ""}
          </StatusChip>
        ) : null}
        {activeJob ? (
          <StatusChip tone="accent" dot="pulse" title={`${activeJob.label}: ${activeJob.message}`}>
            {truncate(activeJob.label, 18)} {fmtPct(activeJob.progress)}
          </StatusChip>
        ) : null}
        {healthChip}
      </div>
    </header>
  );
}
