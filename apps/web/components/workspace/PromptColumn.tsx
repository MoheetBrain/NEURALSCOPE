"use client";

import { useState, type KeyboardEvent } from "react";
import Link from "next/link";
import { Play, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Kbd } from "@/components/ui/Kbd";
import { Panel } from "@/components/ui/Panel";
import { StatusChip } from "@/components/ui/StatusChip";
import { api } from "@/lib/api/client";
import type { RunDetail } from "@/lib/api/types";
import { cn } from "@/lib/cn";
import { fmtSeconds, quoteToken, shortId, splitSpecials, visibleToken } from "@/lib/format";
import { useWorkspaceActions } from "./actions";
import { GuidedDemo } from "./GuidedDemo";
import { useSelection } from "./hooks";
import { TokenStrip } from "./TokenStrip";
import { FLAGSHIP_PROMPT, useWorkspace } from "./WorkspaceProvider";
import { ErrorBox, Field, Loading, NumberInput, SelectInput, StatusBadge, Switch, TextInput } from "./ui";

interface ChatMessage {
  role: string;
  content: string;
}

/** Messages stored on a multi-message run (SelfLab chat / replay runs), or null for a single-prompt run. */
export function runMessages(run: RunDetail | null | undefined): ChatMessage[] | null {
  const m = (run as (RunDetail & { messages?: unknown }) | null | undefined)?.messages;
  if (!Array.isArray(m) || m.length === 0) return null;
  const ok = m.filter(
    (x): x is ChatMessage => typeof x === "object" && x !== null && typeof (x as ChatMessage).role === "string" && typeof (x as ChatMessage).content === "string",
  );
  return ok.length ? ok : null;
}

/** Read-only view of the conversation a multi-message run was given (the prompt box cannot express it). */
export function RunConversation({ run }: { run: RunDetail }) {
  const msgs = runMessages(run);
  if (!msgs) return null;
  return (
    <Panel
      label={`Run context · ${msgs.length} msg${msgs.length === 1 ? "" : "s"}`}
      status="observed"
      statusNote="input"
      className="shrink-0"
      bodyClassName="flex flex-col gap-1.5 p-2.5"
      data-testid="run-conversation"
    >
      <p className="text-[10.5px] leading-snug text-ns-accent-2">
        {String(run.kind).startsWith("selflab") ? "SelfLab chat run" : "Chat run"} — multi-message context; edit in{" "}
        <Link href="/app/selflab" className="underline hover:text-ns-text">
          SelfLab
        </Link>
        . Read-only here.
      </p>
      {msgs.map((m, i) => (
        <div key={i} className="rounded-[3px] border border-ns-border bg-ns-bg px-2 py-1">
          <div className="ns-label text-[9px]">
            #{i} · {m.role}
          </div>
          <div className="max-h-32 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-ns-text-muted">{m.content}</div>
        </div>
      ))}
    </Panel>
  );
}

export function PromptColumn() {
  const { run } = useWorkspace();
  return (
    <div className="ns-scroll flex min-h-0 flex-col gap-2 p-2 pr-1">
      <GuidedDemo />
      {run.data ? <RunConversation run={run.data} /> : null}
      <PromptForm />
      {run.error && !run.data ? <ErrorBox error={run.error} onRetry={run.reload} /> : null}
      {run.data ? <RunOutput run={run.data} stale={run.stale} /> : run.loading ? <Loading label="Loading run" /> : null}
      {run.data ? <TokensPanel run={run.data} /> : null}
      {run.data ? <TargetPanel key={`${run.data.id}-${run.data.target?.token_id}-${run.data.target?.position}`} run={run.data} /> : null}
      {run.data ? <NotesPanel key={`${run.data.id}-notes`} run={run.data} /> : null}
    </div>
  );
}

function PromptForm() {
  const { state, update, submitRun, pending, model, run } = useWorkspace();
  const running = pending?.kind === "run";
  // A multi-message run cannot be re-run from this single-prompt box: say so instead of implying it.
  const multi = runMessages(run.data) !== null;
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      if (!running) void submitRun();
    }
  };
  return (
    <Panel label={multi ? "New run prompt" : "Prompt"} status="observed" statusNote="input" className="shrink-0" bodyClassName="flex flex-col gap-2 p-2.5">
      {multi ? (
        <p className="text-[10.5px] leading-snug text-ns-text-faint" data-testid="new-run-note">
          RUN starts a new single-prompt run. It does not replay the conversation shown above.
        </p>
      ) : null}
      <textarea
        aria-label="Prompt"
        value={state.prompt}
        onChange={(e) => update({ prompt: e.target.value })}
        onKeyDown={onKey}
        rows={3}
        maxLength={4000}
        placeholder="Type a prompt…"
        className="min-h-[64px] w-full resize-y rounded-[3px] border border-ns-border-strong bg-ns-bg px-2 py-1.5 font-mono text-[12px] leading-relaxed text-ns-text placeholder:text-ns-text-faint focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-ns-accent"
      />
      <div className="flex flex-wrap items-center gap-1.5">
        <button
          type="button"
          onClick={() => update({ prompt: FLAGSHIP_PROMPT, chat: true })}
          className="inline-flex h-6 items-center gap-1 rounded-[3px] border border-ns-accent-2/35 bg-ns-accent-2/8 px-1.5 font-mono text-[10px] text-ns-accent-2 hover:border-ns-accent-2/70"
          title={`Preset: ${FLAGSHIP_PROMPT}`}
        >
          <Sparkles aria-hidden="true" className="size-3" /> Flagship: 37 + 58
        </button>
        <Switch checked={state.chat} onChange={(v) => update({ chat: v })} label="chat template" title="Wrap the prompt in the model's chat template" />
      </div>
      <div className="grid grid-cols-2 gap-2">
        <Field label="max new tokens">
          <NumberInput value={state.maxNewTokens} min={1} max={48} step={1} onValue={(v) => update({ maxNewTokens: Math.min(48, Math.max(1, Math.round(v))) })} />
        </Field>
        <Field label="trace level">
          <SelectInput value={state.traceLevel} onChange={(e) => update({ traceLevel: e.target.value as typeof state.traceLevel })}>
            <option value="minimal">minimal</option>
            <option value="standard">standard</option>
            <option value="full">full</option>
          </SelectInput>
        </Field>
      </div>
      <Button
        variant="primary"
        caps
        size="md"
        onClick={() => void submitRun()}
        disabled={running || !model}
        aria-keyshortcuts="Control+Enter Meta+Enter"
        className="w-full"
      >
        <Play aria-hidden="true" /> {running ? "Running…" : "Run"}
        <span className="ml-auto hidden items-center gap-0.5 opacity-70 sm:inline-flex">
          <Kbd className="border-ns-bg/30 bg-ns-bg/15 text-ns-bg">⌘/Ctrl</Kbd>
          <Kbd className="border-ns-bg/30 bg-ns-bg/15 text-ns-bg">↵</Kbd>
        </span>
      </Button>
      {running ? (
        <Loading
          compact
          label="Generating greedily + capturing the trace"
          hint="Real forward passes on CPU: typically 5–8 s."
          startedAt={pending?.startedAt}
        />
      ) : null}
    </Panel>
  );
}

function RunOutput({ run, stale }: { run: RunDetail; stale: boolean }) {
  const { openRun } = useWorkspace();
  const gen = run.generation;
  const isInt = run.kind === "intervention";
  const pieces = splitSpecials(isInt ? (run.free_generated_text ?? run.generated_text) : run.generated_text);
  return (
    <Panel
      label="Output"
      status="observed"
      className={cn("shrink-0", stale && "ns-stale")}
      right={
        run.trace_consistent === true ? (
          <StatusChip tone="ok" dot title="Trace integrity: the teacher-forced capture reproduces every greedy choice of the generation.">
            consistent
          </StatusChip>
        ) : run.trace_consistent === false ? (
          <StatusChip tone="warn" dot title="The captured trace does not reproduce every greedy choice (numerical drift).">
            trace mismatch
          </StatusChip>
        ) : (
          <StatusChip tone="neutral" title="Intervention runs are teacher-forced on the parent's tokens for aligned metrics.">
            teacher-forced
          </StatusChip>
        )
      }
      bodyClassName="p-2.5"
    >
      <div className="mb-1.5 flex flex-wrap items-center gap-1.5 font-mono text-[10px] text-ns-text-faint">
        <span className="text-ns-text-muted">{shortId(run.id)}</span>
        <span>· {run.kind}</span>
        {run.is_demo ? <StatusChip tone="warn">demo</StatusChip> : null}
        <span>· {run.tokens.length} tok</span>
        <span>· {run.trace_level}</span>
        {gen.seconds_generate !== undefined ? <span>· gen {fmtSeconds(gen.seconds_generate)}</span> : null}
        {gen.seconds_capture !== undefined ? <span>· capture {fmtSeconds(gen.seconds_capture)}</span> : null}
      </div>
      {isInt ? (
        <div className="mb-1.5 rounded-[3px] border border-ns-causal/25 bg-ns-causal/5 px-2 py-1 text-[11px] text-ns-text-muted">
          <span className="font-mono text-ns-causal">{run.interventions_label}</span>
          {run.parent_run_id ? (
            <>
              {" "}on{" "}
              <button type="button" className="font-mono text-ns-accent hover:underline" onClick={() => openRun(run.parent_run_id as string)}>
                {shortId(run.parent_run_id)}
              </button>
            </>
          ) : null}
          <div className="mt-0.5 text-[10.5px] text-ns-text-faint">Output below is free-running under the intervention; the trace is teacher-forced on the parent&apos;s tokens.</div>
        </div>
      ) : null}
      <div className="max-h-28 overflow-auto rounded-[3px] border border-ns-border bg-ns-bg px-2 py-1.5 font-mono text-[13px] leading-relaxed text-ns-text" aria-label="Generated text">
        {pieces.length === 0 ? <span className="text-ns-text-faint">∅ (no tokens generated)</span> : null}
        {pieces.map((p, i) =>
          p.special ? (
            <span key={i} className="mx-0.5 inline-flex rounded-[3px] border border-ns-border-strong bg-ns-panel-2 px-1 align-middle font-mono text-[10px] leading-4 text-ns-text-muted">
              {visibleToken(p.text)}
            </span>
          ) : (
            <span key={i} className="whitespace-pre-wrap">
              {p.text}
            </span>
          ),
        )}
      </div>
      <div className="mt-1.5 font-mono text-[10px] text-ns-text-faint">
        greedy · stopped by {gen.stopped_by_free ?? gen.stopped_by ?? "—"} · max {gen.max_new_tokens}
      </div>
      {run.math ? (
        <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-[11px] text-ns-text-muted">
          <span className="font-mono">oracle {run.math.problem.oracle_answer}</span>
          <MathStatusChip status={run.correctness ?? run.math.label ?? run.math.verification.status} />
          <StatusBadge status="observed" note="SymPy oracle" />
        </div>
      ) : null}
    </Panel>
  );
}

export function MathStatusChip({ status }: { status: string | null | undefined }) {
  if (!status) return null;
  const tone = status === "correct" ? "ok" : status === "incorrect" ? "danger" : "neutral";
  const excluded = status !== "correct" && status !== "incorrect";
  return (
    <StatusChip tone={tone} dot title={excluded ? "excluded from comparisons" : undefined}>
      {status}
    </StatusChip>
  );
}

function TokensPanel({ run }: { run: RunDetail }) {
  const { state, update, select } = useWorkspace();
  const sel = useSelection(run);
  return (
    <Panel
      label={`Tokens · ${run.tokens.length}`}
      status="observed"
      className="shrink-0"
      right={<Switch checked={state.hideTemplate} onChange={(v) => update({ hideTemplate: v })} label="hide template" title="Collapse system-prompt template tokens here and in the matrix" />}
      bodyClassName="p-2.5"
    >
      <TokenStrip
        tokens={run.tokens}
        selected={sel?.position ?? null}
        target={run.target?.position ?? null}
        hideTemplate={state.hideTemplate}
        onToggleTemplate={() => update({ hideTemplate: !state.hideTemplate })}
        onSelect={(p) => select({ depth: sel?.depth ?? run.depth_labels.length - 1, position: p })}
      />
    </Panel>
  );
}

function TargetPanel({ run }: { run: RunDetail }) {
  const { pending } = useWorkspace();
  const { setTarget } = useWorkspaceActions();
  const sel = useSelection(run);
  const t = run.target;
  const [text, setText] = useState(t?.token ?? "");
  const [pos, setPos] = useState(t?.position ?? run.prompt_len - 1);
  const [contrast, setContrast] = useState(t?.contrast ?? "");
  const busy = pending?.kind === "target";
  const posToken = run.tokens[pos];

  return (
    <Panel label="Target token" status="observed" statusNote="tokenizer" className="shrink-0" bodyClassName="flex flex-col gap-2 p-2.5">
      {t ? (
        <p className="text-[11.5px] leading-relaxed text-ns-text-muted">
          Tracking <span className="font-mono text-ns-text">{quoteToken(t.token)}</span> (id {t.token_id ?? "—"}) predicted at{" "}
          <span className="font-mono text-ns-text">t{t.position}</span> {posToken ? <span className="font-mono">{quoteToken(run.tokens[t.position]?.text)}</span> : null}
          {t.contrast ? (
            <>
              {" "}vs contrast <span className="font-mono text-ns-text">{quoteToken(t.contrast)}</span>
            </>
          ) : null}
          <span className="text-ns-text-faint"> · source: {t.source}</span>
        </p>
      ) : (
        <p className="text-[11.5px] text-ns-text-faint">No target set.</p>
      )}
      {t?.note ? <p className="text-[10.5px] text-ns-warn">{t.note}</p> : null}
      <form
        className="flex flex-col gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          void setTarget(run, {
            target_text: text || null,
            target_position: pos,
            contrast_text: contrast || null,
          });
        }}
      >
        <div className="grid grid-cols-[1fr_72px] gap-2">
          <Field label="target text" hint="first token is tracked">
            <TextInput value={text} onChange={(e) => setText(e.target.value)} placeholder="e.g. 9" aria-label="Target text" />
          </Field>
          <Field label="position">
            <NumberInput value={pos} min={0} max={run.tokens.length - 1} onValue={(v) => setPos(Math.min(run.tokens.length - 1, Math.max(0, Math.round(v))))} aria-label="Target position" />
          </Field>
        </div>
        <Field label="contrast token (optional)">
          <TextInput value={contrast} onChange={(e) => setContrast(e.target.value)} placeholder="e.g. 8 → metric becomes logit difference" aria-label="Contrast text" />
        </Field>
        <div className="flex flex-wrap gap-1.5">
          <Button type="submit" size="xs" caps variant="secondary" disabled={busy}>
            {busy ? "Setting…" : "Set target"}
          </Button>
          {sel ? (
            <Button type="button" size="xs" variant="ghost" onClick={() => setPos(sel.position)} title="Use the selected matrix column as the prediction position">
              use selected t{sel.position}
            </Button>
          ) : null}
        </div>
        <p className="text-[10.5px] leading-snug text-ns-text-faint">Position t predicts token t+1. Changing the target re-keys the target-rank matrix, Logit Lens and sweeps.</p>
      </form>
    </Panel>
  );
}

function NotesPanel({ run }: { run: RunDetail }) {
  const { notify, bumpRun } = useWorkspace();
  const [notes, setNotes] = useState(run.notes ?? "");
  const [saving, setSaving] = useState(false);
  const dirty = notes !== (run.notes ?? "");
  return (
    <Panel label="Hypothesis / notes" status="inferred" statusNote="yours" className="shrink-0" bodyClassName="flex flex-col gap-2 p-2.5">
      <textarea
        aria-label="Hypothesis notes for this run"
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        rows={3}
        placeholder="Your hypothesis (INFERRED until tested), e.g. “a late head copies the tens digit”…"
        className="w-full resize-y rounded-[3px] border border-ns-inferred/25 bg-ns-bg px-2 py-1.5 text-[12px] leading-relaxed text-ns-text placeholder:text-ns-text-faint focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-ns-accent"
      />
      <div className="flex items-center gap-2">
        <Button
          size="xs"
          caps
          disabled={!dirty || saving}
          onClick={async () => {
            setSaving(true);
            try {
              await api.patchRun(run.id, { notes });
              notify("ok", "Notes saved.");
              bumpRun();
            } catch (e) {
              notify("error", `Saving notes failed: ${e instanceof Error ? e.message : String(e)}`);
            } finally {
              setSaving(false);
            }
          }}
        >
          {saving ? "Saving…" : "Save notes"}
        </Button>
        <span className="text-[10.5px] text-ns-text-faint">Never auto-generated. Stays INFERRED until an intervention tests it.</span>
      </div>
    </Panel>
  );
}
