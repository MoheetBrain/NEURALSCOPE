"use client";

import { useMemo, useState } from "react";
import { BookmarkPlus, ChevronDown, ChevronUp, CornerDownRight, FolderOpen, RotateCw } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { StatusChip } from "@/components/ui/StatusChip";
import { api, errorMessage } from "@/lib/api/client";
import type { Experiment, RunSummary } from "@/lib/api/types";
import { cn } from "@/lib/cn";
import { fmtAgo, shortId, truncate } from "@/lib/format";
import { MathStatusChip } from "./PromptColumn";
import { useResource, useNow } from "./useResource";
import { useWorkspace } from "./WorkspaceProvider";
import { ErrorBox, OutputText, Segmented, StatusBadge, TextInput } from "./ui";

type Filter = "workspace" | "math" | "all";

/** Order runs newest-first with each baseline's intervention children right after it. */
export function orderWithChildren(runs: readonly RunSummary[]): { run: RunSummary; depth: 0 | 1 }[] {
  const ids = new Set(runs.map((r) => r.id));
  const kids = new Map<string, RunSummary[]>();
  for (const r of runs) {
    if (r.parent_run_id && ids.has(r.parent_run_id)) {
      const list = kids.get(r.parent_run_id) ?? [];
      list.push(r);
      kids.set(r.parent_run_id, list);
    }
  }
  const out: { run: RunSummary; depth: 0 | 1 }[] = [];
  for (const r of runs) {
    if (r.parent_run_id && ids.has(r.parent_run_id)) continue;
    out.push({ run: r, depth: 0 });
    for (const k of kids.get(r.id) ?? []) out.push({ run: k, depth: 1 });
  }
  return out;
}

export function Timeline() {
  const { history, state, openRun, bumpHistory } = useWorkspace();
  const [filter, setFilter] = useState<Filter>("workspace");
  const [collapsed, setCollapsed] = useState(false);
  const now = useNow(30000);
  const items = useMemo(() => {
    const runs = (history.data ?? []).filter((r) =>
      filter === "all" ? true : filter === "math" ? r.kind === "math" || r.benchmark_id : r.kind !== "math",
    );
    return orderWithChildren(runs).slice(0, 60);
  }, [history.data, filter]);

  return (
    <section
      aria-label="Run history and experiments"
      className={cn(
        "grid shrink-0 grid-cols-[minmax(0,1fr)_400px] gap-2 border-t border-ns-border bg-ns-bg px-2 pb-2 pt-1.5",
        collapsed ? "h-[46px]" : "h-[164px]",
      )}
    >
      <div className="flex min-h-0 min-w-0 flex-col rounded-[var(--ns-radius)] border border-ns-border bg-ns-panel">
        <div className="flex h-8 shrink-0 items-center gap-2 border-b border-ns-border px-3">
          <button
            type="button"
            aria-expanded={!collapsed}
            aria-label={collapsed ? "Expand timeline" : "Collapse timeline"}
            onClick={() => setCollapsed((c) => !c)}
            className="-ml-1 text-ns-text-faint hover:text-ns-text"
          >
            {collapsed ? <ChevronUp className="size-3.5" aria-hidden="true" /> : <ChevronDown className="size-3.5" aria-hidden="true" />}
          </button>
          <span className="ns-label">Timeline · run history</span>
          <StatusBadge status="observed" note="stored runs" />
          <Segmented
            label="History filter"
            size="xs"
            className="ml-auto"
            value={filter}
            onChange={setFilter}
            items={[
              { value: "workspace", label: "workspace" },
              { value: "math", label: "math lab" },
              { value: "all", label: "all" },
            ]}
          />
          <button type="button" aria-label="Refresh history" onClick={bumpHistory} className="text-ns-text-faint hover:text-ns-text">
            <RotateCw className="size-3.5" aria-hidden="true" />
          </button>
        </div>
        {history.error && !history.data ? <ErrorBox error={history.error} onRetry={history.reload} className="m-2" /> : null}
        <ol className="flex min-h-0 flex-1 items-stretch gap-1.5 overflow-x-auto overflow-y-hidden p-2" aria-label="Runs, newest first">
          {items.length === 0 && history.data ? <li className="self-center px-2 text-[11px] text-ns-text-faint">No runs yet.</li> : null}
          {items.map(({ run: r, depth }) => (
            <li key={r.id} className="flex shrink-0 items-stretch">
              {depth === 1 ? <CornerDownRight aria-hidden="true" className="mr-0.5 mt-1 size-3.5 shrink-0 text-ns-text-faint" /> : null}
              <button
                type="button"
                onClick={() => openRun(r.id)}
                aria-current={state.runId === r.id ? "true" : undefined}
                className={cn(
                  "flex w-[200px] flex-col gap-1 rounded-[3px] border px-2 py-1.5 text-left transition-colors",
                  state.runId === r.id ? "border-ns-accent bg-ns-accent/8" : "border-ns-border bg-ns-bg hover:border-ns-border-strong",
                  depth === 1 && "w-[184px]",
                )}
              >
                <span className="flex items-center gap-1.5">
                  <KindChip run={r} />
                  <span className="font-mono text-[10px] text-ns-text-faint">{shortId(r.id)}</span>
                  {r.is_demo ? <StatusChip tone="warn">demo</StatusChip> : null}
                  <span className="ml-auto font-mono text-[9.5px] text-ns-text-faint">{now ? fmtAgo(r.created_at, now) : ""}</span>
                </span>
                <span className="truncate text-[11px] text-ns-text">{r.kind === "intervention" ? r.interventions_label : r.prompt}</span>
                <span className="flex min-w-0 items-center gap-1 truncate font-mono text-[10.5px] text-ns-text-muted">
                  {r.kind === "intervention" ? (
                    <>
                      <OutputText text={r.generated_text} max={12} /> → <OutputText text={r.free_generated_text ?? ""} max={12} className="text-ns-accent" />
                    </>
                  ) : (
                    <OutputText text={r.generated_text} max={26} />
                  )}
                  {r.correctness ? <MathStatusChip status={r.correctness} /> : null}
                </span>
              </button>
            </li>
          ))}
        </ol>
      </div>
      <ExperimentsBox />
    </section>
  );
}

function KindChip({ run }: { run: RunSummary }) {
  if (run.kind === "intervention") return <StatusChip tone="accent">interv.</StatusChip>;
  if (run.kind === "math") return <StatusChip tone="paper">math</StatusChip>;
  return <StatusChip tone="neutral">{run.kind}</StatusChip>;
}

function ExperimentsBox() {
  const ws = useWorkspace();
  const { state, run, notify, openRun, update } = ws;
  const [mode, setMode] = useState<"save" | "open">("save");
  const [name, setName] = useState("");
  const [notes, setNotes] = useState("");
  const [saving, setSaving] = useState(false);
  const [version, setVersion] = useState(0);
  const exps = useResource<Experiment[]>(mode === "open" ? `experiments:${version}` : null, (signal) => api.listExperiments(signal), { cache: false });

  const r = run.data;
  const runIds = r ? [r.id, ...(state.diffRunId && state.diffRunId !== r.id ? [state.diffRunId] : [])] : [];

  const save = async () => {
    if (!r || !name.trim()) return;
    setSaving(true);
    try {
      const e = await api.createExperiment({
        name: name.trim(),
        run_ids: runIds,
        notes,
        payload: {
          run_id: r.id,
          diff_run_id: state.diffRunId,
          selection: state.selection,
          metric: state.metric,
          tab: state.tab,
          attention: { layer: state.attnLayer, head: state.attnHead },
          prompt: r.prompt,
        },
      });
      notify("ok", `Experiment “${e.name}” saved (${e.id}) with ${e.run_ids.length} run(s).`);
      setName("");
      setNotes("");
      setVersion((v) => v + 1);
    } catch (e) {
      notify("error", `Save failed: ${errorMessage(e)}`);
    } finally {
      setSaving(false);
    }
  };

  const reopen = async (id: string) => {
    try {
      const e = await api.getExperiment(id);
      const p = e.payload ?? {};
      const rid = (typeof p.run_id === "string" && p.run_id) || e.run_ids[0];
      if (!rid) return notify("error", "Experiment has no runs.");
      openRun(rid, { selection: p.selection ?? null, diffRunId: (p.diff_run_id as string | null | undefined) ?? null });
      update({
        ...(p.metric ? { metric: p.metric } : {}),
        ...(p.tab ? { tab: p.tab as typeof state.tab } : {}),
        ...(p.attention ? { attnLayer: p.attention.layer ?? null, attnHead: p.attention.head ?? null } : {}),
      });
      notify("info", `Reopened “${e.name}”${e.notes ? ` — hypothesis: ${truncate(e.notes, 80)}` : ""}`);
    } catch (err) {
      notify("error", `Open failed: ${errorMessage(err)}`);
    }
  };

  return (
    <div className="flex min-h-0 flex-col rounded-[var(--ns-radius)] border border-ns-border bg-ns-panel">
      <div className="flex h-8 shrink-0 items-center gap-2 border-b border-ns-border px-3">
        <span className="ns-label">Experiments</span>
        <Segmented
          label="Experiments mode"
          size="xs"
          className="ml-auto"
          value={mode}
          onChange={setMode}
          items={[
            { value: "save", label: "save" },
            { value: "open", label: "open" },
          ]}
        />
      </div>
      {mode === "save" ? (
        <form
          className="flex min-h-0 flex-1 flex-col gap-1.5 p-2"
          onSubmit={(e) => {
            e.preventDefault();
            void save();
          }}
        >
          <div className="flex gap-1.5">
            <TextInput value={name} onChange={(e) => setName(e.target.value)} placeholder="Experiment name" aria-label="Experiment name" maxLength={200} className="flex-1" />
            <Button type="submit" size="sm" caps variant="secondary" disabled={!r || !name.trim() || saving}>
              <BookmarkPlus aria-hidden="true" /> {saving ? "Saving" : "Save"}
            </Button>
          </div>
          <div className="relative min-h-0 flex-1">
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              aria-label="Hypothesis (inferred)"
              placeholder="Hypothesis — your interpretation, not a result"
              className="h-full min-h-[40px] w-full resize-none rounded-[3px] border border-ns-inferred/25 bg-ns-bg px-2 py-1 pr-24 text-[11.5px] text-ns-text placeholder:text-ns-text-faint"
            />
            <StatusBadge status="inferred" note="hypothesis" className="absolute right-1.5 top-1.5" />
          </div>
          <div className="truncate font-mono text-[10px] text-ns-text-faint">
            runs: {runIds.length ? runIds.map(shortId).join(", ") : "—"} · view: {state.metric}, {state.tab}
            {state.selection ? `, d${state.selection.depth} t${state.selection.position}` : ""}
          </div>
        </form>
      ) : (
        <div className="ns-scroll min-h-0 flex-1 p-1.5">
          {exps.error ? <ErrorBox error={exps.error} onRetry={exps.reload} /> : null}
          {exps.data?.length === 0 ? <p className="p-2 text-[11px] text-ns-text-faint">No saved experiments.</p> : null}
          <ol className="flex flex-col gap-1">
            {exps.data?.map((e) => (
              <li key={e.id}>
                <button
                  type="button"
                  onClick={() => void reopen(e.id)}
                  className="flex w-full items-center gap-2 rounded-[3px] border border-ns-border bg-ns-bg px-2 py-1 text-left hover:border-ns-border-strong"
                >
                  <FolderOpen aria-hidden="true" className="size-3.5 shrink-0 text-ns-text-faint" />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[11px] text-ns-text">{e.name}</span>
                    <span className="block truncate text-[10px] text-ns-inferred/80">{e.notes || "no hypothesis"}</span>
                  </span>
                  <span className="shrink-0 font-mono text-[9.5px] text-ns-text-faint">{e.run_ids.length} run</span>
                </button>
              </li>
            ))}
          </ol>
        </div>
      )}
    </div>
  );
}
