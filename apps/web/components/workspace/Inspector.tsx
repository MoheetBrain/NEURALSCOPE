"use client";

import { Tabs, type TabItem } from "@/components/ui/Tabs";
import { AttentionTab } from "./inspector/AttentionTab";
import { DiffTab } from "./inspector/DiffTab";
import { InterventionsTab } from "./inspector/InterventionsTab";
import { LogitLensTab } from "./inspector/LogitLensTab";
import { TraceTab } from "./inspector/TraceTab";
import { useWorkspace, type InspectorTab } from "./WorkspaceProvider";
import { EmptyState, ErrorBox, Loading } from "./ui";

const TABS: readonly TabItem<InspectorTab>[] = [
  { value: "trace", label: "Trace", status: "observed" },
  { value: "lens", label: "Logit Lens", status: "derived" },
  { value: "attention", label: "Attention", status: "observed" },
  { value: "interventions", label: "Intervene", status: "causal" },
  { value: "diff", label: "Diff", status: "causal" },
];

const ID = "ns-inspector";

export function Inspector() {
  const { state, update, run } = useWorkspace();
  const r = run.data;
  return (
    <section aria-label="Inspector" className="flex h-full min-h-0 flex-col rounded-[var(--ns-radius)] border border-ns-border bg-ns-panel">
      <div className="flex shrink-0 items-center gap-2 px-3 pt-1">
        <span className="ns-label mr-1 hidden text-[9.5px] 2xl:inline">Inspector</span>
        <Tabs items={TABS} value={state.tab} onValueChange={(v) => update({ tab: v })} aria-label="Inspector views" idBase={ID} size="sm" className="min-w-0 flex-1 gap-3" />
      </div>
      <div
        role="tabpanel"
        id={`${ID}-panel-${state.tab}`}
        aria-labelledby={`${ID}-tab-${state.tab}`}
        tabIndex={0}
        className="ns-scroll min-h-0 flex-1 px-3 pb-4 pt-3"
      >
        {!r ? (
          run.error ? (
            <ErrorBox error={run.error} onRetry={run.reload} />
          ) : run.loading ? (
            <Loading label="Loading run" />
          ) : (
            <EmptyState title="Nothing selected">Run a prompt, then click any cell of the matrix to inspect it.</EmptyState>
          )
        ) : state.tab === "trace" ? (
          <TraceTab run={r} />
        ) : state.tab === "lens" ? (
          <LogitLensTab run={r} />
        ) : state.tab === "attention" ? (
          <AttentionTab run={r} />
        ) : state.tab === "interventions" ? (
          <InterventionsTab run={r} />
        ) : (
          <DiffTab run={r} />
        )}
      </div>
    </section>
  );
}
