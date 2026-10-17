"use client";

import { ScanSearch } from "lucide-react";
import { StatusChip } from "@/components/ui/StatusChip";
import { SelectInput } from "@/components/workspace/ui";
import { statusIsFixture, statusModel } from "@/lib/api/selflab";
import { truncate } from "@/lib/format";
import { BottomPanel } from "./BottomPanel";
import { ConversationColumn } from "./ConversationColumn";
import { EvidenceColumn } from "./EvidenceColumn";
import { FeatureBadge, FixtureMark } from "./marks";
import { PlanColumn } from "./PlanColumn";
import { SelfLabProvider, useSelfLab } from "./SelfLabProvider";

function SelfLabBar() {
  const { status, cases, caseId, openCase } = useSelfLab();
  const model = statusModel(status.data);
  const list = cases.data ?? [];
  const inList = list.some((c) => c.id === caseId);
  return (
    <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-1 border-b border-ns-border bg-ns-bg px-3 py-1.5">
      <span className="inline-flex items-center gap-1.5 font-mono text-[11px] uppercase tracking-[0.12em] text-ns-text">
        <ScanSearch aria-hidden="true" className="size-3.5 text-ns-accent" /> SelfLab
      </span>
      <FeatureBadge status="research_prototype" />
      <span className="hidden text-[11px] text-ns-text-muted lg:inline">Audit a model&apos;s claims about its own answers against measured evidence.</span>
      <FixtureMark show={statusIsFixture(status.data)} what="The target model" />
      <div className="ml-auto flex min-w-0 items-center gap-2">
        {model ? (
          <span className="hidden truncate font-mono text-[10.5px] text-ns-text-muted md:inline" title={`${model.model_id} · ${model.device} · ${model.dtype}`}>
            target {model.model_id}
          </span>
        ) : null}
        <label className="flex min-w-0 items-center gap-1.5">
          <span className="ns-label shrink-0 text-[9.5px]">Case</span>
          <SelectInput
            aria-label="Open a case"
            className="h-7 w-[260px] max-w-[46vw]"
            value={caseId ?? ""}
            onChange={(e) => openCase(e.target.value || null)}
          >
            <option value="">{list.length ? "— new investigation —" : "— no cases yet —"}</option>
            {caseId && !inList ? <option value={caseId}>{caseId}</option> : null}
            {list.slice(0, 100).map((c) => (
              <option key={c.id} value={c.id}>
                {c.is_fixture ? "[DEMO] " : ""}
                {truncate(c.title || c.claim.text, 40)} · {c.id}
              </option>
            ))}
          </SelectInput>
        </label>
        {status.error && !status.data ? <StatusChip tone="danger">selflab API unavailable</StatusChip> : null}
      </div>
    </div>
  );
}

/** /app/selflab — conversation (left) · plan & branches (centre) · inspector & evidence (right) · statistics & export (bottom). */
function SelfLabLayout() {
  return (
    <div className="flex min-h-0 flex-1 flex-col" data-testid="selflab-view">
      <SelfLabBar />
      <div className="min-h-0 flex-1 overflow-auto xl:overflow-hidden">
        <div className="grid gap-2 px-2 pt-2 xl:h-full xl:grid-cols-[minmax(300px,24vw)_minmax(0,1fr)_minmax(340px,28vw)]">
          <section aria-label="Target conversation" className="ns-scroll min-h-0 pb-2 xl:pr-1">
            <ConversationColumn />
          </section>
          <section aria-label="Plan and counterfactual branches" className="ns-scroll min-h-0 pb-2 xl:pr-1">
            <PlanColumn />
          </section>
          <section aria-label="Inspector and evidence" className="ns-scroll min-h-0 pb-2 xl:pr-1">
            <EvidenceColumn />
          </section>
        </div>
      </div>
      <BottomPanel />
    </div>
  );
}

export function SelfLabView() {
  return (
    <SelfLabProvider>
      <SelfLabLayout />
    </SelfLabProvider>
  );
}
