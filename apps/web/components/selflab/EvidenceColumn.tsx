"use client";

import { useEffect, useRef, useState } from "react";
import { ExternalLink, Microscope } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Panel } from "@/components/ui/Panel";
import { StatusChip } from "@/components/ui/StatusChip";
import { Caveat, EmptyState, ErrorBox, KV, Loading } from "@/components/workspace/ui";
import { useResource } from "@/components/workspace/useResource";
import { errorMessage } from "@/lib/api/client";
import { selflab, type EvidenceItem, type OpenInLabResponse, type TestStatus, type Trial } from "@/lib/api/selflab";
import { shortId } from "@/lib/format";
import { EvidenceCard, HypothesisCard } from "./EvidenceCard";
import { FixtureMark, TrialStatusChip } from "./marks";
import { useSelfLab } from "./SelfLabProvider";
import { showBlank } from "./spanSelection";
import { ValueTable, fmtScalar } from "./values";

export function EvidenceColumn() {
  return (
    <div className="flex flex-col gap-2">
      <LinkedInspector />
      <HypothesesPanel />
      <EvidencePanel />
    </div>
  );
}

/* ------------------------------------------------------------------ linked inspector */

export function TrialDetail({ trial }: { trial: Trial }) {
  const scalars = (o?: Record<string, unknown>) => Object.fromEntries(Object.entries(o ?? {}).filter(([, v]) => v === null || typeof v !== "object"));
  return (
    <div className="flex flex-col gap-2">
      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
        <KV k="condition" v={trial.condition} />
        <KV k="family" v={trial.family} />
        <KV k="item" v={trial.item_id} />
        <KV k="unit (group)" v={trial.group_id} />
        <KV k="prompt tokens" v={trial.n_prompt_tokens ?? "—"} />
        <KV k="generated" v={trial.n_generated_tokens ?? "—"} />
        <KV k="stopped by" v={trial.stopped_by ?? "—"} />
        <KV k="latency" v={trial.latency_s !== undefined ? `${fmtScalar(trial.latency_s)} s` : "—"} />
      </div>
      <div>
        <h5 className="ns-label mb-0.5 text-[9px]">Output (verbatim)</h5>
        <pre className="max-h-32 overflow-auto whitespace-pre-wrap break-words rounded-[3px] border border-ns-border bg-ns-bg px-2 py-1 font-mono text-[11.5px] text-ns-text">
          {trial.output_text.trim() ? trial.output_text : <span className="text-ns-text-faint">{showBlank(trial.output_text) || "∅"} (blank)</span>}
        </pre>
      </div>
      {trial.error ? <ErrorBox error={trial.error} /> : null}
      {trial.parsed && Object.keys(trial.parsed).length ? (
        <div>
          <h5 className="ns-label mb-0.5 text-[9px]">Parsed (selflab.parsing)</h5>
          <ValueTable data={scalars(trial.parsed)} empty="no scalar fields" />
        </div>
      ) : null}
      {trial.ground_truth && Object.keys(trial.ground_truth).length ? (
        <div>
          <h5 className="ns-label mb-0.5 text-[9px]">Ground truth (separate fields)</h5>
          <ValueTable data={trial.ground_truth} />
        </div>
      ) : null}
      {trial.measurements && Object.keys(trial.measurements).length ? (
        <div>
          <h5 className="ns-label mb-0.5 text-[9px]">Measurements</h5>
          <ValueTable data={trial.measurements} maxDepth={2} />
        </div>
      ) : null}
    </div>
  );
}

function LinkedInspector() {
  const { selectedTrialId, selectTrial, openRunInLab, notify, record, dataVersion } = useSelfLab();
  const trial = useResource<Trial>(selectedTrialId ? `selflab:trial:${selectedTrialId}:${dataVersion}` : null, (s) => selflab.trial(selectedTrialId as string, s), {
    cache: false,
  });
  const [lab, setLab] = useState<{ trialId: string; res: OpenInLabResponse } | null>(null);
  const [opening, setOpening] = useState(false);
  const t = trial.data?.id === selectedTrialId ? trial.data : undefined;
  const labRes = lab && lab.trialId === selectedTrialId ? lab.res : null;
  const anchor = useRef<HTMLDivElement>(null);
  // Bring the inspector into view when a trial is picked elsewhere (evidence card, branch, table). Only the
  // column's own scroll container moves: scrollIntoView would also scroll the shell's overflow-hidden frame.
  useEffect(() => {
    const el = anchor.current;
    const box = el?.closest(".ns-scroll");
    if (!selectedTrialId || !el || !box) return;
    const delta = el.getBoundingClientRect().top - box.getBoundingClientRect().top - 8;
    if (Math.abs(delta) > 4) box.scrollTo?.({ top: box.scrollTop + delta, behavior: "smooth" });
  }, [selectedTrialId]);

  const openInLab = async (tr: Trial) => {
    setOpening(true);
    try {
      const res = await selflab.openInLab(tr.id);
      setLab({ trialId: tr.id, res });
      trial.reload();
      if (res.reproduced) {
        notify("ok", `Replay reproduced trial ${tr.id}. Opening run ${res.run_id} in the debugger.`);
        openRunInLab(res.run_id);
      } else {
        notify("info", `Replay did NOT reproduce trial ${tr.id}. Review the difference before opening the run.`);
      }
    } catch (e) {
      notify("error", `Open in Lab failed: ${errorMessage(e)}`);
    } finally {
      setOpening(false);
    }
  };

  return (
    <div ref={anchor} className="scroll-mt-2">
    <Panel
      label="Linked inspector"
      right={
        selectedTrialId ? (
          <button type="button" onClick={() => selectTrial(null)} className="font-mono text-[10px] uppercase tracking-[0.08em] text-ns-text-faint hover:text-ns-text">
            close
          </button>
        ) : null
      }
      bodyClassName="p-2.5"
    >
      {!selectedTrialId ? (
        <EmptyState title="No trial selected" className="py-3">
          Click a trial id on an evidence card, a branch in COMPARE CONTEXTS, or a row of the trial table. OPEN IN LAB replays it and opens the recorded
          tensors in the neural debugger.
        </EmptyState>
      ) : trial.error && !t ? (
        <ErrorBox error={trial.error} onRetry={trial.reload} />
      ) : !t ? (
        <Loading compact label={`Loading trial ${selectedTrialId}`} />
      ) : (
        <div className="flex flex-col gap-2">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="font-mono text-[11px] text-ns-text">{t.id}</span>
            <TrialStatusChip status={t.status} />
            <FixtureMark show={record?.is_fixture} what="This trial" />
          </div>
          <TrialDetail trial={t} />
          <div className="flex flex-wrap items-center gap-2 border-t border-ns-border pt-2">
            <Button variant="primary" caps size="sm" onClick={() => void openInLab(t)} disabled={opening}>
              <Microscope aria-hidden="true" /> {opening ? "Replaying…" : "Open in Lab"}
            </Button>
            {(t.run_ids ?? []).map((r) => (
              <Button key={r} variant="ghost" size="xs" caps onClick={() => openRunInLab(r)} title={`Open stored run ${r}`}>
                <ExternalLink aria-hidden="true" /> run {shortId(r)}
              </Button>
            ))}
          </div>
          {(t.lab_runs ?? []).length ? (
            <div>
              <h5 className="ns-label mb-0.5 text-[9px]">Lab runs linked to this trial</h5>
              <ul className="flex flex-col gap-0.5">
                {(t.lab_runs ?? []).map((l) => (
                  <li key={`${l.run_id}-${l.created_at ?? ""}`} className="flex flex-wrap items-center gap-1.5 font-mono text-[10.5px] text-ns-text-muted">
                    <button type="button" onClick={() => openRunInLab(l.run_id)} className="text-ns-observed hover:underline">
                      {l.run_id}
                    </button>
                    {l.kind ? <span>{l.kind.replace(/_/g, " ")}</span> : null}
                    {typeof l.reproduced === "boolean" ? (
                      <StatusChip tone={l.reproduced ? "ok" : "warn"}>{l.reproduced ? "reproduced" : "not reproduced"}</StatusChip>
                    ) : null}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          {labRes ? (
            <div className="rounded-[3px] border border-ns-border bg-ns-bg px-2 py-1.5" data-testid="open-in-lab-result">
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-[11px] text-ns-text-muted">Deterministic replay:</span>
                <StatusChip tone={labRes.reproduced ? "ok" : "warn"} dot>
                  {labRes.reproduced ? "reproduced" : "not reproduced"}
                </StatusChip>
                <span className="font-mono text-[10.5px] text-ns-text-muted">
                  run {labRes.run_id}
                  {labRes.compared ? ` · compared ${labRes.compared.replace("_", " ")}` : ""}
                </span>
              </div>
              {!labRes.reproduced ? (
                <>
                  {labRes.diff !== undefined && labRes.diff !== null ? (
                    <pre className="mt-1 max-h-28 overflow-auto whitespace-pre-wrap break-words font-mono text-[10.5px] text-ns-warn">
                      {typeof labRes.diff === "string" ? labRes.diff : JSON.stringify(labRes.diff, null, 2)}
                    </pre>
                  ) : null}
                  <Button variant="secondary" size="xs" caps className="mt-1" onClick={() => openRunInLab(labRes.run_id)}>
                    <ExternalLink aria-hidden="true" /> Open the replay anyway
                  </Button>
                </>
              ) : null}
            </div>
          ) : (
            <Caveat>OPEN IN LAB re-runs this trial deterministically and reports whether the replay reproduces the stored output.</Caveat>
          )}
        </div>
      )}
    </Panel>
    </div>
  );
}

/* ------------------------------------------------------------------ hypotheses */

function HypothesesPanel() {
  const { record } = useSelfLab();
  const hs = record?.committed.spec.hypotheses ?? [];
  if (!record || hs.length === 0) return null;
  return (
    <Panel label="Pre-registered hypotheses" bodyClassName="flex flex-col gap-1.5 p-2.5">
      <Caveat>Committed with the spec before any trial ran. A hypothesis says what the test checks; the evidence below says what it found.</Caveat>
      {hs.map((h) => (
        <HypothesisCard key={h.id} h={h} />
      ))}
    </Panel>
  );
}

/* ------------------------------------------------------------------ evidence */

const GROUPS: { key: string; label: string; statuses: TestStatus[] }[] = [
  { key: "for", label: "For · supported", statuses: ["supported"] },
  { key: "against", label: "Against · contradicted", statuses: ["contradicted"] },
  { key: "open", label: "Unresolved · inconclusive, control failed, untested", statuses: ["inconclusive", "control_failed", "untested"] },
];

export function groupEvidence(items: EvidenceItem[]): Record<string, EvidenceItem[]> {
  const out: Record<string, EvidenceItem[]> = { for: [], against: [], open: [] };
  for (const it of items) {
    const g = GROUPS.find((x) => x.statuses.includes(it.status));
    out[g ? g.key : "open"].push(it);
  }
  return out;
}

function EvidencePanel() {
  const { evidence, caseId, experimentId, caseDetail, experiments, record, selectTrial, openRunInLab } = useSelfLab();
  const detail = caseDetail.data?.case.id === caseId ? caseDetail.data : undefined;
  const items = evidence.data ?? [];
  const groups = groupEvidence(items);
  const fixtureFor = (it: EvidenceItem) => {
    const exp = it.experiment_id ? (record?.id === it.experiment_id ? record : experiments.data?.find((e) => e.id === it.experiment_id)) : null;
    return Boolean(exp?.is_fixture || exp?.committed.provenance?.model.is_demo || detail?.case.is_fixture);
  };
  return (
    <Panel
      label="Evidence"
      right={<span className="font-mono text-[10px] text-ns-text-faint">{experimentId ? `experiment ${experimentId}` : caseId ? `case ${caseId}` : ""}</span>}
      bodyClassName="flex flex-col gap-2 p-2.5"
    >
      {!caseId && !experimentId ? (
        <EmptyState title="No evidence yet" className="py-3">
          Evidence items are produced by analysis code after an experiment runs. Investigate a claim and run a committed experiment to see them here.
        </EmptyState>
      ) : evidence.error && !evidence.data ? (
        <ErrorBox error={evidence.error} onRetry={evidence.reload} />
      ) : !evidence.data ? (
        <Loading compact label="Loading evidence" />
      ) : items.length === 0 ? (
        <EmptyState title="No evidence recorded" className="py-3">
          {record && record.status !== "done"
            ? `Experiment ${record.id} is ${record.status}. Evidence is written when its analysis runs after the last trial.`
            : "Commit and run an experiment on this case. Its analysis writes evidence items with a kind, a test status and a scope."}
        </EmptyState>
      ) : (
        <>
          <Caveat>
            Each item keeps its kind, test status and scope apart. A status holds only within its scope; nothing here is a general verdict on the model.
          </Caveat>
          {GROUPS.map((g) =>
            groups[g.key].length ? (
              <section key={g.key} aria-label={g.label} className="flex flex-col gap-1.5">
                <h4 className="ns-label text-[9.5px] text-ns-text">
                  {g.label} <span className="text-ns-text-faint">({groups[g.key].length})</span>
                </h4>
                {groups[g.key].map((it) => (
                  <EvidenceCard key={it.id} item={it} fixture={fixtureFor(it)} onSelectTrial={selectTrial} onOpenRun={openRunInLab} />
                ))}
              </section>
            ) : null,
          )}
        </>
      )}
    </Panel>
  );
}
