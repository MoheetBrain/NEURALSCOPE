"use client";

import { useState } from "react";
import { Crosshair, GitBranch, Play, Plus, X } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { StatusChip } from "@/components/ui/StatusChip";
import { api, errorMessage } from "@/lib/api/client";
import type { DlaResponse, HeadSweepResult, Intervention, InterventionType, RunDetail } from "@/lib/api/types";
import { cn } from "@/lib/cn";
import { fmtDelta, fmtNum, fmtProb, fmtSeconds, shortId, truncate } from "@/lib/format";
import { layerForDepth } from "@/lib/matrix";
import { describeIntervention, draftToIntervention, useWorkspaceActions } from "../actions";
import { LayerHeadHeatmap } from "../charts/LayerHeadHeatmap";
import { useSelection } from "../hooks";
import { useResource } from "../useResource";
import { useWorkspace, type PositionsMode } from "../WorkspaceProvider";
import { PatchSweepPanel } from "./PatchSweepPanel";
import { Caveat, ErrorBox, Field, Loading, NumberInput, OutputText, ProgressBar, SectionHeader, Segmented, SelectInput, StatusBadge, TextInput, Tok } from "../ui";

const MATH_BUDGET = 24;

const TYPES: readonly { value: InterventionType; label: string; cap: string }[] = [
  { value: "head_ablation", label: "Head", cap: "head_ablation" },
  { value: "mlp_ablation", label: "MLP", cap: "mlp_ablation" },
  { value: "resid_patch", label: "Patch", cap: "activation_patching" },
  { value: "resid_steer", label: "Steer", cap: "steering" },
];

export function InterventionsTab({ run }: { run: RunDetail }) {
  const { state, update } = useWorkspace();
  return (
    <div className="flex flex-col gap-5">
      <Builder run={run} />
      <Children run={run} />
      <section className="flex flex-col gap-2">
        <SectionHeader
          label="Layer × head map"
          right={
            <Segmented
              label="Map"
              size="xs"
              value={state.sweepView}
              onChange={(v) => update({ sweepView: v })}
              items={[
                { value: "sweep", label: "Head sweep" },
                { value: "patch", label: "Patch sweep" },
                { value: "dla", label: "DLA" },
              ]}
            />
          }
        />
        {state.sweepView === "sweep" ? <SweepPanel run={run} /> : state.sweepView === "patch" ? <PatchSweepPanel run={run} /> : <DlaPanel run={run} />}
      </section>
    </div>
  );
}

// ------------------------------------------------------------------ builder

function Builder({ run }: { run: RunDetail }) {
  const { state, update, updateDraft, model, pending, history, notify } = useWorkspace();
  const { intervene } = useWorkspaceActions();
  const sel = useSelection(run);
  const d = state.draft;
  const nLayers = model?.n_layers ?? run.depth_labels.length - 1;
  const nHeads = model?.n_heads ?? 12;
  const caps = model?.capabilities ?? {};
  const [maxNew, setMaxNew] = useState<number | null>(null);
  const busy = pending?.kind === "intervention";
  const isChild = run.kind === "intervention" && Boolean(run.parent_run_id);
  const selPos = sel?.position ?? run.target?.position ?? run.prompt_len - 1;
  // Math Lab runs record max_new_tokens = number of tokens actually generated, so a rerun with the
  // stored value could cut a longer (changed) answer short; use the Math Lab budget instead.
  const defaultMaxNew = run.kind === "math" || run.math ? MATH_BUDGET : null;
  const sources = (history.data ?? []).filter((r) => r.kind !== "intervention");

  const fromSelection = () => {
    if (!sel) return;
    updateDraft({ layer: layerForDepth(sel.depth, nLayers), depth: sel.depth, position: sel.position, plusPosition: sel.position, minusPosition: sel.position });
  };

  const current = (): Intervention | null => {
    try {
      return draftToIntervention(d, selPos, run.tokens.length);
    } catch (e) {
      notify("error", errorMessage(e));
      return null;
    }
  };

  const execute = async () => {
    let specs = state.stack;
    if (specs.length === 0) {
      const c = current();
      if (!c) return;
      specs = [c];
    }
    // Interventions on an intervention run are applied to its baseline together with the run's own
    // interventions, so effects compose and the comparison stays "vs the un-intervened baseline".
    const base = isChild ? { ...run, id: run.parent_run_id as string } : run;
    const all = isChild ? [...run.interventions, ...specs] : specs;
    await intervene(base, all, maxNew ?? defaultMaxNew);
  };

  const positionsControl = (d.type === "head_ablation" || d.type === "mlp_ablation" || d.type === "resid_steer") && (
    <Field label="positions" className="col-span-2">
      <div className="flex flex-wrap items-center gap-2">
        <Segmented<PositionsMode>
          label="Positions"
          size="xs"
          value={d.positionsMode}
          onChange={(v) => updateDraft({ positionsMode: v })}
          items={[
            { value: "all", label: "all" },
            { value: "selected", label: `t${selPos}` },
            { value: "custom", label: "custom" },
          ]}
        />
        {d.positionsMode === "custom" ? (
          <TextInput className="w-40" value={d.customPositions} onChange={(e) => updateDraft({ customPositions: e.target.value })} placeholder="e.g. 27-32, 43" aria-label="Custom positions" />
        ) : null}
      </div>
    </Field>
  );

  return (
    <section className="flex flex-col gap-2.5">
      <SectionHeader
        label="Intervention builder"
        status="causal"
        note="on rerun"
        right={
          sel ? (
            <Button size="xs" variant="ghost" onClick={fromSelection} title="Prefill layer / depth / position from the matrix selection">
              <Crosshair aria-hidden="true" /> from selection ({run.depth_labels[sel.depth]}, t{sel.position})
            </Button>
          ) : null
        }
      />
      <Segmented<InterventionType>
        label="Intervention type"
        value={d.type}
        onChange={(v) => updateDraft({ type: v })}
        items={TYPES.map((t) => ({ value: t.value, label: t.label, disabled: caps[t.cap] === false, title: caps[t.cap] === false ? "not supported by this model" : undefined }))}
      />

      <div className="grid grid-cols-2 gap-2">
        {d.type === "head_ablation" || d.type === "mlp_ablation" ? (
          <>
            <Field label={`layer (0–${nLayers - 1})`}>
              <NumberInput value={d.layer} min={0} max={nLayers - 1} onValue={(v) => updateDraft({ layer: clampInt(v, 0, nLayers - 1) })} aria-label="Layer" />
            </Field>
            {d.type === "head_ablation" ? (
              <Field label={`head (0–${nHeads - 1})`}>
                <NumberInput value={d.head} min={0} max={nHeads - 1} onValue={(v) => updateDraft({ head: clampInt(v, 0, nHeads - 1) })} aria-label="Head" />
              </Field>
            ) : (
              <div />
            )}
            <Field label="factor" hint="0 = zero-ablation; 1 = unchanged; >1 amplifies">
              <NumberInput value={d.factor} step={0.1} onValue={(v) => updateDraft({ factor: v })} aria-label="Factor" />
            </Field>
            <div />
            {positionsControl}
          </>
        ) : null}

        {d.type === "resid_patch" ? (
          <>
            <Field label="source run" className="col-span-2" hint="h[depth, t] ← (1−α)·h + α·h_source[depth, t_src]">
              <SelectInput value={d.sourceRunId} onChange={(e) => updateDraft({ sourceRunId: e.target.value })} aria-label="Source run">
                <option value="">— pick a stored run —</option>
                {sources.map((r) => (
                  <option key={r.id} value={r.id} disabled={r.id === run.id}>
                    {shortId(r.id)} · {truncate(r.prompt, 40)} ({r.n_tokens} tok)
                  </option>
                ))}
              </SelectInput>
            </Field>
            <Field label={`depth (0–${nLayers})`}>
              <NumberInput value={d.depth} min={0} max={nLayers} onValue={(v) => updateDraft({ depth: clampInt(v, 0, nLayers) })} aria-label="Depth" />
            </Field>
            <Field label="position">
              <NumberInput value={d.position} min={0} max={run.tokens.length - 1} onValue={(v) => updateDraft({ position: clampInt(v, 0, run.tokens.length - 1) })} aria-label="Position" />
            </Field>
            <Field label="source position" hint="blank = same position">
              <TextInput value={d.sourcePosition} onChange={(e) => updateDraft({ sourcePosition: e.target.value })} placeholder={`${d.position}`} aria-label="Source position" />
            </Field>
            <Field label="α">
              <NumberInput value={d.alpha} step={0.1} min={0} max={1} onValue={(v) => updateDraft({ alpha: v })} aria-label="Alpha" />
            </Field>
          </>
        ) : null}

        {d.type === "resid_steer" ? (
          <>
            <Field label="+ run" hint="steering vector = h₊ − h₋ at this depth">
              <SelectInput value={d.plusRunId} onChange={(e) => updateDraft({ plusRunId: e.target.value })} aria-label="Plus run">
                <option value="">— run —</option>
                {sources.map((r) => (
                  <option key={r.id} value={r.id}>
                    {shortId(r.id)} · {truncate(r.prompt, 28)}
                  </option>
                ))}
              </SelectInput>
            </Field>
            <Field label="+ position">
              <NumberInput value={d.plusPosition} min={0} onValue={(v) => updateDraft({ plusPosition: clampInt(v, 0, 10000) })} aria-label="Plus position" />
            </Field>
            <Field label="− run">
              <SelectInput value={d.minusRunId} onChange={(e) => updateDraft({ minusRunId: e.target.value })} aria-label="Minus run">
                <option value="">— run —</option>
                {sources.map((r) => (
                  <option key={r.id} value={r.id}>
                    {shortId(r.id)} · {truncate(r.prompt, 28)}
                  </option>
                ))}
              </SelectInput>
            </Field>
            <Field label="− position">
              <NumberInput value={d.minusPosition} min={0} onValue={(v) => updateDraft({ minusPosition: clampInt(v, 0, 10000) })} aria-label="Minus position" />
            </Field>
            <Field label={`depth (0–${nLayers})`}>
              <NumberInput value={d.depth} min={0} max={nLayers} onValue={(v) => updateDraft({ depth: clampInt(v, 0, nLayers) })} aria-label="Steer depth" />
            </Field>
            <Field label="scale">
              <NumberInput value={d.scale} step={0.5} onValue={(v) => updateDraft({ scale: v })} aria-label="Scale" />
            </Field>
            {positionsControl}
          </>
        ) : null}

        <Field label="max new tokens" hint="blank = same as the run">
          <NumberInput value={maxNew ?? NaN} min={1} max={48} placeholder={`${defaultMaxNew ?? run.generation.max_new_tokens}`} onValue={(v) => setMaxNew(clampInt(v, 1, 48))} aria-label="Max new tokens for rerun" />
        </Field>
      </div>

      {state.stack.length > 0 ? (
        <ol className="flex flex-col gap-1 rounded-[3px] border border-ns-border bg-ns-bg/60 p-1.5">
          {state.stack.map((s, i) => (
            <li key={i} className="flex items-center gap-2 font-mono text-[10.5px] text-ns-text">
              <span className="text-ns-text-faint">{i + 1}.</span>
              <span className="min-w-0 flex-1 truncate">{describeIntervention(s)}</span>
              <button type="button" aria-label="Remove from stack" className="text-ns-text-faint hover:text-ns-text" onClick={() => update({ stack: state.stack.filter((_, j) => j !== i) })}>
                <X className="size-3" aria-hidden="true" />
              </button>
            </li>
          ))}
        </ol>
      ) : null}

      {isChild ? (
        <p className="rounded-[3px] border border-ns-warn/30 bg-ns-warn/5 px-2 py-1 text-[10.5px] leading-snug text-ns-warn">
          This is an intervention run. New interventions are applied to its baseline <span className="font-mono">{shortId(run.parent_run_id)}</span> together with “{run.interventions_label}”, so effects compose and are compared against the un-intervened baseline.
        </p>
      ) : null}

      <div className="flex flex-wrap items-center gap-2">
        <Button variant="primary" caps size="sm" onClick={() => void execute()} disabled={busy}>
          <Play aria-hidden="true" /> {busy ? "Rerunning…" : d.type.endsWith("ablation") && state.stack.length === 0 ? "Ablate + rerun" : "Intervene + rerun"}
        </Button>
        <Button
          size="sm"
          variant="ghost"
          onClick={() => {
            const c = current();
            if (c) update({ stack: [...state.stack, c] });
          }}
        >
          <Plus aria-hidden="true" /> add to stack
        </Button>
        <span className="min-w-0 truncate font-mono text-[10.5px] text-ns-text-muted">
          {state.stack.length ? `${state.stack.length} stacked` : safeDescribe(d, selPos, run.tokens.length)}
        </span>
      </div>
      {busy ? <Loading compact label="Free-running generation + teacher-forced capture under the intervention" hint="≈5 s on CPU" startedAt={pending?.startedAt} /> : null}
    </section>
  );
}

function safeDescribe(d: Parameters<typeof draftToIntervention>[0], sel: number, n: number): string {
  try {
    return describeIntervention(draftToIntervention(d, sel, n));
  } catch (e) {
    return errorMessage(e);
  }
}

function clampInt(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, Math.round(v)));
}

// ------------------------------------------------------------------ children

function Children({ run }: { run: RunDetail }) {
  const { state, update, openRun } = useWorkspace();
  const isChild = run.kind === "intervention";
  if (isChild) {
    return (
      <section className="flex flex-col gap-1.5">
        <SectionHeader label="Lineage" />
        <div className="flex items-center gap-2 text-[11px] text-ns-text-muted">
          <GitBranch aria-hidden="true" className="size-3.5" />
          <span className="font-mono text-ns-causal">{run.interventions_label}</span> on
          <button type="button" className="font-mono text-ns-accent hover:underline" onClick={() => openRun(run.parent_run_id as string, { diffRunId: run.id })}>
            {shortId(run.parent_run_id)}
          </button>
          <Button size="xs" variant="ghost" onClick={() => update({ diffRunId: run.id, tab: "diff" })}>
            diff
          </Button>
        </div>
      </section>
    );
  }
  return (
    <section className="flex flex-col gap-1.5">
      <SectionHeader label={`Intervention runs on this baseline (${run.children.length})`} status={run.children.length ? "causal" : undefined} />
      {run.children.length === 0 ? (
        <p className="text-[11px] text-ns-text-faint">None yet — build an intervention above and rerun.</p>
      ) : (
        <ol className="flex flex-col divide-y divide-ns-border rounded-[3px] border border-ns-border">
          {run.children.map((c) => (
            <li key={c.id} className={cn("flex items-center gap-2 px-2 py-1.5", state.diffRunId === c.id && "bg-ns-causal/6")}>
              <div className="min-w-0 flex-1">
                <div className="truncate font-mono text-[10.5px] text-ns-text">{c.interventions_label}</div>
                <div className="truncate font-mono text-[10px] text-ns-text-faint">
                  {shortId(c.id)} · out <OutputText text={c.generated_text} className="text-ns-text-muted" /> → <OutputText text={c.free_generated_text ?? "?"} className={c.free_generated_text !== c.generated_text ? "text-ns-accent" : "text-ns-text-muted"} />
                </div>
              </div>
              <Button size="xs" variant="ghost" onClick={() => update({ diffRunId: c.id, tab: "diff" })}>
                diff
              </Button>
              <Button size="xs" variant="ghost" onClick={() => openRun(c.id, { keepTab: true })} title="Open this intervention run in the workspace (matrix: Δ vs parent)">
                open
              </Button>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

// ------------------------------------------------------------------ sweep

function SweepPanel({ run }: { run: RunDetail }) {
  const { jobs, activeJob, state } = useWorkspace();
  const { launchHeadSweep, prefillHeadAblation } = useWorkspaceActions();
  const sweeps = (run.metrics.sweeps ?? []).filter((s): s is HeadSweepResult => s.kind === "head_ablation");
  const job = jobs.head_sweep;
  const running = job && job.scopeKey === run.id && (job.status === "queued" || job.status === "running") ? job : null;
  const fresh = job && job.scopeKey === run.id && job.status === "done" ? (job.result as HeadSweepResult | undefined) : undefined;
  const sweep: HeadSweepResult | undefined = sweeps[sweeps.length - 1] ?? fresh;
  const [which, setWhich] = useState<"delta_metric" | "delta_prob" | "kl">("delta_metric");
  const noTarget = run.target?.token_id == null;
  const d = state.draft;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          caps
          variant="secondary"
          disabled={Boolean(activeJob) || noTarget}
          onClick={() => void launchHeadSweep(run)}
          title={activeJob ? `Busy: ${activeJob.label}` : noTarget ? "Set a target token first" : undefined}
        >
          {sweep ? "Re-run head sweep" : "Run head sweep"}
        </Button>
        <span className="text-[10.5px] text-ns-text-faint">
          Zero-ablates each of the {sweep ? sweep.delta_metric.length * (sweep.delta_metric[0]?.length ?? 0) : "L×H"} heads one at a time (all positions) · ≈80 s on CPU
        </span>
      </div>
      {running ? (
        <div className="flex flex-col gap-1">
          <ProgressBar value={running.progress} tone="causal" />
          <span className="font-mono text-[10px] text-ns-text-muted">
            {running.status} · {running.message} · {Math.round(running.progress * 100)}%
          </span>
        </div>
      ) : null}
      {job?.status === "error" && job.scopeKey === run.id ? <ErrorBox error={job.error} /> : null}
      {sweep ? (
        <div className="flex flex-col gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge status={sweep.status} note={sweep.scope} />
            <Segmented
              label="Sweep quantity"
              size="xs"
              value={which}
              onChange={setWhich}
              items={[
                { value: "delta_metric", label: sweep.metric === "logit_diff" ? "Δ logit diff" : "Δ log P" },
                { value: "delta_prob", label: "Δ P" },
                { value: "kl", label: "KL" },
              ]}
            />
          </div>
          <p className="text-[10.5px] leading-snug text-ns-text-muted">
            Metric at t{sweep.position}: {sweep.metric === "logit_diff" ? `logit(${sweep.target.text}) − logit(${sweep.contrast?.text})` : `log P(${sweep.target.text})`} · baseline {fmtNum(sweep.baseline_metric, 4)} (P = {fmtProb(sweep.baseline_prob)}) · {fmtSeconds(sweep.seconds)}
            {sweeps.length > 1 ? ` · latest of ${sweeps.length} sweeps` : ""}
          </p>
          <LayerHeadHeatmap
            values={sweep[which]}
            selected={d.type === "head_ablation" ? { layer: d.layer, head: d.head } : null}
            onCellClick={(l, h) => prefillHeadAblation(l, h)}
            colorbarLabel={which}
            negLabel={which === "kl" ? "" : "hurts"}
            posLabel={which === "kl" ? "" : "helps"}
            formatTick={(v) => fmtNum(v, 2)}
            ariaLabel={`Head ablation sweep, ${which}. Most harmful: ${sweep.most_harmful
              .slice(0, 3)
              .map((x) => `L${x.layer}.H${x.head} ${fmtDelta(x.delta, 3)}`)
              .join(", ")}.`}
            tooltip={(l, h) => (
              <div className="flex flex-col">
                <span className="text-ns-accent">
                  L{l}.H{h}
                </span>
                <span>Δ metric {fmtDelta(sweep.delta_metric[l]?.[h] ?? null, 4)}</span>
                <span>Δ P {fmtDelta(sweep.delta_prob[l]?.[h] ?? null, 4)}</span>
                <span>KL {fmtNum(sweep.kl[l]?.[h] ?? null, 4)}</span>
                <span className="text-ns-text-faint">click → prefill ablation</span>
              </div>
            )}
          />
          <div className="grid grid-cols-2 gap-3">
            <RankList title="Most harmful (ablation lowers metric)" items={sweep.most_harmful.slice(0, 6)} onPick={prefillHeadAblation} />
            <RankList title="Most helpful (ablation raises metric)" items={sweep.most_helpful.slice(0, 6)} onPick={prefillHeadAblation} />
          </div>
        </div>
      ) : !running ? (
        <p className="text-[11px] text-ns-text-faint">No sweep for this run yet.</p>
      ) : null}
    </div>
  );
}

function RankList({ title, items, onPick }: { title: string; items: { layer: number; head: number; delta: number | null }[]; onPick: (l: number, h: number) => void }) {
  return (
    <div>
      <div className="ns-label mb-1 text-[9px]">{title}</div>
      <ol className="flex flex-col gap-0.5">
        {items.map((x) => (
          <li key={`${x.layer}-${x.head}`}>
            <button type="button" onClick={() => onPick(x.layer, x.head)} className="flex w-full items-center justify-between gap-2 rounded-[2px] px-1 font-mono text-[10.5px] hover:bg-ns-panel-2">
              <span className="text-ns-text">
                L{x.layer}.H{x.head}
              </span>
              <span className="ns-tabular text-ns-text-muted">{fmtDelta(x.delta, 4)}</span>
            </button>
          </li>
        ))}
      </ol>
    </div>
  );
}

// ------------------------------------------------------------------ DLA

function DlaPanel({ run }: { run: RunDetail }) {
  const { prefillHeadAblation } = useWorkspaceActions();
  const pos = run.target?.position ?? run.prompt_len - 1;
  const tid = run.target?.token_id ?? null;
  const cid = run.target?.contrast_id ?? null;
  const res = useResource<DlaResponse>(tid === null ? null : `dla:${run.id}:${pos}:${tid}:${cid}`, (signal) =>
    api.dla(run.id, { position: pos, target_token_id: tid, contrast_token_id: cid }, signal),
  );
  const d = res.data;
  if (tid === null) return <p className="text-[11px] text-ns-text-faint">Set a target token to decompose its logit.</p>;
  if (res.error && !d) return <ErrorBox error={res.error} onRetry={res.reload} />;
  if (!d) return <Loading label="Decomposing the target logit into head / MLP writes" />;
  return (
    <div className={cn("flex flex-col gap-2", res.stale && "ns-stale")}>
      <div className="flex flex-wrap items-center gap-2">
        <StatusBadge status={d.status} note="direct effects" />
        <span className="font-mono text-[10.5px] text-ns-text-muted">
          {d.contrast ? (
            <>
              logit(<Tok text={d.target.text} />) − logit(<Tok text={d.contrast.text} />)
            </>
          ) : (
            <>
              logit(<Tok text={d.target.text} />)
            </>
          )}{" "}
          at t{d.position}
        </span>
      </div>
      <LayerHeadHeatmap
        values={d.heads}
        onCellClick={(l, h) => prefillHeadAblation(l, h)}
        colorbarLabel="direct logit attribution"
        negLabel="suppresses"
        posLabel="promotes"
        ariaLabel={`Direct logit attribution per head. Top: ${d.top_heads
          .slice(0, 3)
          .map((x) => `L${x.layer}.H${x.head} ${fmtNum(x.value, 3)}`)
          .join(", ")}`}
        tooltip={(l, h) => (
          <div className="flex flex-col">
            <span className="text-ns-accent">
              L{l}.H{h}
            </span>
            <span>DLA {fmtNum(d.heads[l]?.[h] ?? null, 4)}</span>
            <span className="text-ns-text-faint">click → prefill ablation</span>
          </div>
        )}
      />
      <dl className="grid grid-cols-2 gap-x-4 font-mono text-[10.5px]">
        <div className="flex justify-between">
          <dt className="text-ns-text-muted">Σ components</dt>
          <dd>{fmtNum(d.sum_of_components, 4)}</dd>
        </div>
        <div className="flex justify-between">
          <dt className="text-ns-text-muted">actual logit</dt>
          <dd>{fmtNum(d.actual_logit, 4)}</dd>
        </div>
        <div className="flex justify-between">
          <dt className="text-ns-text-muted">embed</dt>
          <dd>{fmtNum(d.embed, 4)}</dd>
        </div>
        <div className="flex justify-between">
          <dt className="text-ns-text-muted">decomp. error</dt>
          <dd>{fmtNum(d.decomposition_error, 2)}</dd>
        </div>
      </dl>
      <div>
        <div className="ns-label mb-1 text-[9px]">Top heads by |DLA|</div>
        <div className="flex flex-wrap gap-1">
          {d.top_heads.slice(0, 8).map((x) => (
            <button
              key={`${x.layer}-${x.head}`}
              type="button"
              onClick={() => prefillHeadAblation(x.layer, x.head)}
              className="rounded-[3px] border border-ns-border px-1.5 py-0.5 font-mono text-[10.5px] text-ns-text hover:border-ns-border-strong"
            >
              L{x.layer}.H{x.head} <span className="text-ns-text-muted">{fmtNum(x.value, 2)}</span>
            </button>
          ))}
        </div>
      </div>
      <Caveat>{d.caveat}</Caveat>
      <StatusChip tone="neutral">DLA is DERIVED — not a causal test</StatusChip>
    </div>
  );
}
