"use client";

import { ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { api } from "@/lib/api/client";
import type { InterventionComparison, RunDetail } from "@/lib/api/types";
import { cn } from "@/lib/cn";
import { diffMiddle, fmtDelta, fmtNum, fmtProb, fmtRank, shortId, splitSpecials, visibleToken } from "@/lib/format";
import { comparisonCacheKey } from "../actions";
import { SeriesChart } from "../charts/SeriesChart";
import { ProbBar } from "../charts/VectorStrip";
import { MathStatusChip } from "../PromptColumn";
import { useResource } from "../useResource";
import { useWorkspace } from "../WorkspaceProvider";
import { Caveat, EmptyState, ErrorBox, Loading, SectionHeader, SelectInput, Stat, StatusBadge, Tok } from "../ui";

const BEFORE = "var(--ns-text-muted)";
const AFTER = "var(--ns-accent)";

export function DiffTab({ run }: { run: RunDetail }) {
  const { state, update, openRun } = useWorkspace();
  const isChild = run.kind === "intervention" && Boolean(run.parent_run_id);
  const childId = isChild ? run.id : (state.diffRunId && run.children.some((c) => c.id === state.diffRunId) ? state.diffRunId : run.children[0]?.id) ?? null;
  const res = useResource<InterventionComparison | null>(childId ? comparisonCacheKey(childId) : null, async (signal) => {
    if (isChild && run.metrics.comparison) return run.metrics.comparison;
    const r = await api.getRun(childId as string, signal);
    return r.metrics.comparison ?? null;
  });
  const c = res.data;

  if (!childId) {
    return (
      <EmptyState title="No intervention yet">
        Build an intervention in the <button type="button" className="text-ns-accent hover:underline" onClick={() => update({ tab: "interventions" })}>Intervene</button> tab and press ABLATE + RERUN — the baseline vs intervention diff appears here.
      </EmptyState>
    );
  }

  const child = isChild ? run : run.children.find((x) => x.id === childId);
  const label = isChild ? run.interventions_label : (child?.interventions_label ?? "");
  const tgt = c?.target?.token ?? run.target?.token ?? "target";

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-1.5">
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge status={c?.status ?? "causal"} size="sm" note={c?.scope.split(";")[0]} />
          {!isChild && run.children.length > 1 ? (
            <SelectInput aria-label="Intervention run to compare" className="h-6 w-auto max-w-[260px] text-[10.5px]" value={childId} onChange={(e) => update({ diffRunId: e.target.value })}>
              {run.children.map((ch) => (
                <option key={ch.id} value={ch.id}>
                  {shortId(ch.id)} · {ch.interventions_label}
                </option>
              ))}
            </SelectInput>
          ) : null}
        </div>
        <p className="font-mono text-[12px] text-ns-causal">{label}</p>
        <div className="flex flex-wrap items-center gap-2 font-mono text-[10px] text-ns-text-faint">
          baseline {shortId(isChild ? run.parent_run_id : run.id)} <ArrowRight className="size-3" aria-hidden="true" /> intervention {shortId(childId)}
          {isChild ? (
            <Button size="xs" variant="ghost" onClick={() => openRun(run.parent_run_id as string, { diffRunId: run.id, keepTab: true })}>
              open baseline
            </Button>
          ) : (
            <Button size="xs" variant="ghost" onClick={() => openRun(childId, { keepTab: true })} title="Open the intervention run (matrix offers Δ vs parent)">
              open intervention run
            </Button>
          )}
        </div>
        {c ? <p className="text-[10.5px] leading-snug text-ns-text-faint">Scope: {c.scope}</p> : null}
      </div>

      {res.error && !c ? <ErrorBox error={res.error} onRetry={res.reload} /> : null}
      {res.data === undefined && !res.error ? <Loading label="Loading comparison" /> : null}
      {res.data === null ? <p className="text-[11px] text-ns-text-faint">This run has no stored comparison.</p> : null}

      {c ? (
        <div className={cn("flex flex-col gap-5", res.stale && "ns-stale")}>
          <OutputDiff run={run} c={c} />

          <section className="flex flex-col gap-2">
            <SectionHeader
              label={
                <>
                  Target <Tok text={tgt} className="normal-case text-ns-accent-2" /> at t{c.target_position}
                </>
              }
              status={c.status}
            />
            {c.before && c.after ? (
              <div className="grid grid-cols-2 gap-2">
                <Stat label="P(target)" value={`${fmtProb(c.before.prob)} → ${fmtProb(c.after.prob)}`} sub={`Δ ${fmtDelta(c.delta?.prob ?? null, 4)}`} tone="accent" />
                <Stat label="rank" value={`${fmtRank(c.before.rank)} → ${fmtRank(c.after.rank)}`} sub={`Δ ${fmtDelta(c.delta?.rank ?? null, 0)}`} />
                <Stat label="log P" value={`${fmtNum(c.before.logprob, 3)} → ${fmtNum(c.after.logprob, 3)}`} sub={`Δ ${fmtDelta(c.delta?.logprob ?? null, 3)}`} />
                <Stat label="logit" value={`${fmtNum(c.before.logit, 2)} → ${fmtNum(c.after.logit, 2)}`} sub={`Δ ${fmtDelta(c.delta?.logit ?? null, 3)}`} />
                {c.before.logit_diff !== undefined ? (
                  <Stat label="logit diff vs contrast" value={`${fmtNum(c.before.logit_diff, 2)} → ${fmtNum(c.after.logit_diff, 2)}`} sub={`Δ ${fmtDelta(c.delta?.logit_diff ?? null, 3)}`} />
                ) : null}
                <Stat label="KL(base ‖ int) at target" value={fmtNum(c.kl_at_target, 4)} sub={`max over generated positions ${fmtNum(c.max_kl, 4)}`} />
              </div>
            ) : (
              <p className="text-[11px] text-ns-text-faint">No target token on this run.</p>
            )}
            <p className="text-[11px] leading-relaxed text-ns-text-muted">
              {c.before && c.after ? (
                <>
                  {label} changed P(<Tok text={tgt} className="text-ns-text" />) by <span className="font-mono text-ns-text">{fmtDelta(c.delta?.prob ?? null, 4)}</span> ({fmtProb(c.before.prob)} → {fmtProb(c.after.prob)}).
                </>
              ) : null}
            </p>
          </section>

          {c.correctness_before !== undefined || c.correctness_after !== undefined ? (
            <section className="flex flex-col gap-1.5">
              <SectionHeader label="Math correctness (SymPy oracle)" status="causal" />
              <div className="flex items-center gap-2">
                <MathStatusChip status={c.correctness_before} />
                <ArrowRight className="size-3.5 text-ns-text-faint" aria-hidden="true" />
                <MathStatusChip status={c.correctness_after} />
              </div>
            </section>
          ) : null}

          <TopKPaired c={c} />

          <section className="flex flex-col gap-1.5">
            <SectionHeader label="KL per generated position (teacher-forced)" status={c.status} />
            <SeriesChart
              x={c.per_position.map((p) => `t${p.position}`)}
              series={[{ id: "kl", label: "KL", color: AFTER, values: c.per_position.map((p) => p.kl) }]}
              kind="bar"
              height={110}
              formatY={(v) => fmtNum(v, 2)}
              formatValue={(v) => fmtNum(v, 5)}
              tooltipExtra={(i) => {
                const p = c.per_position[i];
                return (
                  <div className="font-mono text-[10px] text-ns-text-muted">
                    next {visibleToken(p.next_token ?? "∅")} · top-1 {visibleToken(p.base_top1)} → {visibleToken(p.int_top1)}
                    {p.top1_changed ? <span className="text-ns-accent"> (changed)</span> : null}
                    <br />
                    log P {fmtNum(p.base_logprob, 3)} → {fmtNum(p.int_logprob, 3)}
                  </div>
                );
              }}
              markers={c.per_position.map((p, i) => (p.top1_changed ? { index: i, label: "top-1 flips", color: "var(--ns-accent-2)" } : null)).filter((m): m is { index: number; label: string; color: string } => m !== null)}
              ariaLabel={`KL divergence between baseline and intervention next-token distributions at each generated position: ${c.per_position.map((p) => `t${p.position} ${fmtNum(p.kl, 3)}`).join(", ")}`}
            />
          </section>

          <section className="flex flex-col gap-1.5">
            <SectionHeader
              label="Where the effect starts: max ‖Δh‖/‖h‖ by depth"
              status={c.status}
              right={
                c.earliest_affected_depth !== null ? (
                  <span className="font-mono text-[10.5px] text-ns-text-muted">
                    earliest affected: <span className="text-ns-accent-2">{run.depth_labels[c.earliest_affected_depth]}</span> (depth {c.earliest_affected_depth})
                  </span>
                ) : null
              }
            />
            <SeriesChart
              x={run.depth_labels}
              series={[{ id: "rd", label: "relative Δ", color: AFTER, values: c.resid_delta_by_depth }]}
              kind="bar"
              height={120}
              formatY={(v) => fmtNum(v, 2)}
              formatValue={(v) => fmtNum(v, 5)}
              markers={c.earliest_affected_depth !== null ? [{ index: c.earliest_affected_depth, label: `starts at ${run.depth_labels[c.earliest_affected_depth]}` }] : []}
              ariaLabel={`Relative change of the residual stream by depth (max over positions); the effect first appears at depth ${c.earliest_affected_depth}`}
            />
            <Caveat>Depth d is the residual after block d−1, so an edit in block ℓ first shows at depth ℓ+1. Values are teacher-forced on the baseline tokens.</Caveat>
          </section>
        </div>
      ) : null}
    </div>
  );
}

function OutputDiff({ run, c }: { run: RunDetail; c: InterventionComparison }) {
  const gen = run.tokens.filter((t) => t.role === "generated");
  const changed = new Set(c.changed_token_indices);
  const d = diffMiddle(c.output_before, c.output_after);
  return (
    <section className="flex flex-col gap-2">
      <SectionHeader label="Output" status="observed" note="both runs" right={c.output_changed ? <span className="font-mono text-[10.5px] text-ns-accent">changed</span> : <span className="font-mono text-[10.5px] text-ns-text-faint">unchanged</span>} />
      <div className="grid grid-cols-[64px_1fr] items-start gap-x-2 gap-y-1.5">
        <span className="ns-label pt-1 text-[9px]">baseline</span>
        <div className="flex flex-wrap gap-[3px] rounded-[3px] border border-ns-border bg-ns-bg p-1.5">
          {gen.length ? (
            gen.map((t, i) => (
              <span
                key={t.index}
                className={cn(
                  "rounded-[3px] border px-1 font-mono text-[12px] leading-5",
                  changed.has(i) ? "border-ns-text-muted/60 bg-ns-text-muted/15 text-ns-text line-through decoration-ns-text-faint" : "border-transparent text-ns-text-muted",
                )}
              >
                {visibleToken(t.text)}
              </span>
            ))
          ) : (
            <Plain text={c.output_before} />
          )}
        </div>
        <span className="ns-label pt-1 text-[9px] text-ns-accent">intervened</span>
        <div className="rounded-[3px] border border-ns-accent/30 bg-ns-bg p-1.5 font-mono text-[12px] leading-5 text-ns-text">
          <Plain text={d.prefix} />
          {d.b ? <mark className="rounded-[2px] bg-ns-accent/25 px-0.5 text-ns-text">{<Plain text={d.b} />}</mark> : null}
          <Plain text={d.suffix} />
          {!c.output_after ? <span className="text-ns-text-faint">∅</span> : null}
        </div>
      </div>
      <p className="text-[10.5px] text-ns-text-faint">Baseline tokens that differ are struck through; the changed span of the free-running intervened output is highlighted.</p>
    </section>
  );
}

function Plain({ text }: { text: string }) {
  return (
    <>
      {splitSpecials(text).map((p, i) =>
        p.special ? (
          <span key={i} className="mx-0.5 inline-flex rounded-[3px] border border-ns-border-strong bg-ns-panel-2 px-1 align-middle text-[10px] leading-4 text-ns-text-muted">
            {visibleToken(p.text)}
          </span>
        ) : (
          <span key={i} className="whitespace-pre-wrap">
            {p.text}
          </span>
        ),
      )}
    </>
  );
}

function TopKPaired({ c }: { c: InterventionComparison }) {
  const ids: number[] = [];
  for (const e of [...c.top_before, ...c.top_after]) if (!ids.includes(e.id)) ids.push(e.id);
  const byB = new Map(c.top_before.map((e) => [e.id, e]));
  const byA = new Map(c.top_after.map((e) => [e.id, e]));
  const rows = ids.slice(0, 14).map((id) => ({ id, text: (byB.get(id) ?? byA.get(id))!.text, b: byB.get(id)?.prob ?? null, a: byA.get(id)?.prob ?? null }));
  return (
    <section className="flex flex-col gap-1.5">
      <SectionHeader label="Top-10 next-token distribution at the target position" status={c.status} />
      <div className="flex items-center gap-3 font-mono text-[10px] text-ns-text-muted" aria-hidden="true">
        <span className="inline-flex items-center gap-1">
          <span className="inline-block h-1.5 w-3 rounded-full" style={{ background: BEFORE }} /> baseline
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="inline-block h-1.5 w-3 rounded-full" style={{ background: AFTER }} /> intervened
        </span>
        <span className="text-ns-text-faint">— = outside that run&apos;s top-10</span>
      </div>
      <table className="w-full border-collapse">
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} className="border-t border-ns-border">
              <td className="w-[76px] py-1 pr-2">
                <Tok text={r.text} className="block max-w-[76px] truncate text-[11px] text-ns-text" />
              </td>
              <td className="py-1">
                <div className="flex items-center gap-2">
                  <ProbBar value={r.b} color={BEFORE} className="w-full" />
                  <span className="ns-tabular w-14 shrink-0 text-right font-mono text-[10px] text-ns-text-muted">{r.b === null ? "—" : fmtProb(r.b)}</span>
                </div>
                <div className="mt-0.5 flex items-center gap-2">
                  <ProbBar value={r.a} color={AFTER} className="w-full" />
                  <span className="ns-tabular w-14 shrink-0 text-right font-mono text-[10px] text-ns-text">{r.a === null ? "—" : fmtProb(r.a)}</span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
