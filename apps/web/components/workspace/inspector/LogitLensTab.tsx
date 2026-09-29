"use client";

import { useState } from "react";
import { api } from "@/lib/api/client";
import type { LogitLensResponse, RunDetail } from "@/lib/api/types";
import { cn } from "@/lib/cn";
import { fmtNum, fmtProb, fmtRank } from "@/lib/format";
import { SeriesChart } from "../charts/SeriesChart";
import { ProbBar } from "../charts/VectorStrip";
import { targetId, useSelection } from "../hooks";
import { useResource } from "../useResource";
import { useWorkspace } from "../WorkspaceProvider";
import { Caveat, ErrorBox, Loading, SectionHeader, Segmented, StatusBadge, Tok } from "../ui";

export function LogitLensTab({ run }: { run: RunDetail }) {
  const { select, update, model } = useWorkspace();
  const sel = useSelection(run);
  const tid = targetId(run);
  const [k, setK] = useState<"5" | "10">("5");
  const pos = sel?.position ?? run.target?.position ?? run.prompt_len - 1;
  const res = useResource<LogitLensResponse>(`lens:${run.id}:${pos}:${tid}`, (signal) => api.logitLens(run.id, pos, tid, signal), {
    debounceMs: 120,
  });
  const L = res.data;
  const isTargetPos = run.target?.position === pos;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <StatusBadge status="derived" size="sm" />
        <span className="font-mono text-[12px] text-ns-text">
          t{pos} <Tok text={run.tokens[pos]?.text ?? ""} className="rounded-[3px] border border-ns-border-strong bg-ns-bg px-1" />
        </span>
        <span className="text-[11px] text-ns-text-muted">
          → predicts <Tok text={L?.predicts_next ?? run.tokens[pos + 1]?.text ?? "∅"} className="text-ns-text" />
        </span>
        {!isTargetPos && run.target ? (
          <button
            type="button"
            className="ml-auto font-mono text-[10px] uppercase tracking-[0.08em] text-ns-accent hover:underline"
            onClick={() => select({ depth: sel?.depth ?? run.depth_labels.length - 1, position: run.target!.position })}
          >
            jump to target position t{run.target.position}
          </button>
        ) : null}
      </div>

      {res.error && !L ? <ErrorBox error={res.error} onRetry={res.reload} /> : null}
      {!L && !res.error ? <Loading label="Projecting every depth through the unembedding" /> : null}

      {L ? (
        <div className={cn("flex flex-col gap-4", res.stale && "ns-stale")}>
          {L.target ? (
            <section className="flex flex-col gap-1.5">
              <SectionHeader
                label={
                  <>
                    Rank of target <Tok text={L.target.text} className="normal-case text-ns-accent-2" /> through depth
                  </>
                }
                status="derived"
                right={
                  <span className="font-mono text-[10.5px] text-ns-text-muted">
                    first top-1:{" "}
                    <span className="text-ns-accent-2">
                      {L.first_depth_target_top1 !== null ? `${run.depth_labels[L.first_depth_target_top1]} (depth ${L.first_depth_target_top1})` : "never"}
                    </span>
                  </span>
                }
              />
              <SeriesChart
                x={L.depths.map((d) => d.label)}
                series={[{ id: "rank", label: "target rank", color: "var(--ns-derived)", values: L.depths.map((d) => d.target?.rank ?? null) }]}
                yScale="log"
                yCap={model?.vocab_size}
                formatY={(v) => fmtRank(v)}
                height={170}
                yLabel="rank (log)"
                markers={
                  L.first_depth_target_top1 !== null
                    ? [{ index: L.first_depth_target_top1, label: `top-1 at ${run.depth_labels[L.first_depth_target_top1]}` }]
                    : []
                }
                highlightIndex={sel?.depth ?? null}
                onIndexClick={(i) => {
                  select({ depth: i, position: pos });
                  update({ attnLayer: null });
                }}
                tooltipExtra={(i) => {
                  const d = L.depths[i];
                  return d.target ? (
                    <div className="font-mono text-[10px] text-ns-text-muted">
                      p {fmtProb(d.target.prob)} · logit {fmtNum(d.target.logit, 2)}
                    </div>
                  ) : null;
                }}
                ariaLabel={`Logit Lens rank of target ${L.target.text} at position ${pos} for each depth: ${L.depths
                  .map((d) => `${d.label} ${d.target?.rank ?? "?"}`)
                  .join(", ")}`}
              />
              <SeriesChart
                x={L.depths.map((d) => d.label)}
                series={[{ id: "p", label: "P(target)", color: "var(--ns-derived)", values: L.depths.map((d) => d.target?.prob ?? null) }]}
                kind="bar"
                yDomain={[0, 1]}
                yTicks={[0, 0.5, 1]}
                formatY={(v) => (v === 0 ? "0" : v.toFixed(1))}
                formatValue={(v) => fmtProb(v)}
                height={84}
                yLabel="P"
                highlightIndex={sel?.depth ?? null}
                onIndexClick={(i) => select({ depth: i, position: pos })}
                ariaLabel={`Logit Lens probability of the target at each depth`}
              />
            </section>
          ) : (
            <p className="text-[11px] text-ns-text-faint">No target token — set one in the left column to track its rank through depth.</p>
          )}

          <section className="flex flex-col gap-1.5">
            <SectionHeader
              label="Top tokens decoded by Logit Lens · ↑ deeper"
              status="derived"
              right={<Segmented label="Top-k" size="xs" value={k} onChange={setK} items={[{ value: "5", label: "k=5" }, { value: "10", label: "k=10" }]} />}
            />
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-left">
                <thead>
                  <tr className="font-mono text-[9.5px] uppercase tracking-[0.08em] text-ns-text-faint">
                    <th className="py-1 pr-2 font-normal">depth</th>
                    <th className="py-1 pr-2 font-normal">top-{k} (p)</th>
                    {L.target ? <th className="py-1 pr-2 text-right font-normal">rank</th> : null}
                    {L.target ? <th className="py-1 pr-2 text-right font-normal">P(tgt)</th> : null}
                    <th className="py-1 text-right font-normal">H</th>
                  </tr>
                </thead>
                <tbody>
                  {[...L.depths].reverse().map((d) => {
                    const on = sel?.depth === d.depth;
                    const top1 = d.target?.rank === 1;
                    return (
                      <tr
                        key={d.depth}
                        onClick={() => select({ depth: d.depth, position: pos })}
                        className={cn(
                          "cursor-pointer border-t border-ns-border align-top hover:bg-ns-panel-2",
                          on && "bg-ns-accent/8",
                          L.first_depth_target_top1 === d.depth && "shadow-[inset_2px_0_0_var(--ns-accent-2)]",
                        )}
                      >
                        <td className={cn("py-1 pl-1 pr-2 font-mono text-[10.5px]", on ? "text-ns-accent" : "text-ns-text-muted")}>{d.label}</td>
                        <td className="py-1 pr-2">
                          <div className={cn("flex gap-x-2 gap-y-0.5", k === "10" ? "flex-wrap" : "max-w-[300px] overflow-hidden")}>
                            {d.top.slice(0, Number(k)).map((e) => (
                              <span
                                key={e.id}
                                className={cn(
                                  "inline-flex items-center gap-1 font-mono text-[10.5px]",
                                  L.target && e.id === L.target.id ? "text-ns-accent-2" : "text-ns-text",
                                )}
                                title={`id ${e.id} · logit ${fmtNum(e.logit, 3)} · p ${fmtProb(e.prob)}`}
                              >
                                <Tok text={e.text} className="max-w-[64px] truncate" />
                                <ProbBar value={e.prob} className="w-6" />
                              </span>
                            ))}
                          </div>
                        </td>
                        {L.target ? (
                          <td className={cn("ns-tabular py-1 pr-2 text-right font-mono text-[10.5px]", top1 ? "text-ns-accent-2" : "text-ns-text-muted")}>
                            {fmtRank(d.target?.rank)}
                          </td>
                        ) : null}
                        {L.target ? <td className="ns-tabular py-1 pr-2 text-right font-mono text-[10.5px] text-ns-text-muted">{fmtProb(d.target?.prob)}</td> : null}
                        <td className="ns-tabular py-1 text-right font-mono text-[10.5px] text-ns-text-faint">{fmtNum(d.entropy, 2)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
          <Caveat>
            Method: {L.method} Tokens are &ldquo;decoded by Logit Lens&rdquo;: a fixed projection of an intermediate state, not a readout of intent.
          </Caveat>
        </div>
      ) : null}
    </div>
  );
}
