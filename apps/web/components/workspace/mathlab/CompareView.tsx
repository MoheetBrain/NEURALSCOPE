"use client";

import { api } from "@/lib/api/client";
import type { CompareResponse, CrossRunComparison } from "@/lib/api/types";
import { cn } from "@/lib/cn";
import { fmtNum, fmtRank, shortId } from "@/lib/format";
import { LayerHeadHeatmap } from "../charts/LayerHeadHeatmap";
import { SeriesChart, type ChartSeries } from "../charts/SeriesChart";
import { useResource } from "../useResource";
import { useWorkspace } from "../WorkspaceProvider";
import { Caveat, ErrorBox, Loading, OutputText, SectionHeader, StatusBadge, Tok } from "../ui";

export const COLOR_A = "#8fb3ff";
export const COLOR_B = "#d7c9a6";

function pair(c: CrossRunComparison, pick: (s: CrossRunComparison["series_a"]) => readonly (number | null)[] | undefined): ChartSeries[] {
  return [
    { id: "a", label: "A (correct)", color: COLOR_A, values: pick(c.series_a) ?? [] },
    { id: "b", label: "B (incorrect)", color: COLOR_B, values: pick(c.series_b) ?? [] },
  ];
}

export function CompareView({ runA, runB, onOpen }: { runA: string; runB: string; onOpen: (runId: string) => void }) {
  const { model } = useWorkspace();
  const res = useResource<CompareResponse>(`compare:${runA}:${runB}`, (signal) => api.compare({ run_a: runA, run_b: runB }, signal));
  if (res.error) return <ErrorBox error={res.error} onRetry={res.reload} />;
  if (!res.data) return <Loading label="Comparing residual trajectories (lazy attention capture for both runs)" hint="≈15 s the first time a pair is compared." />;
  const c = res.data;
  if (c.mode !== "cross_run") return <p className="text-[11px] text-ns-text-muted">B is an intervention child of A — see the Diff tab in the workspace.</p>;
  const x = c.depth_labels;

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-2">
        <RunCard label="A" color={COLOR_A} c={c} which="a" onOpen={onOpen} />
        <RunCard label="B" color={COLOR_B} c={c} which="b" onOpen={onOpen} />
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <StatusBadge status={c.status} size="sm" note="correlational" />
        <span className="font-mono text-[10.5px] text-ns-text-muted">final-layer KL(A‖B) at the compared positions {fmtNum(c.final_kl_a_b, 3)}</span>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <section>
          <SectionHeader label="Target rank through depth (Logit Lens, log)" status="derived" />
          <SeriesChart
            x={x}
            series={pair(c, (s) => s.target_rank)}
            yScale="log"
            yCap={model?.vocab_size}
            formatY={(v) => fmtRank(v)}
            height={170}
            yLabel="rank"
            ariaLabel="Logit Lens rank of each run's oracle target token through depth, A versus B"
          />
        </section>
        <section>
          <SectionHeader label="Residual norm ‖h‖ by depth" status="derived" />
          <SeriesChart x={x} series={pair(c, (s) => s.resid_norm)} height={170} formatY={(v) => fmtNum(v, 0)} ariaLabel="Residual stream norm by depth, A versus B" />
        </section>
        <section>
          <SectionHeader label="1 − cos between successive residual states" status="derived" />
          <SeriesChart x={x} series={pair(c, (s) => s.cos_change)} height={150} formatY={(v) => fmtNum(v, 2)} ariaLabel="Rotation of the residual state by each block, A versus B" />
        </section>
        <section>
          <SectionHeader label="Logit Lens entropy (nats)" status="derived" />
          <SeriesChart x={x} series={pair(c, (s) => s.lens_entropy)} height={150} formatY={(v) => fmtNum(v, 1)} ariaLabel="Logit Lens entropy by depth, A versus B" />
        </section>
        <section>
          <SectionHeader label="Cross-run cosine cos(h_A, h_B) by depth" status="derived" />
          <SeriesChart
            x={x}
            series={[{ id: "x", label: "cos(A, B)", color: "var(--ns-derived)", values: c.cross_cosine }]}
            height={150}
            yDomain={[Math.min(0, ...c.cross_cosine.map((v) => v ?? 0)), 1]}
            formatY={(v) => fmtNum(v, 2)}
            ariaLabel="Cosine similarity between the two runs' residual states at the compared positions, by depth"
          />
        </section>
        <section>
          <SectionHeader label="Attention entropy difference A − B at the query [layer × head]" status="derived" />
          {c.attn_entropy_diff ? (
            <LayerHeadHeatmap
              values={c.attn_entropy_diff}
              colorbarLabel="attention entropy A − B (nats)"
              negLabel="A sharper"
              posLabel="B sharper"
              maxCell={20}
              ariaLabel="Attention entropy difference between run A and run B per layer and head"
              tooltip={(l, h) => (
                <div className="flex flex-col">
                  <span className="text-ns-accent">
                    L{l}.H{h}
                  </span>
                  <span>A {fmtNum(c.attn_entropy_a?.[l]?.[h] ?? null, 3)}</span>
                  <span>B {fmtNum(c.attn_entropy_b?.[l]?.[h] ?? null, 3)}</span>
                  <span>A − B {fmtNum(c.attn_entropy_diff?.[l]?.[h] ?? null, 3)}</span>
                </div>
              )}
            />
          ) : (
            <p className="text-[11px] text-ns-text-faint">{c.attn_error ?? "not available"}</p>
          )}
        </section>
      </div>

      <section>
        <SectionHeader label="Top-1 token decoded by Logit Lens per depth" status="derived" />
        <div className="mt-1 overflow-x-auto">
          <table className="border-collapse">
            <tbody>
              <tr>
                <th className="pr-2 text-left font-mono text-[9.5px] font-normal text-ns-text-faint">depth</th>
                {x.map((l) => (
                  <td key={l} className="px-1 text-center font-mono text-[9px] text-ns-text-faint">
                    {l}
                  </td>
                ))}
              </tr>
              {(["a", "b"] as const).map((w) => {
                const s = w === "a" ? c.series_a : c.series_b;
                return (
                  <tr key={w}>
                    <th className="pr-2 text-left font-mono text-[10px] font-normal" style={{ color: w === "a" ? COLOR_A : COLOR_B }}>
                      {w.toUpperCase()}
                    </th>
                    {s.lens_top1.map((t, i) => {
                      const p = s.lens_top1_prob[i] ?? 0;
                      return (
                        <td key={i} className="px-0.5 py-0.5">
                          <span
                            className={cn("block max-w-[44px] truncate rounded-[2px] px-0.5 text-center font-mono text-[10px]", p > 0.5 ? "text-ns-text" : "text-ns-text-muted")}
                            style={{ background: `color-mix(in srgb, var(--ns-derived) ${Math.round(p * 35)}%, transparent)` }}
                            title={`${x[i]}: “${t}” p=${fmtNum(p, 3)}`}
                          >
                            <Tok text={t} />
                          </span>
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>
      <Caveat>{c.caveat}</Caveat>
    </div>
  );
}

function RunCard({ label, color, c, which, onOpen }: { label: string; color: string; c: CrossRunComparison; which: "a" | "b"; onOpen: (id: string) => void }) {
  const run = which === "a" ? c.run_a : c.run_b;
  const tgt = which === "a" ? c.target_a : c.target_b;
  const pos = which === "a" ? c.position_a : c.position_b;
  const tok = which === "a" ? c.token_a : c.token_b;
  return (
    <div className="rounded-[3px] border border-ns-border bg-ns-bg px-2.5 py-2" style={{ boxShadow: `inset 2px 0 0 ${color}` }}>
      <div className="flex items-center gap-2">
        <span className="font-mono text-[12px]" style={{ color }}>
          {label}
        </span>
        <span className="truncate text-[11.5px] text-ns-text">{run.prompt}</span>
        <button type="button" className="ml-auto shrink-0 font-mono text-[10px] uppercase tracking-[0.08em] text-ns-accent hover:underline" onClick={() => onOpen(run.id)}>
          open {shortId(run.id)}
        </button>
      </div>
      <div className="mt-1 font-mono text-[10.5px] text-ns-text-muted">
        output <OutputText text={run.generated_text} className="text-ns-text" /> · {run.correctness}
      </div>
      <div className="font-mono text-[10.5px] text-ns-text-muted">
        compared at t{pos} <Tok text={tok} /> · target <Tok text={tgt.token ?? "—"} className="text-ns-text" />
        {tgt.contrast ? (
          <>
            {" "}vs model <Tok text={tgt.contrast} />
          </>
        ) : null}
      </div>
    </div>
  );
}
