"use client";

import { api } from "@/lib/api/client";
import type { CellResponse, RunDetail } from "@/lib/api/types";
import { cn } from "@/lib/cn";
import { fmtNum, fmtProb, fmtRank } from "@/lib/format";
import { Histogram, ProbBar, VectorStrip } from "../charts/VectorStrip";
import { targetId, useSelection } from "../hooks";
import { useResource } from "../useResource";
import { useWorkspace } from "../WorkspaceProvider";
import { Caveat, ComingSoon, ErrorBox, KV, Loading, SectionHeader, StatusBadge, Tok } from "../ui";

export function TraceTab({ run }: { run: RunDetail }) {
  const { model } = useWorkspace();
  const sel = useSelection(run);
  const tid = targetId(run);
  const key = sel ? `cell:${run.id}:${sel.depth}:${sel.position}:${tid}` : null;
  const res = useResource<CellResponse>(
    key,
    (signal) => api.cell(run.id, sel!.depth, sel!.position, tid, signal),
    { debounceMs: 90 },
  );
  const c = res.data;
  const caps = model?.capabilities ?? {};

  if (!sel) return null;
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <span className="font-mono text-[15px] text-ns-accent">{run.depth_labels[sel.depth]}</span>
        <span className="font-mono text-[12px] text-ns-text-muted">
          depth {sel.depth} · t{sel.position}
        </span>
        <Tok text={run.tokens[sel.position]?.text ?? ""} className="rounded-[3px] border border-ns-border-strong bg-ns-bg px-1 text-[12px] text-ns-text" />
        <span className="font-mono text-[10.5px] text-ns-text-faint">{run.tokens[sel.position]?.role}</span>
      </div>
      <p className="-mt-3 text-[11px] text-ns-text-faint">{c?.layer_note ?? (sel.depth === 0 ? "token embedding (input to block 0)" : `residual stream after block ${sel.depth - 1} (pre final-norm)`)}</p>

      {res.error && !c ? <ErrorBox error={res.error} onRetry={res.reload} /> : null}
      {!c && !res.error ? <Loading label="Reading residual vector" /> : null}

      {c ? (
        <div className={cn("flex flex-col gap-4", res.stale && "ns-stale")}>
          <section className="flex flex-col gap-2">
            <SectionHeader label={`Residual vector · ${c.vector.d_model} dims`} status={c.vector.status} />
            <VectorStrip
              values={c.vector.values}
              highlight={c.vector.top_dims.map((d) => d.dim)}
              ariaLabel={`Residual vector at ${c.depth_label}, position ${c.position}: ${c.vector.d_model} values, norm ${fmtNum(c.vector.norm, 3)}, max |v| ${fmtNum(c.vector.max_abs, 3)}`}
            />
            <div className="grid grid-cols-[1fr_1fr] gap-x-4">
              <dl>
                <KV k="‖h‖₂" v={fmtNum(c.vector.norm, 4)} />
                <KV k="mean" v={fmtNum(c.vector.mean, 4)} />
                <KV k="std" v={fmtNum(c.vector.std, 4)} />
                <KV k="max |v|" v={fmtNum(c.vector.max_abs, 4)} />
              </dl>
              <div>
                <div className="ns-label mb-1 text-[9px]">Histogram (√count)</div>
                <Histogram counts={c.vector.histogram.counts} min={c.vector.histogram.min} max={c.vector.histogram.max} />
              </div>
            </div>
            <div>
              <div className="ns-label mb-1 text-[9px]">Top |v| dimensions</div>
              <div className="flex flex-wrap gap-1">
                {c.vector.top_dims.map((d) => (
                  <span key={d.dim} className="rounded-[3px] border border-ns-border bg-ns-bg px-1.5 py-0.5 font-mono text-[10.5px] text-ns-text-muted">
                    <span className="text-ns-accent-2">{d.dim}</span> {fmtNum(d.value, 3)}
                  </span>
                ))}
              </div>
            </div>
          </section>

          <section className="flex flex-col gap-1">
            <SectionHeader label="Geometry" status={c.geometry.status} />
            <dl className="grid grid-cols-2 gap-x-4">
              <KV k="cos(h, h_prev depth)" v={fmtNum(c.geometry.cos_prev_depth, 4)} />
              <KV k="cos(h, h_next depth)" v={fmtNum(c.geometry.cos_next_depth, 4)} />
              <KV k="cos(h, h_final)" v={fmtNum(c.geometry.cos_final_depth, 4)} />
              <KV k="‖h‖ / ‖h_prev‖" v={fmtNum(c.geometry.norm_ratio_prev, 4)} />
            </dl>
          </section>

          <section className="flex flex-col gap-1.5">
            <SectionHeader label="Decoded by Logit Lens at this cell" status={c.lens.status} right={<span className="font-mono text-[10px] text-ns-text-faint">entropy {fmtNum(c.lens.entropy, 3)} nats</span>} />
            <ol className="flex flex-col gap-0.5">
              {c.lens.top.map((e, i) => (
                <li key={e.id} className="grid grid-cols-[16px_minmax(0,90px)_1fr_56px] items-center gap-2 font-mono text-[11px]">
                  <span className="text-right text-ns-text-faint">{i + 1}</span>
                  <Tok text={e.text} className="truncate text-ns-text" title={`id ${e.id} · logit ${fmtNum(e.logit, 3)}`} />
                  <ProbBar value={e.prob} className="w-full" />
                  <span className="ns-tabular text-right text-ns-text-muted">{fmtProb(e.prob)}</span>
                </li>
              ))}
            </ol>
            <dl className="mt-1 grid grid-cols-1 gap-x-4">
              {c.target ? (
                <KV
                  k={
                    <>
                      target <Tok text={c.target.text} className="text-ns-accent-2" />
                    </>
                  }
                  v={`rank ${fmtRank(c.target.lens_rank)} · p ${fmtProb(c.target.lens_prob)} · logit ${fmtNum(c.target.lens_logit, 3)}`}
                />
              ) : null}
              {c.next_token ? (
                <KV
                  k={
                    <>
                      actual next <Tok text={c.next_token.text} className="text-ns-text" />
                    </>
                  }
                  v={`rank ${fmtRank(c.next_token.lens_rank)} · p ${fmtProb(c.next_token.lens_prob)}`}
                />
              ) : null}
            </dl>
            <Caveat>Logit Lens = final norm + unembedding applied to an intermediate residual state; exact only at the last depth, biased earlier.</Caveat>
          </section>

          <section className="flex flex-col gap-1.5">
            <SectionHeader label="Feature-level views" />
            {!caps.sae_features ? <ComingSoon label="SAE features" note="sparse-autoencoder latents for this state" /> : null}
            {!caps.transcoders ? <ComingSoon label="Transcoders" note="MLP replacement features" /> : null}
            {!caps.gradients ? <ComingSoon label="Gradients / attribution patching" /> : null}
            {!caps.mlp_preactivations ? <ComingSoon label="MLP pre-activations" note="not captured" /> : null}
          </section>
          <p className="text-[10.5px] text-ns-text-faint">
            <StatusBadge status="observed" /> raw tensor values · <StatusBadge status="derived" /> transformations of them.
          </p>
        </div>
      ) : null}
    </div>
  );
}
