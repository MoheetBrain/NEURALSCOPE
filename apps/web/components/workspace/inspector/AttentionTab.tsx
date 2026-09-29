"use client";

import { useMemo, useState } from "react";
import { ChevronLeft, ChevronRight, Scissors } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { api } from "@/lib/api/client";
import type { AttentionResponse, HeadDetailResponse, RunDetail } from "@/lib/api/types";
import { cn } from "@/lib/cn";
import { normalize } from "@/lib/colormap";
import { fmtNum, truncate, visibleToken } from "@/lib/format";
import { layerForDepth } from "@/lib/matrix";
import { useWorkspaceActions } from "../actions";
import { Colorbar } from "../charts/Colorbar";
import { HeatmapCanvas, colorsFromGrid, type HeatCell } from "../charts/HeatmapCanvas";
import { SeriesChart } from "../charts/SeriesChart";
import { useElementSize } from "../charts/useElementSize";
import { ProbBar } from "../charts/VectorStrip";
import { targetId, useAttentionLayer, useColumns, useSelection } from "../hooks";
import { useResource } from "../useResource";
import { useWorkspace } from "../WorkspaceProvider";
import { Caveat, ComingSoon, ErrorBox, KV, Loading, SectionHeader, SelectInput, StatusBadge, Switch, Tok } from "../ui";

const MINI = 70;

function gammaT(v: number | null, sqrt: boolean): number | null {
  if (v === null || !Number.isFinite(v)) return null;
  const t = normalize(v, { kind: "linear", vmin: 0, vmax: 1 });
  return t === null ? null : sqrt ? Math.sqrt(t) : t;
}

export function AttentionTab({ run }: { run: RunDetail }) {
  const { model, state, update, select } = useWorkspace();
  const nLayers = model?.n_layers ?? run.depth_labels.length - 1;
  const sel = useSelection(run);
  const layer = useAttentionLayer(run, nLayers);
  const q = sel?.position ?? run.target?.position ?? run.prompt_len - 1;
  const [sqrt, setSqrt] = useState(true);
  const res = useResource<AttentionResponse>(`attn:${run.id}:${layer}:${q}`, (signal) => api.attention(run.id, layer, q, signal), {
    debounceMs: 100,
  });
  const A = res.data;
  const head = state.attnHead !== null && A && state.attnHead < A.n_heads ? state.attnHead : null;
  const linked = state.attnLayer === null;
  const caps = model?.capabilities ?? {};

  const setLayer = (l: number) => update({ attnLayer: Math.min(nLayers - 1, Math.max(0, l)) });

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1">
          <Button size="xs" variant="ghost" aria-label="Previous layer" onClick={() => setLayer(layer - 1)} disabled={layer <= 0}>
            <ChevronLeft aria-hidden="true" />
          </Button>
          <SelectInput aria-label="Attention layer" className="h-6 w-[76px] text-[11px]" value={layer} onChange={(e) => setLayer(Number(e.target.value))}>
            {Array.from({ length: nLayers }, (_, l) => (
              <option key={l} value={l}>
                L{l}
              </option>
            ))}
          </SelectInput>
          <Button size="xs" variant="ghost" aria-label="Next layer" onClick={() => setLayer(layer + 1)} disabled={layer >= nLayers - 1}>
            <ChevronRight aria-hidden="true" />
          </Button>
        </div>
        <span className="font-mono text-[11px] text-ns-text-muted">
          query t{q} <Tok text={run.tokens[q]?.text ?? ""} className="rounded-[3px] border border-ns-border-strong bg-ns-bg px-1 text-ns-text" />
        </span>
        {linked ? (
          <span className="font-mono text-[10px] text-ns-text-faint">linked to matrix depth</span>
        ) : (
          <button
            type="button"
            className="font-mono text-[10px] uppercase tracking-[0.08em] text-ns-accent hover:underline"
            onClick={() => update({ attnLayer: null })}
            title={sel ? `Follow the matrix selection (layer ${layerForDepth(sel.depth, nLayers)})` : undefined}
          >
            follow matrix
          </button>
        )}
        <Switch checked={sqrt} onChange={setSqrt} label="√ colour" className="ml-auto" title="Square-root colour scale makes small weights visible (values are unchanged)" />
      </div>

      {res.error && !A ? <ErrorBox error={res.error} onRetry={res.reload} /> : null}
      {!A && !res.error ? (
        <Loading
          label={`Reading attention patterns of block ${layer}`}
          hint={run.trace_level === "minimal" ? "This run was stored at trace level minimal — attention is re-captured on first request (deterministic)." : undefined}
        />
      ) : null}

      {A ? (
        <div className={cn("flex flex-col gap-4", res.stale && "ns-stale")}>
          <section className="flex flex-col gap-2">
            <SectionHeader label={`All ${A.n_heads} heads · block ${A.layer}`} status={A.status.patterns} note="patterns" />
            <div className="grid grid-cols-6 gap-1.5">
              {A.heads.map((h) => (
                <MiniHead
                  key={h.head}
                  A={A}
                  head={h.head}
                  active={head === h.head}
                  sqrt={sqrt}
                  q={q}
                  onClick={() => update({ attnHead: h.head, attnLayer: layer })}
                />
              ))}
            </div>
            <div className="flex flex-wrap items-end justify-between gap-2">
              <Colorbar
                cmap="viridis"
                width={150}
                ticks={[0, 0.1, 0.5, 1].map((v) => ({ t: sqrt ? Math.sqrt(v) : v, label: String(v) }))}
                label="attention weight"
              />
              <span className="pb-3 font-mono text-[10px] text-ns-text-faint">rows = queries, cols = keys · ─ marks query t{q} · H = entropy at query (DERIVED)</span>
            </div>
          </section>

          {head !== null ? (
            <HeadView run={run} A={A} head={head} q={q} sqrt={sqrt} onSelectQuery={(p) => select({ depth: sel?.depth ?? run.depth_labels.length - 1, position: p })} />
          ) : (
            <p className="rounded-[3px] border border-dashed border-ns-border-strong px-3 py-2 text-[11px] text-ns-text-muted">Click a head to open its full pattern, query-row weights, vocabulary projection and DLA.</p>
          )}

          <Caveat>{A.caveat} Attention ≠ explanation.</Caveat>
          <div className="flex flex-col gap-1.5">
            {!caps.qkv ? <ComingSoon label="Q / K / V vectors" note="not captured in V1" /> : null}
            {!caps.kv_cache_inspection ? <ComingSoon label="KV-cache inspection" /> : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function MiniHead({
  A,
  head,
  active,
  sqrt,
  q,
  onClick,
}: {
  A: AttentionResponse;
  head: number;
  active: boolean;
  sqrt: boolean;
  q: number;
  onClick: () => void;
}) {
  const T = A.n_tokens;
  const colors = useMemo(() => colorsFromGrid(A.patterns[head], (v) => gammaT(v, sqrt), "viridis"), [A, head, sqrt]);
  const h = A.heads[head];
  const cell = MINI / T;
  const markRows = useMemo(() => [q], [q]);
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      aria-label={`Head ${head}: entropy ${fmtNum(h.entropy, 2)} at the query, top source t${h.top_sources[0]?.position}`}
      className={cn(
        "flex flex-col items-center gap-0.5 rounded-[3px] border p-1 transition-colors",
        active ? "border-ns-accent bg-ns-accent/8" : "border-ns-border hover:border-ns-border-strong",
      )}
    >
      <HeatmapCanvas rows={T} cols={T} colors={colors} cellW={cell} cellH={cell} markRows={markRows} markColor="rgba(215,201,166,0.7)" ariaLabel={`Attention pattern of head ${head}`} />
      <span className="flex w-full items-center justify-between font-mono text-[9.5px]">
        <span className={active ? "text-ns-accent" : "text-ns-text"}>H{head}</span>
        <span className="text-ns-text-faint">H {fmtNum(h.entropy, 1)}</span>
      </span>
    </button>
  );
}

function HeadView({
  run,
  A,
  head,
  q,
  sqrt,
  onSelectQuery,
}: {
  run: RunDetail;
  A: AttentionResponse;
  head: number;
  q: number;
  sqrt: boolean;
  onSelectQuery: (p: number) => void;
}) {
  const { state } = useWorkspace();
  const { prefillHeadAblation } = useWorkspaceActions();
  const columns = useColumns(run);
  const shown = state.hideTemplate ? columns : run.tokens.map((t) => t.index);
  const tid = targetId(run);
  const detail = useResource<HeadDetailResponse>(`head:${run.id}:${A.layer}:${head}:${q}:${tid}`, (signal) =>
    api.head(run.id, A.layer, head, q, tid, signal),
  );
  const [wrapRef, size] = useElementSize<HTMLDivElement>();
  const [hover, setHover] = useState<HeatCell | null>(null);
  const labelW = 58;
  const n = shown.length;
  const cell = Math.max(3, Math.min(18, Math.floor((size.width - labelW - 4) / Math.max(1, n))));
  const grid = useMemo(() => shown.map((qi) => shown.map((ki) => A.patterns[head][qi]?.[ki] ?? null)), [A, head, shown]);
  const colors = useMemo(() => colorsFromGrid(grid, (v) => gammaT(v, sqrt), "viridis"), [grid, sqrt]);
  const qRow = shown.indexOf(q);
  const markRows = useMemo(() => (qRow >= 0 ? [qRow] : []), [qRow]);
  const every = cell >= 9 ? 1 : Math.ceil(9 / cell);
  const h = A.heads[head];

  const row = A.query_rows[head] ?? [];
  const sources = row
    .map((w, p) => ({ p, w: w ?? 0 }))
    .filter((s) => s.p <= q)
    .sort((a, b) => b.w - a.w)
    .slice(0, 10);
  const rest = Math.max(0, 1 - sources.reduce((a, s) => a + s.w, 0));

  const hq = hover ? shown[hover.row] : null;
  const hk = hover ? shown[hover.col] : null;
  const d = detail.data;

  return (
    <section className="flex flex-col gap-3 rounded-[var(--ns-radius)] border border-ns-border-strong p-2.5">
      <SectionHeader
        label={`Head L${A.layer}.H${head}`}
        status={A.status.patterns}
        note="pattern"
        right={
          <Button size="xs" caps variant="secondary" onClick={() => prefillHeadAblation(A.layer, head)}>
            <Scissors aria-hidden="true" /> Ablate this head
          </Button>
        }
      />
      <div ref={wrapRef} className="relative">
        {size.width > 0 ? (
          <div className="flex">
            <div className="shrink-0 pr-1" style={{ width: labelW }} aria-hidden="true">
              {shown.map((p, i) => (
                <div
                  key={p}
                  className={cn("truncate whitespace-pre text-right font-mono text-[9px]", p === q ? "text-ns-accent-2" : "text-ns-text-faint")}
                  style={{ height: cell, lineHeight: `${cell}px` }}
                >
                  {i % every === 0 || p === q ? truncate(visibleToken(run.tokens[p].text), 8) : ""}
                </div>
              ))}
            </div>
            <div>
              <HeatmapCanvas
                rows={n}
                cols={n}
                colors={colors}
                cellW={cell}
                cellH={cell}
                gap={cell >= 10 ? 1 : 0}
                markRows={markRows}
                hovered={hover}
                onHoverCell={(c) => setHover(c)}
                onClickCell={(c) => onSelectQuery(shown[c.row])}
                ariaLabel={`Attention pattern of head L${A.layer}.H${head}: ${n}×${n} (query rows × key columns). Query t${q} attends most to ${sources
                  .slice(0, 3)
                  .map((s) => `t${s.p} (${fmtNum(s.w, 3)})`)
                  .join(", ")}.`}
              />
              <div className="relative" style={{ height: 50, width: n * cell }} aria-hidden="true">
                {shown.map((p, i) =>
                  i % every === 0 ? (
                    <div
                      key={p}
                      className="absolute top-1 overflow-hidden whitespace-pre font-mono text-[9px] leading-none text-ns-text-faint"
                      style={{ left: i * cell + cell / 2 - 5, width: 10, height: 46, writingMode: "vertical-rl", transform: "rotate(180deg)", textAlign: "right" }}
                    >
                      {truncate(visibleToken(run.tokens[p].text), 7)}
                    </div>
                  ) : null,
                )}
              </div>
            </div>
          </div>
        ) : (
          <div style={{ height: 200 }} />
        )}
        {hover && hq !== null && hk !== null ? (
          <div className="pointer-events-none absolute right-0 top-0 z-20 rounded-[3px] border border-ns-border-strong bg-ns-bg/95 px-2 py-1 font-mono text-[10.5px] text-ns-text">
            q t{hq} <Tok text={run.tokens[hq].text} /> ← k t{hk} <Tok text={run.tokens[hk].text} /> ·{" "}
            <span className="text-ns-accent">{fmtNum(A.patterns[head][hq]?.[hk] ?? null, 4)}</span>
          </div>
        ) : null}
        {state.hideTemplate && n < run.tokens.length ? (
          <p className="text-[10px] text-ns-text-faint">
            {run.tokens.length - n} system-template positions hidden (weights not renormalised). Click a row to move the query.
          </p>
        ) : (
          <p className="text-[10px] text-ns-text-faint">Click a row to move the query position.</p>
        )}
      </div>

      <div>
        <SectionHeader label={`Where query t${q} attends (top sources)`} status="observed" note="weights" />
        <ol className="mt-1 flex flex-col gap-0.5">
          {sources.map((s) => (
            <li key={s.p} className="grid grid-cols-[34px_minmax(0,86px)_1fr_48px] items-center gap-2 font-mono text-[10.5px]">
              <span className="text-right text-ns-text-faint">t{s.p}</span>
              <Tok text={run.tokens[s.p].text} className="truncate text-ns-text" />
              <ProbBar value={s.w} color="var(--ns-observed)" className="w-full" />
              <span className="ns-tabular text-right text-ns-text-muted">{fmtNum(s.w, 3)}</span>
            </li>
          ))}
          <li className="grid grid-cols-[34px_minmax(0,86px)_1fr_48px] items-center gap-2 font-mono text-[10px] text-ns-text-faint">
            <span />
            <span>other {Math.max(0, q + 1 - sources.length)}</span>
            <span />
            <span className="text-right">{fmtNum(rest, 3)}</span>
          </li>
        </ol>
      </div>

      <dl className="grid grid-cols-2 gap-x-4">
        <KV k={<>entropy at query <StatusBadge status="derived" /></>} v={`${fmtNum(h.entropy, 3)} nats`} />
        <KV k="attn → t0" v={fmtNum(h.attn_to_first, 3)} />
        <KV k="attn → self" v={fmtNum(h.attn_to_self, 3)} />
        <KV k={<>‖head out‖ at query <StatusBadge status="derived" /></>} v={fmtNum(h.output_norm_at_query, 3)} />
        <KV k="KV group" v={`${h.kv_group} (shares K/V)`} />
        <KV k="query position" v={`t${A.query_position}`} />
      </dl>

      {detail.error && !d ? <ErrorBox error={detail.error} onRetry={detail.reload} /> : null}
      {!d && !detail.error ? <Loading compact label="Head vocabulary projection + DLA" /> : null}
      {d ? (
        <div className={cn("flex flex-col gap-3", detail.stale && "ns-stale")}>
          {d.dla_to_target ? (
            <div className="flex items-center justify-between rounded-[3px] border border-ns-border bg-ns-bg/60 px-2.5 py-1.5">
              <span className="flex items-center gap-2 text-[11px] text-ns-text-muted">
                Direct logit attribution to <Tok text={d.dla_to_target.token} className="text-ns-accent-2" /> <StatusBadge status={d.status.dla} />
              </span>
              <span className="font-mono text-[13px] text-ns-text">
                {(d.dla_to_target.value ?? 0) >= 0 ? "+" : ""}
                {fmtNum(d.dla_to_target.value, 4)}
              </span>
            </div>
          ) : null}
          <div className="grid grid-cols-2 gap-3">
            <TokenValueList title="Promoted (W_U · out)" items={d.promoted} status={d.status.vocab_projection} />
            <TokenValueList title="Suppressed" items={d.suppressed} status={d.status.vocab_projection} negative />
          </div>
          <div>
            <SectionHeader label="Head output norm by position" status={d.status.output_norm} />
            <SeriesChart
              x={d.output_norm_by_position.map((_, i) => `t${i}`)}
              series={[{ id: "norm", label: "‖out‖", color: "var(--ns-derived)", values: d.output_norm_by_position }]}
              kind="bar"
              height={80}
              highlightIndex={q}
              formatY={(v) => fmtNum(v, 1)}
              onIndexClick={(i) => onSelectQuery(i)}
              tooltipExtra={(i) => <div className="font-mono text-[10px] text-ns-text-muted">{visibleToken(run.tokens[i]?.text ?? "")}</div>}
              ariaLabel={`Output norm of head L${A.layer}.H${head} at each position`}
            />
          </div>
          <Caveat>Vocabulary projection and DLA are direct effects only; use ablation (Intervene tab) to measure total causal effect.</Caveat>
        </div>
      ) : null}
    </section>
  );
}

function TokenValueList({
  title,
  items,
  status,
  negative,
}: {
  title: string;
  items: { id: number; text: string; value: number | null }[];
  status: string;
  negative?: boolean;
}) {
  const max = Math.max(1e-9, ...items.map((i) => Math.abs(i.value ?? 0)));
  return (
    <div>
      <SectionHeader label={title} status={status} />
      <ol className="mt-1 flex flex-col gap-0.5">
        {items.slice(0, 8).map((it) => (
          <li key={it.id} className="grid grid-cols-[minmax(0,1fr)_40px_40px] items-center gap-1.5 font-mono text-[10.5px]">
            <Tok text={it.text} className="truncate text-ns-text" />
            <ProbBar value={Math.abs(it.value ?? 0)} max={max} color={negative ? "var(--ns-text-faint)" : "var(--ns-derived)"} className="w-full" />
            <span className="ns-tabular text-right text-ns-text-muted">{fmtNum(it.value, 2)}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

