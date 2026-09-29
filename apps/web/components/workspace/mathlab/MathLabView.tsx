"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { ExternalLink, FlaskConical, Play } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Panel } from "@/components/ui/Panel";
import { StatusChip } from "@/components/ui/StatusChip";
import { api, errorMessage } from "@/lib/api/client";
import type { Benchmark, BenchmarkEvalResult, BenchmarkResultRow, MathVerifyResponse } from "@/lib/api/types";
import { cn } from "@/lib/cn";
import { fmtAgo, fmtDelta, fmtNum, fmtPct, fmtSeconds, shortId, truncate } from "@/lib/format";
import { MathStatusChip } from "../PromptColumn";
import { useResource, useNow } from "../useResource";
import { useWorkspace } from "../WorkspaceProvider";
import { Caveat, EmptyState, ErrorBox, Field, Loading, NumberInput, ProgressBar, SectionHeader, Stat, StatusBadge, TextInput } from "../ui";
import { CompareView } from "./CompareView";

export function MathLabView() {
  const { state, update } = useWorkspace();
  const [benchVersion, setBenchVersion] = useState(0);
  const benches = useResource<Benchmark[]>(`benchmarks:${benchVersion}`, (signal) => api.listBenchmarks(signal), { cache: false });
  const list = benches.data ?? [];
  const selected = list.find((b) => b.id === state.benchmarkId) ?? list[0];

  return (
    <div className="grid min-h-0 flex-1 grid-cols-[300px_minmax(0,1fr)] gap-2 p-2">
      <aside className="ns-scroll flex min-h-0 flex-col gap-2 pr-1">
        <BenchmarkLauncher
          onDone={(b) => {
            setBenchVersion((v) => v + 1);
            update({ benchmarkId: b.id, compareA: null, compareB: null });
          }}
        />
        <Panel label="Benchmarks" status="observed" statusNote="stored" className="shrink-0" bodyClassName="p-1.5">
          {benches.error ? <ErrorBox error={benches.error} onRetry={benches.reload} /> : null}
          {!benches.data && !benches.error ? <Loading compact label="Loading benchmarks" /> : null}
          {benches.data?.length === 0 ? <p className="p-2 text-[11px] text-ns-text-faint">No benchmarks yet — run one above.</p> : null}
          <BenchmarkList list={list} selectedId={selected?.id ?? null} onSelect={(id) => update({ benchmarkId: id, compareA: null, compareB: null })} />
        </Panel>
        <VerifyTool />
      </aside>
      <main className="ns-scroll flex min-h-0 flex-col gap-3 pr-1">
        {selected ? (
          <BenchmarkDetail bench={selected} key={selected.id} onEvalDone={() => setBenchVersion((v) => v + 1)} />
        ) : benches.data ? (
          <div className="ns-dot-bg grid flex-1 place-items-center rounded-[var(--ns-radius)] border border-ns-border">
            <EmptyState title="Math Lab">
              Run a benchmark of generated arithmetic problems. Every answer is checked by a SymPy oracle; then compare a correct and an incorrect run layer by layer, and test candidate heads causally across the whole benchmark.
            </EmptyState>
          </div>
        ) : null}
      </main>
    </div>
  );
}

function BenchmarkLauncher({ onDone }: { onDone: (b: Benchmark) => void }) {
  const { jobs, activeJob, startJob, bumpHistory, notify } = useWorkspace();
  const [n, setN] = useState(25);
  const [seed, setSeed] = useState(0);
  const job = jobs.benchmark;
  const running = job && (job.status === "queued" || job.status === "running") ? job : null;
  const now = useNow(1000, Boolean(running));
  return (
    <Panel label="Run benchmark" status="observed" statusNote="SymPy oracle" className="shrink-0" bodyClassName="flex flex-col gap-2 p-2.5">
      <div className="grid grid-cols-2 gap-2">
        <Field label="problems">
          <NumberInput value={n} min={1} max={100} onValue={(v) => setN(Math.min(100, Math.max(1, Math.round(v))))} aria-label="Number of problems" />
        </Field>
        <Field label="seed">
          <NumberInput value={seed} min={0} onValue={(v) => setSeed(Math.max(0, Math.round(v)))} aria-label="Seed" />
        </Field>
      </div>
      <Button
        variant="primary"
        caps
        disabled={Boolean(activeJob)}
        title={activeJob ? `Busy: ${activeJob.label}` : undefined}
        onClick={() =>
          void startJob<Benchmark>("benchmark", `Benchmark ×${n}`, `bench-seed-${seed}`, () => api.runBenchmark({ n, seed }), (b) => {
            bumpHistory();
            notify("ok", `Benchmark done: ${b.summary?.correct}/${b.summary?.n} correct (${fmtPct(b.summary?.accuracy)}).`);
            onDone(b);
          })
        }
      >
        <Play aria-hidden="true" /> Run {n} problems
      </Button>
      <p className="text-[10.5px] leading-snug text-ns-text-faint">≈4 s per problem on CPU (greedy, ≤24 new tokens). Runs are stored at trace level minimal; views re-capture lazily.</p>
      {running ? (
        <div className="flex flex-col gap-1">
          <ProgressBar value={running.progress} />
          <span className="truncate font-mono text-[10px] text-ns-text-muted">
            {running.status} · {running.message} · {now ? fmtSeconds((now - running.startedAt) / 1000) : ""}
          </span>
        </div>
      ) : null}
      {job?.status === "error" ? <ErrorBox error={job.error} /> : null}
    </Panel>
  );
}

function BenchmarkList({ list, selectedId, onSelect }: { list: Benchmark[]; selectedId: string | null; onSelect: (id: string) => void }) {
  const now = useNow(30000);
  return (
    <ol className="flex flex-col gap-1">
      {list.map((b) => (
        <li key={b.id}>
          <button
            type="button"
            onClick={() => onSelect(b.id)}
            aria-current={b.id === selectedId ? "true" : undefined}
            className={cn(
              "flex w-full flex-col gap-0.5 rounded-[3px] border px-2 py-1.5 text-left",
              b.id === selectedId ? "border-ns-accent bg-ns-accent/8" : "border-ns-border bg-ns-bg hover:border-ns-border-strong",
            )}
          >
            <span className="flex items-center gap-1.5">
              <span className="truncate text-[11px] text-ns-text">{b.name}</span>
              {b.is_demo ? <StatusChip tone="warn">demo</StatusChip> : null}
            </span>
            <span className="flex items-center gap-2 font-mono text-[10px] text-ns-text-muted">
              {b.summary ? `${b.summary.correct}/${b.summary.n} · ${fmtPct(b.summary.accuracy)}` : b.status}
              <span className="ml-auto text-ns-text-faint">{now ? fmtAgo(b.created_at, now) : ""}</span>
            </span>
          </button>
        </li>
      ))}
    </ol>
  );
}

function VerifyTool() {
  const [expr, setExpr] = useState("37 + 58");
  const [answer, setAnswer] = useState("95");
  const [res, setRes] = useState<MathVerifyResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  return (
    <Panel label="Verify" status="observed" statusNote="SymPy oracle" className="shrink-0" bodyClassName="flex flex-col gap-2 p-2.5">
      <form
        className="flex flex-col gap-2"
        onSubmit={async (e) => {
          e.preventDefault();
          setBusy(true);
          setErr(null);
          try {
            setRes(await api.mathVerify({ expression: expr, answer }));
          } catch (x) {
            setRes(null);
            setErr(errorMessage(x));
          } finally {
            setBusy(false);
          }
        }}
      >
        <Field label="expression">
          <TextInput value={expr} onChange={(e) => setExpr(e.target.value)} aria-label="Expression" />
        </Field>
        <Field label="answer">
          <TextInput value={answer} onChange={(e) => setAnswer(e.target.value)} aria-label="Answer to verify" />
        </Field>
        <Button type="submit" size="sm" caps disabled={busy || !expr.trim()}>
          {busy ? "Checking…" : "Verify"}
        </Button>
      </form>
      {err ? <ErrorBox error={err} /> : null}
      {res ? (
        <div className="flex flex-col gap-1 rounded-[3px] border border-ns-border bg-ns-bg px-2 py-1.5">
          <div className="flex items-center gap-2">
            <MathStatusChip status={res.status} />
            <span className="font-mono text-[11px] text-ns-text">oracle = {res.oracle_answer}</span>
          </div>
          <span className="font-mono text-[10px] text-ns-text-muted">parsed {res.parsed_answer ?? "—"} · {res.note}</span>
        </div>
      ) : null}
    </Panel>
  );
}

function BenchmarkDetail({ bench, onEvalDone }: { bench: Benchmark; onEvalDone: () => void }) {
  const { state, update, openRun } = useWorkspace();
  const router = useRouter();
  const s = bench.summary;
  const excluded = s ? s.truncated + s.unparseable : 0;
  const a = bench.results.find((r) => r.run_id === state.compareA) ?? null;
  const b = bench.results.find((r) => r.run_id === state.compareB) ?? null;
  const [compareKey, setCompareKey] = useState<string | null>(null);

  const openInWorkspace = (runId: string) => {
    openRun(runId, { selection: null });
    router.push("/app");
  };

  return (
    <>
      <Panel
        label={bench.name}
        status="observed"
        statusNote="model output + SymPy oracle"
        className="shrink-0"
        right={<span className="font-mono text-[10px] text-ns-text-faint">{bench.id} · {bench.model_id}</span>}
        bodyClassName="flex flex-col gap-3 p-3"
      >
        {bench.is_demo ? <p className="font-mono text-[11px] text-ns-warn">DEMO benchmark — recorded on the random-weight fixture model.</p> : null}
        {s ? (
          <div className="grid grid-cols-[repeat(5,minmax(0,1fr))] gap-2">
            <Stat label="accuracy" value={fmtPct(s.accuracy, 0)} sub={`${s.correct}/${s.n} correct`} tone="accent" />
            <Stat label="correct" value={s.correct} tone="ok" />
            <Stat label="incorrect" value={s.incorrect} tone="danger" />
            <Stat label="truncated" value={s.truncated} sub="excluded from comparisons" />
            <Stat label="unparseable" value={s.unparseable} sub={excluded ? `${excluded} excluded total` : undefined} />
          </div>
        ) : (
          <Loading label="Benchmark running" />
        )}
        {s ? (
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(s.by_kind).map(([k, v]) => (
              <span key={k} className="rounded-[3px] border border-ns-border bg-ns-bg px-2 py-1 font-mono text-[10.5px] text-ns-text-muted">
                <span className="text-ns-text">{k}</span> {v.correct}/{v.n} · {fmtPct(v.accuracy)}
              </span>
            ))}
          </div>
        ) : null}
        <Caveat>Accuracy counts correct / all problems. Truncated (hit the token budget) and unparseable outputs are not treated as wrong answers and are excluded from correct-vs-incorrect comparisons.</Caveat>
      </Panel>

      <Panel
        label="Results"
        status="observed"
        className="shrink-0"
        right={
          <span className="flex items-center gap-2 font-mono text-[10px] text-ns-text-muted">
            pick <span className="text-ns-accent">A = correct</span> and <span className="text-ns-accent-2">B = incorrect</span> to compare
          </span>
        }
        padded={false}
      >
        <div className="max-h-[360px] overflow-auto">
          <table className="w-full border-collapse text-left">
            <thead className="sticky top-0 z-10 bg-ns-panel">
              <tr className="font-mono text-[9.5px] uppercase tracking-[0.08em] text-ns-text-faint">
                <th className="px-2 py-1.5 font-normal">A</th>
                <th className="px-1 py-1.5 font-normal">B</th>
                <th className="px-2 py-1.5 font-normal">kind</th>
                <th className="px-2 py-1.5 font-normal">question</th>
                <th className="px-2 py-1.5 font-normal">oracle</th>
                <th className="px-2 py-1.5 font-normal">model output</th>
                <th className="px-2 py-1.5 font-normal">status</th>
                <th className="px-2 py-1.5 text-right font-normal" title="Teacher-forced log P(oracle answer tokens | prompt)">log P(oracle)</th>
                <th className="px-2 py-1.5 font-normal" />
              </tr>
            </thead>
            <tbody>
              {bench.results.map((r) => (
                <ResultRow
                  key={r.run_id}
                  r={r}
                  isA={state.compareA === r.run_id}
                  isB={state.compareB === r.run_id}
                  onA={() => update({ compareA: r.run_id })}
                  onB={() => update({ compareB: r.run_id })}
                  onOpen={() => openInWorkspace(r.run_id)}
                />
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel
        label="Trajectory comparison · correct A vs incorrect B"
        status="derived"
        statusNote="correlational"
        className="shrink-0"
        bodyClassName="flex flex-col gap-3 p-3"
      >
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-[11px] text-ns-accent">A: {a ? truncate(a.question, 40) : "—"}</span>
          <span className="font-mono text-[11px] text-ns-accent-2">B: {b ? truncate(b.question, 40) : "—"}</span>
          <Button
            size="sm"
            caps
            variant="secondary"
            className="ml-auto"
            disabled={!a || !b || a.run_id === b.run_id}
            onClick={() => a && b && setCompareKey(`${a.run_id}|${b.run_id}`)}
          >
            Compare A vs B
          </Button>
        </div>
        {a && !a.correct ? <p className="text-[10.5px] text-ns-warn">A is not a correct run.</p> : null}
        {b && b.status !== "incorrect" ? <p className="text-[10.5px] text-ns-warn">B is {b.status} — truncated/unparseable runs are excluded from correct-vs-incorrect comparisons.</p> : null}
        {compareKey ? (
          <CompareView key={compareKey} runA={compareKey.split("|")[0]} runB={compareKey.split("|")[1]} onOpen={openInWorkspace} />
        ) : (
          <p className="text-[11px] text-ns-text-faint">Select A and B in the table, then compare. The first comparison of a pair re-captures attention for both runs (≈15 s).</p>
        )}
      </Panel>

      <InterventionEval bench={bench} onDone={onEvalDone} />
    </>
  );
}

function ResultRow({
  r,
  isA,
  isB,
  onA,
  onB,
  onOpen,
}: {
  r: BenchmarkResultRow;
  isA: boolean;
  isB: boolean;
  onA: () => void;
  onB: () => void;
  onOpen: () => void;
}) {
  return (
    <tr className={cn("border-t border-ns-border align-middle", isA && "bg-ns-accent/6", isB && "bg-ns-accent-2/6")}>
      <td className="px-2 py-1">
        <input type="radio" name="compare-a" checked={isA} onChange={onA} aria-label={`Use as A: ${r.question}`} className="accent-[#8fb3ff]" />
      </td>
      <td className="px-1 py-1">
        <input type="radio" name="compare-b" checked={isB} onChange={onB} aria-label={`Use as B: ${r.question}`} className="accent-[#d7c9a6]" />
      </td>
      <td className="px-2 py-1 font-mono text-[10.5px] text-ns-text-muted">{r.kind}</td>
      <td className="max-w-[320px] truncate px-2 py-1 text-[11.5px] text-ns-text" title={r.question}>
        {r.question}
      </td>
      <td className="px-2 py-1 font-mono text-[11px] text-ns-text">{r.oracle_answer}</td>
      <td className="max-w-[220px] truncate px-2 py-1 font-mono text-[11px] text-ns-text-muted" title={r.model_output}>
        {r.model_output.replace(/<\|im_end\|>/g, "⟨im_end⟩").replace(/\n/g, "↵")}
      </td>
      <td className="px-2 py-1">
        <MathStatusChip status={r.status} />
      </td>
      <td className="ns-tabular px-2 py-1 text-right font-mono text-[10.5px] text-ns-text-muted">{fmtNum(r.oracle_logprob, 3)}</td>
      <td className="px-2 py-1">
        <button type="button" onClick={onOpen} className="inline-flex items-center gap-1 font-mono text-[10px] uppercase tracking-[0.08em] text-ns-accent hover:underline" title="Open this run in the workspace">
          open <ExternalLink className="size-3" aria-hidden="true" />
        </button>
      </td>
    </tr>
  );
}

function InterventionEval({ bench, onDone }: { bench: Benchmark; onDone: () => void }) {
  const { jobs, activeJob, startJob, model, notify, state } = useWorkspace();
  // defaults follow the workspace's intervention builder (e.g. a head picked from a sweep)
  const [layer, setLayer] = useState(state.draft.layer);
  const [head, setHead] = useState(state.draft.head);
  const [factor, setFactor] = useState(state.draft.factor);
  const job = jobs.benchmark_eval;
  const mine = job && job.scopeKey === bench.id ? job : null;
  const running = mine && (mine.status === "queued" || mine.status === "running") ? mine : null;
  const result = mine?.status === "done" ? (mine.result as BenchmarkEvalResult | undefined) : undefined;
  const past = bench.summary?.intervention_evals ?? [];
  const nL = model?.n_layers ?? 28;
  const nH = model?.n_heads ?? 12;

  return (
    <Panel label="Intervention eval across the benchmark" status="causal" statusNote={`n=${bench.results.length}`} className="shrink-0" bodyClassName="flex flex-col gap-3 p-3">
      <p className="text-[11px] leading-snug text-ns-text-muted">
        Apply one head ablation to every problem, regenerate, re-verify with the oracle and re-measure log P(oracle answer). The result is CAUSALLY TESTED over this benchmark only.
      </p>
      <div className="flex flex-wrap items-end gap-2">
        <Field label={`layer (0–${nL - 1})`} className="w-24">
          <NumberInput value={layer} min={0} max={nL - 1} onValue={(v) => setLayer(Math.min(nL - 1, Math.max(0, Math.round(v))))} aria-label="Eval layer" />
        </Field>
        <Field label={`head (0–${nH - 1})`} className="w-24">
          <NumberInput value={head} min={0} max={nH - 1} onValue={(v) => setHead(Math.min(nH - 1, Math.max(0, Math.round(v))))} aria-label="Eval head" />
        </Field>
        <Field label="factor" className="w-20">
          <NumberInput value={factor} step={0.1} onValue={setFactor} aria-label="Eval factor" />
        </Field>
        <Button
          caps
          variant="secondary"
          size="md"
          disabled={Boolean(activeJob)}
          title={activeJob ? `Busy: ${activeJob.label}` : undefined}
          onClick={() =>
            void startJob<BenchmarkEvalResult>(
              "benchmark_eval",
              `Eval L${layer}.H${head}`,
              bench.id,
              () => api.benchmarkEval(bench.id, [{ type: "head_ablation", layer, head, factor, positions: null }]),
              (res) => {
                notify("ok", `${res.interventions_label}: accuracy ${fmtPct(res.accuracy_before)} → ${fmtPct(res.accuracy_after)} (n=${res.n}).`);
                onDone();
              },
            )
          }
        >
          <FlaskConical aria-hidden="true" /> Evaluate ablation
        </Button>
        <span className="text-[10.5px] text-ns-text-faint">≈4 s per problem</span>
      </div>
      {running ? (
        <div className="flex flex-col gap-1">
          <ProgressBar value={running.progress} tone="causal" />
          <span className="font-mono text-[10px] text-ns-text-muted">
            {running.status} · {running.message}
          </span>
        </div>
      ) : null}
      {mine?.status === "error" ? <ErrorBox error={mine.error} /> : null}
      {result ? <EvalResult res={result} /> : null}
      {past.length ? (
        <div>
          <SectionHeader label="Previous evaluations on this benchmark" status="causal" />
          <table className="mt-1 w-full border-collapse text-left">
            <thead>
              <tr className="font-mono text-[9.5px] uppercase tracking-[0.08em] text-ns-text-faint">
                <th className="py-1 pr-2 font-normal">intervention</th>
                <th className="py-1 pr-2 font-normal">accuracy</th>
                <th className="py-1 pr-2 font-normal">flips c→i / i→c</th>
                <th className="py-1 pr-2 font-normal">mean Δ log P(oracle) ± SEM</th>
                <th className="py-1 font-normal">scope</th>
              </tr>
            </thead>
            <tbody>
              {past
                .slice()
                .reverse()
                .map((e) => (
                  <tr key={e.id} className="border-t border-ns-border font-mono text-[10.5px]">
                    <td className="py-1 pr-2 text-ns-text">{e.interventions_label}</td>
                    <td className="py-1 pr-2 text-ns-text-muted">
                      {fmtPct(e.accuracy_before)} → {fmtPct(e.accuracy_after)}
                    </td>
                    <td className="py-1 pr-2 text-ns-text-muted">
                      {e.flips_correct_to_incorrect} / {e.flips_incorrect_to_correct}
                    </td>
                    <td className="py-1 pr-2 text-ns-text-muted">
                      {fmtDelta(e.mean_delta_oracle_logprob, 3)} ± {fmtNum(e.sem_delta_oracle_logprob, 3)}
                    </td>
                    <td className="py-1 text-ns-text-faint">{e.scope}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </Panel>
  );
}

function EvalResult({ res }: { res: BenchmarkEvalResult }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <StatusBadge status={res.status} note={res.scope} size="sm" />
        <span className="font-mono text-[11px] text-ns-text">{res.interventions_label}</span>
        <span className="font-mono text-[10px] text-ns-text-faint">{fmtSeconds(res.seconds)}</span>
      </div>
      <div className="grid grid-cols-4 gap-2">
        <Stat label="accuracy" value={`${fmtPct(res.accuracy_before)} → ${fmtPct(res.accuracy_after)}`} sub={`n=${res.n}`} tone="accent" />
        <Stat label="correct → incorrect" value={res.flips_correct_to_incorrect} />
        <Stat label="incorrect → correct" value={res.flips_incorrect_to_correct} />
        <Stat label="mean Δ log P(oracle)" value={fmtDelta(res.mean_delta_oracle_logprob, 3)} sub={`± ${fmtNum(res.sem_delta_oracle_logprob, 3)} SEM`} />
      </div>
      <div className="max-h-56 overflow-auto rounded-[3px] border border-ns-border">
        <table className="w-full border-collapse text-left">
          <thead className="sticky top-0 bg-ns-panel">
            <tr className="font-mono text-[9.5px] uppercase tracking-[0.08em] text-ns-text-faint">
              <th className="px-2 py-1 font-normal">question</th>
              <th className="px-2 py-1 font-normal">before</th>
              <th className="px-2 py-1 font-normal">after</th>
              <th className="px-2 py-1 font-normal">output after</th>
              <th className="px-2 py-1 text-right font-normal">Δ log P(oracle)</th>
            </tr>
          </thead>
          <tbody>
            {res.rows.map((r) => (
              <tr key={r.run_id} className="border-t border-ns-border">
                <td className="max-w-[260px] truncate px-2 py-1 text-[11px] text-ns-text" title={r.question}>
                  {r.question}
                </td>
                <td className="px-2 py-1">
                  <MathStatusChip status={r.correct_before ? "correct" : "incorrect"} />
                </td>
                <td className="px-2 py-1">
                  <MathStatusChip status={r.status_after} />
                </td>
                <td className="max-w-[160px] truncate px-2 py-1 font-mono text-[10.5px] text-ns-text-muted">{r.output_after.replace(/<\|im_end\|>/g, "⟨im_end⟩")}</td>
                <td className="ns-tabular px-2 py-1 text-right font-mono text-[10.5px] text-ns-text">{fmtDelta(r.delta_oracle_logprob, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[10.5px] text-ns-text-faint">
        {shortId(res.id)} · {res.scope}. “before” uses the stored benchmark label (truncated runs count as not correct).
      </p>
    </div>
  );
}
