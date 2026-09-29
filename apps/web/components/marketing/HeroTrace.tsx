"use client";

import { useEffect, useId, useRef, useState, useSyncExternalStore, type CSSProperties, type ReactNode } from "react";
import { AlertTriangle, Pause, Play } from "lucide-react";
import { EpistemicBadge } from "@/components/ui/EpistemicBadge";
import { StatusChip } from "@/components/ui/StatusChip";
import { cn } from "@/lib/cn";
import {
  depthRowForLayer,
  interventionLabel,
  visibleToken,
  type HeroTraceSnapshot,
} from "@/lib/hero-trace";

/* -------------------------------------------------------------------------- */
/* Timeline                                                                    */
/* -------------------------------------------------------------------------- */

export const HERO_STAGES = [
  { key: "prompt", label: "Prompt", ms: 2200 },
  { key: "matrix", label: "Token × layer", ms: 2600 },
  { key: "component", label: "Component", ms: 2100 },
  { key: "intervene", label: "Intervention", ms: 1800 },
  { key: "output", label: "Output", ms: 4800 },
] as const;
const LAST = HERO_STAGES.length - 1;

const ROW_H = 8; // px, matrix row height
const ROW_GAP = 1; // px
const PITCH = ROW_H + ROW_GAP;

const MQ = "(prefers-reduced-motion: reduce)";
function subscribeReducedMotion(onChange: () => void) {
  if (typeof window === "undefined" || typeof window.matchMedia !== "function") return () => {};
  const mq = window.matchMedia(MQ);
  mq.addEventListener("change", onChange);
  return () => mq.removeEventListener("change", onChange);
}
function getReducedMotion() {
  return typeof window !== "undefined" && typeof window.matchMedia === "function" && window.matchMedia(MQ).matches;
}

const fmtP = (p: number) => (p >= 0.01 || p === 0 ? p.toFixed(2) : p < 0.001 ? "<0.001" : p.toFixed(3));

/** Sequential single-hue ramp: panel-2 (0) → accent (1), mixed in OKLab. */
function cellColor(v: number) {
  const pct = Math.round(Math.min(1, Math.max(0, v)) * 100);
  return `color-mix(in oklab, var(--ns-accent) ${pct}%, var(--ns-panel-2))`;
}

/* -------------------------------------------------------------------------- */
/* Component                                                                   */
/* -------------------------------------------------------------------------- */

export interface HeroTraceProps {
  snapshot: HeroTraceSnapshot;
  className?: string;
  /** Start paused (tests, screenshots). */
  initialPlaying?: boolean;
}

/**
 * Animated, data-driven walkthrough of one NeuralScope experiment:
 * prompt → token × layer matrix → selected component → intervention → changed output.
 * Renders entirely from a snapshot; the caption states whether it is a recorded
 * model trace or placeholder data.
 */
export function HeroTrace({ snapshot, className, initialPlaying = true }: HeroTraceProps) {
  const reduced = useSyncExternalStore(subscribeReducedMotion, getReducedMotion, () => false);
  // Start on the final stage so the complete picture is visible before JS / for reduced motion.
  const [stage, setStage] = useState<number>(LAST);
  const [cycle, setCycle] = useState(0);
  const [playing, setPlaying] = useState(initialPlaying);
  const [visible, setVisible] = useState(true);
  const rootRef = useRef<HTMLElement>(null);
  const titleId = useId();

  const running = playing && !reduced && visible;

  // Pause while scrolled out of view.
  useEffect(() => {
    const el = rootRef.current;
    if (!el || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting), { threshold: 0.15 });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    if (!running) return;
    const id = window.setTimeout(() => {
      if (stage >= LAST) {
        setCycle((c) => c + 1);
        setStage(0);
      } else {
        setStage(stage + 1);
      }
    }, HERO_STAGES[stage].ms);
    return () => window.clearTimeout(id);
  }, [running, stage, cycle]);

  const { tokens, depth_labels: depths, matrix, target, intervention: iv, source } = snapshot;
  const T = tokens.length;
  const D = depths.length;
  const col = target.column;
  const row = depthRowForLayer(depths, iv.layer);
  const isPlaceholder = source === "PLACEHOLDER";

  const shown = {
    matrix: stage >= 1,
    lock: stage >= 2,
    ablate: stage >= 3,
    output: stage >= 4,
  };
  const sweeping = stage === 1 && running;
  const lensAtLock = target.prob_by_depth[row];
  const rankAtLock = target.rank_by_depth[row];
  const targetTok = visibleToken(target.token);
  const ivLabel = interventionLabel(iv);
  const delta = iv.target_prob_after - iv.target_prob_before;

  const colCenter = `${((col + 0.5) / T) * 100}%`;
  const rowCenter = row * PITCH + ROW_H / 2;
  const matrixH = D * PITCH - ROW_GAP;

  function jump(i: number) {
    setPlaying(false);
    setStage(i);
  }

  return (
    <figure
      ref={rootRef}
      className={cn(
        "relative min-w-0 rounded-[6px] border border-ns-border bg-ns-panel",
        "shadow-[0_40px_120px_-60px_rgba(0,0,0,0.9)]",
        className,
      )}
      aria-labelledby={titleId}
      data-stage={HERO_STAGES[stage].key}
      data-source={source}
    >
      <CornerTicks />

      {/* ------------------------------------------------------------ header */}
      <div className="flex items-center gap-3 border-b border-ns-border px-3 py-2 sm:px-4">
        <span id={titleId} className="ns-label flex items-center gap-2 text-ns-text">
          <span
            aria-hidden="true"
            className={cn(
              "size-1.5 rounded-full",
              isPlaceholder ? "bg-ns-warn" : "bg-ns-ok",
              running && "animate-[ns-blink_1.6s_ease-in-out_infinite]",
            )}
          />
          Trace
        </span>
        <ol className="flex min-w-0 flex-1 items-center gap-1" aria-label="Experiment stages">
          {HERO_STAGES.map((s, i) => {
            const active = i === stage;
            const done = i < stage;
            return (
              <li key={s.key} className="relative min-w-0 flex-none">
                <button
                  type="button"
                  onClick={() => jump(i)}
                  aria-current={active ? "step" : undefined}
                  aria-label={`Stage ${i + 1}: ${s.label}`}
                  className={cn(
                    "flex h-6 items-center gap-1.5 rounded-[3px] px-1.5 font-mono text-[10px] uppercase tracking-[0.1em] transition-colors",
                    active ? "bg-ns-panel-2 text-ns-text" : done ? "text-ns-text-muted hover:text-ns-text" : "text-ns-text-faint hover:text-ns-text-muted",
                  )}
                >
                  <span className="ns-tabular">{String(i + 1).padStart(2, "0")}</span>
                  <span className={cn("truncate", active ? "inline" : "hidden xl:inline")}>{s.label}</span>
                </button>
                {active ? (
                  <span aria-hidden="true" className="absolute inset-x-1.5 -bottom-[5px] h-px overflow-hidden bg-ns-border">
                    <span
                      key={`${cycle}-${stage}-${running}`}
                      className="block h-full origin-left bg-ns-accent"
                      style={
                        running
                          ? { animation: `ns-progress ${HERO_STAGES[stage].ms}ms linear both` }
                          : { transform: "scaleX(1)" }
                      }
                    />
                  </span>
                ) : null}
              </li>
            );
          })}
        </ol>
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          disabled={reduced}
          aria-label={playing && !reduced ? "Pause animation" : "Play animation"}
          title={reduced ? "Animation disabled (reduced motion)" : undefined}
          className="grid size-6 shrink-0 place-items-center rounded-[3px] border border-ns-border text-ns-text-muted transition-colors hover:border-ns-border-strong hover:text-ns-text disabled:opacity-40"
        >
          {playing && !reduced ? <Pause className="size-3" aria-hidden="true" /> : <Play className="size-3" aria-hidden="true" />}
        </button>
      </div>

      {/* ------------------------------------------------------------ 01 prompt */}
      <section aria-label="Prompt" className="border-b border-ns-border px-3 py-3 sm:px-4">
        <StageHeader n={1} label="Prompt" active={stage === 0}>
          <EpistemicBadge status="observed" size="xs" />
        </StageHeader>
        <ol key={`tok-${cycle}`} className="mt-2.5 flex flex-wrap gap-1" aria-label={`Prompt tokens: ${snapshot.prompt}`}>
          {tokens.map((tok, i) => {
            const isTarget = i === col;
            const animate = stage === 0 && running;
            return (
              <li
                key={i}
                title={`t${i} ${JSON.stringify(tok)}`}
                className={cn(
                  "relative flex h-6 items-center rounded-[3px] border px-1.5 font-mono text-[11px]",
                  "border-ns-border-strong bg-ns-panel-2 text-ns-text",
                  isTarget && shown.matrix && "border-ns-accent-2/70",
                )}
                style={animate ? { animation: `ns-token 700ms ease-out ${i * 110}ms both` } : undefined}
              >
                <TokenText tok={tok} />
                <span className="ml-1 text-[8px] text-ns-text-faint">{i}</span>
              </li>
            );
          })}
        </ol>
      </section>

      {/* ------------------------------------------------------------ 02 matrix */}
      <section aria-label="Token by layer matrix" className="border-b border-ns-border px-3 pb-3 pt-3 sm:px-4">
        <StageHeader n={2} label={matrix.label} active={stage === 1}>
          <EpistemicBadge status="derived" size="xs" />
        </StageHeader>

        <div className="mt-3 grid grid-cols-[28px_minmax(0,1fr)_44px] gap-x-2 sm:grid-cols-[34px_minmax(0,1fr)_64px] sm:gap-x-3">
          {/* column header row */}
          <div aria-hidden="true" />
          <div aria-hidden="true" className="grid h-4 items-end" style={{ gridTemplateColumns: `repeat(${T}, minmax(0, 1fr))`, columnGap: ROW_GAP }}>
            {tokens.map((tok, i) => (
              <span
                key={i}
                className={cn(
                  "truncate text-center font-mono text-[9px] leading-none",
                  i === col && shown.matrix ? "text-ns-accent-2" : "text-ns-text-faint",
                )}
              >
                <span className="sm:hidden">{i}</span>
                <span className="hidden sm:inline">{visibleToken(tok)}</span>
              </span>
            ))}
          </div>
          <div aria-hidden="true" className="truncate text-right font-mono text-[9px] leading-4 text-ns-accent-2">
            p(“{targetTok}”)
          </div>

          {/* row labels */}
          <div aria-hidden="true" className="relative mt-1.5" style={{ height: matrixH }}>
            {depths.map((label, d) => {
              const isLock = d === row && shown.lock;
              const regular = d % 4 === 0 || d === D - 1;
              const hide = shown.lock && d !== row && Math.abs(d - row) < 2;
              if (!(isLock || (regular && !hide))) return null;
              return (
                <span
                  key={d}
                  className={cn(
                    "absolute right-0 -translate-y-1/2 font-mono text-[9px] leading-none",
                    isLock ? (shown.ablate ? "text-ns-danger" : "text-ns-text") : "text-ns-text-faint",
                  )}
                  style={{ top: d * PITCH + ROW_H / 2 }}
                >
                  {label}
                </span>
              );
            })}
          </div>

          {/* matrix */}
          <div className="relative mt-1.5" style={{ height: matrixH }}>
            <div
              aria-hidden="true"
              className="grid h-full"
              style={{
                gridTemplateColumns: `repeat(${T}, minmax(0, 1fr))`,
                gridTemplateRows: `repeat(${D}, ${ROW_H}px)`,
                gap: ROW_GAP,
              }}
            >
              {matrix.values.map((r, d) =>
                r.map((v, t) => (
                  <div
                    key={`${d}-${t}`}
                    title={`${depths[d]} · t${t} ${visibleToken(tokens[t])} · ${matrix.metric} ${v.toFixed(3)}`}
                    className="rounded-[1px] transition-[background-color] duration-300 ease-out"
                    style={{
                      backgroundColor: shown.matrix ? cellColor(v) : "var(--ns-panel-2)",
                      transitionDelay: sweeping ? `${d * 45}ms` : "0ms",
                    }}
                  />
                )),
              )}
            </div>

            {/* target column frame */}
            <div
              aria-hidden="true"
              className={cn(
                "pointer-events-none absolute -inset-y-[3px] rounded-[2px] border border-ns-accent-2/70 transition-opacity duration-500",
                shown.matrix ? "opacity-100" : "opacity-0",
              )}
              style={{ left: `calc(${(col / T) * 100}% - 2px)`, width: `calc(${100 / T}% + 3px)` }}
            />

            {/* scan line */}
            {sweeping ? (
              <div
                key={`scan-${cycle}`}
                aria-hidden="true"
                className="pointer-events-none absolute inset-x-0 h-px bg-ns-accent shadow-[0_0_12px_2px_rgba(143,179,255,0.45)]"
                style={{ animation: `ns-scan ${D * 45 + 500}ms linear both` }}
              />
            ) : null}

            {/* crosshair guides */}
            <div
              aria-hidden="true"
              className={cn(
                "pointer-events-none absolute left-0 h-px border-t border-dashed transition-opacity duration-500",
                shown.ablate ? "border-ns-danger/60" : "border-ns-text/45",
                shown.lock ? "opacity-100" : "opacity-0",
              )}
              style={{ top: rowCenter, width: colCenter }}
            />
            <div
              aria-hidden="true"
              className={cn(
                "pointer-events-none absolute top-0 w-px border-l border-dashed transition-opacity duration-500",
                shown.ablate ? "border-ns-danger/60" : "border-ns-text/45",
                shown.lock ? "opacity-100" : "opacity-0",
              )}
              style={{ left: colCenter, height: rowCenter }}
            />

            {/* reticle */}
            {shown.lock ? (
              <div
                key={`ret-${cycle}`}
                aria-hidden="true"
                className="pointer-events-none absolute"
                style={{
                  left: colCenter,
                  top: rowCenter,
                  transform: "translate(-50%, -50%)",
                  animation: stage === 2 && running ? "ns-reticle-lock 650ms cubic-bezier(.2,.8,.2,1) both" : undefined,
                }}
              >
                <Reticle danger={shown.ablate} />
              </div>
            ) : null}

            {/* ablation tag */}
            <div
              aria-hidden="true"
              className={cn(
                "pointer-events-none absolute whitespace-nowrap transition-opacity duration-300",
                shown.ablate ? "opacity-100" : "opacity-0",
              )}
              style={
                col / T > 0.5
                  ? { top: rowCenter - 30, right: `calc(${100 - ((col + 0.5) / T) * 100}% + 14px)` }
                  : { top: rowCenter - 30, left: `calc(${colCenter} + 14px)` }
              }
            >
              <span className="inline-flex h-5 items-center rounded-[3px] border border-ns-danger/60 bg-ns-bg px-1.5 font-mono text-[10px] font-medium uppercase tracking-[0.1em] text-ns-danger">
                {ivLabel}
              </span>
            </div>
          </div>

          {/* p(target) by depth track */}
          <div aria-hidden="true" className="relative mt-1.5" style={{ height: matrixH }}>
            <div className="absolute inset-y-0 left-0 w-px bg-ns-border" />
            {target.prob_by_depth.map((p, d) => (
              <div
                key={d}
                title={`${depths[d]} · p(${targetTok}) ${fmtP(p)} · rank ${target.rank_by_depth[d]}`}
                className={cn(
                  "absolute left-px rounded-r-[1px] transition-[width] duration-300 ease-out",
                  d === row && shown.lock ? (shown.ablate ? "bg-ns-danger" : "bg-ns-text") : "bg-ns-accent-2/75",
                )}
                style={{
                  top: d * PITCH,
                  height: ROW_H,
                  width: shown.matrix ? `max(${p > 0 ? 1 : 0}px, calc(${p * 100}% - 1px))` : 0,
                  transitionDelay: sweeping ? `${d * 45}ms` : "0ms",
                }}
              />
            ))}
          </div>

          {/* legend row */}
          <div aria-hidden="true" />
          <div aria-hidden="true" className="mt-2 flex items-center gap-2 font-mono text-[9px] text-ns-text-faint">
            <span>0</span>
            <span
              className="h-1.5 w-16 rounded-[1px] sm:w-24"
              style={{ background: "linear-gradient(90deg, var(--ns-panel-2), var(--ns-accent))" }}
            />
            <span>1</span>
            <span className="truncate">{matrix.metric}</span>
          </div>
          <div aria-hidden="true" className="mt-2 text-right font-mono text-[9px] text-ns-text-faint">
            0 → 1
          </div>
        </div>

        {/* 03 component readout */}
        <div className="mt-3 flex min-h-6 flex-wrap items-center gap-x-3 gap-y-1 border-t border-ns-border pt-2.5">
          <StageHeader n={3} label="Component" active={stage === 2} />
          {shown.lock ? (
            <p className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px] text-ns-text-muted">
              <span className="text-ns-text">
                resid[{depths[row]}, t{col}]
              </span>
              <span>
                {iv.head === null ? "block" : "head"}{" "}
                <span className="text-ns-text">{iv.head === null ? `L${iv.layer}` : `L${iv.layer}.H${iv.head}`}</span>
              </span>
              <span>
                lens p(“{targetTok}”) <span className="ns-tabular text-ns-text">{fmtP(lensAtLock)}</span>
              </span>
              <span>
                rank <span className="ns-tabular text-ns-text">{rankAtLock}</span>
              </span>
            </p>
          ) : (
            <span className="font-mono text-[11px] text-ns-text-faint">{shown.matrix ? "scanning depth…" : "—"}</span>
          )}
        </div>
      </section>

      {/* ------------------------------------------------------------ 04 / 05 */}
      <div className="grid sm:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
        <section aria-label="Intervention" className="border-b border-ns-border px-3 py-3 sm:border-b-0 sm:border-r sm:px-4">
          <StageHeader n={4} label="Intervention" active={stage === 3} />
          <div className={cn("mt-2.5 transition-opacity duration-300", shown.ablate ? "opacity-100" : "opacity-35")}>
            <span className="inline-flex h-6 items-center rounded-[3px] border border-ns-danger/60 bg-ns-danger/10 px-2 font-mono text-[11px] font-medium uppercase tracking-[0.1em] text-ns-danger">
              {ivLabel}
            </span>
            <p className="mt-2 text-[12px] leading-snug text-ns-text-muted">{iv.description}</p>
          </div>
        </section>

        <section aria-label="Output" className="px-3 py-3 sm:px-4">
          <StageHeader n={5} label="Output" active={stage === 4}>
            <EpistemicBadge status="causal" size="xs" note="n=1" />
          </StageHeader>
          <div className="mt-2.5 space-y-1.5">
            <ProbBar label="before" value={iv.target_prob_before} shown={shown.ablate} tone="neutral" />
            <ProbBar
              label="after"
              value={iv.target_prob_after}
              shown={shown.output}
              tone="danger"
              from={shown.ablate ? iv.target_prob_before : 0}
            />
          </div>
          <p
            className={cn(
              "mt-2.5 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px] text-ns-text-muted transition-opacity duration-300",
              shown.output ? "opacity-100" : "opacity-0",
            )}
          >
            <span>
              p(“{targetTok}”) Δ <span className="ns-tabular text-ns-text">{delta >= 0 ? "+" : "−"}{Math.abs(delta).toFixed(2)}</span>
            </span>
            <span>
              KL <span className="ns-tabular text-ns-text">{iv.kl.toFixed(2)}</span> nats
            </span>
            <span className="flex items-center gap-1.5">
              out
              <span className="text-ns-text-faint line-through decoration-ns-text-faint">
                <TokenText tok={iv.output_before} />
              </span>
              →
              <span className="text-ns-text">
                <TokenText tok={iv.output_after} />
              </span>
            </span>
          </p>
        </section>
      </div>

      {/* ------------------------------------------------------------ caption */}
      <figcaption className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-t border-ns-border px-3 py-2 font-mono text-[10.5px] text-ns-text-muted sm:px-4">
        {isPlaceholder ? (
          <>
            <StatusChip tone="warn" icon={<AlertTriangle aria-hidden="true" />}>
              Placeholder data — not a model trace
            </StatusChip>
            <span className="text-ns-text-faint">Illustrative values in the recorded-trace schema.</span>
          </>
        ) : (
          <>
            <span className="inline-flex items-center gap-1.5 text-ns-text">
              <span aria-hidden="true" className="size-1.5 rounded-full bg-ns-ok" />
              Recorded trace · {snapshot.model_id} · run {snapshot.run_id}
            </span>
            {snapshot.captured_at ? <span className="text-ns-text-faint">{snapshot.captured_at.slice(0, 10)}</span> : null}
          </>
        )}
        <span className="sr-only">
          {`Prompt "${snapshot.prompt}". ${matrix.label} across ${D} depths and ${T} tokens. Selected component ${ivLabel.replace("ABLATE ", "")} at ${depths[row]}, token ${col}. ${iv.description}. Probability of "${target.token}" changed from ${iv.target_prob_before.toFixed(2)} to ${iv.target_prob_after.toFixed(2)}; output changed from "${iv.output_before}" to "${iv.output_after}".`}
        </span>
      </figcaption>
    </figure>
  );
}

/* -------------------------------------------------------------------------- */
/* Pieces                                                                      */
/* -------------------------------------------------------------------------- */

function StageHeader({
  n,
  label,
  active,
  children,
}: {
  n: number;
  label: string;
  active: boolean;
  children?: ReactNode;
}) {
  return (
    <div className="flex min-w-0 items-center gap-2">
      <span
        className={cn(
          "font-mono text-[10px] font-medium uppercase tracking-[0.12em] transition-colors",
          active ? "text-ns-accent" : "text-ns-text-faint",
        )}
      >
        {String(n).padStart(2, "0")}
      </span>
      <span className={cn("ns-label truncate transition-colors", active && "text-ns-text")}>{label}</span>
      {children}
    </div>
  );
}

function TokenText({ tok }: { tok: string }) {
  const shown = visibleToken(tok);
  return (
    <span className="whitespace-pre">
      {Array.from(shown).map((ch, i) =>
        ch === "␣" || ch === "↵" || ch === "⇥" ? (
          <span key={i} className="text-ns-text-faint">
            {ch}
          </span>
        ) : (
          <span key={i}>{ch}</span>
        ),
      )}
    </span>
  );
}

function ProbBar({
  label,
  value,
  shown,
  tone,
  from,
}: {
  label: string;
  value: number;
  shown: boolean;
  tone: "neutral" | "danger";
  from?: number;
}) {
  const width = shown ? value : from ?? 0;
  const style: CSSProperties = { width: `${width * 100}%` };
  return (
    <div className="grid grid-cols-[44px_minmax(0,1fr)_36px] items-center gap-2 font-mono text-[10.5px]">
      <span className="text-ns-text-muted">{label}</span>
      <span className="relative h-2 overflow-hidden rounded-[1px] bg-ns-panel-2">
        <span
          className={cn(
            "absolute inset-y-0 left-0 rounded-r-[1px] transition-[width] duration-700 ease-out",
            tone === "neutral" ? "bg-ns-text-muted" : "bg-ns-danger",
            !shown && "opacity-30",
          )}
          style={style}
        />
      </span>
      <span className={cn("ns-tabular text-right", shown ? "text-ns-text" : "text-ns-text-faint")}>
        {shown ? value.toFixed(2) : "—"}
      </span>
    </div>
  );
}

function Reticle({ danger }: { danger: boolean }) {
  return (
    <svg
      width="30"
      height="30"
      viewBox="0 0 30 30"
      fill="none"
      className={cn("transition-colors duration-300", danger ? "text-ns-danger" : "text-ns-text")}
    >
      <circle cx="15" cy="15" r="9" stroke="currentColor" strokeWidth="1.25" />
      <path d="M15 1v6M15 23v6M1 15h6M23 15h6" stroke="currentColor" strokeWidth="1.25" />
      <rect x="12.5" y="12.5" width="5" height="5" stroke="currentColor" strokeWidth="1" />
    </svg>
  );
}

function CornerTicks() {
  const base = "pointer-events-none absolute size-2.5 border-ns-text-faint";
  return (
    <>
      <span aria-hidden="true" className={cn(base, "-left-px -top-px rounded-tl-[6px] border-l border-t")} />
      <span aria-hidden="true" className={cn(base, "-right-px -top-px rounded-tr-[6px] border-r border-t")} />
      <span aria-hidden="true" className={cn(base, "-bottom-px -left-px rounded-bl-[6px] border-b border-l")} />
      <span aria-hidden="true" className={cn(base, "-bottom-px -right-px rounded-br-[6px] border-b border-r")} />
    </>
  );
}
