"use client";

import type { InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from "react";
import { AlertTriangle, Info, RotateCw } from "lucide-react";
import { EpistemicBadge } from "@/components/ui/EpistemicBadge";
import { StatusChip } from "@/components/ui/StatusChip";
import { errorMessage } from "@/lib/api/client";
import { toEpistemic } from "@/lib/api/status";
import type { ApiStatus } from "@/lib/api/types";
import { cn } from "@/lib/cn";
import { fmtSeconds, specialName, splitSpecials, visibleToken } from "@/lib/format";
import { useNow } from "./useResource";

/** Epistemic badge fed directly from an API status string ("causal" → CAUSALLY TESTED). */
export function StatusBadge({
  status,
  note,
  size = "xs",
  className,
}: {
  status: ApiStatus | string | null | undefined;
  note?: string;
  size?: "xs" | "sm";
  className?: string;
}) {
  return <EpistemicBadge status={toEpistemic(status)} size={size} note={note} className={className} />;
}

/** Sub-section heading inside an inspector tab. */
export function SectionHeader({
  label,
  status,
  note,
  right,
  className,
}: {
  label: ReactNode;
  status?: ApiStatus | string;
  note?: string;
  right?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex min-h-6 flex-wrap items-center gap-x-2 gap-y-1", className)}>
      <h3 className="ns-label text-ns-text-muted">{label}</h3>
      {status ? <StatusBadge status={status} note={note} /> : null}
      {right ? <div className="ml-auto flex items-center gap-2">{right}</div> : null}
    </div>
  );
}

export function Caveat({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <p className={cn("flex items-start gap-1.5 text-[11px] leading-snug text-ns-text-faint", className)}>
      <Info aria-hidden="true" className="mt-px size-3 shrink-0" />
      <span>{children}</span>
    </p>
  );
}

export function ErrorBox({ error, onRetry, className }: { error: unknown; onRetry?: () => void; className?: string }) {
  return (
    <div
      role="alert"
      className={cn(
        "flex items-start gap-2 rounded-[var(--ns-radius)] border border-ns-danger/40 bg-ns-danger/8 px-2.5 py-2 text-xs text-ns-danger",
        className,
      )}
    >
      <AlertTriangle aria-hidden="true" className="mt-px size-3.5 shrink-0" />
      <span className="min-w-0 flex-1 break-words font-mono text-[11px] leading-relaxed">{errorMessage(error)}</span>
      {onRetry ? (
        <button
          type="button"
          onClick={onRetry}
          className="inline-flex shrink-0 items-center gap-1 font-mono text-[10px] uppercase tracking-[0.1em] text-ns-text-muted hover:text-ns-text"
        >
          <RotateCw aria-hidden="true" className="size-3" /> retry
        </button>
      ) : null}
    </div>
  );
}

/** Loading readout with a scanning bar and an optional elapsed timer. */
export function Loading({
  label = "Loading",
  hint,
  startedAt,
  className,
  compact,
}: {
  label?: string;
  hint?: string;
  startedAt?: number;
  className?: string;
  compact?: boolean;
}) {
  const now = useNow(250, startedAt !== undefined);
  const elapsed = startedAt !== undefined && now > 0 ? Math.max(0, (now - startedAt) / 1000) : null;
  return (
    <div role="status" aria-live="polite" className={cn("flex flex-col gap-1.5", compact ? "py-1" : "py-3", className)}>
      <div className="flex items-center gap-2 font-mono text-[11px] text-ns-text-muted">
        <span aria-hidden="true" className="size-1.5 animate-[ns-blink_1.2s_ease-in-out_infinite] rounded-full bg-ns-accent" />
        <span>{label}…</span>
        {elapsed !== null ? <span className="ns-tabular text-ns-text-faint">{fmtSeconds(elapsed)}</span> : null}
      </div>
      <div aria-hidden="true" className="ns-loading-bar h-px w-full overflow-hidden bg-ns-border" />
      {hint ? <p className="text-[11px] leading-snug text-ns-text-faint">{hint}</p> : null}
    </div>
  );
}

export function ProgressBar({ value, tone = "accent", className }: { value: number; tone?: "accent" | "causal"; className?: string }) {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100);
  return (
    <div
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct}
      className={cn("h-1 w-full overflow-hidden rounded-full bg-ns-border", className)}
    >
      <div
        className={cn("h-full rounded-full transition-[width] duration-300", tone === "causal" ? "bg-ns-causal" : "bg-ns-accent")}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

export function ComingSoon({ label, note, className }: { label: string; note?: string; className?: string }) {
  return (
    <div
      className={cn(
        "ns-hatch flex items-center justify-between gap-2 rounded-[var(--ns-radius)] border border-dashed border-ns-border-strong px-2.5 py-1.5",
        className,
      )}
    >
      <span className="text-[11px] text-ns-text-muted">
        {label}
        {note ? <span className="text-ns-text-faint"> · {note}</span> : null}
      </span>
      <StatusChip tone="future">Coming soon</StatusChip>
    </div>
  );
}

/** Key/value readout (mono, tabular). */
export function KV({ k, v, title, className }: { k: ReactNode; v: ReactNode; title?: string; className?: string }) {
  return (
    <div className={cn("flex min-w-0 items-baseline justify-between gap-3 py-0.5", className)} title={title}>
      <dt className="shrink-0 text-[11px] text-ns-text-muted">{k}</dt>
      <dd className="ns-tabular min-w-0 truncate text-right font-mono text-[11px] text-ns-text">{v}</dd>
    </div>
  );
}

/** Stat tile: small label over a mono value. */
export function Stat({
  label,
  value,
  sub,
  tone,
  className,
}: {
  label: ReactNode;
  value: ReactNode;
  sub?: ReactNode;
  tone?: "ok" | "danger" | "warn" | "accent";
  className?: string;
}) {
  return (
    <div className={cn("min-w-0 rounded-[var(--ns-radius)] border border-ns-border bg-ns-bg/60 px-2.5 py-2", className)}>
      <div className="ns-label truncate text-[9.5px]">{label}</div>
      <div
        className={cn(
          "mt-1.5 truncate font-mono text-[15px] leading-none",
          tone === "ok" && "text-ns-ok",
          tone === "danger" && "text-ns-danger",
          tone === "warn" && "text-ns-warn",
          tone === "accent" && "text-ns-accent",
          !tone && "text-ns-text",
        )}
      >
        {value}
      </div>
      {sub ? <div className="mt-1 truncate font-mono text-[10.5px] text-ns-text-muted">{sub}</div> : null}
    </div>
  );
}

const FIELD =
  "h-7 min-w-0 rounded-[3px] border border-ns-border-strong bg-ns-bg px-2 font-mono text-[11.5px] text-ns-text " +
  "placeholder:text-ns-text-faint focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-ns-accent disabled:opacity-50";

/** Fields are full-width unless the caller sets an explicit width class. */
function widthCls(className?: string): string | null {
  return className && /(^|\s)(w-|flex-1)/.test(className) ? null : "w-full";
}

export function TextInput({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return <input type="text" className={cn(FIELD, widthCls(className), className)} {...rest} />;
}

export function NumberInput({
  value,
  onValue,
  min,
  max,
  step,
  className,
  ...rest
}: Omit<InputHTMLAttributes<HTMLInputElement>, "value" | "onChange" | "type"> & {
  value: number;
  onValue: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <input
      type="number"
      inputMode="decimal"
      className={cn(FIELD, widthCls(className), "ns-tabular pr-1", className)}
      value={Number.isFinite(value) ? value : ""}
      min={min}
      max={max}
      step={step}
      onChange={(e) => {
        const v = e.target.valueAsNumber;
        if (Number.isFinite(v)) onValue(v);
      }}
      {...rest}
    />
  );
}

export function SelectInput({ className, children, ...rest }: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select className={cn(FIELD, widthCls(className), "cursor-pointer pr-6", className)} {...rest}>
      {children}
    </select>
  );
}

/** Labelled field wrapper. */
export function Field({ label, children, className, hint }: { label: ReactNode; children: ReactNode; className?: string; hint?: ReactNode }) {
  return (
    <label className={cn("flex min-w-0 flex-col gap-1", className)}>
      <span className="ns-label text-[9.5px]">{label}</span>
      {children}
      {hint ? <span className="text-[10.5px] leading-snug text-ns-text-faint">{hint}</span> : null}
    </label>
  );
}

export function Switch({
  checked,
  onChange,
  label,
  className,
  title,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  label: ReactNode;
  className?: string;
  title?: string;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      title={title}
      className={cn(
        "inline-flex h-6 items-center gap-1.5 whitespace-nowrap rounded-[3px] px-1 font-mono text-[10.5px] uppercase tracking-[0.08em] text-ns-text-muted hover:text-ns-text",
        className,
      )}
    >
      <span
        aria-hidden="true"
        className={cn(
          "relative inline-block h-3 w-5 rounded-full border transition-colors",
          checked ? "border-ns-accent bg-ns-accent/30" : "border-ns-border-strong bg-ns-bg",
        )}
      >
        <span
          className={cn(
            "absolute top-1/2 size-2 -translate-y-1/2 rounded-full transition-[left]",
            checked ? "left-[9px] bg-ns-accent" : "left-[1px] bg-ns-text-faint",
          )}
        />
      </span>
      {label}
    </button>
  );
}

/** A token rendered with visible whitespace; chat specials as ⟨name⟩ chips. */
export function Tok({ text, className, title }: { text: string; className?: string; title?: string }) {
  const sp = specialName(text);
  if (sp) {
    return (
      <span
        title={title ?? text}
        className={cn(
          "inline-flex items-center rounded-[3px] border border-ns-border-strong bg-ns-panel-2 px-1 font-mono text-[10px] leading-4 text-ns-text-muted",
          className,
        )}
      >
        ⟨{sp}⟩
      </span>
    );
  }
  return (
    <span title={title} className={cn("whitespace-pre font-mono", className)}>
      {visibleToken(text)}
    </span>
  );
}

/** Free text (e.g. generated output) with chat specials rendered as ⟨name⟩ chips. */
export function OutputText({ text, className, max }: { text: string; className?: string; max?: number }) {
  const parts = splitSpecials(max && text.length > max ? `${text.slice(0, max)}…` : text);
  return (
    <span className={cn("font-mono", className)}>
      {parts.map((p, i) =>
        p.special ? (
          <span key={i} className="mx-px inline-flex rounded-[2px] border border-ns-border-strong bg-ns-panel-2 px-0.5 align-middle text-[9px] leading-[13px] text-ns-text-muted">
            {visibleToken(p.text)}
          </span>
        ) : (
          <span key={i} className="whitespace-pre">
            {p.text.replace(/\n/g, "↵")}
          </span>
        ),
      )}
    </span>
  );
}

/** Empty state for panels that need a run. */
export function EmptyState({ title, children, className }: { title: string; children?: ReactNode; className?: string }) {
  return (
    <div className={cn("flex flex-col gap-1.5 px-1 py-6 text-center", className)}>
      <p className="font-mono text-[11px] uppercase tracking-[0.1em] text-ns-text-muted">{title}</p>
      {children ? <div className="text-xs leading-relaxed text-ns-text-faint">{children}</div> : null}
    </div>
  );
}

/** Small segmented control (radio group). */
export function Segmented<T extends string>({
  value,
  onChange,
  items,
  label,
  className,
  size = "sm",
}: {
  value: T;
  onChange: (v: T) => void;
  items: readonly { value: T; label: ReactNode; title?: string; disabled?: boolean }[];
  label: string;
  className?: string;
  size?: "xs" | "sm";
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={cn("inline-flex items-stretch gap-0.5 rounded-[3px] border border-ns-border bg-ns-bg p-0.5", className)}
    >
      {items.map((it) => {
        const on = it.value === value;
        return (
          <button
            key={it.value}
            type="button"
            role="radio"
            aria-checked={on}
            title={it.title}
            disabled={it.disabled}
            onClick={() => onChange(it.value)}
            className={cn(
              "whitespace-nowrap rounded-[2px] font-mono uppercase tracking-[0.08em] transition-colors disabled:cursor-not-allowed disabled:opacity-40",
              size === "xs" ? "h-5 px-1.5 text-[9.5px]" : "h-6 px-2 text-[10px]",
              on ? "bg-ns-panel-2 text-ns-text shadow-[inset_0_0_0_1px_var(--ns-border-strong)]" : "text-ns-text-muted hover:text-ns-text",
            )}
          >
            {it.label}
          </button>
        );
      })}
    </div>
  );
}
