"use client";

import { useRef, type KeyboardEvent, type ReactNode } from "react";
import type { EpistemicStatus } from "@/lib/epistemic";
import { cn } from "@/lib/cn";

export interface TabItem<T extends string = string> {
  value: T;
  label: ReactNode;
  /** Optional coloured dot marking the epistemic status of the tab's view. */
  status?: EpistemicStatus;
  disabled?: boolean;
}

export interface TabsProps<T extends string = string> {
  items: readonly TabItem<T>[];
  value: T;
  onValueChange: (value: T) => void;
  /** Accessible name for the tab list. */
  "aria-label": string;
  /** Prefix used to derive tab / panel ids. Pair with <TabPanel idBase=...>. */
  idBase?: string;
  variant?: "underline" | "segmented";
  size?: "sm" | "md";
  className?: string;
}

const DOT: Record<EpistemicStatus, string> = {
  observed: "bg-ns-observed",
  derived: "bg-ns-derived",
  inferred: "bg-ns-inferred",
  causal: "bg-ns-causal",
};

export const tabId = (idBase: string, value: string) => `${idBase}-tab-${value}`;
export const tabPanelId = (idBase: string, value: string) => `${idBase}-panel-${value}`;

/**
 * Controlled tab list (WAI-ARIA tabs pattern, automatic activation).
 * Arrow keys / Home / End move between enabled tabs.
 */
export function Tabs<T extends string = string>({
  items,
  value,
  onValueChange,
  idBase = "ns-tabs",
  variant = "underline",
  size = "md",
  className,
  ...aria
}: TabsProps<T>) {
  const refs = useRef<Array<HTMLButtonElement | null>>([]);

  const enabled = items.map((it, i) => ({ it, i })).filter(({ it }) => !it.disabled);

  function onKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    if (enabled.length === 0) return;
    const pos = enabled.findIndex(({ it }) => it.value === value);
    let next = -1;
    switch (e.key) {
      case "ArrowRight":
      case "ArrowDown":
        next = (pos + 1) % enabled.length;
        break;
      case "ArrowLeft":
      case "ArrowUp":
        next = (pos - 1 + enabled.length) % enabled.length;
        break;
      case "Home":
        next = 0;
        break;
      case "End":
        next = enabled.length - 1;
        break;
      default:
        return;
    }
    e.preventDefault();
    const target = enabled[next];
    onValueChange(target.it.value);
    refs.current[target.i]?.focus();
  }

  return (
    <div
      role="tablist"
      aria-label={aria["aria-label"]}
      onKeyDown={onKeyDown}
      className={cn(
        "flex min-w-0 items-stretch",
        variant === "underline" && "gap-4 border-b border-ns-border",
        variant === "segmented" && "gap-0.5 rounded-[var(--ns-radius)] border border-ns-border bg-ns-bg p-0.5",
        className,
      )}
    >
      {items.map((it, i) => {
        const selected = it.value === value;
        return (
          <button
            key={it.value}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="tab"
            id={tabId(idBase, it.value)}
            aria-selected={selected}
            aria-controls={tabPanelId(idBase, it.value)}
            tabIndex={selected ? 0 : -1}
            disabled={it.disabled}
            onClick={() => onValueChange(it.value)}
            className={cn(
              "inline-flex items-center gap-1.5 whitespace-nowrap font-mono uppercase tracking-[0.1em] transition-colors",
              "disabled:cursor-not-allowed disabled:opacity-40",
              size === "sm" ? "text-[10px]" : "text-[11px]",
              variant === "underline" && [
                "-mb-px border-b py-2",
                selected
                  ? "border-ns-accent text-ns-text"
                  : "border-transparent text-ns-text-muted hover:text-ns-text",
              ],
              variant === "segmented" && [
                "rounded-[3px] px-2.5",
                size === "sm" ? "h-6" : "h-7",
                selected ? "bg-ns-panel-2 text-ns-text shadow-[inset_0_0_0_1px_var(--ns-border-strong)]" : "text-ns-text-muted hover:text-ns-text",
              ],
            )}
          >
            {it.status ? <span aria-hidden="true" className={cn("size-1.5 rounded-full", DOT[it.status])} /> : null}
            {it.label}
          </button>
        );
      })}
    </div>
  );
}

export interface TabPanelProps {
  idBase?: string;
  value: string;
  active: boolean;
  className?: string;
  children?: ReactNode;
}

/** Panel paired with a tab. Inactive panels are unmounted-free: they are hidden. */
export function TabPanel({ idBase = "ns-tabs", value, active, className, children }: TabPanelProps) {
  return (
    <div
      role="tabpanel"
      id={tabPanelId(idBase, value)}
      aria-labelledby={tabId(idBase, value)}
      hidden={!active}
      tabIndex={0}
      className={className}
    >
      {children}
    </div>
  );
}
