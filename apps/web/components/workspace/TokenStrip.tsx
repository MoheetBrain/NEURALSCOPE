"use client";

import { useMemo, useState } from "react";
import type { TokenInfo } from "@/lib/api/types";
import { cn } from "@/lib/cn";
import { fmtNum, fmtProb, fmtRank, specialName, visibleToken } from "@/lib/format";
import { StatusBadge, Tok } from "./ui";

interface Hover {
  token: TokenInfo;
  left: number;
  top: number;
}

/**
 * Every token of the run as a chip: template tokens dimmed, user tokens normal,
 * generated tokens in accent, chat specials as ⟨name⟩ chips, whitespace visible.
 * Hover / focus shows the id, the next-token log-prob and the top-5 predictions.
 */
export function TokenStrip({
  tokens,
  selected,
  target,
  hideTemplate,
  onToggleTemplate,
  onSelect,
}: {
  tokens: readonly TokenInfo[];
  selected: number | null;
  target: number | null;
  hideTemplate: boolean;
  onToggleTemplate: () => void;
  onSelect: (position: number) => void;
}) {
  const [hover, setHover] = useState<Hover | null>(null);
  const firstUser = useMemo(() => tokens.findIndex((t) => t.role !== "template"), [tokens]);
  const hiddenCount = hideTemplate && firstUser > 0 ? firstUser : 0;

  const show = (t: TokenInfo, el: HTMLElement) => {
    const box = el.closest("[data-token-strip]")?.getBoundingClientRect();
    const r = el.getBoundingClientRect();
    if (!box) return;
    setHover({ token: t, left: Math.min(r.left - box.left, box.width - 230), top: r.bottom - box.top + 4 });
  };

  return (
    <div data-token-strip className="relative">
      <div className="flex flex-wrap items-center gap-[3px]" role="listbox" aria-label="Tokens (click to select a position)">
        {hiddenCount > 0 ? (
          <button
            type="button"
            onClick={onToggleTemplate}
            className="h-5 rounded-[3px] border border-dashed border-ns-border-strong px-1.5 font-mono text-[10px] text-ns-text-faint hover:text-ns-text"
            title="System-prompt template tokens are collapsed. Click to show them."
          >
            ⋯ {hiddenCount} template
          </button>
        ) : null}
        {tokens.map((t) => {
          if (t.index < hiddenCount && t.index !== target) return null;
          const sp = specialName(t.text);
          const isSel = selected === t.index;
          const isTarget = target === t.index;
          return (
            <button
              key={t.index}
              type="button"
              role="option"
              aria-selected={isSel}
              aria-label={`t${t.index} ${visibleToken(t.text)} (${t.role})`}
              onClick={() => onSelect(t.index)}
              onMouseEnter={(e) => show(t, e.currentTarget)}
              onFocus={(e) => show(t, e.currentTarget)}
              onMouseLeave={() => setHover(null)}
              onBlur={() => setHover(null)}
              className={cn(
                "relative inline-flex h-5 max-w-[140px] items-center rounded-[3px] border px-1 font-mono text-[11px] leading-none transition-colors",
                t.role === "template" && "border-transparent bg-transparent text-ns-text-faint hover:border-ns-border-strong",
                t.role === "user" && "border-ns-border bg-ns-panel-2 text-ns-text hover:border-ns-text-faint",
                t.role === "generated" && "border-ns-accent/40 bg-ns-accent/10 text-ns-accent hover:border-ns-accent",
                sp && "text-[9.5px]",
                isSel && "outline outline-1 outline-offset-1 outline-ns-accent",
              )}
            >
              <span className="truncate whitespace-pre">{sp ? `⟨${sp}⟩` : visibleToken(t.text)}</span>
              {isTarget ? (
                <span aria-hidden="true" className="absolute -bottom-[3px] left-0.5 right-0.5 h-[2px] rounded-full bg-ns-accent-2" />
              ) : null}
            </button>
          );
        })}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[9.5px] text-ns-text-faint">
        <span className="inline-flex items-center gap-1">
          <span className="inline-block size-2 rounded-[2px] border border-ns-border-strong" /> template
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="inline-block size-2 rounded-[2px] bg-ns-panel-2 ring-1 ring-ns-border" /> user
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="inline-block size-2 rounded-[2px] bg-ns-accent/40" /> generated
        </span>
        <span className="inline-flex items-center gap-1">
          <span className="inline-block h-0.5 w-2.5 rounded-full bg-ns-accent-2" /> target position
        </span>
        <span>· = space ↵ = newline</span>
      </div>
      {hover ? <TokenTooltip hover={hover} next={tokens[hover.token.index + 1]?.text ?? null} /> : null}
    </div>
  );
}

function TokenTooltip({ hover, next }: { hover: Hover; next: string | null }) {
  const t = hover.token;
  return (
    <div
      role="tooltip"
      className="pointer-events-none absolute z-30 w-[230px] rounded-[3px] border border-ns-border-strong bg-ns-bg/97 p-2 shadow-xl"
      style={{ left: Math.max(0, hover.left), top: hover.top }}
    >
      <div className="flex items-center gap-2">
        <Tok text={t.text} className="text-[12px] text-ns-text" />
        <span className="ml-auto font-mono text-[10px] text-ns-text-faint">
          t{t.index} · id {t.id}
        </span>
      </div>
      <div className="mt-1 flex items-center gap-1.5 font-mono text-[10px] text-ns-text-muted">
        <span>{t.role}</span>
        {t.special ? <span>· special</span> : null}
        <StatusBadge status="observed" className="ml-auto" />
      </div>
      {t.next_logprob !== undefined && t.next_logprob !== null ? (
        <div className="mt-1.5 border-t border-ns-border pt-1.5 font-mono text-[10.5px] text-ns-text">
          log P(next = <Tok text={next ?? ""} className="text-ns-accent-2" />) {fmtNum(t.next_logprob, 4)} · rank {fmtRank(t.next_rank)}
        </div>
      ) : (
        <div className="mt-1.5 border-t border-ns-border pt-1.5 font-mono text-[10.5px] text-ns-text-faint">last token (no next token)</div>
      )}
      {t.top?.length ? (
        <div className="mt-1.5">
          <div className="ns-label mb-1 text-[9px]">Top-5 next (final layer)</div>
          {t.top.map((c) => (
            <div key={c.id} className="flex items-center justify-between gap-2 font-mono text-[10.5px]">
              <Tok text={c.text} className="truncate text-ns-text" />
              <span className="ns-tabular text-ns-text-muted">{fmtProb(c.logprob === null ? null : Math.exp(c.logprob))}</span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
