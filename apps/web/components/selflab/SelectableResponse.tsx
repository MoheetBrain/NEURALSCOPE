"use client";

import { useCallback, useRef } from "react";
import { TextSelect, X } from "lucide-react";
import { cn } from "@/lib/cn";
import { NumberInput } from "@/components/workspace/ui";
import { CHAR_START_ATTR, chunkText, clampSpan, selectionToSpan, showBlank, type CharSpan } from "./spanSelection";

export interface SelectableResponseProps {
  text: string;
  /** The span currently selected for investigation (controlled). */
  value: CharSpan | null;
  onChange: (span: CharSpan | null) => void;
  /** An existing claim to mark (e.g. the open case's claim). */
  claim?: { char_start: number; char_end: number } | null;
  label?: string;
  className?: string;
}

/**
 * The target's response, rendered verbatim (no whitespace or special-token substitution) so DOM text
 * offsets equal response offsets. Mouse/touch selection is read on release; keyboard users set offsets
 * with the numeric fields.
 */
export function SelectableResponse({ text, value, onChange, claim, label = "Target response", className }: SelectableResponseProps) {
  const ref = useRef<HTMLDivElement>(null);

  const capture = useCallback(() => {
    const el = ref.current;
    if (!el || typeof window === "undefined") return;
    const span = selectionToSpan(el, window.getSelection(), text);
    if (span) onChange(span);
  }, [text, onChange]);

  const marked = value ?? claim ?? null;
  const blank = text.length > 0 && !text.trim();
  const pieces = chunkText(text, marked);
  const markIsClaim = !value && Boolean(claim);

  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <div
        ref={ref}
        role="document"
        aria-label={`${label} (select a span to investigate)`}
        data-testid="selectable-response"
        onMouseUp={capture}
        onTouchEnd={capture}
        onKeyUp={capture}
        className="max-h-56 cursor-text select-text overflow-auto whitespace-pre-wrap break-words rounded-[3px] border border-ns-border bg-ns-bg px-2.5 py-2 font-mono text-[12.5px] leading-relaxed text-ns-text"
      >
        {pieces.length === 0 ? <span className="select-none text-ns-text-faint">∅ (empty response)</span> : null}
        {pieces.map((p) => (
          <span
            key={`${p.start}-${p.text.length}-${p.marked}`}
            {...{ [CHAR_START_ATTR]: p.start }}
            data-marked={p.marked ? (markIsClaim ? "claim" : "selection") : undefined}
            className={cn(
              p.marked && !markIsClaim && "rounded-[2px] bg-ns-accent/25 text-ns-text shadow-[inset_0_-1px_0_var(--ns-accent)]",
              p.marked && markIsClaim && "rounded-[2px] bg-ns-accent-2/20 shadow-[inset_0_-1px_0_var(--ns-accent-2)]",
            )}
          >
            {p.text}
          </span>
        ))}
      </div>
      <div className="flex flex-wrap items-end gap-2">
        <label className="flex items-center gap-1 font-mono text-[10px] text-ns-text-muted">
          start
          <NumberInput
            aria-label="Claim start offset"
            className="w-16"
            value={value?.char_start ?? 0}
            min={0}
            max={text.length}
            onValue={(v) => onChange(clampSpan(text, v, value?.char_end ?? text.length, false))}
          />
        </label>
        <label className="flex items-center gap-1 font-mono text-[10px] text-ns-text-muted">
          end
          <NumberInput
            aria-label="Claim end offset"
            className="w-16"
            value={value?.char_end ?? 0}
            min={0}
            max={text.length}
            onValue={(v) => onChange(clampSpan(text, value?.char_start ?? 0, v, false))}
          />
        </label>
        <button
          type="button"
          onClick={() => onChange(clampSpan(text, 0, text.length))}
          disabled={!text}
          className="inline-flex h-7 items-center gap-1 rounded-[3px] border border-ns-border-strong px-2 font-mono text-[10px] uppercase tracking-[0.08em] text-ns-text-muted hover:text-ns-text disabled:opacity-40"
        >
          <TextSelect aria-hidden="true" className="size-3" /> whole response
        </button>
        {value ? (
          <button
            type="button"
            onClick={() => onChange(null)}
            className="inline-flex h-7 items-center gap-1 px-1 font-mono text-[10px] uppercase tracking-[0.08em] text-ns-text-faint hover:text-ns-text"
          >
            <X aria-hidden="true" className="size-3" /> clear
          </button>
        ) : null}
      </div>
      {blank ? (
        <p className="text-[10.5px] leading-snug text-ns-warn" data-testid="blank-response">
          The response is whitespace only ({text.length} chars: “{showBlank(text.length > 40 ? `${text.slice(0, 40)}…` : text)}”). A span of it can
          still be audited; the case will note that the claim is blank.
        </p>
      ) : null}
      <p className="font-mono text-[10.5px] text-ns-text-muted" aria-live="polite" data-testid="selection-readout">
        {value ? (
          <>
            chars [{value.char_start}, {value.char_end}) · {value.char_end - value.char_start} chars · “
            <span className="text-ns-text">{showBlank(value.text.length > 80 ? `${value.text.slice(0, 80)}…` : value.text)}</span>”
          </>
        ) : (
          <span className="text-ns-text-faint">Select a span of the response with the mouse, or set start/end offsets.</span>
        )}
      </p>
    </div>
  );
}
