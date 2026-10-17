/**
 * Character offsets of a DOM selection relative to a response text.
 *
 * The response is rendered as one or more `<span data-char-start="N">` chunks whose text content is
 * exactly `text.slice(N, N + chunk.length)` (no substitution of whitespace or specials). Decorations
 * without `data-char-start` contribute no characters. Offsets are therefore computed from the chunk an
 * endpoint falls in, never by searching the text, so repeated substrings resolve to the right place.
 */

export const CHAR_START_ATTR = "data-char-start";

export interface CharSpan {
  char_start: number;
  char_end: number;
  text: string;
}

/**
 * Trim surrounding whitespace, unless that would leave nothing: a whitespace-only span is kept as is
 * (the backend accepts it and notes it; the DEMO fixture model, for one, answers with newlines).
 */
function trimSpan(text: string, s: number, e: number): [number, number] {
  let a = s;
  let b = e;
  while (a < b && /\s/.test(text[a])) a++;
  while (b > a && /\s/.test(text[b - 1])) b--;
  return b > a ? [a, b] : [s, e];
}

/** Whitespace made visible for read-outs ("↵" for newlines, "·" for spaces) when a span is blank. */
export function showBlank(text: string): string {
  return text.trim() ? text.replace(/\n/g, "↵") : text.replace(/\n/g, "↵").replace(/ /g, "·").replace(/\t/g, "⇥");
}

function chunkOf(container: Element, node: Node): HTMLElement | null {
  let n: Node | null = node;
  while (n && n !== container) {
    if (n.nodeType === Node.ELEMENT_NODE && (n as Element).hasAttribute(CHAR_START_ATTR)) return n as HTMLElement;
    n = n.parentNode;
  }
  return null;
}

function chunkStart(el: Element): number {
  const v = Number(el.getAttribute(CHAR_START_ATTR));
  return Number.isFinite(v) ? v : 0;
}

function chunks(container: Element): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>(`[${CHAR_START_ATTR}]`));
}

/** Characters of `el` that precede the boundary point (node, offset), counting only text nodes. */
function charsBefore(el: Element, node: Node, offset: number): number {
  const doc = el.ownerDocument;
  const range = doc.createRange();
  range.setStart(el, 0);
  range.setEnd(node, offset);
  return range.toString().length;
}

/**
 * Resolve a boundary point to an offset in the response text, or `null` when it lies outside the
 * container. `textLength` clamps the result.
 */
export function resolveBoundary(container: Element, node: Node, offset: number, textLength: number): number | null {
  if (!container.contains(node)) return null;
  const clamp = (v: number) => Math.min(textLength, Math.max(0, v));
  const chunk = chunkOf(container, node);
  if (chunk) return clamp(chunkStart(chunk) + charsBefore(chunk, node, offset));
  // An element boundary (e.g. the container itself, `offset` = child index): use the next chunk that
  // starts at or after the boundary, else the end of the last chunk before it.
  const all = chunks(container);
  const doc = container.ownerDocument;
  const point = doc.createRange();
  point.setStart(node, offset);
  point.collapse(true);
  for (const c of all) {
    const r = doc.createRange();
    r.selectNode(c);
    // c starts at/after the boundary point
    if (r.compareBoundaryPoints(Range.START_TO_START, point) >= 0) return clamp(chunkStart(c));
  }
  const last = all[all.length - 1];
  return last ? clamp(chunkStart(last) + (last.textContent ?? "").length) : 0;
}

/** Offsets of a Range inside `container` (partially-outside ranges are clipped to the container). */
export function rangeToSpan(container: Element, range: Range, text: string, opts: { trim?: boolean } = {}): CharSpan | null {
  const n = text.length;
  const doc = container.ownerDocument;
  const whole = doc.createRange();
  whole.selectNodeContents(container);
  // clip to the container
  const startsBefore = range.compareBoundaryPoints(Range.START_TO_START, whole) < 0;
  const endsAfter = range.compareBoundaryPoints(Range.END_TO_END, whole) > 0;
  const disjoint =
    range.compareBoundaryPoints(Range.END_TO_START, whole) >= 0 || // range starts after container end
    range.compareBoundaryPoints(Range.START_TO_END, whole) <= 0; // range ends before container start
  if (disjoint) return null;
  const a = startsBefore ? 0 : resolveBoundary(container, range.startContainer, range.startOffset, n);
  const b = endsAfter ? n : resolveBoundary(container, range.endContainer, range.endOffset, n);
  if (a === null || b === null) return null;
  let s = Math.min(a, b);
  let e = Math.max(a, b);
  if (opts.trim !== false) [s, e] = trimSpan(text, s, e);
  if (e <= s) return null;
  return { char_start: s, char_end: e, text: text.slice(s, e) };
}

/**
 * Offsets of the current selection inside `container`. Multi-range selections (Firefox) become the
 * union of their ranges. Backwards selections are normalised. Returns `null` when nothing inside the
 * container is selected.
 */
export function selectionToSpan(container: Element, selection: Selection | null, text: string, opts: { trim?: boolean } = {}): CharSpan | null {
  if (!selection || selection.rangeCount === 0 || selection.isCollapsed) return null;
  let s = Infinity;
  let e = -Infinity;
  for (let i = 0; i < selection.rangeCount; i++) {
    const span = rangeToSpan(container, selection.getRangeAt(i), text, { trim: false });
    if (!span) continue;
    s = Math.min(s, span.char_start);
    e = Math.max(e, span.char_end);
  }
  if (!Number.isFinite(s) || !Number.isFinite(e)) return null;
  return clampSpan(text, s, e, opts.trim !== false);
}

/** Validate / normalise manually entered offsets. */
export function clampSpan(text: string, start: number, end: number, trim = true): CharSpan | null {
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
  let s = Math.max(0, Math.min(text.length, Math.round(Math.min(start, end))));
  let e = Math.max(0, Math.min(text.length, Math.round(Math.max(start, end))));
  if (trim) [s, e] = trimSpan(text, s, e);
  if (e <= s) return null;
  return { char_start: s, char_end: e, text: text.slice(s, e) };
}

/** Split `text` into chunks around `highlight` so the claim can be marked without changing offsets. */
export function chunkText(text: string, highlight?: { char_start: number; char_end: number } | null): { start: number; text: string; marked: boolean }[] {
  if (!highlight || highlight.char_end <= highlight.char_start || highlight.char_start >= text.length) {
    return text ? [{ start: 0, text, marked: false }] : [];
  }
  const s = Math.max(0, highlight.char_start);
  const e = Math.min(text.length, highlight.char_end);
  const out: { start: number; text: string; marked: boolean }[] = [];
  if (s > 0) out.push({ start: 0, text: text.slice(0, s), marked: false });
  out.push({ start: s, text: text.slice(s, e), marked: true });
  if (e < text.length) out.push({ start: e, text: text.slice(e), marked: false });
  return out;
}
