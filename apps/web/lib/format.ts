/** Number and token formatting for the workspace (mono, tabular contexts). */

const MINUS = "−";

function withMinus(s: string): string {
  return s.startsWith("-") ? MINUS + s.slice(1) : s;
}

/** Generic number: fixed decimals for mid-range values, scientific for tiny / huge. */
export function fmtNum(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e5 || a < 10 ** -digits) return withMinus(v.toExponential(Math.max(0, digits - 2)).replace("e+", "e"));
  if (a >= 100) return withMinus(v.toFixed(Math.max(0, digits - 2)));
  return withMinus(v.toFixed(digits));
}

/** Probability in [0, 1]: 0.9998, 3.5e-4, <1e-6. */
export function fmtProb(p: number | null | undefined): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return "—";
  if (p === 0) return "<1e-6";
  if (p >= 0.9999995) return "1.000";
  if (p >= 0.001) return p >= 0.995 ? p.toFixed(4) : p.toFixed(3);
  return p.toExponential(1).replace("e-", "e−");
}

export function fmtPct(p: number | null | undefined, digits = 0): string {
  if (p === null || p === undefined || !Number.isFinite(p)) return "—";
  return `${(p * 100).toFixed(digits)}%`;
}

/** Signed delta with a true minus sign: +0.12, −0.9995. */
export function fmtDelta(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  if (v === 0) return "±0";
  const a = Math.abs(v);
  const body = a >= 1e4 || a < 10 ** -digits ? a.toExponential(1) : a.toFixed(digits);
  return `${v > 0 ? "+" : MINUS}${body}`;
}

/** Token rank: 1, 12, 4.9k, 150k. */
export function fmtRank(r: number | null | undefined): string {
  if (r === null || r === undefined || !Number.isFinite(r)) return "—";
  const n = Math.round(r);
  if (n < 1000) return String(n);
  if (n < 10000) return `${(n / 1000).toFixed(1)}k`;
  if (n < 1e6) return `${Math.round(n / 1000)}k`;
  return `${(n / 1e6).toFixed(1)}M`;
}

export function fmtSeconds(s: number | null | undefined): string {
  if (s === null || s === undefined || !Number.isFinite(s)) return "—";
  if (s < 10) return `${s.toFixed(1)} s`;
  if (s < 120) return `${Math.round(s)} s`;
  return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
}

export function fmtInt(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  return Math.round(n).toLocaleString("en-US");
}

/** Params count: 1.54B. */
export function fmtParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  return fmtInt(n);
}

/** Relative time for history entries ("3 min ago"). `now` injected for purity/tests. */
export function fmtAgo(iso: string, now: number): string {
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return iso;
  const s = Math.max(0, (now - t) / 1000);
  if (s < 45) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}

export function fmtTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
}

// ---------------------------------------------------------------- tokens

const SPECIAL_RE = /^<\|([^|>]+)\|>$/;

/** "<|im_end|>" → "im_end" (null if not a chat special token). */
export function specialName(text: string): string | null {
  const m = SPECIAL_RE.exec(text);
  return m ? m[1] : null;
}

/**
 * Make whitespace and special tokens visible: " " → "·", "\n" → "↵", "\t" → "⇥",
 * "<|im_end|>" → "⟨im_end⟩". Empty strings render as "∅".
 */
export function visibleToken(text: string): string {
  if (text === "") return "∅";
  const sp = specialName(text);
  if (sp) return `⟨${sp}⟩`;
  return text.replace(/ /g, "·").replace(/\n/g, "↵").replace(/\t/g, "⇥").replace(/\r/g, "␍");
}

/** Replace special tokens inside generated text with ⟨name⟩ chips-as-text (for plain output rendering). */
export function splitSpecials(text: string): { text: string; special: boolean }[] {
  const out: { text: string; special: boolean }[] = [];
  const re = /<\|[^|>]+\|>/g;
  let last = 0;
  for (let m = re.exec(text); m; m = re.exec(text)) {
    if (m.index > last) out.push({ text: text.slice(last, m.index), special: false });
    out.push({ text: m[0], special: true });
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ text: text.slice(last), special: false });
  return out;
}

/** Truncate for axis labels. */
export function truncate(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, Math.max(1, max - 1))}…`;
}

/** Quote a token for prose: “9”. */
export function quoteToken(text: string | null | undefined): string {
  if (text === null || text === undefined) return "—";
  return `“${visibleToken(text)}”`;
}

/** Character-level diff of two strings: common prefix / changed middle / common suffix. */
export function diffMiddle(a: string, b: string): { prefix: string; a: string; b: string; suffix: string } {
  let p = 0;
  while (p < a.length && p < b.length && a[p] === b[p]) p++;
  let s = 0;
  while (s < a.length - p && s < b.length - p && a[a.length - 1 - s] === b[b.length - 1 - s]) s++;
  return { prefix: a.slice(0, p), a: a.slice(p, a.length - s), b: b.slice(p, b.length - s), suffix: a.slice(a.length - s) };
}

/** "L22" for block index 22; depth label for residual depth (0 → "emb", d → "L{d-1}"). */
export function depthLabel(depth: number): string {
  return depth <= 0 ? "emb" : `L${depth - 1}`;
}

export function depthMeaning(depth: number): string {
  return depth <= 0 ? "token embedding (input to block 0)" : `residual stream after block ${depth - 1}`;
}

export function shortId(id: string | null | undefined): string {
  if (!id) return "—";
  const i = id.indexOf("_");
  return i >= 0 ? id.slice(i + 1, i + 7) : id.slice(0, 6);
}
