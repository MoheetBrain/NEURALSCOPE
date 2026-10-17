/**
 * Renderers for backend-produced numbers (measurements, analysis summaries, cost).
 * They format what the backend sent and never compute, impute or round-trip new statistics.
 */

import type { ReactNode } from "react";
import { cn } from "@/lib/cn";
import { fmtNum } from "@/lib/format";

type Json = unknown;

const isObj = (v: Json): v is Record<string, Json> => typeof v === "object" && v !== null && !Array.isArray(v);
const isNum = (v: Json): v is number => typeof v === "number" && Number.isFinite(v);
const isScalar = (v: Json) => v === null || ["string", "number", "boolean"].includes(typeof v);

export function fmtScalar(v: Json): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isFinite(v) ? (Number.isInteger(v) && Math.abs(v) < 1e7 ? String(v) : fmtNum(v, 3)) : "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "string") return v;
  return JSON.stringify(v);
}

function ciOf(o: Record<string, Json>): [number, number] | null {
  for (const k of ["ci", "ci95", "ci_95", "interval", "wilson", "bootstrap_ci"]) {
    const v = o[k];
    if (Array.isArray(v) && v.length === 2 && isNum(v[0]) && isNum(v[1])) return [v[0], v[1]];
    if (isObj(v) && isNum(v.low) && isNum(v.high)) return [v.low, v.high];
    if (isObj(v) && isNum(v.lo) && isNum(v.hi)) return [v.lo, v.hi];
  }
  if (isNum(o.ci_low) && isNum(o.ci_high)) return [o.ci_low, o.ci_high];
  if (isNum(o.lo) && isNum(o.hi)) return [o.lo, o.hi];
  return null;
}

const POINT_KEYS = ["estimate", "rate", "value", "mean", "effect", "delta", "p", "proportion", "accuracy"];

/** `{rate, ci:[lo,hi], k, n}`-like objects → "0.620 [0.410, 0.790] · 31/50". */
export function asEstimate(v: Json): string | null {
  if (!isObj(v)) return null;
  const key = POINT_KEYS.find((k) => isNum(v[k]));
  const ci = ciOf(v);
  // selflab.stats summaries always carry `estimate` (null when n = 0) plus `ci_low`/`ci_high`.
  const statsShape = "estimate" in v && ("ci_low" in v || "method" in v);
  if (!key && !ci && !statsShape) return null;
  const parts: string[] = [key ? fmtNum(v[key] as number, 3) : "—"];
  if (ci) parts.push(`[${fmtNum(ci[0], 3)}, ${fmtNum(ci[1], 3)}]`);
  const k = isNum(v.k) ? v.k : isNum(v.count) ? v.count : isNum(v.successes) ? v.successes : null;
  const n = isNum(v.n) ? v.n : isNum(v.n_obs) ? v.n_obs : isNum(v.total) ? v.total : null;
  if (k !== null && n !== null) parts.push(`${k}/${n}`);
  else if (n !== null) parts.push(`n=${n}`);
  if (isNum(v.n_units) && v.n_units !== n) parts.push(`${v.n_units} units`);
  return parts.join(" ");
}

/** ContinuationScore-like objects → "Σ −3.210 · n 4 · mean −0.803". */
export function asScore(v: Json): string | null {
  if (!isObj(v) || !isNum(v.sum_logprob)) return null;
  const parts = [`Σ ${fmtNum(v.sum_logprob, 3)}`];
  if (isNum(v.n_tokens)) parts.push(`n ${v.n_tokens}`);
  if (isNum(v.mean_logprob)) parts.push(`mean ${fmtNum(v.mean_logprob, 3)}`);
  return parts.join(" · ");
}

export function fmtCell(v: Json): string {
  if (isScalar(v)) return fmtScalar(v);
  const score = asScore(v);
  if (score) return score;
  const est = asEstimate(v);
  if (est) return est;
  if (Array.isArray(v)) {
    if (v.length <= 8 && v.every(isScalar)) return `[${v.map(fmtScalar).join(", ")}]`;
    return `array(${v.length})`;
  }
  if (isObj(v)) {
    const flat = flatten(v, 1);
    return flat.length <= 4 ? flat.map(([k, x]) => `${k}=${x}`).join(" · ") : `{${Object.keys(v).length} fields}`;
  }
  return String(v);
}

/** Flatten nested objects into [path, formatted] rows; scores/estimates stay whole. */
export function flatten(v: Json, maxDepth = 3, prefix = ""): [string, string][] {
  const out: [string, string][] = [];
  if (!isObj(v)) return prefix ? [[prefix, fmtCell(v)]] : [];
  for (const [k, x] of Object.entries(v)) {
    const path = prefix ? `${prefix}.${k}` : k;
    if (isObj(x) && !asScore(x) && !asEstimate(x) && maxDepth > 1) out.push(...flatten(x, maxDepth - 1, path));
    else out.push([path, fmtCell(x)]);
  }
  return out;
}

export function ValueTable({ data, className, empty = "none recorded", maxDepth = 3 }: { data: Json; className?: string; empty?: ReactNode; maxDepth?: number }) {
  const rows = flatten(data, maxDepth);
  if (rows.length === 0) return <p className={cn("text-[11px] text-ns-text-faint", className)}>{empty}</p>;
  return (
    <dl className={cn("grid grid-cols-[minmax(0,auto)_minmax(0,1fr)] gap-x-3 gap-y-0.5", className)}>
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="truncate font-mono text-[10.5px] text-ns-text-muted" title={k}>
            {k}
          </dt>
          <dd className="ns-tabular min-w-0 break-words text-right font-mono text-[10.5px] text-ns-text" title={v}>
            {v}
          </dd>
        </div>
      ))}
    </dl>
  );
}

/** A row that is itself one estimate collapses to `{estimate, method}` instead of a dozen raw columns. */
function compactRow(r: Record<string, Json>): Record<string, Json> {
  if (asEstimate(r) !== null) {
    const out: Record<string, Json> = { estimate: r };
    if (typeof r.method === "string") out.method = r.method;
    return out;
  }
  return r;
}

/** Rows for a table-like value: object of objects, or array of objects. `null` if not table-like. */
export function tableRows(v: Json): { rowKey: string; cells: Record<string, Json> }[] | null {
  if (Array.isArray(v) && v.length > 0 && v.every(isObj)) {
    return v.map((row, i) => {
      const r = row as Record<string, Json>;
      const name = ["condition", "name", "observer", "id", "key", "label", "family"].find((k) => typeof r[k] === "string");
      // the field that names the row is shown once, as the row header, not again as a column
      const rest = name ? Object.fromEntries(Object.entries(r).filter(([k]) => k !== name)) : r;
      return { rowKey: name ? String(r[name]) : String(i), cells: compactRow(rest) };
    });
  }
  if (isObj(v)) {
    const vals = Object.values(v);
    if (vals.length > 0 && vals.every(isObj) && !asScore(v) && !asEstimate(v)) {
      return Object.entries(v).map(([k, row]) => ({ rowKey: k, cells: compactRow(row as Record<string, Json>) }));
    }
  }
  return null;
}

export function DataTable({ rows, caption, className }: { rows: { rowKey: string; cells: Record<string, Json> }[]; caption?: ReactNode; className?: string }) {
  const cols: string[] = [];
  for (const r of rows) for (const k of Object.keys(r.cells)) if (!cols.includes(k)) cols.push(k);
  return (
    <div className={cn("overflow-x-auto", className)}>
      <table className="w-full border-collapse text-left font-mono text-[10.5px]">
        {caption ? <caption className="sr-only">{caption}</caption> : null}
        <thead>
          <tr className="border-b border-ns-border text-ns-text-muted">
            <th scope="col" className="py-1 pr-3 font-normal">
              &nbsp;
            </th>
            {cols.map((c) => (
              <th key={c} scope="col" className="whitespace-nowrap py-1 pr-3 font-normal">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={`${r.rowKey}-${i}`} className="border-b border-ns-border/60 align-top">
              <th scope="row" className="whitespace-nowrap py-1 pr-3 font-normal text-ns-text">
                {r.rowKey}
              </th>
              {cols.map((c) => (
                <td key={c} className="ns-tabular whitespace-nowrap py-1 pr-3 text-ns-text">
                  {c in r.cells ? fmtCell(r.cells[c]) : <span className="text-ns-text-faint">—</span>}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const TEST_STATUS_VALUES = ["untested", "supported", "contradicted", "inconclusive", "control_failed"];
const STATUS_KEY = /(^|_)status$/;

/** An analysis summary: scalars as a key/value list, table-like entries as tables, nested objects as sub-sections. */
export function SummaryView({
  summary,
  depth = 0,
  renderStatus,
}: {
  summary: Record<string, Json>;
  depth?: number;
  /** Renders a test-status value (e.g. `claim_status`) as a chip instead of plain text. */
  renderStatus?: (status: string) => ReactNode;
}) {
  const scalars: Record<string, Json> = {};
  const statuses: { key: string; value: string }[] = [];
  const blocks: { key: string; node: ReactNode }[] = [];
  for (const [k, v] of Object.entries(summary)) {
    if (renderStatus && typeof v === "string" && STATUS_KEY.test(k) && TEST_STATUS_VALUES.includes(v)) {
      statuses.push({ key: k, value: v });
      continue;
    }
    if (Array.isArray(v) && v.length > 0 && v.every((x) => typeof x === "string")) {
      blocks.push({
        key: k,
        node: (
          <ul className="list-disc pl-4 text-[11px] leading-snug text-ns-text-muted">
            {(v as string[]).map((x, i) => (
              <li key={i}>{x}</li>
            ))}
          </ul>
        ),
      });
      continue;
    }
    const rows = tableRows(v);
    if (rows) blocks.push({ key: k, node: <DataTable rows={rows} caption={k} /> });
    else if (isObj(v) && !asScore(v) && !asEstimate(v))
      blocks.push({
        key: k,
        node: depth < 2 ? <SummaryView summary={v} depth={depth + 1} renderStatus={renderStatus} /> : <ValueTable data={v} />,
      });
    else scalars[k] = v;
  }
  return (
    <div className={cn("flex flex-col gap-3", depth > 0 && "gap-2 border-l border-ns-border pl-3")}>
      {statuses.map((st) => (
        <div key={st.key} className="flex items-center gap-2">
          <span className="font-mono text-[10.5px] text-ns-text-muted">{st.key}</span>
          {renderStatus?.(st.value)}
        </div>
      ))}
      {Object.keys(scalars).length > 0 ? <ValueTable data={scalars} className="max-w-xl" /> : null}
      {blocks.map((b) => (
        <section key={b.key} aria-label={b.key} className="min-w-0">
          <h4 className="ns-label mb-1 text-[9.5px] text-ns-text-muted">{b.key.replace(/_/g, " ")}</h4>
          {b.node}
        </section>
      ))}
    </div>
  );
}

export function fmtBytes(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}
