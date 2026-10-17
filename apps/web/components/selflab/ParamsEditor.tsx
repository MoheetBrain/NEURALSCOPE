"use client";

import { useState } from "react";
import { SelectInput, TextInput } from "@/components/workspace/ui";
import { cn } from "@/lib/cn";
import { fmtCell } from "./values";

/** The subset of JSON Schema that template params use (pydantic `model_json_schema`). */
export interface SchemaProp {
  type?: string;
  anyOf?: SchemaProp[];
  default?: unknown;
  description?: string;
  title?: string;
  maxLength?: number;
  minimum?: number;
  maximum?: number;
  enum?: unknown[];
  items?: SchemaProp;
}

export type FieldKind = "string" | "integer" | "number" | "boolean" | "array" | "enum" | "unknown";

export interface ResolvedField {
  kind: FieldKind;
  nullable: boolean;
  itemType?: string;
  options?: string[];
  long?: boolean;
  min?: number;
  max?: number;
}

/** Resolve a (possibly `anyOf [T, null]`) schema property into an input kind. */
export function resolveField(p: SchemaProp): ResolvedField {
  let nullable = false;
  let base: SchemaProp = p;
  if (Array.isArray(p.anyOf)) {
    const nonNull = p.anyOf.filter((x) => x.type !== "null");
    nullable = nonNull.length < p.anyOf.length;
    base = nonNull.length === 1 ? nonNull[0] : { type: "unknown" };
  }
  if (Array.isArray(base.enum) && base.enum.every((x) => typeof x === "string")) return { kind: "enum", nullable, options: base.enum as string[] };
  switch (base.type) {
    case "string":
      return { kind: "string", nullable, long: (base.maxLength ?? 0) > 300 };
    case "integer":
    case "number":
      return { kind: base.type, nullable, min: base.minimum, max: base.maximum };
    case "boolean":
      return { kind: "boolean", nullable };
    case "array":
      return { kind: "array", nullable, itemType: base.items?.type };
    default:
      return { kind: "unknown", nullable };
  }
}

/** Without a schema entry, infer the input from the current value (null stays read-only: its type is unknown). */
export function inferField(v: unknown): ResolvedField {
  if (typeof v === "string") return { kind: "string", nullable: true, long: v.length > 80 };
  if (typeof v === "number") return { kind: Number.isInteger(v) ? "integer" : "number", nullable: false };
  if (typeof v === "boolean") return { kind: "boolean", nullable: false };
  if (Array.isArray(v) && v.every((x) => typeof x === "string")) return { kind: "array", nullable: true, itemType: "string" };
  if (Array.isArray(v) && v.every((x) => typeof x === "number")) return { kind: "array", nullable: true, itemType: "number" };
  return { kind: "unknown", nullable: true };
}

/** Parse a field's text input into a param value; `undefined` means "invalid, keep the previous value". */
export function parseFieldText(f: ResolvedField, text: string): unknown {
  const t = text.trim();
  if (f.kind === "string") return t === "" && f.nullable ? null : text;
  if (f.kind === "integer" || f.kind === "number") {
    if (t === "") return f.nullable ? null : undefined;
    const v = Number(t);
    if (!Number.isFinite(v) || (f.kind === "integer" && !Number.isInteger(v))) return undefined;
    if (f.min !== undefined && v < f.min) return undefined;
    if (f.max !== undefined && v > f.max) return undefined;
    return v;
  }
  if (f.kind === "array") {
    if (t === "") return f.nullable ? null : [];
    const parts = t.split(",").map((x) => x.trim()).filter(Boolean);
    if (f.itemType === "integer" || f.itemType === "number") {
      const nums = parts.map(Number);
      return nums.every((n) => Number.isFinite(n) && (f.itemType !== "integer" || Number.isInteger(n))) ? nums : undefined;
    }
    return parts;
  }
  return undefined;
}

function toText(f: ResolvedField, v: unknown): string {
  if (v === null || v === undefined) return "";
  if (f.kind === "array" && Array.isArray(v)) return v.join(", ");
  return String(v);
}

function TextField({ name, f, value, onValue }: { name: string; f: ResolvedField; value: unknown; onValue: (v: unknown) => void }) {
  const [text, setText] = useState(() => toText(f, value));
  const [bad, setBad] = useState(false);
  const change = (s: string) => {
    setText(s);
    const v = parseFieldText(f, s);
    setBad(v === undefined);
    if (v !== undefined) onValue(v);
  };
  const cls = cn(bad && "border-ns-danger");
  return f.kind === "string" && f.long ? (
    <textarea
      aria-label={name}
      value={text}
      onChange={(e) => change(e.target.value)}
      rows={3}
      className={cn(
        "w-full resize-y rounded-[3px] border border-ns-border-strong bg-ns-bg px-2 py-1 font-mono text-[11px] text-ns-text placeholder:text-ns-text-faint",
        cls,
      )}
      placeholder={f.nullable ? "empty = none" : undefined}
    />
  ) : (
    <TextInput
      aria-label={name}
      aria-invalid={bad || undefined}
      value={text}
      inputMode={f.kind === "integer" || f.kind === "number" ? "decimal" : undefined}
      onChange={(e) => change(e.target.value)}
      placeholder={f.kind === "array" ? "comma-separated" : f.nullable ? "empty = none" : undefined}
      className={cls}
    />
  );
}

/** Editable template params, driven by the template's `params_schema`. Unknown shapes stay read-only. */
export function ParamsEditor({
  schema,
  params,
  onChange,
}: {
  schema: Record<string, unknown> | null | undefined;
  params: Record<string, unknown>;
  onChange: (params: Record<string, unknown>) => void;
}) {
  const props = (schema && typeof schema.properties === "object" && schema.properties) as Record<string, SchemaProp> | undefined;
  const keys = props ? Object.keys(props) : Object.keys(params);
  if (keys.length === 0) return <p className="text-[11px] text-ns-text-faint">This template takes no params.</p>;
  return (
    <div className="grid gap-x-4 gap-y-2 md:grid-cols-2">
      {keys.map((k) => {
        const p = props?.[k] ?? {};
        const value = k in params ? params[k] : p.default;
        const f = props?.[k] ? resolveField(p) : inferField(value);
        const set = (v: unknown) => onChange({ ...params, [k]: v });
        return (
          <div key={k} className={cn("flex min-w-0 flex-col gap-0.5", f.long && "md:col-span-2")}>
            <span className="font-mono text-[10.5px] text-ns-text">{k}</span>
            {f.kind === "boolean" ? (
              <label className="inline-flex items-center gap-1.5 text-[11px] text-ns-text-muted">
                <input type="checkbox" aria-label={k} checked={Boolean(value)} onChange={(e) => set(e.target.checked)} className="accent-[var(--ns-accent)]" />
                {value ? "true" : "false"}
              </label>
            ) : f.kind === "enum" ? (
              <SelectInput aria-label={k} value={String(value ?? "")} onChange={(e) => set(e.target.value)}>
                {(f.options ?? []).map((o) => (
                  <option key={o} value={o}>
                    {o}
                  </option>
                ))}
              </SelectInput>
            ) : f.kind === "unknown" ? (
              <span className="font-mono text-[10.5px] text-ns-text-muted" title="Not editable here">
                {fmtCell(value)}
              </span>
            ) : (
              <TextField name={k} f={f} value={value} onValue={set} />
            )}
            {p.description ? <span className="text-[10px] leading-snug text-ns-text-faint">{p.description}</span> : null}
          </div>
        );
      })}
    </div>
  );
}
