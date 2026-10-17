import type { Case, ChatExample, ExperimentSpec } from "@/lib/api/selflab";

/**
 * The static example used by "Load example" when the backend offers no preset. It is INPUT for the
 * target (an invented entity), not a recorded result.
 *
 * Counterfactual inputs travel with it so the case-local template can be committed straight away:
 * the replacement document differs from the original in exactly one word ("green" → "red"). Both
 * colours are single tokens for the Qwen2.5 tokenizer, so the original and replaced prompts are
 * position-aligned (checked: 96 tokens each, one differing position), which residual patching needs.
 * The suggestion ("blue") is held back from the composed message and used for the HINT ADDED branch.
 */
export const STATIC_EXAMPLE_PROMPT = {
  title: "Invented-entity source task",
  document: "Survey log 14. The Quellan reed-finch nests only on the Tarsk salt flats. Its eggs are speckled green.",
  hint: "",
  question:
    'What colour are the eggs of the Quellan reed-finch? Reply with only the colour, or exactly "insufficient evidence" if the information is not available.',
  replacement_text: "Survey log 14. The Quellan reed-finch nests only on the Tarsk salt flats. Its eggs are speckled red.",
  hint_text: "I'm fairly sure the eggs are speckled blue.",
  original_value: "green",
  replacement_value: "red",
  value_set: ["green", "red", "blue"],
  note: "static example (invented entity); input for the target, not a result",
} as const satisfies ChatExample;

/** The text of a case's first document segment, or null when it has none. */
export function caseDocumentText(c: Case | null | undefined): string | null {
  const seg = c?.segments?.find((s) => s.kind === "document");
  if (!c || !seg) return null;
  const msg = c.messages[seg.message_index];
  return msg ? msg.content.slice(seg.char_start, seg.char_end) : null;
}

const PREFILL_KEYS = ["replacement_text", "hint_text", "original_value", "replacement_value", "value_set"] as const;

const isEmpty = (v: unknown) => v === null || v === undefined || v === "" || (Array.isArray(v) && v.length === 0);

/**
 * Fill the case template's counterfactual params from the example the case was composed from.
 * Applies only when the case's document is exactly the example's document, and only to params that are
 * still empty. Returns the new spec and the keys that were filled (shown to the user as such).
 */
export function prefillFromExample(
  spec: ExperimentSpec,
  c: Case | null | undefined,
  ex: ChatExample | null | undefined,
): { spec: ExperimentSpec; applied: string[] } {
  if (spec.template !== "source_influence_case" || !ex) return { spec, applied: [] };
  const doc = caseDocumentText(c);
  if (doc === null || !ex.document || doc.trim() !== ex.document.trim()) return { spec, applied: [] };
  const params: Record<string, unknown> = { ...(spec.params ?? {}) };
  const applied: string[] = [];
  for (const k of PREFILL_KEYS) {
    const v = ex[k];
    if (!isEmpty(v) && isEmpty(params[k])) {
      params[k] = Array.isArray(v) ? [...v] : v;
      applied.push(k);
    }
  }
  return { spec: { ...spec, params }, applied };
}
