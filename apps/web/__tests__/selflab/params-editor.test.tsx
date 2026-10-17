import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ParamsEditor, parseFieldText, resolveField } from "@/components/selflab/ParamsEditor";

// Excerpt of the source_influence_case params schema as served by GET /api/selflab/templates.
const SCHEMA = {
  properties: {
    replacement_text: { anyOf: [{ maxLength: 8000, type: "string" }, { type: "null" }], default: null, description: "document text for the document_replaced branch" },
    replacement_value: { anyOf: [{ maxLength: 200, type: "string" }, { type: "null" }], default: null },
    value_set: { anyOf: [{ items: { type: "string" }, type: "array" }, { type: "null" }], default: null },
    report_max_new_tokens: { default: 24, maximum: 24, minimum: 1, type: "integer" },
    max_new_tokens: { anyOf: [{ maximum: 256, minimum: 1, type: "integer" }, { type: "null" }], default: null },
    patch: { default: true, type: "boolean" },
  },
};

describe("params schema resolution", () => {
  it("resolves nullable unions, long strings, bounds and arrays", () => {
    expect(resolveField(SCHEMA.properties.replacement_text)).toMatchObject({ kind: "string", nullable: true, long: true });
    expect(resolveField(SCHEMA.properties.replacement_value)).toMatchObject({ kind: "string", nullable: true, long: false });
    expect(resolveField(SCHEMA.properties.report_max_new_tokens)).toMatchObject({ kind: "integer", nullable: false, min: 1, max: 24 });
    expect(resolveField(SCHEMA.properties.value_set)).toMatchObject({ kind: "array", nullable: true, itemType: "string" });
    expect(resolveField(SCHEMA.properties.patch)).toMatchObject({ kind: "boolean" });
    expect(resolveField({ enum: ["F1", "F2"] })).toMatchObject({ kind: "enum", options: ["F1", "F2"] });
  });

  it("parses text without inventing values", () => {
    const int = resolveField(SCHEMA.properties.report_max_new_tokens);
    expect(parseFieldText(int, "12")).toBe(12);
    expect(parseFieldText(int, "99")).toBeUndefined(); // above the schema maximum: rejected, not clamped
    expect(parseFieldText(int, "1.5")).toBeUndefined();
    expect(parseFieldText(int, "")).toBeUndefined(); // not nullable: keep the previous value
    expect(parseFieldText(resolveField(SCHEMA.properties.max_new_tokens), "")).toBeNull();
    expect(parseFieldText(resolveField(SCHEMA.properties.replacement_value), "  ")).toBeNull();
    expect(parseFieldText(resolveField(SCHEMA.properties.value_set), "green, ochre ,")).toEqual(["green", "ochre"]);
  });
});

describe("<ParamsEditor>", () => {
  it("edits params through the schema and reports each change", () => {
    const onChange = vi.fn();
    const params = { replacement_text: null, replacement_value: null, value_set: null, report_max_new_tokens: 24, max_new_tokens: null, patch: true };
    render(<ParamsEditor schema={SCHEMA} params={params} onChange={onChange} />);
    fireEvent.change(screen.getByLabelText("replacement_text"), { target: { value: "Its eggs are speckled ochre." } });
    expect(onChange).toHaveBeenLastCalledWith({ ...params, replacement_text: "Its eggs are speckled ochre." });
    fireEvent.change(screen.getByLabelText("report_max_new_tokens"), { target: { value: "200" } });
    expect(screen.getByLabelText("report_max_new_tokens")).toHaveAttribute("aria-invalid", "true");
    expect(onChange).toHaveBeenCalledTimes(1);
    fireEvent.click(screen.getByLabelText("patch"));
    expect(onChange).toHaveBeenLastCalledWith({ ...params, patch: false });
    expect(screen.getByText("document text for the document_replaced branch")).toBeInTheDocument();
  });
});

import { STATIC_EXAMPLE_PROMPT, caseDocumentText, prefillFromExample } from "@/components/selflab/example";
import type { Case } from "@/lib/api/selflab";

describe("example prefill for the case template (D-UI1)", () => {
  const content = `Document:\n\n${STATIC_EXAMPLE_PROMPT.document}\n\n${STATIC_EXAMPLE_PROMPT.question}`;
  const start = content.indexOf(STATIC_EXAMPLE_PROMPT.document);
  const exampleCase = {
    id: "case_1",
    messages: [{ role: "user", content }],
    segments: [{ kind: "document", message_index: 0, char_start: start, char_end: start + STATIC_EXAMPLE_PROMPT.document.length, sha256: "x" }],
  } as unknown as Case;

  it("keeps the replacement one word away from the original document", () => {
    const a = STATIC_EXAMPLE_PROMPT.document.split(" ");
    const b = STATIC_EXAMPLE_PROMPT.replacement_text.split(" ");
    expect(a).toHaveLength(b.length);
    expect(a.filter((w, i) => w !== b[i])).toEqual(["green."]);
    expect(STATIC_EXAMPLE_PROMPT.hint).toBe("");
  });

  it("fills empty counterfactual params only when the case document is the example's", () => {
    expect(caseDocumentText(exampleCase)).toBe(STATIC_EXAMPLE_PROMPT.document);
    const spec = { template: "source_influence_case" as const, params: { replacement_text: null, hint_text: "kept", report_family: "F1" } };
    const out = prefillFromExample(spec, exampleCase, STATIC_EXAMPLE_PROMPT);
    expect(out.applied).toEqual(["replacement_text", "original_value", "replacement_value", "value_set"]);
    expect(out.spec.params).toMatchObject({ replacement_text: STATIC_EXAMPLE_PROMPT.replacement_text, hint_text: "kept", report_family: "F1" });
    // other documents, other templates, or no example: untouched
    const edited = { ...exampleCase, messages: [{ role: "user", content: content.replace("green", "teal") }] } as unknown as Case;
    expect(prefillFromExample(spec, edited, STATIC_EXAMPLE_PROMPT).applied).toEqual([]);
    expect(prefillFromExample({ ...spec, template: "csp1_carry" }, exampleCase, STATIC_EXAMPLE_PROMPT).applied).toEqual([]);
    expect(prefillFromExample(spec, exampleCase, null).spec).toBe(spec);
  });
});
