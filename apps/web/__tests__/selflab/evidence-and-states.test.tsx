import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { availableTabs, ExportPreviewView } from "@/components/selflab/BottomPanel";
import { groupBranches } from "@/components/selflab/CompareContexts";
import { EvidenceCard, HypothesisCard } from "@/components/selflab/EvidenceCard";
import { groupEvidence } from "@/components/selflab/EvidenceColumn";
import { FeatureBadge, FixtureMark } from "@/components/selflab/marks";
import { ParticipantsView } from "@/components/selflab/ParticipantsPanel";
import { predictionAvailability, templateAvailability } from "@/components/selflab/PlanColumn";
import { SuggestionsView } from "@/components/selflab/SuggestPanel";
import { SummaryView } from "@/components/selflab/values";
import { normalizeExportPreview, type Case, type EvidenceItem, type ExperimentRecord, type TemplateInfo, type Trial } from "@/lib/api/selflab";

const ITEM: EvidenceItem = {
  id: "ev_1",
  case_id: "case_1",
  experiment_id: "exp_1",
  kind: "deterministic_derivation",
  status: "supported",
  title: "Removing the document changes the answer",
  statement: "With the document removed the answer changed from 'green' to 'insufficient evidence'; with it replaced the replacement value was given.",
  measurement: { delta_sum_logprob: -4.25, n_tokens: 3, ci: [-5.1, -3.2] },
  scope: {
    checkpoint: "Qwen/Qwen2.5-0.5B-Instruct@abc123",
    task_distribution: "single case",
    intervention: "document removed / replaced",
    unit: "single case",
    n_units: 1,
    n_trials: 4,
    controls: ["replay reproduces the case response"],
    tested_generalization: "none",
    split: "debug_pilot",
  },
  trial_ids: ["exp_1:00000", "exp_1:00001"],
  run_ids: ["run_abc"],
  analysis: "neuralscope.selflab.templates.source_influence:case_status@1",
  supports: ["claim"],
  alternatives: [{ explanation: "Answer is driven by the question wording alone", discriminating_test: "document removed", state: "ruled_out" }],
  limitations: ["n = 1 case; says nothing about other inputs"],
  created_at: "2026-10-17T10:00:00Z",
};

describe("EvidenceCard", () => {
  it("renders kind, status and scope as separate fields and never a combined 'validated' badge", () => {
    render(<EvidenceCard item={ITEM} />);
    const card = screen.getByTestId("evidence-card");
    const q = within(card);
    expect(q.getByText("Kind")).toBeInTheDocument();
    expect(q.getByText("Status")).toBeInTheDocument();
    expect(q.getByText("Scope")).toBeInTheDocument();
    expect(card.querySelector("[data-kind]")?.textContent).toBe("Deterministic derivation");
    expect(card.querySelector("[data-test-status]")?.textContent).toBe("Supported");
    // the kind chip and the status chip are different elements
    expect(card.querySelector("[data-kind]")).not.toBe(card.querySelector("[data-test-status]"));
    expect(q.getByTestId("evidence-scope")).toHaveTextContent("n = 1 single case");
    expect(q.getByTestId("evidence-scope")).toHaveTextContent("4 trials");
    expect(card.textContent).not.toMatch(/validated/i);
    expect(card.textContent).not.toMatch(/complete causal explanation/i);
    expect(q.getByText("Ruled out")).toBeInTheDocument();
    expect(q.getByText(/n = 1 case; says nothing about other inputs/)).toBeInTheDocument();
  });

  it("formats measurement numbers as sent and links trials and Lab runs", () => {
    const onSelectTrial = vi.fn();
    const onOpenRun = vi.fn();
    render(<EvidenceCard item={ITEM} onSelectTrial={onSelectTrial} onOpenRun={onOpenRun} />);
    expect(screen.getByText("delta_sum_logprob")).toBeInTheDocument();
    expect(screen.getByText("−4.250")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "#00001" }));
    expect(onSelectTrial).toHaveBeenCalledWith("exp_1:00001");
    fireEvent.click(screen.getByRole("button", { name: /open run abc in lab/i }));
    expect(onOpenRun).toHaveBeenCalledWith("run_abc");
  });

  it("shows the fixture marking when the evidence comes from fixture data", () => {
    const { rerender } = render(<EvidenceCard item={ITEM} />);
    expect(screen.queryByTestId("fixture-mark")).toBeNull();
    rerender(<EvidenceCard item={ITEM} fixture />);
    expect(screen.getByTestId("fixture-mark")).toHaveTextContent(/demo \/ fixture/i);
  });

  it("marks an unscoped item instead of hiding the gap", () => {
    render(<EvidenceCard item={{ ...ITEM, scope: null, status: "inconclusive" }} />);
    expect(screen.getByText(/no scope recorded/i)).toBeInTheDocument();
    expect(screen.getByText("Inconclusive")).toBeInTheDocument();
  });

  it("keeps hypotheses visually and textually distinct from evidence", () => {
    render(<HypothesisCard h={{ id: "H1", statement: "Removing the document changes the answer", metric: "answer_changed", primary: true }} />);
    const h = screen.getByTestId("hypothesis-card");
    expect(h).toHaveTextContent(/hypothesis · not evidence/i);
    expect(h.className).toMatch(/border-dashed/);
    expect(h.querySelector("[data-test-status]")).toBeNull();
  });

  it("groups evidence into for / against / unresolved by test status", () => {
    const g = groupEvidence([
      ITEM,
      { ...ITEM, id: "b", status: "contradicted" },
      { ...ITEM, id: "c", status: "control_failed" },
      { ...ITEM, id: "d", status: "untested" },
    ]);
    expect(g.for.map((x) => x.id)).toEqual(["ev_1"]);
    expect(g.against.map((x) => x.id)).toEqual(["b"]);
    expect(g.open.map((x) => x.id)).toEqual(["c", "d"]);
  });
});

describe("markings", () => {
  it("labels feature statuses honestly", () => {
    const { rerender } = render(<FeatureBadge status="research_prototype" />);
    expect(screen.getByText("Research prototype")).toBeInTheDocument();
    rerender(<FeatureBadge status="roadmap" />);
    expect(screen.getByText("Roadmap")).toBeInTheDocument();
    rerender(<FeatureBadge status="unsupported" />);
    expect(screen.getByText("Unsupported")).toBeInTheDocument();
    rerender(<FeatureBadge status="api_only_view" />);
    expect(screen.getByText("API-only view")).toBeInTheDocument();
    rerender(<FeatureBadge status={undefined} />);
    expect(screen.getByText("status unknown")).toBeInTheDocument();
  });

  it("renders the fixture mark only when flagged", () => {
    const { rerender } = render(<FixtureMark show={false} />);
    expect(screen.queryByTestId("fixture-mark")).toBeNull();
    rerender(<FixtureMark show />);
    expect(screen.getByTestId("fixture-mark")).toBeInTheDocument();
  });
});

describe("export preview", () => {
  it("lists included fields with sizes and flags free text", () => {
    const preview = normalizeExportPreview({
      fields: [
        { name: "committed_spec", bytes: 1400 },
        { name: "trials", bytes: 52_000, count: 40, free_text: true },
        { name: "evidence", bytes: 3_100, count: 3 },
      ],
      total_bytes: 56_500,
      notes: ["Tensors are referenced by run id, not embedded."],
    });
    render(<ExportPreviewView preview={preview} />);
    const box = screen.getByTestId("export-preview");
    expect(within(box).getByText("committed_spec")).toBeInTheDocument();
    expect(within(box).getByText("1.4 KB")).toBeInTheDocument();
    expect(within(box).getByText("50.8 KB")).toBeInTheDocument();
    const rows = box.querySelectorAll("tr[data-free-text='true']");
    expect(rows).toHaveLength(1);
    expect(rows[0]).toHaveTextContent("trials");
    expect(within(box).getByText(/1 field with free text/)).toBeInTheDocument();
    expect(within(box).getByText(/Tensors are referenced by run id/)).toBeInTheDocument();
  });

  it("shows no tabs before an experiment and only tabs with content afterwards", () => {
    const base: ExperimentRecord = {
      id: "exp_1",
      committed: { spec: { template: "csp1_carry" }, spec_sha256: "a", plan_sha256: "b", n_planned: 10, committed_at: "t" },
      status: "committed",
      progress: 0,
      n_done: 0,
      n_errors: 0,
      created_at: "t",
    };
    expect(availableTabs(null)).toEqual([]);
    expect(availableTabs(base)).toEqual(["export"]);
    expect(availableTabs({ ...base, status: "running", n_done: 3 })).toEqual(["trials", "export"]);
    expect(availableTabs({ ...base, status: "done", n_done: 10, analysis: { primary: { estimate: 0.5 } } })).toEqual(["stats", "trials", "export"]);
  });

  it("renders analysis numbers exactly as reported, with intervals", () => {
    render(
      <SummaryView
        summary={{
          analysis_version: "selflab.stats@1",
          per_condition: {
            doc_v: { n_units: 20, n_obs: 40, estimate: 0.875, ci_low: 0.7, ci_high: 0.95, method: "cluster_bootstrap", k: 35 },
            no_doc: { n_units: 20, n_obs: 40, estimate: null, ci_low: null, ci_high: null, method: "cluster_bootstrap" },
          },
          counts: { malformed: 1, refusal: 0, abstention: 4, error: 0 },
        }}
      />,
    );
    expect(screen.getByText("0.875 [0.700, 0.950] 35/40 20 units")).toBeInTheDocument();
    expect(screen.getByText("— n=40 20 units")).toBeInTheDocument();
    expect(screen.getByText("abstention")).toBeInTheDocument();
  });
});

describe("disabled states", () => {
  it("shows API-only access as disabled unless the status enables it", () => {
    const { rerender } = render(<ParticipantsView status={{}} participants={[]} />);
    expect(screen.getByTestId("access-api_only")).toHaveAttribute("data-enabled", "false");
    expect(screen.getByTestId("access-white_box_local")).toHaveAttribute("data-enabled", "true");
    rerender(<ParticipantsView status={{ access_levels: { api_only: { enabled: true } } }} participants={[]} />);
    expect(screen.getByTestId("access-api_only")).toHaveAttribute("data-enabled", "true");
  });

  it("distinguishes the four participant roles and marks the LLM investigator unsupported", () => {
    render(
      <ParticipantsView
        status={{}}
        participants={[
          { role: "target", id: "Qwen/Qwen2.5-0.5B-Instruct", description: "local", sees: ["messages"] },
          { role: "investigator", id: "rules:template-suggester@1", description: "rules", sees: ["claim", "segments"] },
          { role: "statistical_scorer", id: "neuralscope.selflab.stats@1", description: "stats", sees: ["trials"] },
        ]}
      />,
    );
    for (const role of ["target", "investigator", "activation_decoder", "statistical_scorer"]) {
      expect(screen.getByTestId(`role-${role}`)).toBeInTheDocument();
    }
    expect(within(screen.getByTestId("role-activation_decoder")).getByText(/none reported/)).toBeInTheDocument();
    expect(screen.getByTestId("llm-investigator-unsupported")).toHaveTextContent(/unsupported until approved and configured/i);
  });

  it("keeps the LLM investigator button disabled next to rule-based suggestions", () => {
    const onUse = vi.fn();
    render(
      <SuggestionsView
        res={{
          participant: "rules:template-suggester@1",
          suggestions: [{ template: "source_influence_case", title: "Remove or replace the document", rationale: "the claim names the document" }],
          raw: {},
        }}
        onUse={onUse}
        available={() => true}
      />,
    );
    expect(screen.getByText(/not a language model/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /ask an llm investigator/i })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: /draft/i }));
    expect(onUse).toHaveBeenCalledWith("source_influence_case");
  });

  it("disables TEST PREDICTION when csp1_carry is missing or not runnable", () => {
    const t = (id: string, fs: TemplateInfo["feature_status"]): TemplateInfo => ({ id, title: id, description: "", feature_status: fs, requires_case: false });
    expect(predictionAvailability(undefined).ok).toBe(false);
    expect(predictionAvailability([t("source_influence_case", "research_prototype")])).toMatchObject({ ok: false });
    expect(predictionAvailability([t("source_influence_case", "research_prototype")]).reason).toMatch(/not listed/);
    expect(predictionAvailability([t("csp1_carry", "roadmap")]).ok).toBe(false);
    expect(predictionAvailability([t("csp1_carry", "research_prototype")]).ok).toBe(true);
  });

  it("requires a case with a document for the case-local template", () => {
    const tpl: TemplateInfo = { id: "source_influence_case", title: "x", description: "", feature_status: "research_prototype", requires_case: true };
    const c = { id: "case_1", access_level: "white_box_local", segments: [] } as unknown as Case;
    expect(templateAvailability(tpl, null).ok).toBe(false);
    expect(templateAvailability(tpl, c).reason).toMatch(/document segment/);
    const withDoc = { ...c, segments: [{ kind: "document", message_index: 0, char_start: 0, char_end: 5, sha256: "x" }] } as unknown as Case;
    expect(templateAvailability(tpl, withDoc).ok).toBe(true);
    expect(templateAvailability({ ...tpl, feature_status: "unsupported" }, withDoc).ok).toBe(false);
  });

  it("orders counterfactual branches original → removed → replaced → hint", () => {
    const tr = (index: number, condition: string) => ({ id: `e:${index}`, index, condition }) as unknown as Trial;
    const g = groupBranches([tr(3, "hint_added"), tr(1, "document_removed"), tr(0, "original"), tr(2, "document_replaced"), tr(4, "extra")]);
    expect(g.map((b) => b.condition)).toEqual(["original", "document_removed", "document_replaced", "hint_added", "extra"]);
  });
});

describe("unknown analysis keys render generically (H3 keys, v2 labels)", () => {
  const H3 = { n_units: 18, n_obs: 36, estimate: 0.21, ci_low: -0.04, ci_high: 0.44, method: "cluster_bootstrap", assumptions: ["units independent"] };
  const SUMMARY = {
    analysis_version: "source_influence.analysis@2",
    rules: "v2 (see V2_RULES_TEXT)",
    discrimination_contrast: H3,
    n_visible_not_causal: 7,
    self_minus_transcript_scored: { estimate: null, ci_low: null, ci_high: null, method: "paired", n_units: 0, n_obs: 0 },
    hint_induced_abstention: { count: 3, by_condition: { no_doc_hint: 2, doc_v_hint_conflict: 1 }, trial_ids: ["e:1", "e:2", "e:3"] },
    labels: { document: 10, user_suggestion: 4, other_or_unknown: 2, hint_induced_abstention: 3, undetermined: 1 },
    scored_labels: ["document", "user_suggestion", "other_or_unknown"],
    per_condition: [
      { condition: "doc_v", report_accuracy: { estimate: 0.5, ci_low: 0.3, ci_high: 0.7, method: "wilson", k: 10, n_obs: 20 } },
      { condition: "doc_v", note: "duplicate row names must not break rendering" },
    ],
    deeply: { nested: { further: { still: { deeper: { value: 1 } } } } },
    empty_obj: {},
    empty_list: [],
    matrix: [[0.1, 0.2], [0.3, 0.4]],
    nothing: null,
    flag: false,
  };

  it("renders a v2 battery summary with unfamiliar keys without crashing", () => {
    render(<SummaryView summary={SUMMARY} renderStatus={(st) => <span>{st}</span>} />);
    expect(screen.getByText("discrimination_contrast")).toBeInTheDocument();
    expect(screen.getByText("0.210 [−0.040, 0.440] n=36 18 units")).toBeInTheDocument();
    expect(screen.getByText("n_visible_not_causal")).toBeInTheDocument();
    expect(screen.getByText("hint induced abstention")).toBeInTheDocument();
    expect(screen.getAllByText("other_or_unknown").length).toBeGreaterThanOrEqual(2); // labels table + scored_labels list
    expect(screen.getAllByText("doc_v")).toHaveLength(2);
  });

  it("renders the same keys inside an evidence item's measurement", () => {
    render(
      <EvidenceCard
        item={{
          ...ITEM,
          kind: "model_report",
          status: "inconclusive",
          measurement: {
            discrimination_contrast: H3,
            n_visible_not_causal: 7,
            self_minus_transcript_scored: null,
            hint_induced_abstention: { count: 3, trial_ids: ["e:1"] },
          },
        }}
      />,
    );
    const card = screen.getByTestId("evidence-card");
    expect(within(card).getByText("discrimination_contrast")).toBeInTheDocument();
    expect(within(card).getByText("n_visible_not_causal")).toBeInTheDocument();
    expect(within(card).getByText("hint_induced_abstention.count")).toBeInTheDocument();
    expect(card.textContent).not.toMatch(/validated/i);
  });
});
