import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const push = vi.fn();
vi.mock("next/navigation", () => ({
  usePathname: () => "/app/selflab",
  useRouter: () => ({ push, replace: vi.fn(), prefetch: vi.fn() }),
}));

import { SelfLabView } from "@/components/selflab/SelfLabView";
import { WorkspaceShell } from "@/components/workspace/WorkspaceShell";
import { STATIC_EXAMPLE_PROMPT } from "@/components/selflab/ConversationColumn";
import { viewCache } from "@/lib/api/cache";

const RESPONSE = "The eggs are speckled green, according to the survey log.";
// The static example holds its hint back (it pre-fills the hint_added branch), so the composed message is doc + question.
const USER = `Document:\n\n${STATIC_EXAMPLE_PROMPT.document}\n\n${STATIC_EXAMPLE_PROMPT.question}`;
const DOC_START = USER.indexOf(STATIC_EXAMPLE_PROMPT.document);

// Mirrors selflab.service.status_payload (real backend shape).
const STATUS = {
  schema_version: "selflab-1",
  model: { status: "ready", error: null, provenance: { model_id: "demo/random-qwen2-tiny", tokenizer_id: "demo", dtype: "float32", device: "cpu", is_demo: true } },
  is_fixture: true,
  access_levels: {
    white_box_local: { enabled: true, feature_status: "research_prototype", note: "the locally loaded model" },
    api_only: { enabled: false, feature_status: "unsupported", note: "disabled; set the API-only flag to enable pasted transcripts" },
  },
  participants: [
    { role: "target", id: "demo/random-qwen2-tiny", description: "DEMO fixture target", sees: ["messages"] },
    { role: "statistical_scorer", id: "selflab.stats@1", description: "analysis", sees: ["trials"] },
    { role: "investigator", id: "rules:template-suggester@1", description: "rule-based", sees: ["claim_type"] },
  ],
  disabled_participants: [{ id: "llm-investigator", status: "unsupported", reason: "not configured/approved", role: "investigator", enabled: false }],
  features: { llm_investigator: "unsupported" },
};

const SEGMENTS = [
  { kind: "document", message_index: 0, char_start: DOC_START, char_end: DOC_START + STATIC_EXAMPLE_PROMPT.document.length, token_start: 14, token_end: 40, sha256: "d".repeat(64) },
];
const RECEIPTS = [
  {
    id: "rcpt_1",
    kind: "context_delivery",
    statement: "The document segment (26 tokens) is present in the target input.",
    proves: "the document reached the input",
    does_not_prove: "that the answer used it",
  },
];

const CASE = {
  id: "case_1",
  created_at: "2026-10-17T10:00:00Z",
  title: "eggs colour source",
  access_level: "white_box_local",
  run_id: "run_chat1",
  messages: [{ role: "user", content: USER }],
  response_text: RESPONSE,
  claim: { text: "speckled green", char_start: 13, char_end: 27 },
  claim_type: "source_attribution",
  segments: SEGMENTS,
  receipts: RECEIPTS,
  participants: STATUS.participants,
  is_fixture: true,
};

const SPEC = {
  template: "source_influence_case",
  case_id: null,
  seed: 0,
  split: "debug_pilot",
  hypotheses: [{ id: "H1", statement: "Removing or replacing the document changes the answer", metric: "answer_changed", primary: true }],
  primary_metric: "answer_changed",
  controls: ["original replay reproduces the case response"],
  budget: { max_trials: 8, max_new_tokens: 24, max_seconds: 600, max_prompt_tokens: 768 },
};

const TEMPLATES = [
  {
    id: "source_influence_case",
    title: "Source influence (this case)",
    description: "Counterfactual branches on one case.",
    feature_status: "research_prototype",
    requires_case: true,
    access_levels: ["white_box_local"],
    // excerpt of the real params_schema served by GET /api/selflab/templates
    params_schema: {
      properties: {
        replacement_text: { anyOf: [{ maxLength: 8000, type: "string" }, { type: "null" }], default: null },
        replacement_value: { anyOf: [{ maxLength: 200, type: "string" }, { type: "null" }], default: null },
        original_value: { anyOf: [{ maxLength: 200, type: "string" }, { type: "null" }], default: null },
        value_set: { anyOf: [{ items: { type: "string" }, type: "array" }, { type: "null" }], default: null },
        hint_text: { anyOf: [{ maxLength: 1000, type: "string" }, { type: "null" }], default: null },
        patch: { default: true, type: "boolean" },
      },
    },
    default_spec: SPEC,
  },
];

function record(status: string, n_done: number, extra: Record<string, unknown> = {}) {
  return {
    id: "exp_1",
    case_id: "case_1",
    committed: { spec: { ...SPEC, case_id: "case_1" }, spec_sha256: "5".repeat(64), plan_sha256: "7".repeat(64), n_planned: 4, committed_at: "2026-10-17T10:01:00Z" },
    status,
    progress: n_done / 4,
    n_done,
    n_errors: 0,
    created_at: "2026-10-17T10:01:00Z",
    is_fixture: true,
    ...extra,
  };
}

const EVIDENCE = [
  {
    id: "ev_1",
    case_id: "case_1",
    experiment_id: "exp_1",
    kind: "deterministic_derivation",
    status: "inconclusive",
    title: "Document removal and replacement",
    statement: "Removing the document changed the answer; the replacement value was not followed.",
    scope: { checkpoint: "demo/random-qwen2-tiny", task_distribution: "single case", intervention: "document removed / replaced", unit: "single case", n_units: 1, n_trials: 4 },
    trial_ids: ["exp_1:00001"],
    created_at: "2026-10-17T10:02:00Z",
  },
];

function respond(body: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } }));
}

describe("SelfLab workspace flow", () => {
  const fetchMock = vi.fn();
  let expState = "none";
  beforeEach(() => {
    viewCache.clear();
    push.mockReset();
    expState = "none";
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      const path = url.split("?")[0];
      if (path === "/api/selflab/status") return respond(STATUS);
      if (path === "/api/selflab/cases" && method === "GET") return respond(expState === "none" ? [] : [CASE]);
      if (path === "/api/selflab/templates") return respond(TEMPLATES);
      if (path === "/api/selflab/experiments" && method === "GET") return respond(expState === "none" ? [] : [record(expState, expState === "done" ? 4 : 0)]);
      if (path === "/api/selflab/chat")
        return respond({ run_id: "run_chat1", messages: CASE.messages, response_text: RESPONSE, segments: SEGMENTS, receipts: RECEIPTS, is_fixture: true });
      if (path === "/api/selflab/cases" && method === "POST") return respond(CASE);
      if (path === "/api/selflab/cases/case_1") return respond({ case: CASE, experiments: [], evidence: [] });
      if (path === "/api/selflab/experiments" && method === "POST") {
        expState = "committed";
        return respond(record("committed", 0));
      }
      if (path === "/api/selflab/experiments/exp_1/run") {
        expState = "done";
        return respond(record("queued", 0));
      }
      if (path === "/api/selflab/experiments/exp_1")
        return respond(
          expState === "done"
            ? record("done", 4, { analysis: { summary: { claim_status: "inconclusive", reasons: ["replacement value not followed"] }, complete: true, n_trials_analyzed: 4, n_errors: 0, analysis_version: "si@1" } })
            : record(expState, 0),
        );
      if (path === "/api/selflab/evidence") return respond(expState === "done" ? EVIDENCE : []);
      if (path === "/api/selflab/experiments/exp_1/trials") return respond([]);
      return respond({ detail: `not mocked: ${method} ${url}` }, 404);
    });
  });
  afterEach(() => vi.unstubAllGlobals());

  const bodyOf = (pathname: string, method = "POST") => {
    const hit = fetchMock.mock.calls.find(([u, i]) => (u as string).split("?")[0] === pathname && (i?.method ?? "GET") === method);
    return hit ? JSON.parse(hit[1].body as string) : undefined;
  };

  it("mounts inside the workspace shell with the SelfLab tab active and the DEMO banner", async () => {
    const base = fetchMock.getMockImplementation() as (u: string, i?: RequestInit) => Promise<Response>;
    fetchMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url.startsWith("/api/health"))
        return respond({ status: "ready", model_id: "demo/random-qwen2-tiny", is_demo: true, error: null, trace_level_default: "standard", version: "0.1.0" });
      if (url.startsWith("/api/model"))
        return respond({
          model_id: "demo/random-qwen2-tiny", architecture: "qwen2", n_layers: 2, n_heads: 4, n_kv_heads: 2, d_model: 64, d_head: 16, d_mlp: 128,
          vocab_size: 1000, dtype: "float32", device: "cpu", n_params: 123456, tied_embeddings: true, norm_type: "RMSNorm", is_demo: true,
          capabilities: {}, n_depths: 3, load_seconds: 0.1,
        });
      if (url.startsWith("/api/runs")) return respond([]);
      return base(url, init);
    });
    render(
      <WorkspaceShell>
        <SelfLabView />
      </WorkspaceShell>,
    );
    expect(await screen.findByTestId("selflab-view")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /selflab/i })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: /workspace/i })).not.toHaveAttribute("aria-current");
    expect(screen.getByRole("link", { name: /math lab/i })).toBeInTheDocument();
    expect(screen.queryByLabelText("Open run from history")).toBeNull();
    expect(screen.getByTestId("demo-banner")).toBeInTheDocument();
  });

  it("chats, investigates a selected span, commits and runs an experiment, and shows evidence", async () => {
    render(<SelfLabView />);

    // research-prototype marking and DEMO target marking are visible up front
    expect(await screen.findByText("Research prototype")).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByTestId("fixture-mark").length).toBeGreaterThan(0));
    // API-only is disabled with the backend's reason
    expect(screen.getByRole("radio", { name: /api-only view/i })).toBeDisabled();
    // TEST PREDICTION is disabled because csp1_carry is not listed
    expect(screen.getByRole("button", { name: /test prediction/i })).toBeDisabled();
    expect(screen.getByTestId("test-prediction-help")).toHaveTextContent(/not listed/);

    // static example is labelled as an example prompt, not a result
    fireEvent.click(screen.getByRole("button", { name: /load example/i }));
    expect(screen.getByTestId("example-label")).toHaveTextContent(/not a result/i);
    fireEvent.click(screen.getByRole("button", { name: /send to target/i }));

    const box = await screen.findByTestId("selectable-response");
    expect(box.textContent).toBe(RESPONSE);
    expect(bodyOf("/api/selflab/chat")).toMatchObject({ question: STATIC_EXAMPLE_PROMPT.question, document: STATIC_EXAMPLE_PROMPT.document, hint: null });
    // context segments and the receipt caveat
    expect(screen.getByText(/proves delivery, not causal use/i)).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "Context segments" })).toHaveTextContent("[14, 40)");

    // select "speckled green" and investigate
    const t = box.querySelector("[data-char-start]")?.firstChild as Text;
    const r = document.createRange();
    r.setStart(t, 13);
    r.setEnd(t, 27);
    const sel = window.getSelection() as Selection;
    sel.removeAllRanges();
    sel.addRange(r);
    fireEvent.mouseUp(box);
    fireEvent.click(screen.getByRole("button", { name: /^investigate claim$/i }));
    await waitFor(() => expect(bodyOf("/api/selflab/cases")).toEqual({ run_id: "run_chat1", char_start: 13, char_end: 27, claim_type: "source_attribution" }));

    // claim header, then draft the case-local template via COMPARE CONTEXTS
    expect(await screen.findByText("“speckled green”")).toBeInTheDocument();
    const compare = await screen.findByRole("button", { name: /compare contexts/i });
    await waitFor(() => expect(compare).not.toBeDisabled());
    fireEvent.click(compare);
    expect(await screen.findByText("not committed")).toBeInTheDocument();
    expect(screen.getAllByTestId("hypothesis-card").length).toBeGreaterThan(0);
    // D-UI1: the example's controlled counterfactual inputs pre-fill the draft and stay editable
    expect(screen.getByTestId("prefill-note")).toHaveTextContent("replacement_text, hint_text, original_value, replacement_value, value_set");
    expect(screen.queryByTestId("replacement-warning")).toBeNull();
    expect(screen.getByLabelText("replacement_text")).toHaveValue(STATIC_EXAMPLE_PROMPT.replacement_text);
    fireEvent.change(screen.getByLabelText("hint_text"), { target: { value: "I think they are speckled blue." } });

    fireEvent.click(screen.getByRole("button", { name: /commit spec/i }));
    await waitFor(() =>
      expect(bodyOf("/api/selflab/experiments")).toMatchObject({
        spec: {
          template: "source_influence_case",
          case_id: "case_1",
          params: {
            replacement_text: STATIC_EXAMPLE_PROMPT.replacement_text,
            hint_text: "I think they are speckled blue.",
            original_value: "green",
            replacement_value: "red",
            value_set: ["green", "red", "blue"],
          },
        },
      }),
    );
    const committed = await screen.findByTestId("committed-spec");
    expect(within(committed).getByText("spec_sha256")).toBeInTheDocument();
    expect(within(committed).getByText("committed_at")).toBeInTheDocument();

    fireEvent.click(within(committed).getByRole("button", { name: /^run$/i }));
    const card = await screen.findByTestId("evidence-card", {}, { timeout: 3000 });
    expect(card).toHaveTextContent("Inconclusive");
    expect(card).toHaveTextContent("Deterministic derivation");
    expect(within(card).getByTestId("fixture-mark")).toBeInTheDocument();
    expect(card.textContent).not.toMatch(/validated/i);
    // D-UI2: with an experiment selected, evidence is filtered by experiment_id alone (never both filters)
    const evidenceUrls = fetchMock.mock.calls.map((c) => c[0] as string).filter((u) => u.startsWith("/api/selflab/evidence"));
    expect(evidenceUrls).toContain("/api/selflab/evidence?case_id=case_1");
    expect(evidenceUrls).toContain("/api/selflab/evidence?experiment_id=exp_1");
    expect(evidenceUrls.some((u) => u.includes("case_id") && u.includes("experiment_id"))).toBe(false);

    // bottom panel offers statistics (analysis present) and export, no empty tabs
    expect(await screen.findByRole("tab", { name: "Statistics" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Export evidence" })).toBeInTheDocument();
    const stats = screen.getByRole("tabpanel");
    expect(within(stats).getByText("claim_status")).toBeInTheDocument();
    expect(within(stats).getByText("replacement value not followed")).toBeInTheDocument();
    expect(within(stats).getByText(/4 trials analysed/)).toBeInTheDocument();
  });
});
