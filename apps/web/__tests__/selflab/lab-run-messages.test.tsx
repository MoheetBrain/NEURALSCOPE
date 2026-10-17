import { render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("next/navigation", () => ({
  usePathname: () => "/app",
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
}));

import { PromptColumn, RunConversation, runMessages } from "@/components/workspace/PromptColumn";
import { WorkspaceProvider } from "@/components/workspace/WorkspaceProvider";
import { viewCache } from "@/lib/api/cache";
import type { RunDetail } from "@/lib/api/types";

const MESSAGES = [
  { role: "system", content: "You are a careful assistant." },
  { role: "user", content: "Document:\n\nSurvey log 14. Its eggs are speckled green.\n\nWhat colour are the eggs?" },
];

function detail(messages: unknown): RunDetail {
  const tokens = ["<|im_start|>", "user", "\n", "What", " colour", "?", "<|im_end|>", " green"].map((text, index) => ({
    index,
    id: 100 + index,
    text,
    special: text.startsWith("<|"),
    role: index < 3 ? ("template" as const) : index < 7 ? ("user" as const) : ("generated" as const),
    next_logprob: -1,
    next_rank: 1,
    entropy: 1,
    top: [],
  }));
  return {
    id: "run_chat1",
    created_at: "2026-10-17T10:00:00Z",
    kind: "selflab_chat",
    model_id: "demo/random-qwen2-tiny",
    is_demo: true,
    prompt: "Document: … What colour are the eggs?",
    generated_text: " green",
    free_generated_text: null,
    parent_run_id: null,
    interventions_label: "",
    n_tokens: tokens.length,
    trace_level: "minimal",
    correctness: null,
    benchmark_id: null,
    notes: null,
    problem_kind: null,
    prompt_text: "",
    chat: true,
    system_prompt: null,
    generation: { decoding: "greedy", max_new_tokens: 24, stopped_by: "eos" },
    input_ids: tokens.map((t) => t.id),
    prompt_len: 7,
    user_span: [3, 6],
    generated_ids: [107],
    target: null,
    interventions: [],
    metrics: {},
    math: null,
    tags: [],
    trace_consistent: true,
    tokens,
    depth_labels: ["emb", "L0", "L1"],
    children: [],
    status: {},
    ...(messages === undefined ? {} : { messages }),
  } as RunDetail;
}

describe("runMessages / RunConversation", () => {
  it("returns the stored messages of a multi-message run and null otherwise", () => {
    expect(runMessages(detail(MESSAGES))).toEqual(MESSAGES);
    expect(runMessages(detail(undefined))).toBeNull();
    expect(runMessages(detail(null))).toBeNull();
    expect(runMessages(detail([{ nope: 1 }]))).toBeNull();
    expect(runMessages(null)).toBeNull();
  });

  it("shows each message read-only with its role and the SelfLab note", () => {
    render(<RunConversation run={detail(MESSAGES)} />);
    const box = screen.getByTestId("run-conversation");
    expect(box).toHaveTextContent("SelfLab chat run — multi-message context; edit in SelfLab");
    expect(within(box).getByText("#0 · system")).toBeInTheDocument();
    expect(within(box).getByText("#1 · user")).toBeInTheDocument();
    expect(within(box).getByText(/Its eggs are speckled green/)).toBeInTheDocument();
    expect(within(box).getByRole("link", { name: "SelfLab" })).toHaveAttribute("href", "/app/selflab");
    expect(box.querySelector("textarea, input")).toBeNull();
  });
});

describe("Lab prompt column for a SelfLab chat run (/app?run=)", () => {
  const fetchMock = vi.fn();
  const respond = (body: unknown, status = 200) =>
    Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } }));

  beforeEach(() => {
    viewCache.clear();
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
    window.history.replaceState(null, "", "/app?run=run_chat1");
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    window.history.replaceState(null, "", "/");
  });

  function mockBackend(messages: unknown) {
    const d = detail(messages);
    fetchMock.mockImplementation((url: string) => {
      if (url.startsWith("/api/health"))
        return respond({ status: "ready", model_id: "demo/random-qwen2-tiny", is_demo: true, error: null, trace_level_default: "standard", version: "0.1.0" });
      if (url.startsWith("/api/model"))
        return respond({
          model_id: "demo/random-qwen2-tiny", architecture: "qwen2", n_layers: 2, n_heads: 4, n_kv_heads: 2, d_model: 64, d_head: 16, d_mlp: 128,
          vocab_size: 1000, dtype: "float32", device: "cpu", n_params: 123456, tied_embeddings: true, norm_type: "RMSNorm", is_demo: true,
          capabilities: {}, n_depths: 3, load_seconds: 0.1,
        });
      if (url.startsWith("/api/runs/run_chat1")) return respond(d);
      if (url.startsWith("/api/runs")) return respond([{ ...d, tokens: undefined }]);
      return respond({ detail: "not mocked" }, 404);
    });
  }

  it("shows the run's conversation and labels the prompt box as a new run", async () => {
    mockBackend(MESSAGES);
    render(
      <WorkspaceProvider>
        <PromptColumn />
      </WorkspaceProvider>,
    );
    const conv = await screen.findByTestId("run-conversation");
    expect(within(conv).getByText("#0 · system")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "New run prompt" })).toBeInTheDocument();
    expect(screen.getByTestId("new-run-note")).toHaveTextContent(/does not replay the conversation/);
  });

  it("keeps the plain Prompt panel for single-prompt runs", async () => {
    mockBackend(undefined);
    render(
      <WorkspaceProvider>
        <PromptColumn />
      </WorkspaceProvider>,
    );
    expect(await screen.findByLabelText("Generated text")).toBeInTheDocument();
    expect(screen.queryByTestId("run-conversation")).toBeNull();
    expect(screen.getByRole("region", { name: "Prompt" })).toBeInTheDocument();
  });
});
