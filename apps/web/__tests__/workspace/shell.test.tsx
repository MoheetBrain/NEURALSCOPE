import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("next/navigation", () => ({
  usePathname: () => "/app",
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
}));

import { WorkspaceShell } from "@/components/workspace/WorkspaceShell";
import { viewCache } from "@/lib/api/cache";

const MODEL = {
  model_id: "demo/random-qwen2-tiny",
  architecture: "qwen2",
  n_layers: 2,
  n_heads: 4,
  n_kv_heads: 2,
  d_model: 64,
  d_head: 16,
  d_mlp: 128,
  vocab_size: 1000,
  dtype: "float32",
  device: "cpu",
  n_params: 123456,
  tied_embeddings: true,
  norm_type: "RMSNorm",
  is_demo: true,
  capabilities: { qkv: false },
  n_depths: 3,
  load_seconds: 0.1,
};

function respond(body: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } }));
}

describe("WorkspaceShell", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    viewCache.clear();
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => vi.unstubAllGlobals());

  it("shows the permanent DEMO banner when the backend is a demo fixture", async () => {
    fetchMock.mockImplementation((url: string) => {
      if (url.startsWith("/api/health"))
        return respond({ status: "ready", model_id: MODEL.model_id, is_demo: true, error: null, trace_level_default: "standard", version: "0.1.0" });
      if (url.startsWith("/api/model")) return respond(MODEL);
      if (url.startsWith("/api/runs")) return respond([]);
      return respond({ detail: "not mocked" }, 404);
    });
    render(
      <WorkspaceShell>
        <p>workspace body</p>
      </WorkspaceShell>,
    );
    const banner = await screen.findByTestId("demo-banner");
    expect(banner).toHaveTextContent("DEMO — RANDOM-WEIGHT FIXTURE MODEL · outputs carry no scientific meaning");
    expect(await screen.findByText("workspace body")).toBeInTheDocument();
    expect(screen.getByText(/2L×4H/)).toBeInTheDocument();
  });

  it("gates the workspace while the model is loading", async () => {
    fetchMock.mockImplementation((url: string) => {
      if (url.startsWith("/api/health"))
        return respond({ status: "loading", model_id: "Qwen/Qwen2.5-1.5B-Instruct", is_demo: false, error: null, trace_level_default: "standard", version: "0.1.0" });
      return respond({ detail: "model is still loading — poll /api/health" }, 503);
    });
    render(
      <WorkspaceShell>
        <p>workspace body</p>
      </WorkspaceShell>,
    );
    await waitFor(() => expect(screen.getAllByText("Model loading").length).toBeGreaterThan(0));
    expect(screen.getByRole("heading", { name: "Qwen/Qwen2.5-1.5B-Instruct" })).toBeInTheDocument();
    expect(screen.queryByText("workspace body")).toBeNull();
    expect(screen.queryByTestId("demo-banner")).toBeNull();
  });

  it("surfaces a model load error", async () => {
    fetchMock.mockImplementation(() =>
      respond({ status: "error", model_id: "Qwen/Qwen2.5-1.5B-Instruct", is_demo: false, error: "OSError: weights not found", trace_level_default: "standard", version: "0.1.0" }),
    );
    render(
      <WorkspaceShell>
        <p>workspace body</p>
      </WorkspaceShell>,
    );
    expect(await screen.findByText("OSError: weights not found")).toBeInTheDocument();
    expect(screen.queryByText("workspace body")).toBeNull();
  });
});
