import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, buildUrl, errorMessage, formatDetail, isAbortError, pollJob, request } from "@/lib/api/client";
import type { Job } from "@/lib/api/types";

function jsonResponse(body: unknown, status = 200, statusText = "OK"): Response {
  return new Response(typeof body === "string" ? body : JSON.stringify(body), {
    status,
    statusText,
    headers: { "content-type": "application/json" },
  });
}

describe("API client", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("builds same-origin URLs and drops empty query params", () => {
    expect(buildUrl("/runs/run_1/matrix", { metric: "target_rank", target_token_id: null, x: undefined, y: "" })).toBe(
      "/api/runs/run_1/matrix?metric=target_rank",
    );
    expect(buildUrl("runs", { limit: 5, kind: "math" })).toBe("/api/runs?limit=5&kind=math");
  });

  it("returns parsed JSON and sends JSON bodies as POST", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ id: "run_1" }));
    const r = await api.createRun({ prompt: "hi" });
    expect(r).toEqual({ id: "run_1" });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/api/runs");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ prompt: "hi" });
    expect(init.headers["content-type"]).toBe("application/json");
  });

  it("surfaces the backend's string `detail` as an ApiError", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: "run nope not found" }, 404, "Not Found"));
    const err = await api.getRun("nope").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(404);
    expect(err.message).toBe("run nope not found");
    expect(errorMessage(err)).toBe("run nope not found");
  });

  it("formats pydantic validation errors", async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(
        { detail: [{ type: "string_too_short", loc: ["body", "prompt"], msg: "String should have at least 1 character" }] },
        422,
        "Unprocessable Entity",
      ),
    );
    const err = await api.createRun({ prompt: "" }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(422);
    expect(err.detail).toBe("prompt: String should have at least 1 character");
  });

  it("falls back to the status line for non-JSON errors", async () => {
    fetchMock.mockResolvedValueOnce(new Response("<html>bad gateway</html>".repeat(40), { status: 502, statusText: "Bad Gateway" }));
    const err = (await request("/health").catch((e) => e)) as ApiError;
    expect(err.detail).toBe("502 Bad Gateway");
  });

  it("maps network failures to a clear 'unreachable' error", async () => {
    fetchMock.mockRejectedValueOnce(new TypeError("fetch failed"));
    const err = await api.health().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(0);
    expect(err.message).toMatch(/unreachable/i);
  });

  it("re-throws aborts untouched so superseded requests can be ignored", async () => {
    fetchMock.mockRejectedValueOnce(new DOMException("The operation was aborted.", "AbortError"));
    const ctl = new AbortController();
    const err = await api.matrix("run_1", "lens_top1_prob", null, ctl.signal).catch((e) => e);
    expect(isAbortError(err)).toBe(true);
    expect(err).not.toBeInstanceOf(ApiError);
  });

  it("formatDetail handles objects and empty values", () => {
    expect(formatDetail({ a: 1 }, "fb")).toBe('{"a":1}');
    expect(formatDetail("", "fb")).toBe("fb");
    expect(formatDetail(undefined, "fb")).toBe("fb");
  });
});

describe("pollJob", () => {
  const job = (status: Job["status"], progress: number, extra: Partial<Job> = {}): Job<{ ok: boolean }> => ({
    id: "job_1",
    kind: "head_sweep",
    status,
    progress,
    message: `${Math.round(progress * 100)}%`,
    result: null,
    error: null,
    created_at: 0,
    started_at: null,
    finished_at: null,
    meta: {},
    ...extra,
  }) as Job<{ ok: boolean }>;

  it("polls until done and reports progress", async () => {
    const seq = [job("queued", 0), job("running", 0.5), job("done", 1, { result: { ok: true } })];
    const fetchJob = vi.fn(async () => seq.shift()!);
    const updates: number[] = [];
    const done = await pollJob<{ ok: boolean }>("job_1", { fetchJob, sleep: async () => {}, onUpdate: (j) => updates.push(j.progress) });
    expect(done.result).toEqual({ ok: true });
    expect(updates).toEqual([0, 0.5, 1]);
    expect(fetchJob).toHaveBeenCalledTimes(3);
  });

  it("rejects with the job's error text", async () => {
    const fetchJob = vi.fn(async () => job("error", 0.3, { error: "ServiceError: sweep needs a target token" }));
    const err = await pollJob("job_1", { fetchJob, sleep: async () => {} }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.message).toBe("ServiceError: sweep needs a target token");
  });

  it("tolerates transient proxy errors but not a missing job", async () => {
    const transient = vi
      .fn()
      .mockRejectedValueOnce(new ApiError(502, "Bad Gateway"))
      .mockResolvedValueOnce(job("done", 1, { result: { ok: true } }));
    await expect(pollJob("job_1", { fetchJob: transient, sleep: async () => {} })).resolves.toMatchObject({ status: "done" });

    const missing = vi.fn().mockRejectedValue(new ApiError(404, "job not found"));
    await expect(pollJob("job_1", { fetchJob: missing, sleep: async () => {} })).rejects.toThrow("job not found");
  });

  it("stops when aborted", async () => {
    const ctl = new AbortController();
    const fetchJob = vi.fn(async () => job("running", 0.2));
    const p = pollJob("job_1", { fetchJob, signal: ctl.signal, intervalMs: 5 });
    ctl.abort();
    const err = await p.catch((e) => e);
    expect(isAbortError(err)).toBe(true);
  });
});
