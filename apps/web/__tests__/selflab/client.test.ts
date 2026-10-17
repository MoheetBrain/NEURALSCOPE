import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "@/lib/api/client";
import {
  accessLevels,
  allParticipants,
  chatExamples,
  featureStatusOf,
  isApiOnlyEnabled,
  normalizeCaseDetail,
  normalizeExportPreview,
  normalizeSuggestions,
  pollExperiment,
  selflab,
  splitAnalysis,
  statusIsFixture,
  statusModel,
  type ExperimentRecord,
  type SelfLabStatus,
} from "@/lib/api/selflab";

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

const CASE = {
  id: "case_1",
  created_at: "2026-10-17T10:00:00Z",
  title: "t",
  access_level: "white_box_local",
  run_id: "run_1",
  messages: [{ role: "user", content: "q" }],
  response_text: "green",
  claim: { text: "green", char_start: 0, char_end: 5 },
  claim_type: "source_attribution",
};

function rec(status: ExperimentRecord["status"], n_done = 0): ExperimentRecord {
  return {
    id: "exp_1",
    committed: {
      spec: { template: "source_influence_case" },
      spec_sha256: "a".repeat(64),
      plan_sha256: "b".repeat(64),
      n_planned: 4,
      committed_at: "2026-10-17T10:00:00Z",
    },
    status,
    progress: n_done / 4,
    n_done,
    n_errors: 0,
    created_at: "2026-10-17T10:00:00Z",
  };
}

describe("selflab client endpoints", () => {
  const fetchMock = vi.fn();
  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => vi.unstubAllGlobals());

  const call = (i = 0) => {
    const [url, init] = fetchMock.mock.calls[i];
    return { url: url as string, method: init.method as string, body: init.body ? JSON.parse(init.body) : undefined };
  };

  it("hits every §5 route with the right method, path and body", async () => {
    fetchMock.mockImplementation(() => Promise.resolve(json({})));
    await selflab.status();
    await selflab.chat({ question: "q", document: "d", hint: null, max_new_tokens: 16 });
    await selflab.createCase({ run_id: "run_1", char_start: 2, char_end: 7, claim_type: "source_attribution" });
    await selflab.templates("case_1");
    await selflab.commitExperiment({ template: "csp1_carry", seed: 3 });
    await selflab.runExperiment("exp_1");
    await selflab.cancelExperiment("exp_1");
    await selflab.resumeExperiment("exp_1");
    await selflab.getExperiment("exp_1");
    await selflab.trials("exp_1", { offset: 25, limit: 25 });
    await selflab.trial("exp_1:00003");
    await selflab.openInLab("exp_1:00003");
    await selflab.evidence({ case_id: "case_1", experiment_id: null });
    await selflab.suggest("case_1");
    await selflab.listCases();
    await selflab.listExperiments(null);

    const seen = fetchMock.mock.calls.map((_, i) => {
      const c = call(i);
      return `${c.method} ${c.url}`;
    });
    expect(seen).toEqual([
      "GET /api/selflab/status",
      "POST /api/selflab/chat",
      "POST /api/selflab/cases",
      "GET /api/selflab/templates?case_id=case_1",
      "POST /api/selflab/experiments",
      "POST /api/selflab/experiments/exp_1/run",
      "POST /api/selflab/experiments/exp_1/cancel",
      "POST /api/selflab/experiments/exp_1/resume",
      "GET /api/selflab/experiments/exp_1",
      "GET /api/selflab/experiments/exp_1/trials?offset=25&limit=25",
      "GET /api/selflab/trials/exp_1%3A00003",
      "POST /api/selflab/trials/exp_1%3A00003/open-in-lab",
      "GET /api/selflab/evidence?case_id=case_1",
      "POST /api/selflab/investigate/suggest",
      "GET /api/selflab/cases",
      "GET /api/selflab/experiments",
    ]);
    expect(call(1).body).toEqual({ question: "q", document: "d", hint: null, max_new_tokens: 16 });
    expect(call(2).body).toEqual({ run_id: "run_1", char_start: 2, char_end: 7, claim_type: "source_attribution" });
    expect(call(4).body).toEqual({ spec: { template: "csp1_carry", seed: 3 } });
    expect(call(13).body).toEqual({ case_id: "case_1" });
  });

  it("requests the export preview with ?preview=1 and the full record without it", async () => {
    fetchMock.mockImplementation(() => Promise.resolve(json({ fields: [] })));
    await selflab.exportPreview("exp_1");
    await selflab.exportRecord("exp_1");
    expect(call(0).url).toBe("/api/selflab/experiments/exp_1/export?preview=1");
    expect(call(1).url).toBe("/api/selflab/experiments/exp_1/export");
  });

  it("surfaces backend detail messages (e.g. a refused resume)", async () => {
    fetchMock.mockResolvedValueOnce(json({ detail: "plan_sha256 differs: refusing to resume" }, 409));
    const err = await selflab.resumeExperiment("exp_1").catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(409);
    expect(err.message).toMatch(/plan_sha256 differs/);
  });

  it("accepts list endpoints as bare arrays or wrapped objects", async () => {
    fetchMock.mockResolvedValueOnce(json([CASE])).mockResolvedValueOnce(json({ cases: [CASE] }));
    expect(await selflab.listCases()).toHaveLength(1);
    expect(await selflab.listCases()).toHaveLength(1);
  });

  it("normalises the case detail envelope in both shapes", async () => {
    const wrapped = normalizeCaseDetail({ case: CASE, experiments: [rec("done")], evidence: [] });
    expect(wrapped.case.id).toBe("case_1");
    expect(wrapped.experiments).toHaveLength(1);
    const inline = normalizeCaseDetail({ ...CASE, experiments: [], evidence: [{ id: "ev1" }] });
    expect(inline.case.id).toBe("case_1");
    expect("evidence" in inline.case).toBe(false);
    expect(inline.evidence).toHaveLength(1);
    expect(() => normalizeCaseDetail({ nope: true })).toThrow(ApiError);
  });
});

describe("selflab status helpers", () => {
  it("keeps API-only disabled unless the backend says it is enabled", () => {
    expect(isApiOnlyEnabled(undefined)).toBe(false);
    expect(isApiOnlyEnabled({})).toBe(false);
    expect(isApiOnlyEnabled({ access_levels: { api_only: { enabled: false, reason: "not configured" } } })).toBe(false);
    expect(accessLevels({ access_levels: { api_only: { enabled: false, reason: "not configured" } } }).find((a) => a.level === "api_only")?.reason).toBe(
      "not configured",
    );
    expect(isApiOnlyEnabled({ access_levels: { api_only: true } })).toBe(true);
    expect(isApiOnlyEnabled({ access_levels: [{ level: "api_only", enabled: true }] })).toBe(true);
    expect(accessLevels({})[0]).toEqual({ level: "white_box_local", enabled: true });
  });

  it("reads feature statuses, presets and fixture flags without inventing values", () => {
    const st = {
      feature_statuses: { llm_investigator: { status: "unsupported", note: "needs approval" }, csp1_carry: "research_prototype" },
      examples: [{ question: "What colour?", document: "doc" }, { title: "broken" }],
      model: { model_id: "demo/random-qwen2-tiny", tokenizer_id: "x", dtype: "float32", device: "cpu", is_demo: true },
    } as unknown as SelfLabStatus; // deliberately includes a malformed preset
    expect(featureStatusOf(st, "llm_investigator")).toEqual({ status: "unsupported", note: "needs approval" });
    expect(featureStatusOf(st, "csp1_carry")).toEqual({ status: "research_prototype" });
    expect(featureStatusOf(st, "missing")).toBeNull();
    expect(chatExamples(st)).toEqual([{ question: "What colour?", document: "doc" }]);
    expect(chatExamples({})).toEqual([]);
    expect(statusIsFixture(st)).toBe(true);
    expect(statusIsFixture({ is_demo: false })).toBe(false);
  });

  it("normalises export previews from a field list or a field dict", () => {
    const a = normalizeExportPreview({
      fields: [
        { name: "spec", bytes: 812 },
        { name: "trials", bytes: 20480, count: 40, free_text: true },
      ],
      total_bytes: 21292,
    });
    expect(a.fields.map((f) => [f.name, f.bytes, f.free_text])).toEqual([
      ["spec", 812, false],
      ["trials", 20480, true],
    ]);
    expect(a.free_text_fields).toEqual(["trials"]);
    const b = normalizeExportPreview({ included: { evidence: { size_bytes: 300 } }, free_text_fields: ["evidence", "messages"] });
    expect(b.fields.find((f) => f.name === "evidence")).toMatchObject({ bytes: 300, free_text: true });
    expect(b.fields.find((f) => f.name === "messages")).toMatchObject({ bytes: null, free_text: true });
    expect(b.total_bytes).toBeNull();
  });

  it("normalises rule-based suggestions", () => {
    const s = normalizeSuggestions({
      participant: "rules:template-suggester@1",
      suggestions: [{ template: "source_influence_case", title: "Remove the document", rationale: "claim cites a document", rule: "R1" }],
    });
    expect(s.participant).toBe("rules:template-suggester@1");
    expect(s.suggestions[0]).toMatchObject({ template: "source_influence_case", rule: "R1" });
    expect(normalizeSuggestions([]).suggestions).toEqual([]);
  });
});

// The shapes below mirror apps/api/neuralscope_api/selflab_routes.py + selflab/service.py (status_payload,
// build_export(preview=True), investigator.suggest) as of the SelfLab integration.
describe("real backend envelopes", () => {
  const STATUS = {
    schema_version: "selflab-1",
    model: {
      status: "ready",
      error: null,
      provenance: { model_id: "demo/random-qwen2-tiny", tokenizer_id: "demo", dtype: "float32", device: "cpu", is_demo: true },
    },
    is_fixture: true,
    access_levels: {
      white_box_local: { enabled: true, feature_status: "research_prototype", note: "the locally loaded model" },
      api_only: { enabled: false, feature_status: "unsupported", note: "disabled; set NEURALSCOPE_SELFLAB_API_ONLY=1" },
    },
    participants: [{ role: "target", id: "demo/random-qwen2-tiny", description: "DEMO", sees: ["messages"] }],
    disabled_participants: [
      { id: "llm-investigator", status: "unsupported", reason: "not configured/approved", role: "investigator", enabled: false },
      { id: "activation-decoders", role: "activation_decoder", enabled: false, status: "unsupported", reason: "none configured" },
    ],
    features: { llm_investigator: "unsupported", export: "implemented" },
  } as unknown as SelfLabStatus;

  it("reads model provenance, features, access notes and disabled participants", () => {
    expect(statusModel(STATUS)?.model_id).toBe("demo/random-qwen2-tiny");
    expect(statusIsFixture(STATUS)).toBe(true);
    expect(featureStatusOf(STATUS, "llm_investigator")).toEqual({ status: "unsupported" });
    const levels = accessLevels(STATUS);
    expect(levels.find((a) => a.level === "white_box_local")).toMatchObject({ enabled: true, description: "the locally loaded model" });
    expect(levels.find((a) => a.level === "api_only")).toMatchObject({ enabled: false, reason: expect.stringMatching(/disabled/) });
    const ps = allParticipants(STATUS, null);
    expect(ps.map((p) => `${p.role}:${p.id}:${p.enabled !== false}`)).toEqual([
      "target:demo/random-qwen2-tiny:true",
      "investigator:llm-investigator:false",
      "activation_decoder:activation-decoders:false",
    ]);
  });

  it("normalises the backend export preview (sections by path, free-text path objects)", () => {
    const p = normalizeExportPreview({
      preview: true,
      experiment_id: "exp_1",
      fields: [
        { path: "committed_spec", bytes: 900, free_text: false },
        { path: "trials", bytes: 40000, free_text: true, count: 4, free_text_fields: ["trials[].output_text", "trials[].measurements.branch_messages"] },
        { path: "evidence", bytes: 2000, free_text: true, count: 2, free_text_fields: ["evidence[].statement"] },
      ],
      total_bytes: 43000,
      free_text_fields: [
        { path: "trials[].output_text", bytes: 300 },
        { path: "evidence[].statement", bytes: 500 },
      ],
      free_text_bytes: 800,
      exclude_free_text_bytes: 42400,
      redaction: "key-based: values under these keys are replaced by {redacted, sha256, n_chars}: output_text, statement",
      raw_tensors_included: false,
      n_tensor_references: 5,
      trial_status_counts: { ok: 3, malformed: 1 },
      is_fixture: true,
    });
    expect(p.fields.map((f) => [f.name, f.free_text])).toEqual([
      ["committed_spec", false],
      ["trials", true],
      ["evidence", true],
    ]);
    expect(p.fields[1].free_text_paths).toHaveLength(2);
    expect(p.free_text_fields).toEqual(["trials", "evidence"]);
    expect(p.free_text_paths).toEqual([
      { path: "trials[].output_text", bytes: 300 },
      { path: "evidence[].statement", bytes: 500 },
    ]);
    expect(p).toMatchObject({ total_bytes: 43000, free_text_bytes: 800, redacted_bytes: 42400, n_tensor_references: 5, raw_tensors_included: false });
    expect(p.trial_status_counts).toEqual({ ok: 3, malformed: 1 });
  });

  it("keeps rule-based suggestion applicability and what is missing", () => {
    const r = normalizeSuggestions({
      participant: "rules:template-suggester@1",
      kind: "interpretive_hypothesis",
      disclaimer: "Rule-based suggestions … not evidence about the target.",
      suggestions: [
        { template: "source_influence_case", rank_rule: 0, reason: "remove or replace the document", missing: [], applicable: true, available: true, feature_status: "research_prototype", title: "Case" },
        { template: "csp1_carry", rank_rule: 2, reason: "arithmetic only", missing: ["arithmetic process / self-prediction claim"], applicable: false, available: false, feature_status: "roadmap", title: "csp1_carry", note: "template not registered on this server" },
      ],
      llm_investigator: { id: "llm-investigator", status: "unsupported", reason: "not configured/approved" },
    });
    expect(r.disclaimer).toMatch(/not evidence/);
    expect(r.suggestions[0]).toMatchObject({ applicable: true, available: true, rationale: "remove or replace the document" });
    expect(r.suggestions[1]).toMatchObject({ applicable: false, available: false, missing: ["arithmetic process / self-prediction claim"] });
    expect(r.llm_investigator?.status).toBe("unsupported");
  });

  it("splits the engine's analysis wrapper from the template summary", () => {
    const { meta, summary } = splitAnalysis({ summary: { claim_status: "inconclusive" }, complete: false, n_trials_analyzed: 3, analysis_version: "x@1" });
    expect(summary).toEqual({ claim_status: "inconclusive" });
    expect(meta).toMatchObject({ complete: false, n_trials_analyzed: 3 });
    expect(splitAnalysis({ claim_status: "supported" }).summary).toEqual({ claim_status: "supported" });
    expect(splitAnalysis(null)).toEqual({ meta: {}, summary: {} });
  });

  it("asks for a redacted export when free text is excluded", async () => {
    const fetchMock = vi.fn(() => Promise.resolve(json({})));
    vi.stubGlobal("fetch", fetchMock);
    await selflab.exportRecord("exp_1", { excludeFreeText: true });
    expect((fetchMock.mock.calls[0] as unknown[])[0]).toBe("/api/selflab/experiments/exp_1/export?exclude_free_text=1");
    vi.unstubAllGlobals();
  });
});

describe("pollExperiment", () => {
  it("polls while queued/running and resolves on a settled status", async () => {
    const seq = [rec("queued"), rec("running", 1), rec("running", 3), rec("done", 4)];
    const fetchRecord = vi.fn(async () => seq.shift() as ExperimentRecord);
    const updates: string[] = [];
    const done = await pollExperiment("exp_1", {
      fetchRecord,
      sleep: async () => {},
      onUpdate: (r) => updates.push(`${r.status}:${r.n_done}`),
    });
    expect(done.status).toBe("done");
    expect(updates).toEqual(["queued:0", "running:1", "running:3", "done:4"]);
    expect(fetchRecord).toHaveBeenCalledTimes(4);
  });

  it("returns immediately for a committed (not yet run) record", async () => {
    const fetchRecord = vi.fn(async () => rec("committed"));
    expect((await pollExperiment("exp_1", { fetchRecord, sleep: async () => {} })).status).toBe("committed");
    expect(fetchRecord).toHaveBeenCalledTimes(1);
  });

  it("tolerates transient errors but not a 404", async () => {
    const fetchRecord = vi
      .fn()
      .mockRejectedValueOnce(new ApiError(502, "bad gateway"))
      .mockResolvedValueOnce(rec("cancelled", 2));
    expect((await pollExperiment("exp_1", { fetchRecord, sleep: async () => {} })).status).toBe("cancelled");
    const missing = vi.fn().mockRejectedValue(new ApiError(404, "experiment exp_9 not found"));
    await expect(pollExperiment("exp_9", { fetchRecord: missing, sleep: async () => {} })).rejects.toThrow(/not found/);
  });
});
