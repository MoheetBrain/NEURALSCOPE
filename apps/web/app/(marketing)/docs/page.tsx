import type { Metadata } from "next";
import Link from "next/link";
import { Callout, CodeBlock, DocHeading, Formula, PageHeader, Prose, Toc, type TocItem } from "@/components/docs/DocPrimitives";
import { Container } from "@/components/marketing/Section";
import { EpistemicBadge } from "@/components/ui/EpistemicBadge";
import { StatusChip } from "@/components/ui/StatusChip";
import { EPISTEMIC, EPISTEMIC_ORDER, type EpistemicStatus } from "@/lib/epistemic";

export const metadata: Metadata = {
  title: "Docs",
  description:
    "NeuralScope documentation: quickstart, core concepts (token × layer matrix, residual stream, Logit Lens, attention, head ablation, activation patching), epistemic statuses, API overview and limitations.",
};

const TOC: TocItem[] = [
  { id: "quickstart", label: "Quickstart" },
  {
    id: "concepts",
    label: "Concepts",
    children: [
      { id: "token-layer-matrix", label: "Token × layer matrix" },
      { id: "residual-stream", label: "Residual stream" },
      { id: "logit-lens", label: "Logit Lens" },
      { id: "attention", label: "Attention" },
      { id: "head-ablation", label: "Head ablation" },
      { id: "activation-patching", label: "Activation patching" },
    ],
  },
  { id: "epistemic-statuses", label: "Epistemic statuses" },
  { id: "api", label: "API overview" },
  { id: "limitations", label: "Limitations" },
];

const ENV = [
  { k: "NEURALSCOPE_API_URL", d: "http://127.0.0.1:8000", v: "Web only. Where Next.js proxies /api/* requests." },
  { k: "NEURALSCOPE_MODEL", d: "Qwen/Qwen2.5-1.5B-Instruct", v: "Model to load. demo/random-qwen2-tiny enables demo mode." },
  { k: "NEURALSCOPE_DEVICE", d: "auto", v: "Torch device (cpu, cuda, mps, auto)." },
  { k: "NEURALSCOPE_DTYPE", d: "float32", v: "Weights dtype. float32 keeps small intervention deltas above rounding noise." },
  { k: "TRACE_LEVEL", d: "standard", v: "minimal · standard · full. How much internal state each run stores." },
  { k: "NEURALSCOPE_DATA_DIR", d: "./data", v: "Run metadata database and tensor files." },
];

const API_GROUPS: { group: string; rows: [string, string, string][] }[] = [
  {
    group: "Model",
    rows: [
      ["GET", "/api/health", "Load status (loading · ready · error), model id, demo flag."],
      ["GET", "/api/model", "Architecture: layers, heads, KV heads, d_model, vocabulary, dtype."],
      ["POST", "/api/tokenize", "Tokenise a prompt (optionally with the chat template)."],
    ],
  },
  {
    group: "Runs",
    rows: [
      ["POST", "/api/runs", "Generate under instrumentation and store the trace."],
      ["GET", "/api/runs · /api/runs/{id}", "List runs; run detail with tokens, target and output."],
      ["POST", "/api/runs/{id}/target", "Set the target (and contrast) token tracked by views."],
    ],
  },
  {
    group: "Views",
    rows: [
      ["GET", "/api/runs/{id}/matrix", "Token × layer matrix for a metric, e.g. lens_top1_prob."],
      ["GET", "/api/runs/{id}/cell/{depth}/{position}", "Everything about one residual cell."],
      ["GET", "/api/runs/{id}/logit-lens", "Top-k Logit Lens readout by depth; target rank and probability."],
      ["GET", "/api/runs/{id}/attention/{layer}", "Attention patterns for every head in a layer."],
      ["GET", "/api/runs/{id}/heads/{layer}/{head}", "Single-head detail."],
      ["GET", "/api/runs/{id}/dla", "Direct logit attribution by component."],
    ],
  },
  {
    group: "Interventions",
    rows: [
      ["POST", "/api/interventions", "Rerun a base run with one or more intervention specs."],
      ["POST", "/api/compare", "Compare two runs (probabilities, KL, per-depth deltas)."],
      ["POST", "/api/sweeps/heads · /api/sweeps/patch", "Background sweeps over heads or patch sites; poll /api/jobs/{id}."],
    ],
  },
  {
    group: "Math Lab",
    rows: [
      ["POST", "/api/math/problems", "Generate problems with known answers (seeded)."],
      ["POST", "/api/math/run", "Run a problem under instrumentation and verify the answer."],
      ["POST", "/api/math/verify", "SymPy oracle check of an answer."],
      ["POST", "/api/math/benchmarks", "Benchmark a problem set; then evaluate interventions on it."],
    ],
  },
];

const ASSIGNED_BY: Record<EpistemicStatus, string> = {
  observed: "Tokens, captured activations, attention weights, logits, generated text, oracle verdicts.",
  derived: "Logit Lens, norms, entropy, KL divergence, cosine change, direct logit attribution, token × layer summaries.",
  inferred: "Researcher notes, hypotheses and any LLM-generated description of a component.",
  causal: "Results of reruns with a stored intervention spec, measured against the clean run.",
};

export default function DocsPage() {
  return (
    <>
      <PageHeader kicker="Documentation" title="NeuralScope docs">
        <p>
          How to run the V1 neural debugger, the concepts behind each view, what the epistemic labels mean, and where the
          current system stops.
        </p>
      </PageHeader>

      <Container className="py-16 sm:py-20">
        <div className="grid gap-12 lg:grid-cols-[220px_minmax(0,1fr)] lg:gap-16">
          <aside className="hidden lg:block">
            <div className="sticky top-24">
              <Toc items={TOC} />
            </div>
          </aside>

          <div className="min-w-0 space-y-24">
            {/* ======================================================= quickstart */}
            <section aria-labelledby="quickstart">
              <DocHeading id="quickstart" kicker="01">
                Quickstart
              </DocHeading>
              <Prose>
                <p>
                  NeuralScope has two processes: a Python backend that loads the model and captures its internal state with
                  PyTorch forward hooks, and this web app. The browser only talks to the web app; requests to{" "}
                  <code>/api/*</code> are proxied to the backend.
                </p>
                <p>
                  Requirements: Python 3.10+, Node.js 20.9+ (22 recommended), and enough memory for
                  Qwen2.5-1.5B-Instruct in float32 (roughly 6–7 GB for weights alone). A GPU is optional.
                </p>
              </Prose>
              <div className="mt-8 grid gap-4 xl:grid-cols-2">
                <CodeBlock label="1 · Backend (repo root)">{`python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn neuralscope_api.server:app --port 8000`}</CodeBlock>
                <CodeBlock label="2 · Web app">{`cd apps/web
npm install
npm run dev        # http://localhost:3000`}</CodeBlock>
              </div>
              <Prose className="mt-6">
                <p>
                  The model downloads from Hugging Face on first start and loads in the background;{" "}
                  <code>GET /api/health</code> reports <code>loading</code> until it is <code>ready</code>. Then open{" "}
                  <Link href="/app">/app</Link>.
                </p>
              </Prose>

              <h3 className="mb-4 mt-12 text-[18px] font-medium text-ns-text">Configuration</h3>
              <div className="overflow-hidden rounded-[var(--ns-radius)] border border-ns-border">
                <ul className="divide-y divide-ns-border">
                  {ENV.map((e) => (
                    <li key={e.k} className="grid gap-1 px-4 py-3 md:grid-cols-[240px_200px_minmax(0,1fr)] md:gap-4">
                      <code className="break-all font-mono text-[12.5px] text-ns-text">{e.k}</code>
                      <code className="break-all font-mono text-[12px] text-ns-text-faint">{e.d}</code>
                      <span className="text-[13.5px] leading-relaxed text-ns-text-muted">{e.v}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <Callout title="Demo mode" tone="warn">
                <code>NEURALSCOPE_MODEL=demo/random-qwen2-tiny</code> loads a tiny, randomly initialised Qwen2-architecture model
                for fast UI work and tests. Its outputs carry no scientific meaning, and every run it produces is flagged{" "}
                <code>is_demo</code> end to end.
              </Callout>
            </section>

            {/* ======================================================= concepts */}
            <section aria-labelledby="concepts">
              <DocHeading id="concepts" kicker="02">
                Concepts
              </DocHeading>
              <Prose>
                <p>
                  V1 instruments a decoder-only transformer. For Qwen2.5-1.5B-Instruct that is 28 blocks, a 1,536-dimensional
                  residual stream, and 12 query heads per layer sharing 2 key/value heads (grouped-query attention).
                </p>
              </Prose>

              <DocHeading id="token-layer-matrix" level={3}>
                Token × layer matrix
              </DocHeading>
              <Prose>
                <p>
                  The central view. Columns are token positions; rows are <strong>depths</strong>. Depth 0 is the token
                  embedding; depth <em>d</em> is the residual stream after block <em>d</em>−1, so the V1 model has 29 depths
                  (<code>emb</code>, <code>L0</code> … <code>L27</code>). Each cell summarises one residual vector
                  h[d, t] with a scalar metric, such as the Logit Lens top-1 probability (<code>lens_top1_prob</code>) or the
                  rank of a chosen target token.
                </p>
                <p>
                  The vector in a cell is <EpistemicBadge status="observed" size="xs" />. The scalar drawn in the matrix is a
                  function of it, so the matrix is <EpistemicBadge status="derived" size="xs" />.
                </p>
              </Prose>

              <DocHeading id="residual-stream" level={3}>
                Residual stream
              </DocHeading>
              <Prose>
                <p>
                  Each block reads from and adds to a shared vector per token. Attention heads and MLPs never overwrite it; they
                  write increments. This additivity is what makes component-level measurement and intervention well defined.
                </p>
              </Prose>
              <Formula note="pre-norm block; Σ over heads h; RMSNorm as in Qwen2">
                h′ = h<sub>d</sub> + Σ<sub>h</sub> W<sub>O</sub><sup>h</sup> z<sup>h</sup> ,&nbsp;&nbsp; h<sub>d+1</sub> = h′ +
                MLP(RMSNorm(h′))
              </Formula>

              <DocHeading id="logit-lens" level={3}>
                Logit Lens
              </DocHeading>
              <Formula note="the model's own final RMSNorm and unembedding, applied at every depth">
                lens(h<sub>d</sub>) = W<sub>U</sub> · RMSNorm<sub>final</sub>(h<sub>d</sub>)
              </Formula>
              <Prose>
                <p>
                  At the last depth this is exactly the model&apos;s output distribution. At earlier depths it reads an
                  intermediate state through a readout the model was never trained to use there, and it is systematically
                  biased in early layers (Belrose et al., 2023, &ldquo;tuned lens&rdquo;). Logit Lens views are{" "}
                  <EpistemicBadge status="derived" size="xs" /> and are never described as what the model &ldquo;was
                  thinking&rdquo;.
                </p>
              </Prose>

              <DocHeading id="attention" level={3}>
                Attention
              </DocHeading>
              <Formula note="causal mask; per head; queries and keys after rotary position embedding">
                A<sup>l,h</sup> = softmax( q k<sup>⊤</sup> / √d<sub>head</sub> + M )
              </Formula>
              <Prose>
                <p>
                  Attention weights are <EpistemicBadge status="observed" size="xs" />. They say where a head reads from, not
                  whether what it read mattered: the value vectors and the output projection decide what is written. Treat
                  attention as a place to look, then test with an intervention.
                </p>
              </Prose>

              <DocHeading id="head-ablation" level={3}>
                Head ablation
              </DocHeading>
              <Formula note="applied at the input of the attention output projection (o_proj); factor = 0 is zero-ablation">
                z<sup>l,h</sup> ← factor · z<sup>l,h</sup> &nbsp;⇒&nbsp; head write W<sub>O</sub><sup>l,h</sup> z<sup>l,h</sup> is
                scaled exactly
              </Formula>
              <Prose>
                <p>
                  Because <code>o_proj</code> is linear and bias-free in Qwen2, scaling one head&apos;s slice of its input scales
                  that head&apos;s contribution to the residual stream and nothing else. Ablations can target all positions or a
                  subset. The rerun is compared with the clean run on target probability, target rank, KL divergence and the
                  generated output. A single prompt gives a{" "}
                  <EpistemicBadge status="causal" size="xs" note="n=1" /> result; claims about a mechanism need many controlled
                  examples and a baseline such as random heads.
                </p>
              </Prose>

              <DocHeading id="activation-patching" level={3}>
                Activation patching
              </DocHeading>
              <Formula note="source vector from another stored run; α = 1 is a full patch">
                h[d, t] ← (1 − α) · h[d, t] + α · h<sub>src</sub>[d, t′]
              </Formula>
              <Prose>
                <p>
                  Patching copies internal state from one run into another, typically from a clean (correct) run into a corrupt
                  (incorrect) one, and asks whether behaviour is restored. It tests whether the state at (d, t) carries
                  information sufficient for the output. V1 supports single residual patches, and depth × position sweeps between
                  position-aligned pairs (e.g. &ldquo;37 + 58&rdquo; vs &ldquo;37 + 59&rdquo;) scored by how much of the
                  clean-vs-corrupt logit difference each patch restores. Automated activation patching over components is on the
                  V2 roadmap.
                </p>
              </Prose>
            </section>

            {/* ======================================================= epistemics */}
            <section aria-labelledby="epistemic-statuses">
              <DocHeading id="epistemic-statuses" kicker="03">
                Epistemic statuses
              </DocHeading>
              <Prose>
                <p>
                  Every view, number and claim carries exactly one label. The badge&apos;s tooltip repeats its definition. See{" "}
                  <Link href="/research#epistemology">Research → Epistemology</Link> for what each label permits you to claim.
                </p>
              </Prose>
              <div className="mt-8 overflow-hidden rounded-[var(--ns-radius)] border border-ns-border">
                <ul className="divide-y divide-ns-border">
                  {EPISTEMIC_ORDER.map((s) => (
                    <li key={s} className="grid gap-3 px-4 py-4 md:grid-cols-[170px_minmax(0,1fr)_minmax(0,1fr)] md:gap-6">
                      <div>
                        <EpistemicBadge status={s} />
                      </div>
                      <p className="text-[14px] leading-relaxed text-ns-text">{EPISTEMIC[s].definition}</p>
                      <p className="text-[13.5px] leading-relaxed text-ns-text-muted">
                        <span className="ns-label mr-2">Assigned to</span>
                        {ASSIGNED_BY[s]}
                      </p>
                    </li>
                  ))}
                </ul>
              </div>
              <Callout title="For UI contributors" tone="accent">
                Use <code>&lt;EpistemicBadge status=&quot;observed&quot; | &quot;derived&quot; | &quot;inferred&quot; |
                &quot;causal&quot; /&gt;</code> and the <code>EPISTEMIC</code> map in <code>lib/epistemic.ts</code>. Colours
                are the <code>--ns-observed</code>, <code>--ns-derived</code>, <code>--ns-inferred</code> and{" "}
                <code>--ns-causal</code> tokens; use them only to mean the status.
              </Callout>
            </section>

            {/* ======================================================= api */}
            <section aria-labelledby="api">
              <DocHeading id="api" kicker="04">
                API overview
              </DocHeading>
              <Prose>
                <p>
                  The backend is a FastAPI app. The web app proxies <code>/api/:path*</code> to{" "}
                  <code>NEURALSCOPE_API_URL</code>, so the browser always talks same-origin. Tensors stay server-side; endpoints
                  return browser-sized summaries. The interactive OpenAPI reference is served by the backend at{" "}
                  <code>http://127.0.0.1:8000/docs</code> and is the source of truth.
                </p>
              </Prose>
              <p className="mt-4 flex items-center gap-2">
                <StatusChip tone="neutral">v0.1</StatusChip>
                <span className="font-mono text-[11px] text-ns-text-faint">Endpoints may change while V1 is built.</span>
              </p>
              <div className="mt-6 space-y-4">
                {API_GROUPS.map((g) => (
                  <div key={g.group} className="overflow-hidden rounded-[var(--ns-radius)] border border-ns-border">
                    <p className="ns-label border-b border-ns-border bg-ns-panel px-4 py-2.5">{g.group}</p>
                    <ul className="divide-y divide-ns-border">
                      {g.rows.map(([method, path, desc]) => (
                        <li key={path} className="grid gap-1 px-4 py-2.5 md:grid-cols-[52px_minmax(0,330px)_minmax(0,1fr)] md:items-baseline md:gap-4">
                          <span className="font-mono text-[10.5px] font-medium text-ns-accent">{method}</span>
                          <code className="break-all font-mono text-[12.5px] text-ns-text">{path}</code>
                          <span className="text-[13.5px] leading-relaxed text-ns-text-muted">{desc}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
              <h3 className="mb-4 mt-12 text-[18px] font-medium text-ns-text">Example: ablate one head</h3>
              <CodeBlock label="shell">{`# 1. capture a run (target token tracked by every view)
curl -s localhost:3000/api/runs -H 'content-type: application/json' \\
  -d '{"prompt": "What is 37 + 48?", "target_text": "8", "max_new_tokens": 8}'

# 2. rerun it with head 5 of layer 17 zero-ablated
curl -s localhost:3000/api/interventions -H 'content-type: application/json' \\
  -d '{"base_run_id": "<run id>",
       "interventions": [{"type": "head_ablation", "layer": 17, "head": 5, "factor": 0.0}]}'`}</CodeBlock>
              <Prose className="mt-6">
                <p>
                  Intervention specs are declarative and stored with the run, so every intervention can be replayed and audited.
                  V1 types: <code>head_ablation</code>, <code>mlp_ablation</code>, <code>resid_patch</code> and{" "}
                  <code>resid_steer</code>.
                </p>
              </Prose>
            </section>

            {/* ======================================================= limitations */}
            <section aria-labelledby="limitations">
              <DocHeading id="limitations" kicker="05">
                Limitations
              </DocHeading>
              <Prose>
                <p>What V1 does not do, and where its measurements need care.</p>
              </Prose>
              <ul className="mt-6 max-w-[76ch] divide-y divide-ns-border border-y border-ns-border">
                {[
                  ["One model", "V1 targets Qwen2.5-1.5B-Instruct (plus the demo model). Findings do not transfer to other models without re-testing."],
                  ["Logit Lens bias", "Early-depth readouts are systematically biased. V1 has no tuned lens; weigh early layers accordingly."],
                  ["Attention ≠ attribution", "Attention weights show where a head reads, not what mattered. Test with ablation."],
                  ["Direct effects only", "Direct logit attribution misses components that act through later layers; only interventions measure total effects."],
                  ["Ablation artefacts", "Zero-ablation can push activations off-distribution. Compare scaled ablations and random-component baselines before concluding."],
                  ["n = 1 is an anecdote", "A single prompt shows that an effect exists for that prompt. Mechanism claims need controlled sets."],
                  ["No feature-level views yet", "No SAEs, transcoders or probes until V2. Individual neurons are often polysemantic."],
                  ["Local compute", "Prompt length and generation length are capped; full trace levels store large tensors on disk."],
                  ["Oracle scope", "The Math Lab SymPy oracle verifies final answers to problems with computable answers. It does not check reasoning steps or prove theorems."],
                ].map(([k, v]) => (
                  <li key={k} className="grid gap-1 py-4 sm:grid-cols-[200px_minmax(0,1fr)] sm:gap-6">
                    <span className="text-[14px] font-medium text-ns-text">{k}</span>
                    <span className="text-[14px] leading-relaxed text-ns-text-muted">{v}</span>
                  </li>
                ))}
              </ul>
            </section>
          </div>
        </div>
      </Container>
    </>
  );
}
