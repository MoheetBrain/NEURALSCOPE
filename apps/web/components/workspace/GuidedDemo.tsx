"use client";

import { useState } from "react";
import { Check, ChevronDown, ChevronRight, Compass } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { api, errorMessage } from "@/lib/api/client";
import type { RunDetail } from "@/lib/api/types";
import { cn } from "@/lib/cn";
import { useWorkspaceActions } from "./actions";
import { defaultSelection } from "./hooks";
import { FLAGSHIP_PROMPT, useWorkspace } from "./WorkspaceProvider";

interface Step {
  title: string;
  body: string;
  action?: string;
  alt?: string;
}

const STEPS: Step[] = [
  { title: "Tokenise + run", body: `Run “${FLAGSHIP_PROMPT}” through the model: greedy generation, then a teacher-forced capture of every residual state.`, action: "Run flagship prompt" },
  { title: "Output + tokens", body: "Read the output and the token strip (OBSERVED). Hover a token for its id, next-token log-prob and top-5. The target is the first answer token, predicted at the last prompt position.", action: "Select the answer position" },
  { title: "Target-rank matrix", body: "Colour every (depth, token) cell by the Logit Lens rank of the target token (DERIVED). Bright = rank 1.", action: "Show target rank" },
  { title: "Rank through depth", body: "Follow the target's Logit Lens rank at the prediction position from the embeddings to the last block, and find the first depth where it is top-1.", action: "Open Logit Lens" },
  { title: "Attention", body: "Open the layer and head with the largest direct logit attribution to the target (DERIVED, direct effects only) and see where it reads from. Attention ≠ explanation.", action: "Open top-DLA head" },
  { title: "Find a causal head", body: "Zero-ablate every head one at a time and measure Δ log P(target) (≈80 s on CPU), or skip and use the top-DLA head as the candidate.", action: "Run head sweep", alt: "Use top-DLA head" },
  { title: "Ablate + rerun", body: "Zero-ablate the candidate head at all positions, regenerate freely and re-measure the target (CAUSALLY TESTED, n=1).", action: "Ablate + rerun" },
  { title: "Diff", body: "Compare baseline vs intervention: output, target probability, KL, top-10 and where in depth the effect starts.", action: "Open Diff" },
];

export function GuidedDemo() {
  const ws = useWorkspace();
  const { state, update, run, submitRun, notify, pending, activeJob } = ws;
  const actions = useWorkspaceActions();
  const [busy, setBusy] = useState(false);
  const step = Math.min(state.guideStep, STEPS.length);
  const r = run.data;
  const needsRun = step >= 1 && !r;

  const go = (n: number) => update({ guideStep: n });

  async function topDlaHead(detail: RunDetail) {
    const dla = await api.dla(detail.id, {
      position: detail.target?.position ?? null,
      target_token_id: detail.target?.token_id ?? null,
    });
    const top = dla.top_heads[0];
    if (!top) throw new Error("DLA returned no heads");
    return top;
  }

  async function perform(which: "main" | "alt") {
    setBusy(true);
    try {
      switch (step) {
        case 0: {
          update({ prompt: FLAGSHIP_PROMPT, chat: true, metric: "lens_top1_prob", tab: "trace" });
          const d = await submitRun({ prompt: FLAGSHIP_PROMPT });
          if (d) go(1);
          break;
        }
        case 1: {
          if (!r) break;
          update({ selection: defaultSelection(r), tab: "trace" });
          go(2);
          break;
        }
        case 2: {
          if (!r) break;
          const P = r.prompt_len;
          const firstGen = r.generated_ids[0];
          if (firstGen !== undefined && (r.target?.position !== P - 1 || r.target?.token_id !== firstGen)) {
            await actions.setTarget(r, { target_token_id: firstGen, target_position: P - 1 });
          }
          update({ metric: "target_rank", selection: { depth: r.depth_labels.length - 1, position: P - 1 } });
          go(3);
          break;
        }
        case 3: {
          if (!r) break;
          update({ tab: "lens", selection: { depth: r.depth_labels.length - 1, position: r.target?.position ?? r.prompt_len - 1 } });
          go(4);
          break;
        }
        case 4: {
          if (!r) break;
          const top = await topDlaHead(r);
          update({ tab: "attention", attnLayer: top.layer, attnHead: top.head });
          notify("info", `Top direct logit attribution to the target: L${top.layer}.H${top.head} (DERIVED).`);
          go(5);
          break;
        }
        case 5: {
          if (!r) break;
          if (which === "alt") {
            const top = await topDlaHead(r);
            actions.prefillHeadAblation(top.layer, top.head);
            go(6);
          } else {
            update({ tab: "interventions" });
            const res = await actions.launchHeadSweep(r);
            const worst = res?.most_harmful[0];
            if (worst) {
              actions.prefillHeadAblation(worst.layer, worst.head);
              notify("info", `Most harmful single-head ablation: L${worst.layer}.H${worst.head} — prefilled in the builder.`);
              go(6);
            }
          }
          break;
        }
        case 6: {
          if (!r) break;
          const d = state.draft;
          const res = await actions.intervene(r, [{ type: "head_ablation", layer: d.layer, head: d.head, factor: d.factor, positions: null }]);
          if (res) go(7);
          break;
        }
        case 7: {
          update({ tab: "diff" });
          go(8);
          break;
        }
      }
    } catch (e) {
      notify("error", `Guided demo: ${errorMessage(e)}`);
    } finally {
      setBusy(false);
    }
  }

  const blocked = busy || Boolean(pending) || (step === 5 && Boolean(activeJob));

  return (
    <section aria-label="Guided flagship demo" className="shrink-0 rounded-[var(--ns-radius)] border border-ns-accent-2/30 bg-ns-panel">
      <button
        type="button"
        className="flex h-8 w-full items-center gap-2 px-2.5 text-left"
        aria-expanded={state.guideOpen}
        onClick={() => update({ guideOpen: !state.guideOpen })}
      >
        <Compass aria-hidden="true" className="size-3.5 text-ns-accent-2" />
        <span className="ns-label text-ns-accent-2">Guided demo</span>
        <span className="font-mono text-[10px] text-ns-text-faint">
          {Math.min(step + 1, STEPS.length)}/{STEPS.length}
        </span>
        {state.guideOpen ? <ChevronDown aria-hidden="true" className="ml-auto size-3.5 text-ns-text-faint" /> : <ChevronRight aria-hidden="true" className="ml-auto size-3.5 text-ns-text-faint" />}
      </button>
      {state.guideOpen ? (
        <ol className="border-t border-ns-border px-2.5 py-2">
          {STEPS.map((s, i) => {
            const done = i < step;
            const current = i === step;
            return (
              <li key={s.title} className="relative flex gap-2 pb-1.5 last:pb-0">
                <button
                  type="button"
                  onClick={() => go(i)}
                  aria-label={`Go to step ${i + 1}: ${s.title}`}
                  className={cn(
                    "mt-0.5 grid size-4 shrink-0 place-items-center rounded-full border font-mono text-[9px]",
                    done && "border-ns-accent-2/60 bg-ns-accent-2/15 text-ns-accent-2",
                    current && "border-ns-accent-2 text-ns-accent-2",
                    !done && !current && "border-ns-border-strong text-ns-text-faint",
                  )}
                >
                  {done ? <Check className="size-2.5" aria-hidden="true" /> : i + 1}
                </button>
                <div className="min-w-0 flex-1">
                  <div className={cn("text-[11.5px]", current ? "text-ns-text" : done ? "text-ns-text-muted" : "text-ns-text-faint")}>{s.title}</div>
                  {current ? (
                    <div className="mt-1 flex flex-col gap-1.5">
                      <p className="text-[11px] leading-snug text-ns-text-muted">{s.body}</p>
                      {needsRun ? <p className="text-[10.5px] text-ns-warn">Run the flagship prompt first (step 1).</p> : null}
                      <div className="flex flex-wrap gap-1.5">
                        {s.action ? (
                          <Button size="xs" caps variant="secondary" disabled={blocked || needsRun} onClick={() => void perform("main")}>
                            {busy ? "Working…" : s.action}
                          </Button>
                        ) : null}
                        {s.alt ? (
                          <Button size="xs" caps variant="ghost" disabled={blocked || needsRun} onClick={() => void perform("alt")}>
                            {s.alt}
                          </Button>
                        ) : null}
                        <Button size="xs" variant="ghost" onClick={() => go(i + 1)}>
                          skip
                        </Button>
                      </div>
                    </div>
                  ) : null}
                </div>
              </li>
            );
          })}
          {step >= STEPS.length ? (
            <li className="flex items-center justify-between pt-1 text-[11px] text-ns-text-muted">
              Demo complete — every number came from the live model.
              <Button size="xs" variant="ghost" onClick={() => go(0)}>
                restart
              </Button>
            </li>
          ) : null}
        </ol>
      ) : null}
    </section>
  );
}
