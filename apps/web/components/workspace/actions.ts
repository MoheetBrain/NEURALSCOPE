"use client";

import { api } from "@/lib/api/client";
import { viewCache } from "@/lib/api/cache";
import type { HeadSweepResult, Intervention, InterventionResponse, RunDetail, RunTarget, SetTargetRequest } from "@/lib/api/types";
import { fmtDelta, fmtProb } from "@/lib/format";
import { useWorkspace, type InterventionDraft } from "./WorkspaceProvider";

/** Parse "3, 5-7" → [3,5,6,7]. Returns null when empty; throws on garbage. */
export function parsePositions(text: string, maxPos: number): number[] | null {
  const t = text.trim();
  if (!t) return null;
  const out = new Set<number>();
  for (const part of t.split(/[\s,]+/).filter(Boolean)) {
    const m = /^(\d+)(?:-(\d+))?$/.exec(part);
    if (!m) throw new Error(`cannot parse position "${part}"`);
    const a = Number(m[1]);
    const b = m[2] !== undefined ? Number(m[2]) : a;
    for (let p = Math.min(a, b); p <= Math.max(a, b); p++) {
      if (p > maxPos) throw new Error(`position ${p} is beyond the last token (${maxPos})`);
      out.add(p);
    }
  }
  return [...out].sort((x, y) => x - y);
}

/** Build the API intervention spec from the builder draft. */
export function draftToIntervention(d: InterventionDraft, selectedPosition: number, nTokens: number): Intervention {
  const positions = (): number[] | null => {
    if (d.positionsMode === "all") return null;
    if (d.positionsMode === "selected") return [selectedPosition];
    return parsePositions(d.customPositions, nTokens - 1);
  };
  switch (d.type) {
    case "head_ablation":
      return { type: "head_ablation", layer: d.layer, head: d.head, factor: d.factor, positions: positions() };
    case "mlp_ablation":
      return { type: "mlp_ablation", layer: d.layer, factor: d.factor, positions: positions() };
    case "resid_patch": {
      if (!d.sourceRunId) throw new Error("pick a source run for the residual patch");
      const sp = d.sourcePosition.trim() === "" ? null : Number(d.sourcePosition);
      if (sp !== null && !Number.isInteger(sp)) throw new Error("source position must be an integer");
      return { type: "resid_patch", depth: d.depth, position: d.position, source_run_id: d.sourceRunId, source_position: sp, alpha: d.alpha };
    }
    case "resid_steer":
      if (!d.plusRunId || !d.minusRunId) throw new Error("pick both the + and − runs for the steering vector");
      return {
        type: "resid_steer",
        depth: d.depth,
        positions: positions(),
        plus_run_id: d.plusRunId,
        plus_position: d.plusPosition,
        minus_run_id: d.minusRunId,
        minus_position: d.minusPosition,
        scale: d.scale,
      };
  }
}

/** Short human label for an intervention spec (mirrors the backend's describe()). */
export function describeIntervention(i: Intervention): string {
  const pos = (p: number[] | null | undefined) =>
    p === null || p === undefined ? " (all positions)" : p.length <= 4 ? ` @ t${p.join(",")}` : ` @ ${p.length} positions`;
  switch (i.type) {
    case "head_ablation":
      return `${!i.factor ? "zero-ablate" : `scale ×${i.factor}`} head L${i.layer}.H${i.head}${pos(i.positions)}`;
    case "mlp_ablation":
      return `${!i.factor ? "zero-ablate" : `scale ×${i.factor}`} MLP L${i.layer}${pos(i.positions)}`;
    case "resid_patch":
      return `patch resid[d${i.depth}, t${i.position}] ← ${i.source_run_id.slice(0, 12)}[t${i.source_position ?? i.position}]${i.alpha !== undefined && i.alpha !== 1 ? ` (α=${i.alpha})` : ""}`;
    case "resid_steer":
      return `steer resid[d${i.depth}] += ${i.scale ?? 1}·(${i.plus_run_id.slice(0, 12)}[t${i.plus_position}] − ${i.minus_run_id.slice(0, 12)}[t${i.minus_position}])${pos(i.positions)}`;
  }
}

export function comparisonCacheKey(childRunId: string): string {
  return `cmp:${childRunId}`;
}

export function useWorkspaceActions() {
  const ws = useWorkspace();

  async function intervene(base: RunDetail, interventions: Intervention[], maxNewTokens?: number | null): Promise<InterventionResponse | null> {
    const res = await ws.withPending({ kind: "intervention", label: "Intervention + rerun" }, () =>
      api.intervene({ base_run_id: base.id, interventions, max_new_tokens: maxNewTokens ?? null }),
    );
    if (res) {
      viewCache.set(comparisonCacheKey(res.run.id), res.comparison);
      ws.update((s) => ({
        diffRunId: res.run.id,
        tab: "diff",
        stack: [],
        runVersion: s.runVersion + 1,
        historyVersion: s.historyVersion + 1,
      }));
      const c = res.comparison;
      const tgt = c.target?.token ?? "target";
      ws.notify(
        "ok",
        c.before && c.after
          ? `${res.run.interventions_label}: P(${tgt}) ${fmtProb(c.before.prob)} → ${fmtProb(c.after.prob)} (${fmtDelta(c.delta?.prob ?? null)}). Output ${c.output_changed ? "changed" : "unchanged"}.`
          : `${res.run.interventions_label} done.`,
      );
    }
    return res;
  }

  async function launchHeadSweep(run: RunDetail): Promise<HeadSweepResult | null> {
    return ws.startJob<HeadSweepResult>(
      "head_sweep",
      "Head sweep",
      run.id,
      () =>
        api.headSweep({
          run_id: run.id,
          position: run.target?.position ?? null,
          target_token_id: run.target?.token_id ?? null,
          contrast_token_id: run.target?.contrast_id ?? null,
        }),
      () => {
        ws.bumpRun();
        ws.notify("ok", "Head sweep finished — Δ heatmap updated (CAUSALLY TESTED, n=1).");
      },
    );
  }

  async function setTarget(run: RunDetail, body: SetTargetRequest): Promise<RunTarget | null> {
    const tgt = await ws.withPending({ kind: "target", label: "Set target" }, () => api.setTarget(run.id, body));
    if (tgt) {
      ws.bumpRun();
      ws.notify("ok", `Target set: ${JSON.stringify(tgt.token)} at t${tgt.position}${tgt.note ? ` — ${tgt.note}` : ""}`);
    }
    return tgt;
  }

  function prefillHeadAblation(layer: number, head: number) {
    ws.update((s) => ({
      draft: { ...s.draft, type: "head_ablation", layer, head, factor: 0, positionsMode: "all" },
      tab: "interventions",
    }));
  }

  return { intervene, launchHeadSweep, setTarget, prefillHeadAblation };
}
