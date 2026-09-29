"use client";

import { useMemo } from "react";
import type { RunDetail } from "@/lib/api/types";
import { layerForDepth, visibleColumns, type MatrixSelection } from "@/lib/matrix";
import { useWorkspace } from "./WorkspaceProvider";

/** Default selection for a run: last depth at the target (prediction) position. */
export function defaultSelection(run: RunDetail): MatrixSelection {
  const D = run.depth_labels.length;
  const pos = run.target?.position ?? Math.max(0, run.prompt_len - 1);
  return { depth: D - 1, position: Math.min(pos, run.tokens.length - 1) };
}

/** The selection all linked panels use (explicit selection, else the default). */
export function useSelection(run: RunDetail | undefined): MatrixSelection | null {
  const { state } = useWorkspace();
  if (!run) return null;
  const s = state.selection;
  if (s && s.depth >= 0 && s.depth < run.depth_labels.length && s.position >= 0 && s.position < run.tokens.length) return s;
  return defaultSelection(run);
}

/** Attention layer linked to the selection (or the explicit override). */
export function useAttentionLayer(run: RunDetail | undefined, nLayers: number): number {
  const { state } = useWorkspace();
  const sel = useSelection(run);
  if (state.attnLayer !== null) return Math.min(Math.max(0, state.attnLayer), nLayers - 1);
  return sel ? layerForDepth(sel.depth, nLayers) : nLayers - 1;
}

/** Visible matrix columns (system template collapsed when requested). */
export function useColumns(run: RunDetail | undefined): number[] {
  const { state } = useWorkspace();
  const target = run?.target?.position;
  return useMemo(
    () => (run ? visibleColumns(run.tokens, state.hideTemplate, target !== undefined ? [target] : []) : []),
    [run, state.hideTemplate, target],
  );
}

/** Target token id for view requests (explicit so caches key on it). */
export function targetId(run: RunDetail | undefined): number | null {
  return run?.target?.token_id ?? null;
}
