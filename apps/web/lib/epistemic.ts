/**
 * Epistemic status: every quantity or claim shown in NeuralScope carries exactly
 * one of these labels. They are ordered by how close the claim is to a tested
 * mechanism, NOT by numerical certainty: an OBSERVED activation is an exact
 * measurement, while an INFERRED interpretation is a hypothesis.
 *
 * Colours mirror the --ns-* CSS variables in app/globals.css.
 */

export type EpistemicStatus = "observed" | "derived" | "inferred" | "causal";

export interface EpistemicInfo {
  status: EpistemicStatus;
  /** Uppercase label shown in badges. */
  label: string;
  /** CSS custom property holding the colour, e.g. "--ns-observed". */
  colorVar: `--ns-${string}`;
  /** Ready-to-use CSS colour value, e.g. "var(--ns-observed)". */
  color: string;
  /** Literal hex (for canvas or other non-CSS consumers). */
  hex: string;
  /** One-sentence definition (used for tooltips). */
  definition: string;
  /** Canonical example used across docs and marketing. */
  example: string;
}

export const EPISTEMIC_ORDER: readonly EpistemicStatus[] = ["observed", "derived", "inferred", "causal"] as const;

export const EPISTEMIC: Readonly<Record<EpistemicStatus, EpistemicInfo>> = {
  observed: {
    status: "observed",
    label: "Observed",
    colorVar: "--ns-observed",
    color: "var(--ns-observed)",
    hex: "#7cc7f5",
    definition: "Direct tensor or model measurement recorded during a forward pass.",
    example: "Residual-stream activation at layer 14, token 7.",
  },
  derived: {
    status: "derived",
    label: "Derived",
    colorVar: "--ns-derived",
    color: "var(--ns-derived)",
    hex: "#b39dfa",
    definition: "Mathematical transformation of observed quantities (e.g. a projection, norm or decomposition).",
    example: "Logit Lens projection of that activation onto the vocabulary.",
  },
  inferred: {
    status: "inferred",
    label: "Inferred",
    colorVar: "--ns-inferred",
    color: "var(--ns-inferred)",
    hex: "#f2b45a",
    definition: "Semantic interpretation or hypothesis about what a component computes. Not yet tested.",
    example: "Possible arithmetic carry representation (hypothesis).",
  },
  causal: {
    status: "causal",
    label: "Causally tested",
    colorVar: "--ns-causal",
    color: "var(--ns-causal)",
    hex: "#5fd3a0",
    definition: "Claim that survived an actual intervention: the model was rerun with the component changed and the effect was measured.",
    example: "Ablating this component lowers correct-answer probability across 100 controlled examples.",
  },
};

export function isEpistemicStatus(value: unknown): value is EpistemicStatus {
  return typeof value === "string" && value in EPISTEMIC;
}
