import type { EpistemicStatus } from "@/lib/epistemic";
import type { ApiStatus } from "./types";

/**
 * Map an API status string onto the UI's epistemic ladder.
 * The backend uses "causal" for CAUSALLY TESTED results (interventions, sweeps,
 * benchmark evals); the badge label for it is "Causally tested".
 * Unknown values fall back to DERIVED (never upgrade an unknown claim to causal).
 */
export function toEpistemic(status: ApiStatus | string | null | undefined): EpistemicStatus {
  switch (status) {
    case "observed":
      return "observed";
    case "derived":
      return "derived";
    case "inferred":
      return "inferred";
    case "causal":
    case "causally_tested":
    case "causally tested":
      return "causal";
    default:
      return "derived";
  }
}
