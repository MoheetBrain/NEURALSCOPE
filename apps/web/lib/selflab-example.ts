import type { EvidenceKind, Split, TestStatus } from "@/lib/api/selflab";

/**
 * A GENUINE recorded SelfLab example shown on the landing page.
 *
 * Fill this ONLY from an exported reproduction record of a real run
 * (`GET /api/selflab/experiments/{id}/export`). Copy values verbatim; do not round by hand, do not
 * summarise into new numbers, do not invent text. While it is `null` the landing page says the pilot
 * record is pending and links to the workspace instead.
 */
export interface SelfLabRecordedExample {
  /** Always "recorded": there is no placeholder variant. */
  source: "recorded";
  experiment_id: string;
  case_id?: string | null;
  /** Template id, e.g. "source_influence_battery". */
  template: string;
  split: Split;
  /** ISO timestamp of the run (committed_at or finished_at from the record). */
  recorded_at: string;
  /** Target model id from the record's provenance. */
  model_id: string;
  /** True when the run used the DEMO fixture model: the page then marks it as fixture data. */
  is_demo: boolean;
  code_commit?: string | null;
  spec_sha256: string;
  plan_sha256: string;
  /** Repo-relative path of the committed export JSON, when it is checked in. */
  export_path?: string | null;
  /** Optional single task shown verbatim from the record (an invented entity). */
  task?: {
    document?: string | null;
    hint?: string | null;
    question: string;
    response: string;
    claim?: string | null;
    self_report?: string | null;
  } | null;
  /** Rows copied from the analysis summary, already formatted exactly as the record states them. */
  results: { label: string; value: string; detail?: string }[];
  /** Evidence items from the record: kind, status and scope are kept separate. */
  evidence: { title: string; kind: EvidenceKind; status: TestStatus; statement: string; scope: string }[];
  limitations: string[];
}

export const SELFLAB_EXAMPLE: SelfLabRecordedExample | null = null;
