"use client";

import { useState } from "react";
import { Bot, Lightbulb } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Panel } from "@/components/ui/Panel";
import { StatusChip } from "@/components/ui/StatusChip";
import { ErrorBox, Loading } from "@/components/workspace/ui";
import { selflab, type SuggestResponse } from "@/lib/api/selflab";
import { FeatureBadge } from "./marks";
import { useSelfLab } from "./SelfLabProvider";

/** Rule-based suggestions with the LLM investigator shown as unsupported. */
export function SuggestionsView({
  res,
  onUse,
  available,
}: {
  res: SuggestResponse;
  onUse: (template: string) => void;
  available: (template: string) => boolean;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <p className="flex flex-wrap items-center gap-1.5 text-[10.5px] text-ns-text-muted">
        <StatusChip tone="neutral">Rule-based</StatusChip>
        <span>
          participant <code className="text-ns-text">{res.participant}</code> · deterministic rules, not a language model.{" "}
          {res.disclaimer ?? "Suggestions are hypotheses about which test would be informative; they are not evidence."}
        </span>
      </p>
      {res.suggestions.length === 0 ? (
        <p className="text-[11px] text-ns-text-faint">The rules found no applicable template for this case.</p>
      ) : (
        <ul className="flex flex-col gap-1" aria-label="Suggested tests">
          {res.suggestions.map((s, i) => (
            <li
              key={`${s.template}-${i}`}
              data-applicable={s.applicable !== false}
              className={`flex items-start gap-2 rounded-[3px] border border-ns-border bg-ns-bg px-2 py-1.5 ${s.applicable === false ? "opacity-70" : ""}`}
            >
              <Lightbulb aria-hidden="true" className="mt-0.5 size-3.5 shrink-0 text-ns-inferred" />
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[11.5px] text-ns-text">{s.title}</span>
                  <code className="text-[10px] text-ns-text-faint">{s.template}</code>
                  {s.feature_status ? <FeatureBadge status={s.feature_status} /> : null}
                  {s.applicable === false ? <StatusChip tone="neutral">not applicable</StatusChip> : null}
                </div>
                {s.rationale ? <p className="text-[10.5px] leading-snug text-ns-text-muted">{s.rationale}</p> : null}
                {s.missing && s.missing.length ? <p className="text-[10.5px] leading-snug text-ns-warn">missing: {s.missing.join(", ")}</p> : null}
                {s.note ? <p className="text-[10.5px] leading-snug text-ns-text-faint">{s.note}</p> : null}
                {s.rule ? <p className="font-mono text-[9.5px] text-ns-text-faint">rule {s.rule}</p> : null}
              </div>
              <Button
                size="xs"
                variant="secondary"
                caps
                disabled={s.applicable === false || s.available === false || !available(s.template)}
                onClick={() => onUse(s.template)}
              >
                Draft
              </Button>
            </li>
          ))}
        </ul>
      )}
      <div className="flex flex-wrap items-center gap-1.5 rounded-[3px] border border-dashed border-ns-border-strong px-2 py-1.5" data-testid="llm-investigator">
        <Button size="xs" variant="ghost" caps disabled aria-disabled="true" title="Unsupported until approved and configured">
          <Bot aria-hidden="true" /> Ask an LLM investigator
        </Button>
        <StatusChip tone="danger">Unsupported</StatusChip>
        <span className="text-[10.5px] text-ns-text-faint">
          {res.llm_investigator?.reason ?? "An LLM investigator is unsupported until it is approved and configured."}
        </span>
      </div>
    </div>
  );
}

export function SuggestPanel() {
  const { caseId, templates, selectTemplate } = useSelfLab();
  const [res, setRes] = useState<{ caseId: string; data: SuggestResponse } | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  if (!caseId) return null;
  const current = res && res.caseId === caseId ? res.data : null;
  const ask = async () => {
    setBusy(true);
    setErr(null);
    try {
      const data = await selflab.suggest(caseId);
      setRes({ caseId, data });
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
    }
  };
  const available = (id: string) => {
    const t = templates.data?.find((x) => x.id === id);
    return Boolean(t && t.feature_status !== "unsupported" && t.feature_status !== "roadmap");
  };
  return (
    <Panel label="What would test this explanation?" bodyClassName="flex flex-col gap-2 p-2.5">
      {!current ? (
        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" variant="secondary" caps onClick={() => void ask()} disabled={busy}>
            <Lightbulb aria-hidden="true" /> {busy ? "Applying rules…" : "Suggest tests"}
          </Button>
          <span className="text-[10.5px] text-ns-text-faint">Deterministic rule-based suggestions (no language model involved).</span>
        </div>
      ) : (
        <SuggestionsView res={current} onUse={(t) => selectTemplate(t)} available={available} />
      )}
      {busy && current ? <Loading compact label="Applying rules" /> : null}
      {err ? <ErrorBox error={err} onRetry={() => void ask()} /> : null}
    </Panel>
  );
}
