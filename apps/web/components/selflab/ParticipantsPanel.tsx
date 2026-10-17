"use client";

import { Lock, Unlock } from "lucide-react";
import { Panel } from "@/components/ui/Panel";
import { StatusChip } from "@/components/ui/StatusChip";
import { ErrorBox, Loading } from "@/components/workspace/ui";
import { accessLevels, allParticipants, statusModel, type Participant, type ParticipantRole, type SelfLabStatus } from "@/lib/api/selflab";
import { cn } from "@/lib/cn";
import { FixtureMark, ROLE_LABEL, ROLE_MEANING } from "./marks";
import { useSelfLab } from "./SelfLabProvider";

const ROLE_ORDER: ParticipantRole[] = ["target", "investigator", "activation_decoder", "statistical_scorer", "observer"];

/** An LLM investigator is any enabled investigator that is not a rule set. */
export function llmInvestigatorEnabled(participants: Participant[]): boolean {
  return participants.some((p) => p.role === "investigator" && p.enabled !== false && !p.id.startsWith("rules:"));
}

export function ParticipantsView({ status, participants }: { status: SelfLabStatus | undefined; participants: Participant[] }) {
  const levels = accessLevels(status);
  const model = statusModel(status);
  return (
    <div className="flex flex-col gap-2.5">
      <section aria-label="Access level">
        <h4 className="ns-label mb-1 text-[9.5px]">Access level</h4>
        <ul className="flex flex-col gap-1">
          {levels.map((a) => (
            <li
              key={a.level}
              data-testid={`access-${a.level}`}
              data-enabled={a.enabled}
              className={cn(
                "flex items-start gap-2 rounded-[3px] border px-2 py-1.5",
                a.enabled ? "border-ns-border bg-ns-bg" : "ns-hatch border-dashed border-ns-border-strong opacity-80",
              )}
            >
              {a.enabled ? <Unlock aria-hidden="true" className="mt-0.5 size-3 text-ns-ok" /> : <Lock aria-hidden="true" className="mt-0.5 size-3 text-ns-text-faint" />}
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-1.5">
                  <span className="text-[11.5px] text-ns-text">{a.level === "white_box_local" ? "White-box local" : "API-only view"}</span>
                  <StatusChip tone={a.enabled ? "ok" : "neutral"}>{a.enabled ? "enabled" : "disabled"}</StatusChip>
                </div>
                <p className="mt-0.5 text-[10.5px] leading-snug text-ns-text-faint">
                  {a.description ??
                    (a.level === "white_box_local"
                      ? "The target is loaded locally: every token, activation and intervention can be recorded."
                      : "Transcripts from a model reachable only through an API: no activations, black-box tests only.")}
                  {!a.enabled && a.reason ? ` Disabled: ${a.reason}.` : null}
                </p>
              </div>
            </li>
          ))}
        </ul>
      </section>
      {ROLE_ORDER.map((role) => {
        const ps = participants.filter((p) => p.role === role);
        if (role === "observer" && ps.length === 0) return null;
        return (
          <section key={role} aria-label={ROLE_LABEL[role]} data-testid={`role-${role}`}>
            <h4 className="ns-label text-[9.5px] text-ns-text">{ROLE_LABEL[role]}</h4>
            <p className="mt-0.5 text-[10.5px] leading-snug text-ns-text-faint">{ROLE_MEANING[role]}</p>
            {ps.length === 0 ? (
              <p className="mt-1 font-mono text-[10.5px] text-ns-text-muted">
                {role === "target" && model ? model.model_id : "none reported for this context"}
              </p>
            ) : (
              <ul className="mt-1 flex flex-col gap-1">
                {ps.map((p) => (
                  <li key={`${p.role}-${p.id}`} className={cn("rounded-[3px] border border-ns-border bg-ns-bg px-2 py-1", p.enabled === false && "opacity-60")}>
                    <div className="flex items-center gap-1.5">
                      <span className="truncate font-mono text-[11px] text-ns-text" title={p.id}>
                        {p.id}
                      </span>
                      {p.status === "unsupported" ? <StatusChip tone="danger">unsupported</StatusChip> : null}
                      {p.enabled === false ? <StatusChip tone="neutral">disabled</StatusChip> : null}
                    </div>
                    {p.description ? <p className="text-[10.5px] leading-snug text-ns-text-muted">{p.description}</p> : null}
                    {p.reason ? <p className="text-[10.5px] leading-snug text-ns-text-faint">{p.reason}</p> : null}
                    {p.sees && p.sees.length > 0 ? (
                      <p className="mt-0.5 flex flex-wrap gap-1">
                        <span className="text-[10px] text-ns-text-faint">sees:</span>
                        {p.sees.map((s) => (
                          <code key={s} className="rounded-[2px] bg-ns-panel-2 px-1 text-[9.5px] text-ns-text-muted">
                            {s}
                          </code>
                        ))}
                      </p>
                    ) : null}
                  </li>
                ))}
              </ul>
            )}
            {role === "investigator" && !llmInvestigatorEnabled(participants) ? (
              <p className="mt-1 flex items-center gap-1.5 text-[10.5px] text-ns-text-faint" data-testid="llm-investigator-unsupported">
                <StatusChip tone="danger">Unsupported</StatusChip>
                LLM investigator: unsupported until approved and configured.
              </p>
            ) : null}
          </section>
        );
      })}
    </div>
  );
}

export function ParticipantsPanel() {
  const { status, caseId, caseDetail } = useSelfLab();
  const detail = caseDetail.data?.case.id === caseId ? caseDetail.data : undefined;
  const participants = allParticipants(status.data, detail?.case.participants);
  const model = statusModel(status.data);
  return (
    <Panel
      label="Participants & access"
      right={<FixtureMark show={Boolean(model?.is_demo || status.data?.is_demo)} what="The target" />}
      bodyClassName="p-2.5"
    >
      {status.error && !status.data ? <ErrorBox error={status.error} onRetry={status.reload} /> : null}
      {!status.data && !status.error ? <Loading compact label="Loading /api/selflab/status" /> : null}
      {status.data ? <ParticipantsView status={status.data} participants={participants} /> : null}
    </Panel>
  );
}
