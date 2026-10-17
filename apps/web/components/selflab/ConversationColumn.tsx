"use client";

import { useState, type KeyboardEvent } from "react";
import { ExternalLink, FileText, ScanSearch, Send, Sparkles, X } from "lucide-react";
import { Button } from "@/components/ui/Button";
import { Panel } from "@/components/ui/Panel";
import { StatusChip } from "@/components/ui/StatusChip";
import { Caveat, EmptyState, Field, Loading, NumberInput, Segmented, SelectInput, TextInput } from "@/components/workspace/ui";
import {
  CLAIM_TYPES,
  chatExamples,
  isApiOnlyEnabled,
  accessLevels,
  type ClaimType,
  type ContextSegment,
  type Message,
  type Receipt,
} from "@/lib/api/selflab";
import { cn } from "@/lib/cn";
import { shortId } from "@/lib/format";
import { FixtureMark, Hash } from "./marks";
import { SelectableResponse } from "./SelectableResponse";
import { useSelfLab } from "./SelfLabProvider";
import type { CharSpan } from "./spanSelection";
import { STATIC_EXAMPLE_PROMPT } from "./example";
import { ParticipantsPanel } from "./ParticipantsPanel";

export { STATIC_EXAMPLE_PROMPT } from "./example";

const CLAIM_TYPE_LABEL: Record<ClaimType, string> = {
  source_attribution: "source attribution",
  verification: "verification",
  process_description: "process description",
  self_prediction: "self-prediction",
  other: "other",
};

const TEXTAREA =
  "w-full resize-y rounded-[3px] border border-ns-border-strong bg-ns-bg px-2 py-1.5 font-mono text-[11.5px] leading-relaxed text-ns-text " +
  "placeholder:text-ns-text-faint focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-ns-accent disabled:opacity-50";

export function ConversationColumn() {
  return (
    <div className="flex flex-col gap-2">
      <Composer />
      <ResponsePanel />
      <ContextPanel />
      <ParticipantsPanel />
    </div>
  );
}

/* ------------------------------------------------------------------ composer */

function Composer() {
  const { status, mode, setMode, sendChat, busy, chat, caseId, clearConversation, rememberExample } = useSelfLab();
  const [document, setDocument] = useState("");
  const [hint, setHint] = useState("");
  const [question, setQuestion] = useState("");
  const [maxNew, setMaxNew] = useState(24);
  const [exampleSource, setExampleSource] = useState<"backend" | "static" | null>(null);
  const apiOnly = isApiOnlyEnabled(status.data);
  const apiInfo = accessLevels(status.data).find((a) => a.level === "api_only");
  const examples = chatExamples(status.data);
  const sending = busy === "chat";

  const loadExample = () => {
    // A backend preset wins; the static example is the fallback. Either way its counterfactual inputs
    // (replacement document, hint for the hint-added branch, values) pre-fill the case template's params.
    const ex = examples[0] ?? STATIC_EXAMPLE_PROMPT;
    setDocument(ex.document ?? "");
    setHint(ex.hint ?? "");
    setQuestion(ex.question);
    setExampleSource(examples[0] ? "backend" : "static");
    rememberExample(ex);
  };

  const submit = () => {
    if (!question.trim() || sending) return;
    void sendChat({
      question: question.trim(),
      document: document.trim() || null,
      hint: hint.trim() || null,
      max_new_tokens: maxNew,
    });
  };
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <Panel
      label="Target conversation"
      right={
        chat || caseId ? (
          <button
            type="button"
            onClick={clearConversation}
            className="inline-flex items-center gap-1 font-mono text-[10px] uppercase tracking-[0.08em] text-ns-text-muted hover:text-ns-text"
          >
            <X aria-hidden="true" className="size-3" /> new
          </button>
        ) : null
      }
      bodyClassName="flex flex-col gap-2 p-2.5"
    >
      <Segmented<"white_box_local" | "api_only">
        label="Access level"
        size="xs"
        value={mode}
        onChange={setMode}
        items={[
          { value: "white_box_local", label: "White-box local", title: "The target runs locally; activations are recorded for every turn." },
          {
            value: "api_only",
            label: "API-only view",
            disabled: !apiOnly,
            title: apiOnly ? "Paste a transcript from an API-only model (no activations)." : `Disabled: ${apiInfo?.reason ?? "not enabled by the backend"}`,
          },
        ]}
      />
      {mode === "api_only" && apiOnly ? (
        <ApiOnlyComposer />
      ) : (
        <>
          <Field label="Document (optional)" hint="Delivered inside the user message; its span is recorded.">
            <textarea
              aria-label="Document"
              value={document}
              onChange={(e) => setDocument(e.target.value)}
              onKeyDown={onKey}
              rows={3}
              maxLength={6000}
              placeholder="Paste a short document…"
              className={TEXTAREA}
            />
          </Field>
          <Field label="User hint (optional)">
            <TextInput aria-label="User hint" value={hint} onChange={(e) => setHint(e.target.value)} maxLength={600} placeholder="e.g. I think the answer is …" />
          </Field>
          <Field label="Question">
            <textarea
              aria-label="Question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={onKey}
              rows={2}
              maxLength={2000}
              placeholder="Ask the target…"
              className={TEXTAREA}
            />
          </Field>
          <div className="flex flex-wrap items-end gap-2">
            <button
              type="button"
              onClick={loadExample}
              className="inline-flex h-7 items-center gap-1 rounded-[3px] border border-ns-accent-2/35 bg-ns-accent-2/8 px-2 font-mono text-[10px] text-ns-accent-2 hover:border-ns-accent-2/70"
              title={examples[0] ? "Preset offered by /api/selflab/status" : "Static example prompt (input text, not a result)"}
            >
              <Sparkles aria-hidden="true" className="size-3" /> Load example
            </button>
            <Field label="max new tokens" className="w-32">
              <NumberInput value={maxNew} min={1} max={64} onValue={(v) => setMaxNew(Math.min(64, Math.max(1, Math.round(v))))} aria-label="Max new tokens" />
            </Field>
          </div>
          {exampleSource ? (
            <p className="rounded-[3px] border border-dashed border-ns-accent-2/40 px-2 py-1 text-[10.5px] leading-snug text-ns-accent-2" data-testid="example-label">
              {exampleSource === "backend"
                ? "Example prompt loaded from the backend preset. It is input for the target, not a result."
                : `Example prompt (static text, invented entity). It is input for the target, not a result. Its replacement document ("…speckled ${STATIC_EXAMPLE_PROMPT.replacement_value}.") and suggestion ("${STATIC_EXAMPLE_PROMPT.hint_text}") are held back and pre-fill COMPARE CONTEXTS.`}
            </p>
          ) : null}
          <Button variant="primary" caps onClick={submit} disabled={!question.trim() || sending} className="w-full" aria-keyshortcuts="Control+Enter Meta+Enter">
            <Send aria-hidden="true" /> {sending ? "Sending…" : "Send to target"}
          </Button>
          {sending ? <Loading compact label="Generating greedily and recording the run" /> : null}
        </>
      )}
    </Panel>
  );
}

/* ------------------------------------------------------------------ API-only composer */

function ApiOnlyComposer() {
  const { createCase, busy } = useSelfLab();
  const [userMsg, setUserMsg] = useState("");
  const [response, setResponse] = useState("");
  const [span, setSpan] = useState<CharSpan | null>(null);
  const [claimType, setClaimType] = useState<ClaimType>("source_attribution");
  return (
    <div className="flex flex-col gap-2">
      <p className="rounded-[3px] border border-ns-border-strong px-2 py-1 text-[10.5px] leading-snug text-ns-text-muted">
        <StatusChip tone="neutral" className="mr-1.5">
          API-only view
        </StatusChip>
        No activations are available. Only black-box comparisons can be run on this case.
      </p>
      <Field label="User message">
        <textarea aria-label="User message" value={userMsg} onChange={(e) => setUserMsg(e.target.value)} rows={3} className={TEXTAREA} />
      </Field>
      <Field label="Response (as returned by the API)">
        <textarea
          aria-label="Pasted response"
          value={response}
          onChange={(e) => {
            setResponse(e.target.value);
            setSpan(null);
          }}
          rows={3}
          className={TEXTAREA}
        />
      </Field>
      {response ? <SelectableResponse text={response} value={span} onChange={setSpan} label="Pasted response" /> : null}
      <ClaimTypeSelect value={claimType} onChange={setClaimType} />
      <Button
        variant="primary"
        caps
        disabled={!span || !userMsg.trim() || busy === "case"}
        onClick={() =>
          span &&
          void createCase({
            access_level: "api_only",
            messages: [{ role: "user", content: userMsg }],
            response_text: response,
            char_start: span.char_start,
            char_end: span.char_end,
            claim_type: claimType,
          })
        }
      >
        <ScanSearch aria-hidden="true" /> Investigate claim
      </Button>
    </div>
  );
}

function ClaimTypeSelect({ value, onChange }: { value: ClaimType; onChange: (v: ClaimType) => void }) {
  return (
    <Field label="Claim type">
      <SelectInput aria-label="Claim type" value={value} onChange={(e) => onChange(e.target.value as ClaimType)}>
        {CLAIM_TYPES.map((c) => (
          <option key={c} value={c}>
            {CLAIM_TYPE_LABEL[c]}
          </option>
        ))}
      </SelectInput>
    </Field>
  );
}

/* ------------------------------------------------------------------ response + investigate */

function ResponsePanel() {
  const { chat, caseId, caseDetail, createCase, busy, mode, openRunInLab } = useSelfLab();
  const [span, setSpan] = useState<CharSpan | null>(null);
  const [claimType, setClaimType] = useState<ClaimType>("source_attribution");
  const [title, setTitle] = useState("");
  const detail = caseDetail.data?.case.id === caseId ? caseDetail.data : undefined;
  const c = detail?.case;
  const text = c ? c.response_text : chat?.response_text;
  const runId = c ? c.run_id : chat?.run_id;
  const fixture = c ? c.is_fixture : chat?.is_fixture;
  const sourceKey = c ? `case:${c.id}` : chat ? `chat:${chat.run_id}` : "none";
  const [spanKey, setSpanKey] = useState(sourceKey);
  // reset the pending selection when the response changes (derived during render, no effect)
  if (spanKey !== sourceKey) {
    setSpanKey(sourceKey);
    setSpan(null);
  }

  if (mode === "api_only" && !c) return null;
  if (caseId && caseDetail.loading && !detail) return <Loading label="Loading case" />;
  if (text === undefined) {
    return (
      <Panel label="Target response">
        <EmptyState title="No response yet">
          Send a question to the target (or load the example). Then select the span of its answer you want to audit and press INVESTIGATE CLAIM.
        </EmptyState>
      </Panel>
    );
  }

  return (
    <Panel
      label="Target response"
      status="observed"
      right={<FixtureMark show={fixture} what="This response" />}
      bodyClassName="flex flex-col gap-2 p-2.5"
    >
      <div className="flex flex-wrap items-center gap-1.5 font-mono text-[10px] text-ns-text-faint">
        {runId ? (
          <button
            type="button"
            onClick={() => openRunInLab(runId)}
            className="inline-flex items-center gap-0.5 text-ns-observed hover:underline"
            title="Open this Lab run in the neural debugger"
          >
            <ExternalLink aria-hidden="true" className="size-3" /> run {shortId(runId)}
          </button>
        ) : (
          <span>no Lab run (API-only)</span>
        )}
        {!c && chat ? (
          <span>
            · greedy · {chat.n_generated_tokens ?? "?"} tok · stopped by {chat.stopped_by ?? "—"}
          </span>
        ) : null}
        {c ? (
          <>
            <span>· case {c.id}</span>
            <StatusChip tone={c.access_level === "api_only" ? "neutral" : "accent"}>
              {c.access_level === "api_only" ? "API-only view" : "white-box local"}
            </StatusChip>
          </>
        ) : null}
      </div>
      <SelectableResponse text={text} value={span} onChange={setSpan} claim={c?.claim ?? null} />
      {c ? (
        <p className="text-[10.5px] leading-snug text-ns-text-faint">
          The marked span is the claim under audit. It is output from the target: the object of the test, not evidence.
        </p>
      ) : null}
      {runId ? (
        <>
          <div className="grid grid-cols-2 gap-2">
            <ClaimTypeSelect value={claimType} onChange={setClaimType} />
            <Field label="Title (optional)">
              <TextInput aria-label="Case title" value={title} onChange={(e) => setTitle(e.target.value)} maxLength={120} />
            </Field>
          </div>
          <Button
            variant="primary"
            caps
            disabled={!span || busy === "case"}
            title={span ? undefined : "Select a span of the response first"}
            onClick={() =>
              span &&
              void createCase({
                run_id: runId,
                char_start: span.char_start,
                char_end: span.char_end,
                claim_type: claimType,
                title: title.trim() || undefined,
              })
            }
          >
            <ScanSearch aria-hidden="true" /> {busy === "case" ? "Creating case…" : c ? "Investigate another claim" : "Investigate claim"}
          </Button>
        </>
      ) : null}
    </Panel>
  );
}

/* ------------------------------------------------------------------ context segments + receipts */

const SEGMENT_CLASS: Record<ContextSegment["kind"], string> = {
  document: "bg-ns-accent-2/15 shadow-[inset_0_-1px_0_var(--ns-accent-2)]",
  user_hint: "bg-ns-warn/12 shadow-[inset_0_-1px_0_var(--ns-warn)]",
  question: "bg-ns-accent/12 shadow-[inset_0_-1px_0_var(--ns-accent)]",
  system: "bg-ns-panel-2",
  other: "",
};

const SEGMENT_LABEL: Record<ContextSegment["kind"], string> = {
  document: "document",
  user_hint: "user hint",
  question: "question",
  system: "system",
  other: "other",
};

function MessageWithSegments({ message, index, segments }: { message: Message; index: number; segments: ContextSegment[] }) {
  const segs = segments.filter((s) => s.message_index === index).sort((a, b) => a.char_start - b.char_start);
  const parts: { text: string; seg: ContextSegment | null }[] = [];
  let at = 0;
  for (const s of segs) {
    if (s.char_start < at || s.char_end > message.content.length) continue;
    if (s.char_start > at) parts.push({ text: message.content.slice(at, s.char_start), seg: null });
    parts.push({ text: message.content.slice(s.char_start, s.char_end), seg: s });
    at = s.char_end;
  }
  if (at < message.content.length) parts.push({ text: message.content.slice(at), seg: null });
  return (
    <div className="rounded-[3px] border border-ns-border bg-ns-bg px-2 py-1.5">
      <div className="ns-label mb-1 text-[9px]">
        #{index} · {message.role}
      </div>
      <div className="max-h-40 overflow-auto whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-ns-text-muted">
        {parts.map((p, i) =>
          p.seg ? (
            <mark key={i} className={cn("rounded-[2px] bg-transparent text-ns-text", SEGMENT_CLASS[p.seg.kind])} title={SEGMENT_LABEL[p.seg.kind]}>
              {p.text}
            </mark>
          ) : (
            <span key={i}>{p.text}</span>
          ),
        )}
      </div>
    </div>
  );
}

function ContextPanel() {
  const { chat, caseId, caseDetail } = useSelfLab();
  const detail = caseDetail.data?.case.id === caseId ? caseDetail.data : undefined;
  const messages = detail?.case.messages ?? chat?.messages;
  const segments = detail?.case.segments ?? chat?.segments ?? [];
  const receipts = detail?.case.receipts ?? chat?.receipts ?? [];
  if (!messages) return null;
  const tokWarning = !detail && chat?.tokenization_consistent === false;
  return (
    <Panel label="Context delivered" status="observed" statusNote="receipts" bodyClassName="flex flex-col gap-2 p-2.5">
      {tokWarning ? (
        <p role="note" className="rounded-[3px] border border-ns-warn/40 bg-ns-warn/8 px-2 py-1 text-[10.5px] leading-snug text-ns-warn">
          Re-tokenising the composed messages did not reproduce the run&apos;s input ids, so token spans are withheld rather than guessed.
        </p>
      ) : null}
      {messages.map((m, i) => (
        <MessageWithSegments key={i} message={m} index={i} segments={segments} />
      ))}
      {segments.length > 0 ? (
        <table className="w-full border-collapse font-mono text-[10.5px]" aria-label="Context segments">
          <thead>
            <tr className="border-b border-ns-border text-left text-ns-text-faint">
              <th className="py-0.5 pr-2 font-normal">segment</th>
              <th className="py-0.5 pr-2 font-normal">chars</th>
              <th className="py-0.5 pr-2 font-normal">tokens</th>
              <th className="py-0.5 font-normal">sha256</th>
            </tr>
          </thead>
          <tbody>
            {segments.map((s, i) => (
              <tr key={i} className="border-b border-ns-border/50">
                <td className="py-0.5 pr-2">
                  <span className={cn("rounded-[2px] px-1 text-ns-text", SEGMENT_CLASS[s.kind])}>{SEGMENT_LABEL[s.kind]}</span>
                  <span className="text-ns-text-faint"> #{s.message_index}</span>
                </td>
                <td className="ns-tabular py-0.5 pr-2 text-ns-text-muted">
                  [{s.char_start}, {s.char_end})
                </td>
                <td className="ns-tabular py-0.5 pr-2 text-ns-text-muted">
                  {s.token_start !== null && s.token_start !== undefined && s.token_end !== null && s.token_end !== undefined ? (
                    `[${s.token_start}, ${s.token_end})`
                  ) : (
                    <span className="text-ns-text-faint" title="No token span recorded (e.g. API-only view)">
                      —
                    </span>
                  )}
                </td>
                <td className="py-0.5">
                  <Hash value={s.sha256} label="segment sha256" />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="text-[11px] text-ns-text-faint">No labelled segments in this context.</p>
      )}
      <Receipts receipts={receipts} />
    </Panel>
  );
}

export function Receipts({ receipts }: { receipts: Receipt[] }) {
  if (receipts.length === 0) return <p className="text-[11px] text-ns-text-faint">No receipts recorded.</p>;
  return (
    <div className="flex flex-col gap-1.5">
      <Caveat>A delivery receipt proves the text reached the target&apos;s input. It proves delivery, not causal use.</Caveat>
      {receipts.map((r) => (
        <div key={r.id} className="rounded-[3px] border border-ns-border bg-ns-bg px-2 py-1.5" data-testid="receipt">
          <div className="flex items-center gap-1.5">
            <FileText aria-hidden="true" className="size-3 text-ns-observed" />
            <span className="ns-label text-[9px]">{r.kind.replace(/_/g, " ")}</span>
            <span className="ml-auto font-mono text-[9.5px] text-ns-text-faint">{r.id}</span>
          </div>
          <p className="mt-1 text-[11px] leading-snug text-ns-text">{r.statement}</p>
          <dl className="mt-1 grid grid-cols-[70px_minmax(0,1fr)] gap-x-2 text-[10.5px] leading-snug">
            <dt className="text-ns-ok">proves</dt>
            <dd className="text-ns-text-muted">{r.proves}</dd>
            <dt className="text-ns-warn">does not</dt>
            <dd className="text-ns-text-muted">{r.does_not_prove}</dd>
          </dl>
        </div>
      ))}
    </div>
  );
}
