"use client";

import { attemptError } from "@/lib/model-asset/journal";

import type { ActionMessage, LLMTrace, RunningAction } from "@nof1-causal-lab/api-types";
import { LoaderCircle, X } from "lucide-react";
import { useMemo } from "react";
import { ChatMessages } from "@/components/ui/custom/chat-messages";
import { useAttemptProgress } from "@/lib/hooks/use-attempt-progress";
import type { TimelineRevision } from "@nof1-causal-lab/api-types";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { humanize } from "@/lib/model-asset/selection";
import { timelineTickLabel } from "@/lib/model-asset/timeline-presentation";
import { cn } from "@/lib/utils";
import { traceToUIMessages } from "@/lib/utils/trace-to-ui-messages";
import { ActionFindings } from "./action-findings";
import { Hint, Section } from "./scope-primitives";
import { DataComparisonOutcome, DataDetails } from "./scopes/data-details";
import { EditDetails } from "./scopes/edit-details";
import { QuestionDetails } from "./scopes/question-details";
import { FitOutcome } from "./scopes/fit-details";

export type ActionTraceState =
  | { status: "loading" }
  | { status: "ready"; trace: LLMTrace }
  | { status: "absent" };
export type UseActionTrace = (seq: number, enabled: boolean) => ActionTraceState;

function ActionReasoning({ reasoning }: { reasoning: string | null | undefined }) {
  if (!reasoning) return null;
  return (
    <section aria-label="Reasoning" className="px-4 py-1">
      <h3 className="mb-1 text-[11px] font-semibold">Reasoning</h3>
      <p className="whitespace-pre-wrap break-words text-[11.5px] leading-relaxed">{reasoning}</p>
    </section>
  );
}

function ActionLabels({ messages }: { messages: readonly ActionMessage[] }) {
  return (
    <ul className="space-y-1 px-4 py-1 text-[10px] font-mono" aria-label="Action messages">
      {messages.map((message, index) => (
        <li
          key={`${message.timestamp}:${index}`}
          className={cn(
            "flex flex-wrap gap-x-2 text-muted-foreground",
            message.severity === "warning" && "text-amber-700",
            message.severity === "error" && "text-destructive",
          )}
        >
          <time dateTime={message.timestamp} title={message.timestamp}>
            {new Date(message.timestamp).toLocaleTimeString()}
          </time>
          <span>{message.severity}</span>
          <span className="break-all">{message.code}</span>
          <span>{message.detail}</span>
        </li>
      ))}
    </ul>
  );
}

/**
 * What a running data preparation has reported: each step's latest status and the extraction
 * workers' counts. No other action reports progress, so their running entries show messages alone.
 */
function AttemptProgress({
  workspaceId,
  running,
}: {
  workspaceId: string;
  running: RunningAction;
}) {
  const progress = useAttemptProgress(workspaceId, running);
  const view = progress.data;
  // Nothing retained for this attempt: progress is unknown, not zero.
  if (view.cursor === null)
    return <p className="px-4 text-[10px] text-muted-foreground">Progress unavailable.</p>;
  const finished = Object.values(view.workers).filter(
    (worker) => worker.state === "completed" || worker.state === "failed",
  );
  return (
    <dl
      aria-label="Progress"
      className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 px-4 py-1 font-mono text-[10px] text-muted-foreground"
    >
      {view.extraction && (
        <>
          <dt>extraction</dt>
          <dd
            className={cn("break-all", view.extraction.status === "failed" && "text-destructive")}
          >
            {view.extraction.status}
            {view.extraction.error &&
              ` · ${view.extraction.error.type}: ${view.extraction.error.message}`}
          </dd>
        </>
      )}
      {view.snapshot && (
        <>
          <dt>workers</dt>
          <dd>
            {view.snapshot.completed_workers}/{view.snapshot.total_workers} completed ·{" "}
            {view.snapshot.running_workers} running · {view.snapshot.pending_workers} pending ·{" "}
            {view.snapshot.failed_workers} failed
          </dd>
        </>
      )}
      {finished.length > 0 && (
        <>
          <dt>LLM calls</dt>
          <dd>
            {finished.reduce((calls, worker) => calls + (worker.n_llm_calls ?? 0), 0)} reported by{" "}
            {finished.length} finished {finished.length === 1 ? "worker" : "workers"}
          </dd>
        </>
      )}
    </dl>
  );
}

/** The model calls recorded inside the action, failed attempts included. */
function ActionTrace({
  tick,
  useActionTrace,
}: {
  tick: TimelineRevision;
  useActionTrace: UseActionTrace;
}) {
  const traceState = useActionTrace(tick.record.seq, tick.record.trace_ids.length > 0);
  const messages = useMemo(
    () => (traceState.status === "ready" ? traceToUIMessages(traceState.trace) : []),
    [traceState],
  );
  if (tick.record.trace_ids.length === 0) return null;
  if (traceState.status === "loading")
    return (
      <p role="status" className="px-4 text-xs text-muted-foreground">
        Loading conversation…
      </p>
    );
  if (traceState.status === "absent")
    return <p className="px-4 text-xs text-muted-foreground">Conversation unavailable.</p>;
  return (
    <div className="px-4 [&_.prose]:text-[11.5px] [&_pre]:text-[10px]">
      <ChatMessages messages={messages} />
    </div>
  );
}

/**
 * What the selected action did and what came of it, stated rather than shown: its verdicts, its
 * warnings with their reasons, and the results that matter. The state the action left belongs to
 * the graph and the details pane.
 * - edit_question: the question it set, as the root of every lineage.
 * - edit_model: what changed, in model terms, and what its checks concluded (identification kept
 *   or lost, which predictive checks failed and why).
 * - prepare_data: which data arrived (source, window, variables, volume) and the problems found.
 * - fit: how it ran, in one line, and its verdict, such as parameter convergence failing for 7 of
 *   7 parameters with a worst R-hat of 2.3.
 * - simulate: the simulator's log and nothing more; everything the simulation produced is state.
 * - data_diff: which saved observations were compared, the failing checks and their reasons,
 *   and why a variable could not be compared. The comparison evidence belongs to the state panes.
 * Executing calls have no completed timeline node yet, so each streams its own progress.
 */
export function ActionRecord({
  workspaceId,
  context,
  tick,
  running,
  useActionTrace,
}: {
  workspaceId: string;
  context: ScopeContext | null;
  tick: TimelineRevision | undefined;
  running: readonly RunningAction[];
  useActionTrace: UseActionTrace;
}) {
  const request = tick?.record.attempt.request;
  const call =
    tick?.record.attempt.outcome.status === "applied" && tick.record.attempt.request !== null
      ? context?.result
      : undefined;
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="flex flex-none items-center gap-2 border-b px-3 py-2.5">
        <h2 className="text-xs font-semibold">Action record</h2>
        {tick && (
          <span className="ml-auto flex min-w-0 items-center gap-1 font-mono text-[10px] text-muted-foreground">
            {tick.record.attempt.outcome.status !== "applied" && (
              <X className="size-3 flex-none text-destructive" aria-hidden="true" />
            )}
            <span className="truncate">{timelineTickLabel(tick)}</span>
            {tick.record.attempt.outcome.status !== "applied" && (
              <span className="sr-only"> · failed</span>
            )}
          </span>
        )}
      </div>
      <div
        key={tick?.record.seq}
        className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto px-3 py-2"
      >
        {tick ? (
          <>
            <ActionReasoning reasoning={tick.record.attempt.request?.reasoning} />
            {tick.record.attempt.outcome.status !== "applied" ? (
              <Section
                title={`${humanize(tick.record.attempt.action).replace(/^./, (letter) => letter.toUpperCase())} failed`}
              >
                <Hint issue>{attemptError(tick.record.attempt.outcome)}</Hint>
                <ActionLabels messages={tick.record.messages} />
              </Section>
            ) : tick.record.attempt.request === null ? (
              <Section title="Unknown call">
                <Hint>This attempt has no retained call arguments.</Hint>
                <ActionLabels messages={tick.record.messages} />
              </Section>
            ) : tick.record.attempt.action === "simulate" ? (
              <section role="log" aria-label="Simulator log">
                <ActionLabels messages={tick.record.messages} />
              </section>
            ) : context ? (
              <>
                {call?.action === "edit_question" && (
                  <QuestionDetails context={context} question={call.body.question} />
                )}
                {tick.record.attempt.action === "edit_model" && (
                  <EditDetails context={context} tick={tick} />
                )}
                {call?.action === "prepare_data" && request?.action === "prepare_data" && (
                  <DataDetails applied={call.body} source={request.input.source} />
                )}
                {tick.record.attempt.action === "fit" && <FitOutcome context={context} />}
                {call?.action === "data_diff" && context.dataDiff && (
                  <DataComparisonOutcome context={context} report={context.dataDiff} />
                )}
                {call?.action === "model_diff" && (
                  <Section title="Model comparison">
                    <Hint>
                      Compared the selected model definitions, laws, checks and saved evidence.
                    </Hint>
                    <ActionLabels messages={tick.record.messages} />
                  </Section>
                )}
                {call && <ActionFindings context={context} call={call} />}
              </>
            ) : null}
            {tick.record.attempt.action !== "simulate" && tick.record.trace_ids.length > 0 && (
              <details className="px-3 text-xs">
                <summary className="cursor-pointer text-muted-foreground">
                  Agent conversation
                </summary>
                <ActionTrace tick={tick} useActionTrace={useActionTrace} />
              </details>
            )}
          </>
        ) : (
          running.length === 0 && (
            <p className="px-4 text-xs text-muted-foreground">No actions yet.</p>
          )
        )}
        {running.map((call) => (
          <section
            key={call.call_id}
            role="log"
            aria-live="polite"
            aria-label={`${call.action} · running`}
            className="border-t pt-2 first:border-t-0 first:pt-0"
          >
            <h3 className="flex items-center gap-2 px-4 pb-1 text-[11px] font-medium">
              <LoaderCircle className="size-3.5 animate-spin" aria-hidden="true" />
              {call.action} · running
            </h3>
            <ActionReasoning reasoning={call.request.reasoning} />
            <ActionLabels messages={call.messages.filter((message) => message.kind === "log")} />
            {call.action === "prepare_data" && (
              <AttemptProgress workspaceId={workspaceId} running={call} />
            )}
          </section>
        ))}
      </div>
    </div>
  );
}
