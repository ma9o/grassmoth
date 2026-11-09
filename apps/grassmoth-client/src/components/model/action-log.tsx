"use client";

import type {
  ActionId,
  CallId,
  ExecutionMessage,
  LLMTrace,
  ProgressEvent,
} from "@grassmoth/api-types";
import { useMemo } from "react";
import { ChatMessages } from "@/components/ui/custom/chat-messages";
import { applyProgressEvents, EMPTY_ATTEMPT_PROGRESS } from "@/lib/attempt-progress";
import type { UseActionPoll } from "@/lib/hooks/use-action-poll";
import { cn } from "@/lib/utils";
import { traceToUIMessages } from "@/lib/utils/trace-to-ui-messages";

function Conversation({ trace }: { trace: LLMTrace }) {
  const messages = useMemo(() => traceToUIMessages(trace), [trace]);
  return (
    <div className="mt-2 [&_.prose]:text-[11.5px] [&_pre]:text-[10px]">
      <p className="mb-2 text-muted-foreground">
        {trace.model} · {trace.total_time_seconds}s · {trace.usage.input_tokens} input tokens ·{" "}
        {trace.usage.output_tokens} output tokens
        {trace.usage.reasoning_tokens !== null &&
          ` · ${trace.usage.reasoning_tokens} reasoning tokens`}
      </p>
      <ChatMessages messages={messages} />
    </div>
  );
}

function ProgressDetails({ event }: { event: ProgressEvent }) {
  switch (event.event) {
    case "grassmoth.step":
      return (
        <span>
          extraction · {event.status}
          {event.error && ` · ${event.error.type}: ${event.error.message}`}
        </span>
      );
    case "grassmoth.extraction.plan":
      return (
        <span>
          {event.total_workers} workers planned
          {event.max_concurrent_workers !== null &&
            ` · ${event.max_concurrent_workers} maximum concurrent workers`}
        </span>
      );
    case "grassmoth.extraction.worker":
      return (
        <span>
          worker {event.worker_id} · {event.state} · {event.n_windows} windows
          {event.n_extractions !== null && ` · ${event.n_extractions} extractions`}
          {event.n_llm_calls !== null && ` · ${event.n_llm_calls} LLM calls`}
          {event.error !== null && ` · ${event.error}`}
        </span>
      );
    case "grassmoth.extraction.snapshot":
      return (
        <span>
          {event.completed_workers}/{event.total_workers} completed · {event.running_workers}{" "}
          running · {event.pending_workers} pending · {event.failed_workers} failed
        </span>
      );
    default:
      throw new Error(`Unhandled progress event: ${event satisfies never}`);
  }
}

function ExecutionEntry({ message }: { message: ExecutionMessage }) {
  switch (message.kind) {
    case "log":
      return (
        <div
          className={cn(
            "space-y-0.5",
            message.severity === "warning" && "text-amber-700",
            message.severity === "error" && "text-destructive",
          )}
        >
          <p className="break-all">
            <span>{message.severity}</span> · {message.code} ·{" "}
            {typeof message.subject === "string"
              ? message.subject
              : JSON.stringify(message.subject)}
          </p>
          <p className="whitespace-pre-wrap break-words">{message.detail}</p>
        </div>
      );
    case "progress":
      return (
        <div className="space-y-0.5">
          <p
            className="break-all"
            title={`Attempt ${message.progress.attempt_id} · ${message.progress.cursor}`}
          >
            {message.progress.event}
          </p>
          <ProgressDetails event={message.progress} />
        </div>
      );
    case "trace":
      return (
        <details>
          <summary className="cursor-pointer break-all">
            Agent conversation · {message.trace_id}
          </summary>
          <Conversation trace={message.trace} />
        </details>
      );
    case "failure": {
      const failure = message.failure;
      return (
        <div className="space-y-1 text-destructive">
          {failure.status === "rejected" ? (
            <>
              <p className="break-all">
                {failure.code} ·{" "}
                {typeof failure.subject === "string"
                  ? failure.subject
                  : JSON.stringify(failure.subject)}
              </p>
              <p className="whitespace-pre-wrap break-words">{failure.detail}</p>
            </>
          ) : (
            <>
              <p className="whitespace-pre-wrap break-words">
                {failure.error_type}: {failure.error_message}
              </p>
              {failure.details.map((detail, index) => (
                <pre key={index} className="whitespace-pre-wrap break-words">
                  {detail}
                </pre>
              ))}
            </>
          )}
        </div>
      );
    }
    default:
      throw new Error(`Unhandled execution message: ${message satisfies never}`);
  }
}

/** Render the server's ordered accumulator intact for both active and completed calls. */
export function ActionLog({ messages }: { messages: readonly ExecutionMessage[] }) {
  return (
    <ol aria-label="Action messages" className="space-y-3 px-4 py-1 font-mono text-[10px]">
      {messages.map((message, index) => (
        <li
          key={
            message.kind === "trace"
              ? `trace:${message.trace_id}`
              : message.kind === "progress"
                ? `progress:${message.progress.cursor}`
                : `${message.kind}:${message.timestamp}:${index}`
          }
        >
          <time
            className="text-muted-foreground"
            dateTime={message.timestamp}
            title={message.timestamp}
          >
            {new Date(message.timestamp).toLocaleTimeString()}
          </time>
          <ExecutionEntry message={message} />
        </li>
      ))}
    </ol>
  );
}

export function ActionExecution({
  action,
  callId,
  useActionPoll,
}: {
  action: ActionId;
  callId: CallId | null;
  useActionPoll: UseActionPoll;
}) {
  const query = useActionPoll(action, callId);
  return (
    <section
      role="log"
      aria-live="polite"
      aria-relevant="additions text"
      aria-label={action === "simulate" ? "Simulator log" : "Action log"}
    >
      {callId === null ? (
        <p className="px-4 text-xs text-muted-foreground">
          This attempt has no retained call identity.
        </p>
      ) : null}
      {query.error && (
        <p role="alert" className="px-4 text-xs text-destructive">
          Cannot load action log: {query.error.message}
        </p>
      )}
      {callId !== null && !query.data && !query.error && (
        <p role="status" className="px-4 text-xs text-muted-foreground">
          Loading action log…
        </p>
      )}
      {query.data && (
        <>
          {action === "prepare_data" && <AttemptProgress messages={query.data.messages} />}
          {query.data.messages.length === 0 ? (
            <p className="px-4 text-xs text-muted-foreground">No execution messages recorded.</p>
          ) : (
            <ActionLog messages={query.data.messages} />
          )}
        </>
      )}
    </section>
  );
}

/**
 * What a running data preparation has reported: each step's latest status and the extraction
 * workers' counts. No other action reports progress, so their running entries show messages alone.
 */
function AttemptProgress({ messages }: { messages: readonly ExecutionMessage[] }) {
  const view = applyProgressEvents(
    EMPTY_ATTEMPT_PROGRESS,
    messages.flatMap((message) => (message.kind === "progress" ? [message.progress] : [])),
  );
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
