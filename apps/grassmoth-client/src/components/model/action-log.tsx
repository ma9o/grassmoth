"use client";

import type {
  ActionId,
  ActionMessage,
  CallId,
  ExecutionMessage,
  FindingSubject,
  LLMTrace,
  ProgressEvent,
} from "@grassmoth/api-types";
import { type ReactNode, useMemo } from "react";
import { ChatMessages } from "@/components/ui/custom/chat-messages";
import type { UseActionPoll } from "@/lib/hooks/use-action-poll";
import { cn } from "@/lib/utils";
import { traceToUIMessages } from "@/lib/utils/trace-to-ui-messages";
import { Hint, RecordSection } from "./scope-primitives";

function Conversation({ trace }: { readonly trace: LLMTrace }) {
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

function ProgressDetails({ event }: { readonly event: ProgressEvent }) {
  switch (event.event) {
    case "grassmoth.step": {
      return (
        <span>
          extraction · {event.status}
          {event.error && ` · ${event.error.type}: ${event.error.message}`}
        </span>
      );
    }
    case "grassmoth.extraction.plan": {
      return (
        <span>
          {event.total_workers} workers planned
          {event.max_concurrent_workers !== null &&
            ` · ${event.max_concurrent_workers} maximum concurrent workers`}
        </span>
      );
    }
    case "grassmoth.extraction.worker": {
      return (
        <span>
          worker {event.worker_id} · {event.state} · {event.n_windows} windows
          {event.n_extractions !== null && ` · ${event.n_extractions} extractions`}
          {event.n_llm_calls !== null && ` · ${event.n_llm_calls} LLM calls`}
          {event.error !== null && ` · ${event.error}`}
        </span>
      );
    }
    case "grassmoth.extraction.snapshot": {
      return (
        <span>
          {event.completed_workers}/{event.total_workers} completed · {event.running_workers}{" "}
          running · {event.pending_workers} pending · {event.failed_workers} failed
        </span>
      );
    }
  }
}

/** A message's local clock time, to the second. */
export function clockTime(timestamp: string): string {
  return new Date(timestamp).toLocaleTimeString(undefined, { hourCycle: "h23" });
}

/** A finding subject as the identifiers or labels the API gives it. */
function subjectText(subject: FindingSubject): string {
  if (typeof subject === "string") return subject;
  if ("id" in subject) return subject.id;
  if ("construct_id" in subject) {
    return [
      subject.construct_id,
      typeof subject.target === "string" ? subject.target : subject.target.id,
    ]
      .filter((part) => part !== null)
      .join(" · ");
  }
  if ("parameter" in subject || "state" in subject) return subject.label;
  if ("outcome" in subject) return subject.outcome.id;
  if ("kind" in subject && subject.kind === "query_target") {
    return `Core query · ${subject.target.id}`;
  }
  if ("kind" in subject) return "Core query";
  return subject.target.id;
}

const SEVERITY_TONE: Record<ActionMessage["severity"], "plain" | "warning" | "error"> = {
  debug: "plain",
  info: "plain",
  warning: "warning",
  error: "error",
};

/**
 * One message as `[time] [level] [id] subject` above its text. Only the level carries color, so
 * a run of warnings stays readable. Messages without a severity use their kind or failure status.
 */
function LogEntry({
  timestamp,
  level,
  tone,
  id,
  subject,
  children,
}: {
  readonly timestamp: string;
  readonly level: string;
  readonly tone: "plain" | "warning" | "error";
  readonly id: string;
  readonly subject: string | null;
  readonly children: ReactNode;
}) {
  return (
    <>
      <p className="-indent-[2ch] pl-[2ch] [overflow-wrap:anywhere]">
        <time className="text-muted-foreground" dateTime={timestamp} title={timestamp}>
          {`[${clockTime(timestamp)}]`}
        </time>{" "}
        <span
          className={cn(
            tone === "warning" && "text-amber-700",
            tone === "error" && "text-destructive",
            tone === "plain" && "text-muted-foreground",
          )}
        >
          {`[${level}]`}
        </span>{" "}
        <span className="font-medium">{`[${id}]`}</span>
        {subject !== null && ` ${subject}`}
      </p>
      <div className="whitespace-pre-wrap [overflow-wrap:anywhere]">{children}</div>
    </>
  );
}

function ExecutionEntry({ message }: { readonly message: ExecutionMessage }) {
  switch (message.kind) {
    case "log": {
      return (
        <LogEntry
          timestamp={message.timestamp}
          level={message.severity}
          tone={SEVERITY_TONE[message.severity]}
          id={message.code}
          subject={subjectText(message.subject)}
        >
          {message.detail}
        </LogEntry>
      );
    }
    case "progress": {
      return (
        <LogEntry
          timestamp={message.timestamp}
          level="progress"
          tone="plain"
          id={message.progress.event}
          subject={null}
        >
          <ProgressDetails event={message.progress} />
        </LogEntry>
      );
    }
    case "trace": {
      return (
        <LogEntry
          timestamp={message.timestamp}
          level="trace"
          tone="plain"
          id={message.trace_id}
          subject={null}
        >
          <details>
            <summary className="cursor-pointer">Agent conversation</summary>
            <Conversation trace={message.trace} />
          </details>
        </LogEntry>
      );
    }
    case "failure": {
      const failure = message.failure;
      return failure.status === "rejected" ? (
        <LogEntry
          timestamp={message.timestamp}
          level={failure.status}
          tone="error"
          id={failure.code}
          subject={subjectText(failure.subject)}
        >
          {failure.detail}
        </LogEntry>
      ) : (
        <LogEntry
          timestamp={message.timestamp}
          level={failure.status}
          tone="error"
          id={failure.error_type}
          subject={null}
        >
          {failure.error_message}
          {failure.details.map((detail, index) => (
            // eslint-disable-next-line react/no-array-index-key -- a failure's recorded details are positional
            <pre key={index} className="mt-1 whitespace-pre-wrap [overflow-wrap:anywhere]">
              {detail}
            </pre>
          ))}
        </LogEntry>
      );
    }
  }
}

function messageKey(message: ExecutionMessage, index: number): string {
  if (message.kind === "trace") return `trace:${message.trace_id}`;
  if (message.kind === "progress") return `progress:${message.progress.cursor}`;
  return `${message.kind}:${message.timestamp}:${index}`;
}

/** Render the server's ordered accumulator intact for both active and completed calls. */
export function ActionLog({ messages }: { readonly messages: readonly ExecutionMessage[] }) {
  return (
    <ol aria-label="Action messages" className="space-y-3 font-mono text-[10px]">
      {messages.map((message, index) => (
        <li key={messageKey(message, index)}>
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
  readonly action: ActionId;
  readonly callId: CallId | null;
  readonly useActionPoll: UseActionPoll;
}) {
  const query = useActionPoll(action, callId);
  return (
    <RecordSection
      title="Log"
      role="log"
      aria-live="polite"
      aria-relevant="additions text"
      aria-label={action === "simulate" ? "Simulator log" : "Action log"}
    >
      {callId === null ? <Hint>This attempt has no retained call identity.</Hint> : null}
      {query.error && (
        <p role="alert" className="text-[11px] leading-relaxed text-destructive">
          Cannot load action log: {query.error.message}
        </p>
      )}
      {callId !== null && !query.data && !query.error && (
        <p role="status" className="text-[11px] leading-relaxed text-muted-foreground">
          Loading action log…
        </p>
      )}
      {query.data &&
        (query.data.messages.length === 0 ? (
          <Hint>No execution messages recorded.</Hint>
        ) : (
          <ActionLog messages={query.data.messages} />
        ))}
    </RecordSection>
  );
}
