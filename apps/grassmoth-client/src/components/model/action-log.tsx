"use client";

import type {
  ActionId,
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

/** A message's local clock time, to the second. */
export function clockTime(timestamp: string): string {
  return new Date(timestamp).toLocaleTimeString(undefined, { hourCycle: "h23" });
}

/** A finding subject as the identifiers or labels the API gives it. */
function subjectText(subject: FindingSubject): string {
  if (typeof subject === "string") return subject;
  if ("id" in subject) return subject.id;
  if ("construct_id" in subject)
    return [
      subject.construct_id,
      typeof subject.target === "string" ? subject.target : subject.target.id,
    ]
      .filter((part) => part !== null)
      .join(" · ");
  if ("parameter" in subject || "state" in subject) return subject.label;
  if ("outcome" in subject) return subject.outcome.id;
  if ("kind" in subject && subject.kind === "query_target")
    return `Core query · ${subject.target.id}`;
  if ("kind" in subject) return "Core query";
  return subject.target.id;
}

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
  timestamp: string;
  level: string;
  tone: "plain" | "warning" | "error";
  id: string;
  subject: string | null;
  children: ReactNode;
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

function ExecutionEntry({ message }: { message: ExecutionMessage }) {
  switch (message.kind) {
    case "log":
      return (
        <LogEntry
          timestamp={message.timestamp}
          level={message.severity}
          tone={
            message.severity === "warning"
              ? "warning"
              : message.severity === "error"
                ? "error"
                : "plain"
          }
          id={message.code}
          subject={subjectText(message.subject)}
        >
          {message.detail}
        </LogEntry>
      );
    case "progress":
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
    case "trace":
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
            <pre key={index} className="mt-1 whitespace-pre-wrap [overflow-wrap:anywhere]">
              {detail}
            </pre>
          ))}
        </LogEntry>
      );
    }
    default:
      throw new Error(`Unhandled execution message: ${message satisfies never}`);
  }
}

/** Render the server's ordered accumulator intact for both active and completed calls. */
export function ActionLog({ messages }: { messages: readonly ExecutionMessage[] }) {
  return (
    <ol aria-label="Action messages" className="space-y-3 font-mono text-[10px]">
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
