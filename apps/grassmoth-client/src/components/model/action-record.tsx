"use client";

import type { RecordDependency, RunningAction, TimelineRevision } from "@grassmoth/api-types";
import { LoaderCircle } from "lucide-react";
import type { ReactNode } from "react";
import type { UseActionPoll } from "@/lib/hooks/use-action-poll";
import { inputSources } from "@/lib/model-asset/call-dependencies";
import { timelineTickLabel } from "@/lib/model-asset/timeline-presentation";
import { ActionInput } from "./action-input";
import { ActionExecution, clockTime } from "./action-log";
import { ACTION_STYLE, ActionMark } from "./action-mark";
import { Hint } from "./scope-primitives";
import { PaneToggle, usePane } from "./workbench-panes";

/**
 * One call's input above its log. Each scrolls on its own, and the rows share the height: a
 * section shorter than half keeps its natural height and the other takes the rest.
 */
function CallSections({ children }: { readonly children: ReactNode }) {
  return (
    <div className="grid min-h-0 flex-1 grid-rows-[minmax(0,auto)_minmax(0,auto)] content-start divide-y">
      {children}
    </div>
  );
}

function RecordHeading({ tick }: { readonly tick: TimelineRevision }) {
  const { action, outcome } = tick.record.attempt;
  const style = ACTION_STYLE[action];
  return (
    <span className="flex min-w-0 items-center gap-1.5 font-mono text-[10px] whitespace-nowrap text-muted-foreground">
      <ActionMark icon={style.icon} color={style.color} failed={outcome.status !== "applied"} />
      <span className="truncate">{timelineTickLabel(tick)}</span>
      {outcome.status !== "applied" && (
        <span className="flex-none text-destructive">· {outcome.status}</span>
      )}
      <time
        dateTime={tick.record.ts}
        title={new Date(tick.record.ts).toLocaleString()}
        className="flex-none"
      >
        · {clockTime(tick.record.ts)}
      </time>
    </span>
  );
}

/**
 * The selected action's recorded input followed by its ordered execution messages.
 * Running calls have no completed timeline node yet, so each shows its own input and log.
 */
export function ActionRecord({
  tick,
  attempts,
  dependencies,
  running,
  useActionPoll,
  hrefAt,
  onSelectVersion,
}: {
  readonly tick: TimelineRevision | undefined;
  readonly attempts: readonly TimelineRevision[];
  readonly dependencies: readonly RecordDependency[];
  readonly running: readonly RunningAction[];
  readonly useActionPoll: UseActionPoll;
  readonly hrefAt: (seq: number) => string;
  readonly onSelectVersion: (seq: number) => void;
}) {
  if (usePane()?.collapsed) {
    return (
      <div className="flex min-h-0 flex-1 flex-col items-center gap-2 py-2">
        <PaneToggle title="action record" />
        <h2 className="text-xs font-semibold whitespace-nowrap [writing-mode:vertical-rl]">
          Action record
        </h2>
      </div>
    );
  }
  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col">
      <div className="flex min-h-11 flex-none items-center gap-2 border-b px-3 py-1.5">
        <h2 className="mr-auto text-xs font-semibold">Action record</h2>
        {tick && <RecordHeading tick={tick} />}
        <PaneToggle title="action record" />
      </div>
      {/* Calls share the pane equally and scroll together only when they no longer fit. */}
      <div key={tick?.record.seq} className="flex min-h-0 flex-1 flex-col divide-y overflow-y-auto">
        {tick ? (
          <div className="flex min-h-64 flex-1 flex-col">
            <CallSections>
              <ActionInput
                request={tick.record.attempt.request}
                sources={{
                  calls: inputSources(tick, attempts, dependencies),
                  hrefAt,
                  select: onSelectVersion,
                }}
              />
              <ActionExecution
                action={tick.record.attempt.action}
                callId={tick.call_id}
                useActionPoll={useActionPoll}
              />
            </CallSections>
          </div>
        ) : (
          running.length === 0 && (
            <div className="px-3 py-2">
              <Hint>No actions yet.</Hint>
            </div>
          )
        )}
        {running.map((call) => (
          <section
            key={call.call_id}
            aria-label={`${call.action} · running`}
            className="flex min-h-64 flex-1 flex-col"
          >
            <h3 className="flex flex-none items-center gap-2 px-3 pt-2 text-xs font-semibold">
              <LoaderCircle className="size-3.5 animate-spin" aria-hidden="true" />
              {call.action} · running
            </h3>
            <CallSections>
              {/* A running call has no timeline node, so its references have no recorded source yet. */}
              <ActionInput
                request={call.request}
                sources={{
                  calls: new Map<string, readonly TimelineRevision[]>(),
                  hrefAt,
                  select: onSelectVersion,
                }}
              />
              <ActionExecution
                action={call.action}
                callId={call.call_id}
                useActionPoll={useActionPoll}
              />
            </CallSections>
          </section>
        ))}
      </div>
    </div>
  );
}
