"use client";

import { attemptError } from "@/lib/model-asset/journal";

import type { RunningAction } from "@grassmoth/api-types";
import { LoaderCircle, X } from "lucide-react";
import type { UseActionPoll } from "@/lib/hooks/use-action-poll";
import type { TimelineRevision } from "@grassmoth/api-types";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { humanize } from "@/lib/model-asset/selection";
import { timelineTickLabel } from "@/lib/model-asset/timeline-presentation";
import { ActionExecution } from "./action-log";
import { ActionFindings } from "./action-findings";
import { Hint, Section } from "./scope-primitives";
import { DataComparisonOutcome, DataDetails } from "./scopes/data-details";
import { EditDetails } from "./scopes/edit-details";
import { QuestionDetails } from "./scopes/question-details";
import { FitOutcome } from "./scopes/fit-details";

function ActionReasoning({ reasoning }: { reasoning: string | null | undefined }) {
  if (!reasoning) return null;
  return (
    <section aria-label="Reasoning" className="px-4 py-1">
      <h3 className="mb-1 text-[11px] font-semibold">Reasoning</h3>
      <p className="whitespace-pre-wrap break-words text-[11.5px] leading-relaxed">{reasoning}</p>
    </section>
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
  context,
  tick,
  running,
  useActionPoll,
}: {
  context: ScopeContext | null;
  tick: TimelineRevision | undefined;
  running: readonly RunningAction[];
  useActionPoll: UseActionPoll;
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
              </Section>
            ) : tick.record.attempt.request === null ? (
              <Section title="Unknown call">
                <Hint>This attempt has no retained call arguments.</Hint>
              </Section>
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
                  </Section>
                )}
                {call && <ActionFindings context={context} call={call} />}
              </>
            ) : null}
            <ActionExecution
              action={tick.record.attempt.action}
              callId={tick.call_id}
              useActionPoll={useActionPoll}
            />
          </>
        ) : (
          running.length === 0 && (
            <p className="px-4 text-xs text-muted-foreground">No actions yet.</p>
          )
        )}
        {running.map((call) => (
          <section
            key={call.call_id}
            aria-label={`${call.action} · running`}
            className="border-t pt-2 first:border-t-0 first:pt-0"
          >
            <h3 className="flex items-center gap-2 px-4 pb-1 text-[11px] font-medium">
              <LoaderCircle className="size-3.5 animate-spin" aria-hidden="true" />
              {call.action} · running
            </h3>
            <ActionReasoning reasoning={call.request.reasoning} />
            <ActionExecution
              action={call.action}
              callId={call.call_id}
              useActionPoll={useActionPoll}
            />
          </section>
        ))}
      </div>
    </div>
  );
}
