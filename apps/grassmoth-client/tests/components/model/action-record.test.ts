import type {
  ActionPoll,
  GitOid,
  RunningAction,
  SimulateRequest,
  TimelineRevision,
} from "@grassmoth/api-types";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ActionRecord } from "@/components/model/action-record";
import { ActionExecution } from "@/components/model/action-log";

import { runningPoll, simulatedResult, failedPoll } from "@/lib/__fixtures__/action-results";
import { executionMessages, executionFailure } from "@/lib/__fixtures__/execution-log";

const intent = "Check <revised> assumptions.\nGoal: choose a model to fit.";
const request: SimulateRequest<GitOid> = {
  action: "simulate",
  input: {
    dynamical_model_spec_ref: "1".repeat(40),
    simulation: { start: "2026-01-01", horizon: "1d", interventions: [] },
    seed: 0,
    num_replicates: 100,
  },
  reasoning: intent,
};
const tick: TimelineRevision = {
  call_id: `call:${"2".repeat(64)}`,
  commit_id: "2".repeat(40),
  parent_ids: ["1".repeat(40)],
  record: {
    seq: 2,
    ts: "2026-10-05T12:00:00Z",
    attempt: {
      action: "simulate",
      request: { ...request, reasoning: null },
      outcome: {
        status: "applied",
        result: null,
        effects: { produced: [], retracted: [], reports: {} },
      },
    },
  },
};
const render = (
  selected: TimelineRevision | undefined,
  running: readonly RunningAction[] = [],
  poll: ActionPoll = { ...runningPoll, messages: executionMessages },
) =>
  renderToStaticMarkup(
    createElement(ActionRecord, {
      context: null,
      tick: selected,
      running,
      useActionPoll: () => ({ data: poll, error: null }),
    }),
  );

describe("action reasoning", () => {
  it.each([
    { status: "applied", result: null, effects: { produced: [], retracted: [], reports: {} } },
    {
      status: "rejected",
      code: "INPUT_UNAVAILABLE",
      subject: "inputs",
      detail: "The model is incomplete",
    },
    { status: "raised", error_type: "FitError", error_message: "Fit failed", details: [] },
  ] as const)("shows the caller's intent first for $status calls", (outcome) => {
    const selected: TimelineRevision = {
      ...tick,
      record: { ...tick.record, attempt: { action: "simulate", request, outcome } },
    };
    const html = render(selected);
    expect(html).toContain("Check &lt;revised&gt; assumptions.\nGoal: choose a model to fit.");
    expect(html.indexOf('aria-label="Reasoning"')).toBeLessThan(
      html.indexOf(outcome.status === "applied" ? 'aria-label="Simulator log"' : "Simulate failed"),
    );
  });

  it("shows intent before a running call's messages", () => {
    const html = render(undefined, [
      {
        action: "simulate",
        call_id: `call:${"2".repeat(64)}`,
        request,
      },
    ]);
    expect(html.indexOf('aria-label="Reasoning"')).toBeLessThan(
      html.indexOf("FIT_CAPACITY_ESTIMATE"),
    );
  });

  it("omits the section when no reasoning was supplied", () => {
    expect(render(tick)).not.toContain('aria-label="Reasoning"');
  });

  it("shows every concurrent call with its own reasoning", () => {
    const html = render(
      undefined,
      [1, 2].map((index) => ({
        action: "simulate",
        call_id: `call:${String(index).repeat(64)}`,
        request: { ...request, reasoning: `Arm ${index}` },
      })),
    );
    expect(html.match(/aria-label="simulate · running"/g)).toHaveLength(2);
    expect(html).toContain("Arm 1");
    expect(html).toContain("Arm 2");
    expect(html).not.toContain("No actions yet.");
  });
});

it("keeps the complete execution log visible after a running action succeeds", () => {
  const live = render(undefined, [{ action: "simulate", call_id: runningPoll.call_id, request }]);
  const completed = render(tick, [], {
    call_id: runningPoll.call_id,
    action: "simulate",
    status: "success",
    commit_id: tick.commit_id,
    body: simulatedResult,
    messages: executionMessages,
  });
  for (const html of [live, completed]) {
    for (const text of [
      "FIT_CAPACITY_ESTIMATE",
      "128 MiB",
      "grassmoth.step",
      "2 workers planned",
      "worker 7",
      "4 windows",
      "8 extractions",
      "3 LLM calls",
      "1/2 completed",
      "Agent conversation",
      "Inspecting retained observations.",
    ]) {
      expect(html).toContain(text);
    }
    expect(html.indexOf("FIT_CAPACITY_ESTIMATE")).toBeLessThan(html.indexOf("grassmoth.step"));
    expect(html.indexOf("grassmoth.step")).toBeLessThan(html.indexOf("Agent conversation"));
  }
});

it("shows retained conversations and complete failure details for failed calls", () => {
  const html = render(tick, [], {
    ...failedPoll,
    messages: [...executionMessages, executionFailure],
  });
  expect(html).toContain("Inspecting retained observations.");
  expect(html).toContain("ExtractionError: Worker failed");
  expect(html).toContain("Retained provider error details.");
});

it("preserves progress counters and received messages when a later poll fails", () => {
  const html = renderToStaticMarkup(
    createElement(ActionExecution, {
      action: "prepare_data",
      callId: runningPoll.call_id,
      useActionPoll: () => ({
        data: { ...runningPoll, action: "prepare_data" as const, messages: executionMessages },
        error: new Error("Connection interrupted"),
      }),
    }),
  );
  expect(html).toContain("Cannot load action log: Connection interrupted");
  expect(html).toContain('aria-label="Progress"');
  expect(html).toContain("3 reported by");
  expect(html).toContain("1/2 completed");
  expect(html).toContain("FIT_CAPACITY_ESTIMATE");
});
