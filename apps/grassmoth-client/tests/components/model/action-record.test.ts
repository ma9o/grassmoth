import type {
  ActionPoll,
  GitOid,
  RecordDependency,
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
    simulation: { window: { start: "2026-01-01", horizon: "1d" }, interventions: [] },
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
  journal: {
    attempts: readonly TimelineRevision[];
    dependencies: readonly RecordDependency[];
  } = { attempts: selected ? [selected] : [], dependencies: [] },
) =>
  renderToStaticMarkup(
    createElement(ActionRecord, {
      tick: selected,
      ...journal,
      running,
      useActionPoll: () => ({ data: poll, error: null }),
      hrefAt: (seq) => `?seq=${seq}`,
      onSelectVersion: () => {},
    }),
  );

describe("action input", () => {
  it.each([
    { status: "applied", result: null, effects: { produced: [], retracted: [], reports: {} } },
    {
      status: "rejected",
      code: "INPUT_UNAVAILABLE",
      subject: "inputs",
      detail: "The model is incomplete",
    },
    { status: "raised", error_type: "FitError", error_message: "Fit failed", details: [] },
  ] as const)("shows the recorded request before logs for $status calls", (outcome) => {
    const selected: TimelineRevision = {
      ...tick,
      record: { ...tick.record, attempt: { action: "simulate", request, outcome } },
    };
    const html = render(selected);
    expect(html).toContain('aria-label="Action input"');
    expect(html).toContain("Check &lt;revised&gt; assumptions.\nGoal: choose a model to fit.");
    expect(html).toContain(">dynamical_model_spec_ref<");
    expect(html).toContain("1".repeat(40));
    expect(html).toMatch(/>simulation<\/span><span[^>]*>interventions, window</);
    expect(html).toMatch(/>num_replicates<\/span><span[^>]*>100</);
    expect(html.indexOf('aria-label="Action input"')).toBeLessThan(
      html.indexOf('aria-label="Simulator log"'),
    );
    expect(html).not.toContain('aria-label="Reasoning"');
    expect(html).not.toContain("Simulate failed");
  });

  it("shows input before a running call's messages", () => {
    const html = render(undefined, [
      {
        action: "simulate",
        call_id: `call:${"2".repeat(64)}`,
        request,
      },
    ]);
    expect(html).toMatch(/>num_replicates<\/span><span[^>]*>100</);
    expect(html.indexOf('aria-label="Action input"')).toBeLessThan(
      html.indexOf("FIT_CAPACITY_ESTIMATE"),
    );
  });

  it("shows input when no reasoning was supplied", () => {
    const html = render(tick);
    expect(html).toContain('aria-label="Action input"');
    expect(html).toMatch(/>reasoning<\/span><span[^>]*><span[^>]*>null</);
    expect(html).toMatch(/>num_replicates<\/span><span[^>]*>100</);
  });

  it("shows every concurrent call with its own input", () => {
    const html = render(
      undefined,
      [1, 2].map((index) => ({
        action: "simulate",
        call_id: `call:${String(index).repeat(64)}`,
        request: {
          ...request,
          input: { ...request.input, seed: index },
          reasoning: `Arm ${index}`,
        },
      })),
    );
    expect(html.match(/aria-label="simulate · running"/g)).toHaveLength(2);
    expect(html).toContain("Arm 1");
    expect(html).toContain("Arm 2");
    expect(html).toMatch(/>seed<\/span><span[^>]*>1</);
    expect(html).toMatch(/>seed<\/span><span[^>]*>2</);
    expect(html).not.toContain("No actions yet.");
  });

  it("shows each input reference as the timeline node that published it", () => {
    const source: TimelineRevision = {
      ...tick,
      call_id: `call:${"3".repeat(64)}`,
      commit_id: "3".repeat(40),
      record: {
        ...tick.record,
        seq: 1,
        attempt: { action: "edit_model", request: null, outcome: tick.record.attempt.outcome },
      },
    };
    const html = render(tick, [], undefined, {
      attempts: [source, tick],
      dependencies: [{ seq: 2, source_seq: 1, argument: "dynamical_model_spec" }],
    });
    expect(html).toMatch(/<a [^>]*aria-label="edit_model · 3333333"[^>]*href="\?seq=1"/);
    expect(html).not.toContain("1".repeat(40));
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
  expect(html).toContain("[ExtractionError]");
  expect(html).toContain("Worker failed");
  expect(html).toContain("Retained provider error details.");
});

it("preserves progress messages without a separate summary when a later poll fails", () => {
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
  expect(html).not.toContain('aria-label="Progress"');
  expect(html).toContain("3 LLM calls");
  expect(html).toContain("1/2 completed");
  expect(html).toContain("FIT_CAPACITY_ESTIMATE");
});
