import type {
  GitOid,
  RunningAction,
  SimulateRequest,
  TimelineRevision,
} from "@nof1-causal-lab/api-types";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ActionRecord, type ActionTraceState } from "./action-record";

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
    messages: [],
    trace_ids: [],
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
const render = (selected: TimelineRevision | undefined, running: readonly RunningAction[] = []) =>
  renderToStaticMarkup(
    createElement(ActionRecord, {
      workspaceId: "STORYBOOK",
      context: null,
      tick: selected,
      running,
      useActionTrace: (): ActionTraceState => ({ status: "absent" }),
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
        messages: [
          {
            kind: "log",
            timestamp: tick.record.ts,
            severity: "info",
            code: "SIMULATE_STARTED",
            subject: "action",
            detail: "",
          },
        ],
      },
    ]);
    expect(html.indexOf('aria-label="Reasoning"')).toBeLessThan(html.indexOf("SIMULATE_STARTED"));
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
        messages: [],
      })),
    );
    expect(html.match(/aria-label="simulate · running"/g)).toHaveLength(2);
    expect(html).toContain("Arm 1");
    expect(html).toContain("Arm 2");
    expect(html).not.toContain("No actions yet.");
  });
});
