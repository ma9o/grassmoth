import { createModelClient, type ActionSuccess, type TimelineResponse } from "@grassmoth/api-types";
import { setupServer } from "msw/node";
import { afterAll, beforeAll, expect, it } from "vitest";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import {
  scenarioResponses,
  scenarioResults,
  scenarioTimeline,
  scenarioWorkspace,
} from "@/components/__fixtures__/generated/study";
import {
  scenarioHandlers,
  scenarioResult,
  scenarioRevision,
  scenarioSnapshot,
} from "@/components/__fixtures__/study-scenario";
import { callDependencies } from "@/lib/model-asset/call-dependencies";
import { composeCallView } from "@/lib/model-asset/compose-call-view";
import { pathsView, drawsView } from "@/lib/model-asset/result-values";

const server = setupServer(...scenarioHandlers());
beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterAll(() => server.close());

it("reads every synthetic action through HTTP and composes its saved dependency graph", async () => {
  const client = createModelClient({ baseUrl: "http://scenario.test" });
  const response = await client.GET("/api/studies/{workspace_id}/timeline", {
    params: { path: { workspace_id: scenarioWorkspace } },
  });
  const timeline: TimelineResponse = fixtureValue(response.data);
  expect(timeline).toEqual(scenarioTimeline);
  const results = new Map<number, ActionSuccess>();
  const actions = new Set<string>();
  const statuses = new Set<string>();
  for (const call of Object.values(scenarioResponses)) {
    const response = await client.GET("/api/studies/{workspace_id}/{action}/{call_id}", {
      params: {
        path: { workspace_id: scenarioWorkspace, action: call.action, call_id: call.callId },
      },
    });
    expect(response.response.headers.get("Content-Type")).toBe("application/msgpack");
    const result = fixtureValue(response.data);
    expect(result.call_id).toBe(call.callId);
    expect(result).toEqual(
      Object.values(scenarioResults).find((entry) => entry.call_id === call.callId),
    );
    expect(result.action).toBe(call.action);
    actions.add(result.action);
    statuses.add(result.status);
    const revision = timeline.attempts.find((entry) => entry.call_id === result.call_id);
    if (result.status === "success") {
      expect(revision?.record.attempt.outcome.status).toBe("applied");
      results.set(fixtureValue(revision).record.seq, result);
    } else if (result.status === "failed") {
      expect(result.body).toBeNull();
      expect(result.messages.some((message) => message.kind === "failure")).toBe(true);
      expect(revision?.record.attempt.outcome.status).not.toBe("applied");
    } else {
      expect(result.commit_id).toBeNull();
      expect(timeline.running.map((entry) => entry.call_id)).toContain(result.call_id);
      expect(revision).toBeUndefined();
    }
  }
  expect([...actions].sort()).toEqual([
    "data_diff",
    "edit_model",
    "edit_question",
    "fit",
    "model_diff",
    "prepare_data",
    "simulate",
  ]);
  expect([...statuses].sort()).toEqual(["failed", "running", "success"]);
  for (const selected of timeline.attempts) {
    if (selected.record.attempt.outcome.status !== "applied") continue;
    const calls = callDependencies(selected, timeline.attempts, timeline.dependencies);
    expect(calls.every((call) => results.has(call.record.seq))).toBe(true);
    const view = composeCallView(
      scenarioWorkspace,
      selected,
      timeline.attempts,
      timeline.dependencies,
      results,
    );
    expect(view.commit_id).toBe(selected.commit_id);
    expect(view.question?.text).toContain("Synthetic example");
  }
  // Binary plot values and aligned atoms survive the actual transport and production decoder.
  const simulated = fixtureValue(results.get(scenarioRevision("simulated").record.seq));
  const fitted = fixtureValue(results.get(scenarioRevision("fitted").record.seq));
  if (simulated.action !== "simulate" || fitted.action !== "fit")
    throw new Error("Scenario action mismatch");
  const paths = pathsView(
    simulated.body,
    fixtureValue(scenarioSnapshot("simulated").dynamical_model_spec),
  );
  expect(paths.states["construct:00000000000000000003"]?.action[1]?.values).toEqual([2, 5, 6, 7]);
  expect(drawsView(fitted.body).map((column) => column.values)).toHaveLength(2);
  expect(scenarioResult("effectUnavailable").body.report.evidence.arms.kind).toBe("paired");
});
