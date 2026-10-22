/** Synthetic server-validated HTTP examples shared by stories and contract tests. */
import type { ActionSuccess, ModelSnapshot, TimelineRevision } from "@grassmoth/api-types";
import { HttpResponse, http } from "msw";
import { composeCallView } from "@/lib/model-asset/compose-call-view";
import {
  scenarioResponses,
  scenarioResults,
  scenarioTimeline,
  scenarioWorkspace,
  type ScenarioResults,
} from "./generated/study";
import { fixtureValue } from "./fixture-value";

export type ScenarioName = keyof ScenarioResults;

/** Typed values and HTTP bytes are emitted from the same validated ActionPoll owner. */
export function scenarioResult<Name extends ScenarioName>(name: Name): ScenarioResults[Name] {
  return scenarioResults[name];
}

export function scenarioRevision(name: ScenarioName): TimelineRevision {
  return fixtureValue(
    scenarioTimeline.attempts.find((entry) => entry.call_id === scenarioResponses[name].callId),
  );
}

/** Pure unit tests compose the same response values as HTTP consumers. */
export function scenarioSnapshot(name: ScenarioName): ModelSnapshot {
  const results = new Map<number, ActionSuccess>();
  for (const result of Object.values(scenarioResults)) {
    if (result.status === "success")
      results.set(
        fixtureValue(scenarioTimeline.attempts.find((entry) => entry.call_id === result.call_id))
          .record.seq,
        result,
      );
  }
  return composeCallView(
    scenarioWorkspace,
    scenarioRevision(name),
    scenarioTimeline.attempts,
    scenarioTimeline.dependencies,
    results,
  );
}

/** MSW forwards the generated JSON and MessagePack through the normal API client. */
export function scenarioHandlers() {
  return [
    http.get(`*/api/studies/${scenarioWorkspace}/timeline`, () =>
      HttpResponse.json(scenarioTimeline),
    ),
    http.get(`*/api/studies/${scenarioWorkspace}/:action/:callId`, ({ params }) => {
      const response = Object.values(scenarioResponses).find(
        (entry) => entry.action === params.action && entry.callId === params.callId,
      );
      if (!response)
        return HttpResponse.json({ detail: "Unknown synthetic call" }, { status: 404 });
      return new HttpResponse(response.payload.slice(), {
        headers: { "Content-Type": "application/msgpack" },
      });
    }),
  ];
}
