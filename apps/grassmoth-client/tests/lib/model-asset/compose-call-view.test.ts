import type { ActionSuccess, RecordDependency, TimelineRevision } from "@grassmoth/api-types";
import { describe, expect, it } from "vitest";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { fittedResult, initialModelResult } from "@/lib/__fixtures__/action-results";
import {
  dependencies,
  entry,
  journal,
  oid,
  question,
  savedResult,
} from "@/lib/__fixtures__/recorded-calls";
import { callDependencies, producingCall } from "@/lib/model-asset/call-dependencies";
import { composeCallView } from "@/lib/model-asset/compose-call-view";

const WORKSPACE = "TEST";

function compose(
  selected: TimelineRevision,
  attempts = journal,
  edges: readonly RecordDependency[] = dependencies,
  extra: ReadonlyMap<number, ActionSuccess> = new Map(),
) {
  const calls = callDependencies(selected, attempts, edges);
  const results = new Map(
    calls.map((call) => [
      call.record.seq,
      extra.get(call.record.seq) ?? savedResult(call.record.seq),
    ]),
  );
  return {
    calls,
    view: composeCallView(WORKSPACE, selected, { attempts, dependencies: edges, results }),
  };
}

describe("views composed from recorded call dependencies", () => {
  it("shows the initial outcome model and its saved findings", () => {
    const { view } = compose(entry(1));
    expect(view.dynamical_model_spec).toEqual(initialModelResult.dynamical_model_spec);
    expect(view.state.current.model?.revision).toBe(oid(101));
    expect(view.specification).toEqual(initialModelResult.checks.specification);
    expect(view.identification).toEqual(initialModelResult.identification);
    expect(Object.keys(fixtureValue(view.dynamical_model_spec).constructs)).toEqual([
      question.outcome,
    ]);
  });
  it("keeps a simulation's fitted model despite another model owning its data and newer edits", () => {
    const fitted = savedResult(8);
    const simulated = savedResult(9);
    if (fitted.action !== "fit" || simulated.action !== "simulate") throw new Error("Fixture");
    const { calls, view } = compose(entry(9));
    expect(view.dynamical_model_spec).toEqual(fitted.body.dynamical_model_spec);
    expect(view.fit).toEqual(fittedResult.inference.core);
    expect(view.simulation).toEqual(simulated.body.report);
    expect(view.metadata).toBeNull();
    expect(view.state.data).toBeNull();
    expect(view.state.current.panel).toBeUndefined();
    expect(view.dynamical_model_spec?.question).toEqual(question.question);
    expect(calls.map((call) => call.record.seq).toSorted((a, b) => a - b)).toEqual([4, 5, 7, 8, 9]);
    expect(producingCall(journal, view.state.current.model?.revision)).toEqual(entry(8));
  });

  it("reads an edited model with its question without fetching the base's fit and data", () => {
    const { calls, view } = compose(entry(11));
    expect(calls.map((call) => call.record.seq)).toEqual([11]);
    expect(view.dynamical_model_spec?.question).toEqual(question.question);
    expect(view.fit).toBeNull();
    expect(view.metadata).toBeNull();
    expect(view.state.data).toBeNull();
    expect(view.simulation).toBeNull();
  });

  it("keeps the input model reference when a fit publishes no replacement model", () => {
    const template = entry(10);
    const fitOutcome = template.record.attempt.outcome;
    if (fitOutcome.status !== "applied") throw new Error("Fixture");
    const selected: TimelineRevision = {
      ...template,
      record: {
        ...template.record,
        attempt: {
          ...template.record.attempt,
          outcome: { ...fitOutcome, effects: { ...fitOutcome.effects, produced: [] } },
        },
      },
    };
    const { view } = compose(
      selected,
      journal.map((call) => (call.record.seq === selected.record.seq ? selected : call)),
    );
    expect(view.state.current.model?.revision).toBe(oid(107));
    expect(view.state.current.panel?.revision).toBe(oid(5));
  });

  it("keeps a fit's exact simulation replicate without adopting the simulation's model", () => {
    const template = entry(10);
    const request = template.record.attempt.request;
    if (request?.action !== "fit" || request.input.target.kind !== "new") {
      throw new Error("Fixture");
    }
    const selected: TimelineRevision = {
      ...template,
      commit_id: "e".repeat(40),
      call_id: `call:${"e".repeat(64)}`,
      record: {
        ...template.record,
        seq: 14,
        attempt: {
          ...template.record.attempt,
          request: {
            ...request,
            input: {
              ...request.input,
              target: {
                ...request.input.target,
                data_ref: { revision: entry(9).commit_id, replicate_index: 1 },
              },
            },
          },
        },
      },
    };
    const fitted = savedResult(10);
    if (fitted.action !== "fit") throw new Error("Fixture");
    const { view } = compose(
      selected,
      [...journal, selected],
      [
        ...dependencies,
        { seq: 14, source_seq: 7, argument: "dynamical_model_spec" },
        { seq: 14, source_seq: 9, argument: "data" },
      ],
      new Map([[14, fitted]]),
    );
    expect(view.dynamical_model_spec).toEqual(fitted.body.dynamical_model_spec);
    expect(view.state.data).toEqual({ revision: entry(9).commit_id, replicate_index: 1 });
    expect(view.metadata).toBeNull();
    expect(view.simulation).toBeNull();
  });

  it("uses the recorded comparison endpoint and reports missing dependencies", () => {
    const compared = compose(entry(13)).view;
    expect(compared.dynamical_model_spec).toEqual(compose(entry(9)).view.dynamical_model_spec);
    expect(compared.commit_id).toBe(entry(13).commit_id);
    expect(() =>
      compose(
        entry(9),
        journal.filter((call) => call.record.seq !== 8),
      ),
    ).toThrow("Missing recorded dependency 8");
  });
});
