import type { ActionSuccess, RecordDependency, TimelineRevision } from "@grassmoth/api-types";
import { describe, expect, it } from "vitest";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { scenarioResult } from "@/components/__fixtures__/study-scenario";
import { outcome } from "@/lib/__fixtures__/model";
import { authoredSnapshot } from "@/lib/__fixtures__/snapshot";
import {
  callDependencies,
  modelReference,
  producingCall,
} from "@/lib/model-asset/call-dependencies";
import { composeCallView } from "@/lib/model-asset/compose-call-view";

const simulationReport = scenarioResult("simulated").body.report;

const WORKSPACE = "TEST";
const oid = (seq: number) => seq.toString(16).padStart(40, "0");
const question = fixtureValue(authoredSnapshot.question);
const model = () => scenarioResult("authored").body;
const observation = fixtureValue(outcome.indicators[0]).observation;
const extraction = {
  kind: "semantic" as const,
  how_to_measure: "Read the value.",
  source_columns: [],
};
const source = { files: ["input/test.csv"] as const, hashes: {}, start: null, end: null };
const design = { start: "2026-01-01", horizon: "1d", interventions: [] };
const settings = {
  hardware: null,
  time_limit_seconds: null,
  algorithm: null,
  num_samples_per_chain: null,
  num_warmup: null,
  num_chains: null,
  num_particles: null,
  seed: 0,
};
const dependencies: RecordDependency[] = [
  { seq: 4, source_seq: 1, argument: "parent" },
  { seq: 5, source_seq: 4, argument: "dynamical_model_spec" },
  { seq: 7, source_seq: 1, argument: "parent" },
  { seq: 8, source_seq: 7, argument: "dynamical_model_spec" },
  { seq: 8, source_seq: 5, argument: "data" },
  { seq: 9, source_seq: 8, argument: "dynamical_model_spec" },
  { seq: 9, source_seq: 5, argument: "panel" },
  { seq: 10, source_seq: 7, argument: "dynamical_model_spec" },
  { seq: 10, source_seq: 5, argument: "data" },
  { seq: 11, source_seq: 10, argument: "parent" },
  { seq: 12, source_seq: 11, argument: "dynamical_model_spec" },
  { seq: 13, source_seq: 5, argument: "left" },
  { seq: 13, source_seq: 9, argument: "right" },
];
const journal: TimelineRevision[] = [1, 4, 5, 7, 8, 9, 10, 11, 12, 13].map((seq) => {
  const request: NonNullable<TimelineRevision["record"]["attempt"]["request"]> =
    seq === 1
      ? { action: "edit_question", reasoning: null, input: { question: question } }
      : seq === 5
        ? {
            action: "prepare_data",
            reasoning: null,
            input: {
              dynamical_model_spec_ref: oid(4),
              source,
              extraction: { [observation.id]: extraction },
              context: "",
            },
          }
        : seq === 8 || seq === 10
          ? {
              action: "fit",
              reasoning: null,
              input: {
                target: {
                  kind: "new",
                  dynamical_model_spec_ref: oid(7),
                  data_ref: { revision: oid(5), replicate_index: 0 },
                },
                repeat: 0,

                settings,
              },
            }
          : seq === 9 || seq === 12
            ? {
                action: "simulate",
                reasoning: null,
                input: {
                  dynamical_model_spec_ref: oid(seq - 1),
                  simulation: design,
                  seed: 0,
                  num_replicates: 100,
                },
              }
            : seq === 13
              ? {
                  action: "data_diff",
                  reasoning: null,
                  input: {
                    left_ref: [{ revision: oid(5), replicate_index: 0 }],
                    right_ref: [{ revision: oid(9), replicate_index: 1 }],
                  },
                }
              : {
                  action: "edit_model",
                  reasoning: null,
                  input: {
                    parent_ref: seq === 11 ? oid(10) : oid(1),
                    dynamical_model_spec: model().dynamical_model_spec,
                  },
                };
  return {
    call_id: `call:${seq.toString(16).padStart(64, "0")}`,
    commit_id: oid(seq),
    parent_ids: [oid(seq - 1)],
    record: {
      seq,
      ts: "2026-01-01T00:00:00Z",
      messages: [],
      trace_ids: [],
      attempt: {
        action: request.action,
        request,
        outcome: {
          status: "applied",
          result: null,
          effects: {
            produced:
              request.action === "edit_model" || request.action === "fit"
                ? [
                    {
                      artifact_id: "model",
                      revision: oid(seq + 100),
                      derived_from: {},
                      produced_by: request.action,
                      created_at: "2026-01-01T00:00:00Z",
                      source: { kind: "files" },
                    },
                  ]
                : request.action === "edit_question" || request.action === "prepare_data"
                  ? [
                      {
                        artifact_id: request.action === "edit_question" ? "question" : "panel",
                        revision: oid(seq),
                        derived_from: {},
                        produced_by: request.action,
                        created_at: "2026-01-01T00:00:00Z",
                        source: { kind: "files" },
                      },
                    ]
                  : [],
            retracted: [],
            reports: {},
          },
        },
      },
    },
  };
});

function savedResult(seq: number): ActionSuccess {
  const call = fixtureValue(journal.find((item) => item.record.seq === seq));
  const envelope = {
    status: "success" as const,
    call_id: fixtureValue(call.call_id),
    commit_id: call.commit_id,
    messages: [],
  };
  switch (call.record.attempt.action) {
    case "edit_question":
      return { ...envelope, action: "edit_question", body: { question } };
    case "edit_model":
      return { ...envelope, action: "edit_model", body: model() };
    case "prepare_data":
      return {
        ...envelope,
        action: "prepare_data",
        body: {
          profile: { indicators: {}, findings: [] },
          data: {},
          extraction: { workers: [], extraction_reused: null },
          metadata: {
            preparation: {
              default_window: "1d",
              variables: [{ observation, extraction }],
              context: "",
            },
            time_origin: "2026-01-01T00:00:00Z",
            variables: [{ ...observation, observation_window: "1d" }],
          },
        },
      };
    case "fit":
      return {
        ...envelope,
        action: "fit",
        body: {
          ...scenarioResult("fitted").body,
        },
      };
    case "simulate":
      return {
        ...envelope,
        action: "simulate",
        body: { report: simulationReport, data: [{}, {}] },
      };
    case "data_diff":
      return {
        ...envelope,
        action: "data_diff",
        body: {
          report: {
            left: [{ revision: oid(5), replicate_index: 0 }],
            right: [{ revision: oid(9), replicate_index: 1 }],
            variables: [],
          },
        },
      };
    case "model_diff":
      throw new Error("No model comparison in this test");
  }
}

const entry = (seq: number) => fixtureValue(journal.find((call) => call.record.seq === seq));

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
    view: composeCallView(WORKSPACE, selected, attempts, edges, results),
  };
}

describe("views composed from recorded call dependencies", () => {
  it("keeps a simulation's fitted model despite another model owning its data and newer edits", () => {
    const fitted = savedResult(8);
    const simulated = savedResult(9);
    if (fitted.action !== "fit" || simulated.action !== "simulate") throw new Error("Fixture");
    const { calls, view } = compose(entry(9));
    expect(view.dynamical_model_spec).toEqual(fitted.body.dynamical_model_spec);
    expect(view.fit).toEqual(scenarioResult("fitted").body.inference.core);
    expect(view.simulation).toEqual(simulated.body.report);
    expect(view.metadata).toBeNull();
    expect(view.state.data).toBeNull();
    expect(view.state.current.panel).toBeUndefined();
    expect(view.question).toEqual(question);
    expect(calls.map((call) => call.record.seq).sort((a, b) => a - b)).toEqual([1, 4, 5, 7, 8, 9]);
    expect(producingCall(journal, view.state.current.model?.revision)).toEqual(entry(8));
  });

  it("reads an edited model with its question without fetching the base's fit and data", () => {
    const { calls, view } = compose(entry(11));
    expect(calls.map((call) => call.record.seq)).toEqual([1, 11]);
    expect(view.question).toEqual(question);
    expect(view.fit).toBeNull();
    expect(view.metadata).toBeNull();
    expect(view.state.data).toBeNull();
    expect(view.simulation).toBeNull();
  });

  it("keeps the input model reference when a fit publishes no replacement model", () => {
    const template = entry(10);
    const outcome = template.record.attempt.outcome;
    if (outcome.status !== "applied") throw new Error("Fixture");
    const selected: TimelineRevision = {
      ...template,
      record: {
        ...template.record,
        attempt: {
          ...template.record.attempt,
          outcome: { ...outcome, effects: { ...outcome.effects, produced: [] } },
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
    if (request?.action !== "fit" || request.input.target.kind !== "new")
      throw new Error("Fixture");
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

it("addresses models by their outputs or recorded inputs, including failed calls", () => {
  expect(modelReference(entry(1), journal)).toBe(oid(1));
  expect(modelReference(entry(4), journal)).toBe(oid(104));
  expect(modelReference(entry(5), journal)).toBe(oid(4));
  expect(modelReference(entry(9), journal)).toBe(oid(8));
  expect(modelReference(entry(13), journal)).toBe(oid(8));
  const failed: TimelineRevision = {
    ...entry(10),
    parent_ids: [oid(999)],
    record: {
      ...entry(10).record,
      attempt: {
        ...entry(10).record.attempt,
        outcome: { status: "raised", error_type: "FitError", error_message: "Failed", details: [] },
      },
    },
  };
  expect(modelReference(failed, journal)).toBe(oid(7));
});
