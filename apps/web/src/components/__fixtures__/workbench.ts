import { encode } from "@msgpack/msgpack";
import type {
  ActionAttempt,
  ActionPoll,
  ArtifactRecord,
  CallId,
  DataDiffOutput,
  ExecutionMessage,
  FileSourceRef,
  GitOid,
  ModelSnapshot,
  DynamicalModelSpec,
  PrepareDataRequest,
  RecordDependency,
  SimulationReport,
  SimulationSpec,
  SpecificationAssessment,
  StudyRevision,
  TimelineResponse,
  TimelineRevision,
} from "@nof1-causal-lab/api-types";
import { HttpResponse, http } from "msw";
import { decodeFixture, fixtureValue } from "@/components/__fixtures__/fixture-value";
import { modelConstructs } from "@/lib/model-accessors";
import { modelResult } from "./action-results";
import { demoModelSnapshot, demoSnapshotAt } from "./demo-artifacts";
import { demoTraces } from "./demo-traces";
import rawComparisonFixture from "./workbench-comparisons.json";
import rawVisualFixture from "./workbench-visuals.json";

const visualFixture = decodeFixture(rawVisualFixture);
const comparisonFixture = decodeFixture(rawComparisonFixture);
export const WORKBENCH_WORKSPACE = "STORYBOOK";
const stamp = "2026-09-16T12:00:00Z";
// Illustrative interface data. Parameter decisions are staged;
// retained HEALTHDEMO evidence is reused for presentation, not claimed as new inference.
const freeDynamicalModelSpec = structuredClone(
  fixtureValue(demoModelSnapshot.dynamical_model_spec),
);
// Generated and validated by scripts/fixtures/study.py.
const pinnedDynamicalModelSpec = comparisonFixture.pinned_dynamical_model_spec;
const checks: SpecificationAssessment[] = [
  {
    code: "specification",
    kind: "not_evaluated",
    subject: "specification",
    reason: "MODEL_INCOMPLETE",
    detail: "Specification checks have not been run.",
  },
];
const commitId = (seq: number) => (0xc000000 + seq).toString(16).padEnd(40, "c");
const callId = (seq: number): CallId => `call:${seq.toString(16).padStart(64, "0")}`;
const modelId = (ordinal: number): string =>
  ordinal <= 4
    ? fixtureValue(demoSnapshotAt(fixtureValue([2, 3, 4, 7][ordinal - 1])).state.current.model)
        .revision
    : ordinal.toString(16).padStart(40, "a");
const questionId = fixtureValue(demoModelSnapshot.state.current.question).revision;
const panelId = fixtureValue(demoModelSnapshot.state.current.panel).revision;
function record(seq: number, attempt: ActionAttempt, trace_ids: string[] = []): StudyRevision {
  return {
    commit_id: commitId(seq),
    parent_ids: [commitId(seq - 1)],
    record: {
      seq,
      attempt_id: null,
      ts: stamp,
      attempt,
      trace_ids,
      messages: [
        {
          kind: "log",
          timestamp: stamp,
          severity: "info",
          code: "ACTION_COMPLETED",
          subject: "action",
          detail: "",
        },
      ],
    },
  };
}
function metadata(revision: string, parent: string, produced_by: string): ArtifactRecord {
  return {
    artifact_id: "model",
    revision,
    derived_from: {
      question: questionId,
      model: parent,
      ...(produced_by === "fit" ? { panel: panelId } : {}),
    },
    produced_by,
    created_at: stamp,
    source: { kind: "result", result: commitId(0) },
  };
}
const simulationDesign: SimulationSpec = {
  start: "2026-09-01",
  horizon: "1w",
  interventions: [{ target: "construct:27f64faaddd59b2ad491", after: null, value: 10.0 }],
};
function simulation(revision: string): SimulationReport {
  const outcome = fixtureValue(
    modelConstructs(freeDynamicalModelSpec).find(
      (item) => item.name === "internalizing_symptom_burden",
    ),
  );
  const indicator = fixtureValue(
    outcome.indicators.find((item) => item.observation.name === "gad7_screening_score"),
  );
  return {
    ...structuredClone(visualFixture.simulation.report),
    evidence: {
      ...structuredClone(visualFixture.simulation.report.evidence),
      arms:
        revision === modelId(7) && visualFixture.simulation.report.evidence.arms.kind === "paired"
          ? {
              ...visualFixture.simulation.report.evidence.arms,
              causal: {
                kind: "not_evaluated",
                code: "causal_effect",
                subject: "causal_effect",
                reason: "CAUSAL_EVALUATION_FAILED",
                detail: "This edited model has no committed production fit at this revision.",
              },
            }
          : visualFixture.simulation.report.evidence.arms,
    },
    findings: [
      { kind: "construct" as const, id: outcome.id },
      { kind: "indicator" as const, id: indicator.observation.id },
    ].map((target) => ({
      code: "dispersion",
      kind: "not_evaluated" as const,
      subject: { target, construct_id: outcome.id },
      reason: "COMPARISON_INPUTS_MISSING" as const,
      detail: "This illustrative record has no observed comparison for the saved simulation draws.",
    })),
  };
}
const simulations = new Map([
  [4, simulation(modelId(4))],
  [7, simulation(modelId(7))],
]);
const models = new Map<string, DynamicalModelSpec>(
  [1, 2, 3, 4].map((revision) => {
    const seq = fixtureValue([2, 3, 4, 7][revision - 1]);
    return [modelId(revision), fixtureValue(demoSnapshotAt(seq).dynamical_model_spec)];
  }),
);
models.set(modelId(5), freeDynamicalModelSpec);
models.set(modelId(6), freeDynamicalModelSpec);
models.set(modelId(7), pinnedDynamicalModelSpec);
const snapshots = new Map<number, ModelSnapshot>(
  [0, 2, 3, 4, 5, 7].map((seq) => {
    const snapshot = demoSnapshotAt(seq);
    return [seq, { ...snapshot, workspace_id: WORKBENCH_WORKSPACE }];
  }),
);
snapshots.set(1, { ...demoSnapshotAt(0), workspace_id: WORKBENCH_WORKSPACE, selected_seq: 1 });
function illustratedSnapshot(
  seq: number,
  info: ArtifactRecord,
  fitted: boolean,
  report?: SimulationReport,
): ModelSnapshot {
  const snapshot = demoModelSnapshot;
  const dynamicalModelSpec = fixtureValue(models.get(info.revision));
  return {
    ...snapshot,
    selected_seq: seq,
    workspace_id: WORKBENCH_WORKSPACE,
    state: { ...snapshot.state, current: { ...snapshot.state.current, model: info } },
    dynamical_model_spec: dynamicalModelSpec,
    specification: checks,
    fit: fitted ? snapshot.fit : null,
    simulation: report ? report : null,
  };
}
const authored = fixtureValue(demoModelSnapshot.state.current.model);
const v7 = metadata(modelId(7), modelId(4), "edit_model");
snapshots.set(8, illustratedSnapshot(8, authored, false));
snapshots.set(9, illustratedSnapshot(9, authored, false, simulations.get(4)));
snapshots.set(10, illustratedSnapshot(10, authored, false, simulations.get(4)));
snapshots.set(11, illustratedSnapshot(11, v7, false));
snapshots.set(12, illustratedSnapshot(12, v7, false, simulations.get(7)));
const comparisonIndicator = fixtureValue(
  modelConstructs(freeDynamicalModelSpec)
    .flatMap((construct) => construct.indicators)
    .find((indicator) => indicator.observation.name === "phq9_screening_score"),
);
// These illustrative histories are returned by their producing actions below.
const comparedHistory = (values: number[]) => ({
  label: comparisonIndicator.observation.name,
  times: values.map((_, index) => index),
  values,
  support_start: values.map(() => null),
  support_end: values.map(() => null),
  time_origin: "2026-01-01T00:00:00Z",
  levels: null,
  empirical: [],
});
function comparisonReplicate(index: number) {
  return {
    ...fixtureValue(visualFixture.simulation.data[index]),
    ...(index < 3
      ? { [comparisonIndicator.observation.id]: comparedHistory([index, index + 1, index + 2]) }
      : {}),
  };
}
const dataComparison: DataDiffOutput = {
  report: {
    left: [{ revision: panelId, replicate_index: 0 }],
    right: [0, 1, 2].map((replicate_index) => ({ revision: commitId(9), replicate_index })),
    variables: [
      {
        indicator_id: comparisonIndicator.observation.id,
        changes: [],
        findings: [],
        statistics: [
          {
            statistic: "mean",

            left: [2.67],
            right: [1, 2, 3],
          },
        ],
        predictive: {
          reference_side: "left",
          evaluation: {
            n_subsample: 3,
            findings: [
              {
                code: "calibration",
                kind: "evaluated",
                subject: {
                  target: { kind: "indicator", id: comparisonIndicator.observation.id },
                },
                outcome: "failed",
                evidence: {
                  criterion: "calibration",
                  note: "Observed values fall outside the replicated range.",
                  value: 0.67,
                  lower: 0.7,
                  upper: 0.98,
                  lower_inclusive: true,
                  upper_inclusive: true,
                  display_value: "",
                  band_label: "",
                },
              },
            ],
            test_stats: [
              {
                indicator_id: comparisonIndicator.observation.id,
                stat_name: "mean",
                observed_value: 2.67,
                rep_values: [1, 2, 3],
                p_value: 0.33,
                histogram: [1, 2, 3].map((value) => ({
                  bin_center: value,
                  bin_start: value - 0.5,
                  bin_end: value + 0.5,
                  count: 1,
                })),
                frame: [1, 3],
              },
            ],
            overlays: [
              {
                indicator_id: comparisonIndicator.observation.id,
                times: [0, 1, 2],
                time_origin: "2026-09-01T00:00:00Z",
                standardized: false,
                observed: [1, 4, 3],
                median: [1, 2, 3],
                spaghetti_draws: [
                  [0, 1, 2],
                  [1, 2, 3],
                  [2, 3, 4],
                ],
                frame: [0, 4],
              },
            ],
          },
        },
      },
    ],
  },
};

const demoMetadata = fixtureValue(demoModelSnapshot.metadata);
const retainedMetadata = demoMetadata;
function preparation(seq: number): PrepareDataRequest<GitOid, FileSourceRef> {
  return {
    action: "prepare_data",
    reasoning: null,
    input: {
      dynamical_model_spec_ref: modelId(3),
      source: {
        files: ["input/observations.csv"],
        hashes: { "input/observations.csv": "0".repeat(64) },
        start: null,
        end: null,
      },
      extraction: Object.fromEntries(
        retainedMetadata.preparation.variables.map((variable) => [
          variable.observation.id,
          variable.extraction,
        ]),
      ),
      context: `Illustrative preparation ${seq}`,
    },
  };
}
const settings = (seed: number) => ({
  hardware: null,
  time_limit_seconds: null,
  algorithm: null,
  num_samples_per_chain: null,
  num_warmup: null,
  num_chains: null,
  num_particles: null,
  seed,
});
const workbenchRecords: StudyRevision[] = [
  record(1, {
    action: "edit_question",
    request: {
      action: "edit_question",
      reasoning: null,
      input: { question: fixtureValue(demoModelSnapshot.question) },
    },
    outcome: {
      status: "applied",
      result: commitId(0),
      effects: {
        produced: [fixtureValue(demoModelSnapshot.state.current.question)],
        retracted: [],
        reports: {},
      },
    },
  }),
  ...[2, 3, 4].map((seq) =>
    record(
      seq,
      {
        action: "edit_model",
        request: {
          action: "edit_model",
          reasoning: null,
          input: {
            parent_ref: seq === 2 ? questionId : modelId(seq - 2),
            dynamical_model_spec: fixtureValue(
              fixtureValue(snapshots.get(seq)).dynamical_model_spec,
            ),
          },
        },
        outcome: {
          status: "applied",
          result: commitId(0),
          effects: {
            produced: [fixtureValue(fixtureValue(snapshots.get(seq)).state.current.model)],
            retracted: [],
            reports: {},
          },
        },
      },
      seq === 2 ? [] : [seq === 3 ? "latent_structure" : "measurement_structure"],
    ),
  ),
  record(5, {
    action: "prepare_data",
    request: preparation(5),
    outcome: {
      status: "applied",
      result: commitId(0),
      effects: {
        produced: [
          fixtureValue(fixtureValue(snapshots.get(5)).state.current.raw_data),
          fixtureValue(fixtureValue(snapshots.get(5)).state.current.panel),
        ],
        retracted: [],
        reports: {},
      },
    },
  }),
  record(6, {
    action: "edit_model",
    request: null,
    outcome: {
      status: "raised",
      error_type: "ProposalError",
      error_message:
        "Parameter proposal failed.\nTraceback: illustrative full failure details are retained here.",
      details: [],
    },
  }),
  record(
    7,
    {
      action: "edit_model",
      request: {
        action: "edit_model",
        reasoning: null,
        input: {
          parent_ref: modelId(3),
          dynamical_model_spec: fixtureValue(models.get(modelId(4))),
        },
      },
      outcome: {
        status: "applied",
        result: commitId(0),
        effects: {
          produced: [fixtureValue(fixtureValue(snapshots.get(7)).state.current.model)],
          retracted: [],
          reports: {},
        },
      },
    },
    ["statistical_model_spec"],
  ),
  record(8, {
    action: "fit",
    request: {
      action: "fit",
      reasoning: null,
      input: {
        target: {
          kind: "new",
          dynamical_model_spec_ref: modelId(4),
          data_ref: { revision: panelId, replicate_index: 0 },
        },
        repeat: 0,
        settings: settings(8),
      },
    },
    outcome: {
      status: "raised",
      error_type: "IllustrativeFitFailure",
      error_message: "This illustrative fit failed before retaining a result.",
      details: [],
    },
  }),
  record(9, {
    action: "simulate",
    request: {
      action: "simulate",
      reasoning: null,
      input: {
        dynamical_model_spec_ref: modelId(4),
        simulation: simulationDesign,
        seed: 0,
        num_replicates: 100,
      },
    },
    outcome: {
      status: "applied",
      result: commitId(0),
      effects: { produced: [], retracted: [], reports: {} },
    },
  }),
  record(10, {
    action: "fit",
    request: {
      action: "fit",
      reasoning: null,
      input: {
        target: {
          kind: "new",
          dynamical_model_spec_ref: modelId(4),
          data_ref: { revision: panelId, replicate_index: 0 },
        },
        repeat: 0,
        settings: settings(10),
      },
    },
    outcome: {
      status: "raised",
      error_type: "IllustrativeFitFailure",
      error_message: "This illustrative fit failed before retaining a result.",
      details: [],
    },
  }),
  record(11, {
    action: "edit_model",
    request: {
      action: "edit_model",
      reasoning: null,
      input: {
        parent_ref: modelId(4),
        dynamical_model_spec: pinnedDynamicalModelSpec,
      },
    },
    outcome: {
      status: "applied",
      result: commitId(0),
      effects: { produced: [v7], retracted: [], reports: {} },
    },
  }),
  record(12, {
    action: "simulate",
    request: {
      action: "simulate",
      reasoning: null,
      input: {
        dynamical_model_spec_ref: modelId(7),
        simulation: simulationDesign,
        seed: 0,
        num_replicates: 100,
      },
    },
    outcome: {
      status: "applied",
      result: commitId(0),
      effects: { produced: [], retracted: [], reports: {} },
    },
  }),
  record(13, {
    action: "data_diff",
    request: {
      action: "data_diff",
      reasoning: null,
      input: {
        left_ref: [fixtureValue(dataComparison.report.left[0])],
        right_ref: dataComparison.report.right,
      },
    },
    outcome: {
      status: "applied",
      result: commitId(0),
      effects: { produced: [], retracted: [], reports: {} },
    },
  }),
];
const journal = workbenchRecords;
// Failed attempts and comparisons are leaves; successful scientific calls form one journal.
for (const [seq, snapshot] of snapshots)
  snapshots.set(seq, { ...snapshot, commit_id: commitId(seq) });
const commitParents: Record<number, number> = {
  1: 0,
  2: 1,
  3: 2,
  4: 3,
  5: 4,
  6: 5,
  7: 5,
  8: 7,
  9: 7,
  10: 9,
  11: 9,
  12: 11,
  13: 12,
};
for (const [index, entry] of journal.entries())
  journal[index] = {
    ...entry,
    parent_ids: [commitId(fixtureValue(commitParents[entry.record.seq]))],
  };
snapshots.set(13, {
  ...fixtureValue(snapshots.get(12)),
  selected_seq: 13,
  commit_id: commitId(13),
});
// Backend-shaped argument dependencies for the journal above, as the timeline route serves them.
export const workbenchDependencies: RecordDependency[] = (
  [
    [2, 1, "parent"],
    [5, 4, "dynamical_model_spec"],
    [3, 2, "parent"],
    [4, 3, "parent"],
    [7, 4, "parent"],
    [8, 7, "dynamical_model_spec"],
    [8, 5, "data"],
    [9, 7, "dynamical_model_spec"],
    [10, 7, "dynamical_model_spec"],
    [10, 5, "data"],
    [11, 7, "parent"],
    [12, 11, "dynamical_model_spec"],
    [13, 5, "left"],
    [13, 9, "right"],
  ] as const
).map(([seq, source_seq, argument]) => ({ seq, source_seq, argument }));
export const workbenchTraces = new Map([
  [3, demoTraces.latent_structure],
  [4, demoTraces.measurement_structure],
  [7, demoTraces.statistical_model_spec],
]);
export const workbenchQuestion = demoModelSnapshot.question?.text;
export const workbenchJournal: TimelineRevision[] = journal.map((entry) => {
  const outcome = entry.record.attempt.outcome;
  return {
    ...entry,
    call_id: entry.record.attempt.request === null ? null : callId(entry.record.seq),
    record: {
      seq: entry.record.seq,
      ts: entry.record.ts,
      messages: entry.record.messages,
      trace_ids: entry.record.trace_ids,
      attempt: {
        ...entry.record.attempt,
        outcome:
          outcome.status === "applied"
            ? { ...outcome, result: null, effects: { ...outcome.effects } }
            : outcome,
      },
    },
  };
});

/** Complete retained results backing the seven public call routes. */
export function workbenchResult(seq: number): ActionPoll {
  const entry = fixtureValue(journal.find((item) => item.record.seq === seq));
  const attempt = entry.record.attempt;
  if (attempt.outcome.status !== "applied")
    return {
      call_id: callId(seq),
      action: attempt.action,
      status: "failed",
      commit_id: entry.commit_id,
      body: null,
      messages: [{ kind: "failure", timestamp: stamp, failure: attempt.outcome }],
    };
  const snapshot = fixtureValue(snapshots.get(seq));
  const messages: ExecutionMessage[] = [
    ...entry.record.messages,
    ...(attempt.action === "prepare_data"
      ? [
          {
            kind: "log" as const,
            timestamp: stamp,
            severity: "warning" as const,
            code: "EXTRACTION_PARTIAL",
            subject: "action",
            detail: "",
          },
        ]
      : []),
    ...entry.record.trace_ids.map((id) => ({
      kind: "trace" as const,
      timestamp: stamp,
      trace_id: id,
      trace: fixtureValue(workbenchTraces.get(seq)),
    })),
  ];
  const envelope = {
    call_id: callId(seq),
    status: "success" as const,
    commit_id: entry.commit_id,
    messages,
  };
  switch (attempt.action) {
    case "edit_question":
      return {
        ...envelope,
        action: attempt.action,
        body: { question: fixtureValue(snapshot.question) },
      };
    case "edit_model":
      return { ...envelope, action: attempt.action, body: modelResult(snapshot) };
    case "prepare_data":
      return {
        ...envelope,
        action: attempt.action,
        body: {
          extraction: { workers: [], extraction_reused: null },
          metadata: fixtureValue(snapshot.metadata),
          profile: fixtureValue(snapshot.profile),
          data: {
            ...visualFixture.observations,
            [comparisonIndicator.observation.id]: comparedHistory([1, 4, 3]),
          },
        },
      };
    case "fit":
      throw new Error("This story has no retained successful fit");
    case "simulate":
      return {
        ...envelope,
        action: attempt.action,
        body: {
          ...visualFixture.simulation,
          data: [
            comparisonReplicate(0),
            ...visualFixture.simulation.data
              .slice(1)
              .map((_, index) => comparisonReplicate(index + 1)),
          ],
          report: fixtureValue(snapshot.simulation),
        },
      };
    case "data_diff":
      return { ...envelope, action: "data_diff", body: dataComparison };
    case "model_diff":
      throw new Error("This story has no saved model comparison");
  }
}

/** Exercise the real UI requests with isolated, explicit story responses. */
export function workbenchHandlers() {
  const timeline = (): TimelineResponse => ({
    attempts: workbenchJournal,
    dependencies: workbenchDependencies,
    running: [
      {
        call_id: callId(14),
        action: "fit",
        request: {
          action: "fit",
          reasoning: null,
          input: {
            target: {
              kind: "new",
              dynamical_model_spec_ref: modelId(7),
              data_ref: { revision: panelId, replicate_index: 0 },
            },
            repeat: 0,
            settings: settings(13),
          },
        },
        messages: [
          {
            kind: "log",
            timestamp: "2026-09-16T12:05:00Z",
            severity: "info",
            code: "FIT_STARTED",
            subject: "action",
            detail: "",
          },
        ],
      },
    ],
  });
  return [
    http.get(`/api/studies/${WORKBENCH_WORKSPACE}/timeline`, () => HttpResponse.json(timeline())),
    http.get(`/api/studies/${WORKBENCH_WORKSPACE}/:action/:callId`, ({ params }) => {
      const entry = journal.find(
        (item) =>
          item.record.attempt.action === params.action && callId(item.record.seq) === params.callId,
      );
      return entry
        ? new HttpResponse(encode(workbenchResult(entry.record.seq)), {
            headers: { "Content-Type": "application/msgpack" },
          })
        : HttpResponse.json({ detail: "No saved story call" }, { status: 403 });
    }),
  ];
}
