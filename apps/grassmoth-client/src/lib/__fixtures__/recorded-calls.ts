/** A recorded study's calls, the inputs they named and their saved results. */
import type {
  ActionSuccess,
  ArtifactRecord,
  RecordDependency,
  TimelineRevision,
} from "@grassmoth/api-types";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import {
  authoredResult,
  fittedResult,
  initialModelResult,
  simulatedResult,
} from "./action-results";
import { outcome } from "./model";

const simulationReport = simulatedResult.report;

export const oid = (seq: number) => seq.toString(16).padStart(40, "0");
export const question = initialModelResult.dynamical_model_spec;
const model = () => authoredResult;
const observation = fixtureValue(outcome.indicators[0]).observation;
const extraction = {
  kind: "semantic" as const,
  how_to_measure: "Read the value.",
  source_columns: [],
};
const source = { files: ["input/test.csv"] as const, hashes: {}, start: null, end: null };
const design = { window: { start: "2026-01-01", horizon: "1d" }, interventions: [] };
const settings = {
  profile: false,
  thinning: null,
  hardware: null,
  time_limit_seconds: null,
  algorithm: null,
  num_samples_per_chain: null,
  num_warmup: null,
  num_chains: null,
  seed: 0,
};
export const dependencies: RecordDependency[] = [
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
type RecordedRequest = NonNullable<TimelineRevision["record"]["attempt"]["request"]>;

function recordedRequest(seq: number): RecordedRequest {
  switch (seq) {
    case 1: {
      return {
        action: "edit_model",
        reasoning: null,
        input: { parent_ref: null, dynamical_model_spec: question },
      };
    }
    case 5: {
      return {
        action: "prepare_data",
        reasoning: null,
        input: {
          dynamical_model_spec_ref: oid(4),
          source,
          extraction: { [observation.id]: extraction },
          context: "",
        },
      };
    }
    case 8:
    case 10: {
      return {
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
      };
    }
    case 9:
    case 12: {
      return {
        action: "simulate",
        reasoning: null,
        input: {
          dynamical_model_spec_ref: oid(seq - 1),
          simulation: design,
          seed: 0,
          num_replicates: 100,
        },
      };
    }
    case 13: {
      return {
        action: "data_diff",
        reasoning: null,
        input: {
          left_ref: [{ revision: oid(5), replicate_index: 0 }],
          right_ref: [{ revision: oid(9), replicate_index: 1 }],
        },
      };
    }
    default: {
      return {
        action: "edit_model",
        reasoning: null,
        input: {
          parent_ref: seq === 11 ? oid(10) : oid(1),
          dynamical_model_spec: model().dynamical_model_spec,
        },
      };
    }
  }
}

function producedArtifacts(seq: number, request: RecordedRequest): ArtifactRecord[] {
  if (request.action === "edit_model" || request.action === "fit") {
    return [
      {
        artifact_id: "model",
        revision: oid(seq + 100),
        derived_from: {},
        produced_by: request.action,
        created_at: "2026-01-01T00:00:00Z",
        source: { kind: "files" },
      },
    ];
  }
  if (request.action === "prepare_data") {
    return [
      {
        artifact_id: "panel",
        revision: oid(seq),
        derived_from: {},
        produced_by: request.action,
        created_at: "2026-01-01T00:00:00Z",
        source: { kind: "files" },
      },
    ];
  }
  return [];
}

export const journal: TimelineRevision[] = [1, 4, 5, 7, 8, 9, 10, 11, 12, 13].map((seq) => {
  const request = recordedRequest(seq);
  return {
    call_id: `call:${seq.toString(16).padStart(64, "0")}`,
    commit_id: oid(seq),
    parent_ids: [oid(seq - 1)],
    record: {
      seq,
      ts: "2026-01-01T00:00:00Z",
      attempt: {
        action: request.action,
        request,
        outcome: {
          status: "applied",
          result: null,
          effects: {
            produced: producedArtifacts(seq, request),
            retracted: [],
            reports: {},
          },
        },
      },
    },
  };
});

export function savedResult(seq: number): ActionSuccess {
  const call = fixtureValue(journal.find((item) => item.record.seq === seq));
  const envelope = {
    status: "success" as const,
    call_id: fixtureValue(call.call_id),
    commit_id: call.commit_id,
    messages: [],
  };
  switch (call.record.attempt.action) {
    case "edit_model": {
      return { ...envelope, action: "edit_model", body: seq === 1 ? initialModelResult : model() };
    }
    case "prepare_data": {
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
    }
    case "fit": {
      return {
        ...envelope,
        action: "fit",
        body: {
          ...fittedResult,
        },
      };
    }
    case "simulate": {
      return {
        ...envelope,
        action: "simulate",
        body: { report: simulationReport, data: [{}, {}] },
      };
    }
    case "data_diff": {
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
    }
    case "model_diff": {
      throw new Error("No model comparison in this test");
    }
  }
}

export const entry = (seq: number) => fixtureValue(journal.find((call) => call.record.seq === seq));
