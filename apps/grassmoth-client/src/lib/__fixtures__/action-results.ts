/** Small typed values for projection cases absent from France's saved happy path. */
import type {
  ActionSuccess,
  EditModelOutput,
  FailedPoll,
  FitOutput,
  RunningPoll,
  SimulateOutput,
} from "@grassmoth/api-types";
import { dump } from "npyjs";
import { initialOutcome, modelFixture, outcome } from "./model";
import { fittedSnapshot } from "./snapshot";

const { id: initialOutcomeId, ...initialDefinition } = initialOutcome;
export const initialModelResult: EditModelOutput = {
  dynamical_model_spec: {
    constructs: { [initialOutcomeId]: { ...initialDefinition, indicators: {}, dynamics: {} } },
    edges: {},
    parameters: {},
    distributions: {},
    law_layouts: {},
    question: modelFixture.question,
    outcome: initialOutcomeId,
    time: modelFixture.time,
    core_query: modelFixture.core_query,
  },
  checks: { specification: [], question: { findings: [] } },
  identification: { outcome: initialOutcomeId, treatments: {} },
  pruning: { constructs: [], edges: [], parameters: [], distributions: [] },
};

export const authoredResult: EditModelOutput = {
  dynamical_model_spec: modelFixture,
  checks: { specification: [], question: { findings: [] } },
  identification: { outcome: outcome.id, treatments: {} },
  pruning: { constructs: [], edges: [], parameters: [], distributions: [] },
};

if (fittedSnapshot.fit === null || fittedSnapshot.fit_checks === null)
  throw new Error("The fitted unit fixture must own its report and checks");

export const fittedResult: FitOutput = {
  dynamical_model_spec: modelFixture,
  checks: fittedSnapshot.fit_checks,
  profile: null,
  inference: {
    core: fittedSnapshot.fit,
    detail: { trace_data: [], rank_histograms: [], pareto_k: [], loo_pit: [] },
    evidence: {
      checkpoint: null,
      latent_free_mask: null,
      chain_extra_fields: {},
      observation_log_probs: null,
      observed_rows: null,
      exact_observation_rows: null,
      phase_extra_fields: {},
      warmup_complete_log_posterior_history: null,
      all_complete_log_posterior_history: null,
      initial_latent_delta: null,
      final_latent_delta: null,
    },
  },
};

const array = (values: number[], shape: number[], dtype: "f8" | "b1" = "f8") => ({
  npy: new Uint8Array(dump(values, shape, { dtype })),
});
const paths = array([1, 3, 4, 5, 2, 5, 6, 7], [2, 4, 1]);

export const simulatedResult: SimulateOutput = {
  data: [{}, {}],
  report: {
    evidence: {
      assignments: [],
      time_origin: "2026-01-01T00:00:00Z",
      times: [0, 1, 2, 3],
      draws: 2,
      seed: 0,
      state_ids: [outcome.id],
      parameter_draws: {},
      arms: { kind: "single", action: { latent_paths: paths, observations: paths } },
      observation_layout: {
        variables: outcome.indicators.map(({ observation }) => ({
          ...observation,
          observation_window: "1d",
        })),
        support_start_times: array([-1, 0, 1, 2], [4, 1]),
        support_end_times: array([0, 1, 2, 3], [4, 1]),
        mask: array(
          Array.from({ length: 8 }, () => 1),
          [2, 4, 1],
          "b1",
        ),
      },
    },
    summary: {
      state_frames: { [outcome.id]: [1, 7] },
      indicator_frames: {},
      action_category_probabilities: {},
      reference_category_probabilities: {},
    },
    law: { kind: "authored", interpretation: "prior_predictive" },
    findings: [],
  },
};

export const unavailableEffect: SimulateOutput = {
  ...simulatedResult,
  report: {
    ...simulatedResult.report,
    evidence: {
      ...simulatedResult.report.evidence,
      arms: {
        kind: "paired",
        action: simulatedResult.report.evidence.arms.action,
        reference: simulatedResult.report.evidence.arms.action,
        causal: {
          kind: "not_evaluated",
          code: "causal_effect",
          subject: "causal_effect",
          reason: "CAUSAL_EVALUATION_FAILED",
          detail: "The recorded causal evaluation was unavailable.",
        },
      },
    },
  },
};

export const failedPoll: FailedPoll = {
  action: "edit_model",
  call_id: `call:${"a".repeat(64)}`,
  commit_id: "a".repeat(40),
  status: "failed",
  body: null,
  messages: [
    {
      kind: "failure",
      timestamp: "2026-01-01T00:00:00Z",
      failure: {
        status: "raised",
        error_type: "ProposalError",
        error_message: "Proposal failed",
        details: [],
      },
    },
  ],
};
export const rejectedPoll: FailedPoll = {
  ...failedPoll,
  action: "fit",
  call_id: `call:${"b".repeat(64)}`,
  commit_id: "b".repeat(40),
  messages: [
    {
      kind: "failure",
      timestamp: "2026-01-01T00:00:00Z",
      failure: {
        status: "rejected",
        code: "TEST_REJECTION",
        subject: "fit",
        detail: "Selected inputs cannot be fitted",
      },
    },
  ],
};
export const runningPoll: RunningPoll = {
  action: "fit",
  call_id: `call:${"c".repeat(64)}`,
  commit_id: null,
  status: "running",
  body: null,
  messages: [],
};
export const modelComparison: Extract<ActionSuccess, { action: "model_diff" }> = {
  action: "model_diff",
  call_id: `call:${"d".repeat(64)}`,
  commit_id: "d".repeat(40),
  status: "success",
  body: { changes: { time: { step: "2d" } } },
  messages: [],
};
