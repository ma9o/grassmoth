/** Explicit served projections for unit tests; no saved study or numerical execution. */
import type { InferenceReportCore, ModelSnapshot } from "@nof1-causal-lab/api-types";
import { decay, diffusion, modelFixture, outcome } from "./model";

export const dynamicalModelSpecRef = {
  workspace_id: "TEST",
  revision: "1".repeat(40),
  path: "model.json",
};

export const emptySnapshot: ModelSnapshot = {
  workspace_id: "TEST",
  commit_id: "0".repeat(40),
  selected_seq: 0,
  question: null,
  dynamical_model_spec: null,
  state: { current: {}, data: null },
  metadata: null,
  profile: null,
  identification: null,
  fit_checks: null,
  fit: null,
  specification: null,
  question_checks: null,
  simulation: null,
};

const density = { x: [0, 0.5, 1], density: [0, 2, 0] };
export const authoredSnapshot: ModelSnapshot = {
  ...emptySnapshot,
  commit_id: "3".repeat(40),
  selected_seq: 3,
  dynamical_model_spec: modelFixture,
  question: { text: "How does treatment change the outcome?", outcome: outcome.id, queries: {} },
  state: {
    current: {
      question: {
        artifact_id: "question",
        revision: "2".repeat(40),
        derived_from: {},
        produced_by: "edit_question",
        created_at: "2026-01-01T00:00:00Z",
        source: { kind: "files" },
      },
      model: {
        artifact_id: "model",
        revision: dynamicalModelSpecRef.revision,
        derived_from: { question: "2".repeat(40) },
        produced_by: "edit_model",
        created_at: "2026-01-01T00:00:00Z",
        source: { kind: "files" },
      },
    },
    data: null,
  },
  identification: { outcome: outcome.id, treatments: {} },
  question_checks: { findings: [] },
  specification: [],
};

const fit: InferenceReportCore = {
  inference_metadata: {
    distribution: "distribution:00000000000000000003",
    num_samples_total: 3,
    num_chains: 1,
    duration_seconds: 0,
    engine: { method: "custom" },
    sampler_diagnostics: null,
  },
  inference_diagnostics: null,
  convergence: { findings: [], checked: 0, status: "not_evaluated", messages: [] },
  loo_diagnostics: null,
  posterior_marginals: [decay, diffusion].map((parameter) => ({
    parameter: parameter.name,
    subject: {
      parameter_id: parameter.id,
      element_id: `element:${parameter.id.slice("parameter:".length)}` as const,
    },
    density_curve: density,
    empirical: [
      { value: 0.1, probability: 1 / 3 },
      { value: 0.5, probability: 2 / 3 },
      { value: 0.9, probability: 1 },
    ],
    mean: 0.5,
    lower: 0.1,
    upper: 0.9,
    interval_kind: "hdi" as const,
    interval_mass: 0.9,
    sd: 0.2,
  })),
  prior_densities: { [decay.id]: density, [diffusion.id]: density },
};
export const fittedSnapshot: ModelSnapshot = {
  ...authoredSnapshot,
  commit_id: "4".repeat(40),
  selected_seq: 4,
  fit: fit,
  state: { ...authoredSnapshot.state, data: { revision: "5".repeat(40), replicate_index: 0 } },
  fit_checks: {
    data: { indicators: {}, findings: [] },
    preflight: [],
    question: { findings: [] },
  },
  question_checks: {
    findings: [],
  },
};
