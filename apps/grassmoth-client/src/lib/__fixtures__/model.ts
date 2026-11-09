/** Small authored graph owned by the graph, entity and law contract tests. */
import type { ConstructSpec, DynamicalModelSpec, ParameterSpec } from "@grassmoth/api-types";

export const decay: ParameterSpec = {
  id: "parameter:00000000000000000001",
  name: "outcome_persistence",
  transform: { kind: "dt_persistence_to_ct_decay", interval_days: 1 },
  distribution: "distribution:00000000000000000001",
  reasoning: {
    text: "Persistence of the outcome over one day. A symmetric Beta law keeps synthetic daily persistence between zero and one.",
    reference_urls: [],
  },
};
export const diffusion: ParameterSpec = {
  ...decay,
  id: "parameter:00000000000000000002",
  name: "outcome_diffusion",
  transform: { kind: "identity" },
  distribution: "distribution:00000000000000000002",
  reasoning: {
    text: "Outcome process noise. A unit HalfNormal law gives the synthetic process noise a positive scale.",
    reference_urls: [],
  },
};

export const baseline: ConstructSpec = {
  id: "construct:00000000000000000001",
  name: "baseline",
  reasoning: { text: "A static parent of treatment.", reference_urls: [] },
  role: "endogenous",
  temporal_status: "time_invariant",
  indicators: [],
  dynamics: [],
  coefficients: [],
  innovation_family: "gaussian",
  distribution: null,
};
export const treatment: ConstructSpec = {
  ...baseline,
  id: "construct:00000000000000000002",
  name: "treatment",
  reasoning: { text: "A varying treatment.", reference_urls: [] },
  temporal_status: "time_varying",
  indicators: [
    {
      observation: {
        id: "indicator:00000000000000000001",
        name: "treatment_reading",
        measurement: { dtype: "continuous" },
        aggregation: "mean",
        observation_window: null,
      },
      reasoning: {
        text: "A synthetic reading tracks its construct for this interface example.",
        reference_urls: [],
      },
      construct_polarity: "positive",
      likelihood: null,
    },
  ],
};
export const outcome: ConstructSpec = {
  ...treatment,
  id: "construct:00000000000000000003",
  name: "outcome",
  reasoning: { text: "The response to treatment.", reference_urls: [] },
  indicators: [
    {
      observation: {
        id: "indicator:00000000000000000002",
        name: "outcome_reading",
        measurement: { dtype: "continuous" },
        aggregation: "mean",
        observation_window: null,
      },
      reasoning: {
        text: "A synthetic reading tracks its construct for this interface example.",
        reference_urls: [],
      },
      construct_polarity: "positive",
      likelihood: null,
    },
  ],
  dynamics: [
    {
      id: "mechanism:00000000000000000001",
      kind: "drift",
      expression: {
        kind: "binary",
        operator: "multiply",
        left: { kind: "coefficient", role: "decay", value: decay.id, construct_ids: [] },
        right: {
          kind: "binary",
          operator: "subtract",
          left: { kind: "literal", value: 0 },
          right: { kind: "state", construct_id: "construct:00000000000000000003" },
        },
      },
    },
  ],
  coefficients: [
    { kind: "coefficient", role: "diffusion_scale", value: diffusion.id, construct_ids: [] },
  ],
};

export const initialOutcome: ConstructSpec = {
  ...outcome,
  indicators: [],
  dynamics: [],
  coefficients: [],
  distribution: null,
};

function constructEntry({ id, indicators, dynamics, ...value }: ConstructSpec) {
  return [
    id,
    {
      ...value,
      indicators: Object.fromEntries(
        indicators.map(({ observation: { id, ...observation }, ...indicator }) => [
          id,
          { ...indicator, observation },
        ]),
      ),
      dynamics: Object.fromEntries(dynamics.map(({ id, ...mechanism }) => [id, mechanism])),
    },
  ] as const;
}
function parameterEntry({ id, ...value }: ParameterSpec) {
  return [id, value] as const;
}

export const modelFixture: DynamicalModelSpec = {
  constructs: Object.fromEntries([baseline, treatment, outcome].map(constructEntry)),
  edges: {
    "edge:00000000000000000001": {
      cause: baseline.id,
      effect: treatment.id,
      mechanisms: {},
      reasoning: { text: "Baseline affects treatment.", reference_urls: [] },
    },
    "edge:00000000000000000002": {
      cause: treatment.id,
      effect: outcome.id,
      mechanisms: {},
      reasoning: { text: "Treatment affects the outcome.", reference_urls: [] },
    },
  },
  parameters: Object.fromEntries([decay, diffusion].map(parameterEntry)),
  distributions: {
    "distribution:00000000000000000001": {
      distribution: "Beta",
      params: { concentration1: 2, concentration0: 2 },
    },
    "distribution:00000000000000000002": {
      distribution: "HalfNormal",
      params: { scale: 1 },
    },
  },
  law_layouts: {},
  question: "How does treatment affect the outcome?",
  outcome: outcome.id,
  time: { window: { start: "2026-01-01", horizon: "2d" }, step: "1d" },
  core_query: { window: { start: "2026-01-01", horizon: "2d" }, interventions: [] },
};
