/** Small authored graph owned by the graph, entity and law contract tests. */
import type { ConstructSpec, DynamicalModelSpec, ParameterSpec } from "@grassmoth/api-types";

export const decay: ParameterSpec = {
  id: "parameter:00000000000000000001",
  name: "outcome_persistence",
  description: "Persistence of the outcome over one day.",
  transform: { kind: "dt_persistence_to_ct_decay", interval_days: 1 },
  distribution: "distribution:00000000000000000001",
  reasoning: null,
  sources: [],
};
export const diffusion: ParameterSpec = {
  ...decay,
  id: "parameter:00000000000000000002",
  name: "outcome_diffusion",
  description: "Outcome process noise.",
  transform: { kind: "identity" },
  distribution: "distribution:00000000000000000002",
};

export const baseline: ConstructSpec = {
  id: "construct:00000000000000000001",
  name: "baseline",
  description: "A static parent of treatment.",
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
  description: "A varying treatment.",
  temporal_status: "time_varying",
  indicators: [
    {
      observation: {
        id: "indicator:00000000000000000001",
        name: "treatment_reading",
        measurement_dtype: "continuous",
        aggregation: "mean",
        observation_window: null,
        ordinal_levels: null,
        categorical_levels: null,
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
  description: "The response to treatment.",
  indicators: [
    {
      observation: {
        id: "indicator:00000000000000000002",
        name: "outcome_reading",
        measurement_dtype: "continuous",
        aggregation: "mean",
        observation_window: null,
        ordinal_levels: null,
        categorical_levels: null,
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
      description: "Baseline affects treatment.",
      sources: [],
    },
    "edge:00000000000000000002": {
      cause: treatment.id,
      effect: outcome.id,
      mechanisms: {},
      description: "Treatment affects the outcome.",
      sources: [],
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
  measurement_clock: "1d",
};
