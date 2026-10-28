import { causalEffect } from "@/lib/simulation-report";
import { readNumericalArray } from "@grassmoth/api-types";
import type {
  ArrayVector,
  FitOutput,
  NumericalArray,
  InferenceReport,
  InferenceReportDetail,
  ObservationHistory,
  EmpiricalPoint,
  ParameterRef,
  NumPyroValue,
  NumPyroDistribution,
  ScalarValues,
  SimulateOutput,
  DynamicalModelSpec,
  SimulationSummary,
} from "@grassmoth/api-types";

export type HistoryView = Omit<ObservationHistory, "values" | "support_start" | "support_end"> & {
  readonly values: readonly (number | null)[];
  readonly support_start: readonly (number | null)[];
  readonly support_end: readonly (number | null)[];
};
export interface PathView {
  readonly draw: number;
  readonly values: readonly (number | null)[];
}
export interface PathSeriesView {
  readonly label: string;
  readonly action: readonly PathView[];
  readonly reference: readonly PathView[];
  readonly levels: readonly string[] | null;
  readonly frame: readonly [number, number] | null;
}
export interface SimulationPathsView {
  readonly times: readonly number[];
  readonly time_origin: string;
  readonly total_draws: number;
  readonly start: number;
  readonly count: number;
  readonly states: Readonly<Record<string, PathSeriesView | undefined>>;
  readonly indicators: Readonly<Record<string, PathSeriesView | undefined>>;
  readonly effect: PathSeriesView | null;
  readonly action_category_probabilities: SimulationSummary["action_category_probabilities"];
  readonly reference_category_probabilities: SimulationSummary["reference_category_probabilities"];
}
export interface DrawColumnView {
  readonly label: string;
  readonly subject: ParameterRef;
  readonly values: readonly number[];
  readonly empirical: readonly EmpiricalPoint[];
}

/** Decode a recorded vector selection; all statistical summaries already belong to the result. */
export function scalarValues(value: ScalarValues): readonly (number | null)[] {
  if (!("array" in value)) return value;
  const selection: ArrayVector = value;
  const array = readNumericalArray(selection.array);
  const axis = selection.indices.indexOf(null);
  const strides = array.shape.map((_, index) =>
    array.shape.slice(index + 1).reduce((a, b) => a * b, 1),
  );
  const offset = selection.indices.reduce<number>(
    (sum, index, dimension) => sum + (index ?? 0) * required(strides[dimension]),
    0,
  );
  const mask = selection.mask ? scalarValues(selection.mask) : null;
  const stop = selection.stop ?? required(array.shape[axis]);
  return Array.from({ length: stop - selection.start }, (_, index) => {
    if (mask && !mask[index]) return null;
    const scalar = required(
      array.values[offset + (index + selection.start) * required(strides[axis])],
    );
    const numeric = Number(scalar);
    return Number.isFinite(numeric) ? numeric : null;
  });
}

export function historyView(history: ObservationHistory): HistoryView {
  return {
    ...history,
    values: scalarValues(history.values),
    support_start: scalarValues(history.support_start),
    support_end: scalarValues(history.support_end),
  };
}

const simulationPaths = new WeakMap<
  SimulateOutput,
  { readonly model: DynamicalModelSpec; readonly paths: SimulationPathsView }
>();

/** Share one projection of immutable evidence across charts; reductions stay in the report. */
export function pathsView(
  result: SimulateOutput,
  dynamicalModelSpec: DynamicalModelSpec,
): SimulationPathsView {
  const cached = simulationPaths.get(result);
  if (cached?.model === dynamicalModelSpec) return cached.paths;
  const { evidence, summary } = result.report;
  const causal = causalEffect(result.report);
  const reference = evidence.arms.kind === "paired" ? evidence.arms.reference : null;
  const paths = (array: NumericalArray, column?: number, observed = false): readonly PathView[] =>
    Array.from({ length: evidence.draws }, (_, draw) => ({
      draw,
      values: scalarValues({
        array,
        indices: column === undefined ? [draw, null] : [draw, null, column],
        start: 0,
        stop: null,
        mask:
          observed && column !== undefined
            ? {
                array: evidence.observation_layout.mask,
                indices: [draw, null, column],
                start: 0,
                stop: null,
                mask: null,
              }
            : null,
      }),
    }));
  const view: SimulationPathsView = {
    times: evidence.times,
    time_origin: evidence.time_origin,
    total_draws: evidence.draws,
    start: 0,
    count: evidence.draws,
    states: Object.fromEntries(
      evidence.state_ids.map((id, column) => [
        id,
        {
          label: required(dynamicalModelSpec.constructs[id]).name,
          action: paths(evidence.arms.action.latent_paths, column),
          reference: reference ? paths(reference.latent_paths, column) : [],
          frame: summary.state_frames[id] ?? null,
          levels: null,
        },
      ]),
    ),
    indicators: Object.fromEntries(
      evidence.observation_layout.variables.map((variable, column) => [
        variable.id,
        {
          label: variable.name,
          action: paths(evidence.arms.action.observations, column, true),
          reference: reference ? paths(reference.observations, column, true) : [],
          frame: summary.indicator_frames[variable.id] ?? null,
          levels: "levels" in variable.measurement ? variable.measurement.levels : null,
        },
      ]),
    ),
    effect:
      causal !== undefined
        ? {
            label: required(causal.labels[causal.outcome]),
            action: paths(causal.differences),
            reference: [],
            levels: null,
            frame: causal.frame,
          }
        : null,
    action_category_probabilities: summary.action_category_probabilities,
    reference_category_probabilities: summary.reference_category_probabilities,
  };
  simulationPaths.set(result, { model: dynamicalModelSpec, paths: view });
  return view;
}

const isArray: (value: NumPyroValue) => value is readonly NumPyroValue[] = Array.isArray;

function object(value: NumPyroValue | undefined): NumPyroDistribution["params"] {
  if (
    value === undefined ||
    value === null ||
    typeof value !== "object" ||
    isArray(value) ||
    "npy" in value
  )
    throw new Error("Saved posterior law has an invalid constructor");
  return value;
}

/** Select the model's original joint atoms; empirical summaries stay backend-owned. */
export function drawsView(result: FitOutput): readonly DrawColumnView[] {
  const identity = result.inference.core.inference_metadata.distribution;
  if (identity === null || result.dynamical_model_spec === null) return [];
  const law = required(result.dynamical_model_spec.distributions[identity]);
  const layout = required(result.dynamical_model_spec.law_layouts[identity]);
  const component = object(law.params.component_distribution);
  const atoms = object(component.params).v;
  if (
    law.distribution !== "MixtureSameFamily" ||
    component.distribution !== "Delta" ||
    atoms === undefined ||
    atoms === null ||
    typeof atoms !== "object" ||
    !("npy" in atoms) ||
    !(atoms.npy instanceof Uint8Array)
  )
    throw new Error("Saved posterior law must own its retained joint atoms");
  const array = { npy: atoms.npy };
  const coordinates = layout.parameters.flatMap(([parameter, elements]) =>
    elements.map((element) => ({ parameter_id: parameter, element_id: element })),
  );
  return coordinates.map((subject, column) => ({
    label: required(layout.labels[subject.element_id]),
    subject,
    values: scalarValues({ array, indices: [null, column], start: 0, stop: null, mask: null }).map(
      required,
    ),
    empirical: required(
      result.inference.core.posterior_marginals.find(
        (marginal) =>
          marginal.subject.parameter_id === subject.parameter_id &&
          marginal.subject.element_id === subject.element_id,
      ),
    ).empirical,
  }));
}

/** Missing coordinates are a corrupt result, never an alternate scientific value. */
function required<T>(value: T | null | undefined): T {
  if (value === undefined || value === null)
    throw new Error("Saved numerical result has an invalid coordinate");
  return value;
}

export type InferenceDetailView = Omit<InferenceReportDetail, "trace_data"> & {
  readonly trace_data: readonly {
    readonly subject: ParameterRef;
    readonly chains: readonly (readonly number[])[];
  }[];
  readonly divergent: readonly boolean[] | null;
  readonly initial_latent_delta: readonly (readonly number[])[] | null;
  readonly final_latent_delta: readonly (readonly number[])[] | null;
};
export type InferenceView = Omit<InferenceReport, "detail"> & {
  readonly detail: InferenceDetailView;
};

/** Resolve stored chart coordinates without calculating any diagnostic. */
export function inferenceView(result: FitOutput): InferenceView {
  const report = result.inference;
  const detail = report.detail;
  const evidence = report.evidence;
  const rows = (array: NumericalArray | null): readonly (readonly number[])[] | null =>
    array === null
      ? null
      : Array.from({ length: required(readNumericalArray(array).shape[0]) }, (_, row) =>
          scalarValues({ array, indices: [row, null], start: 0, stop: null, mask: null }).map(
            required,
          ),
        );
  const divergent = evidence.chain_extra_fields.diverging;
  const vector = (value: ScalarValues) => scalarValues(value).map(required);
  return {
    ...report,
    detail: {
      ...detail,
      trace_data: detail.trace_data.map((trace) => ({
        ...trace,
        chains: trace.chains.map(vector),
      })),
      divergent:
        divergent === undefined ? null : Array.from(readNumericalArray(divergent).values, Boolean),
      initial_latent_delta: rows(evidence.initial_latent_delta),
      final_latent_delta: rows(evidence.final_latent_delta),
    },
  };
}
