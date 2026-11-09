import type {
  ConstructId,
  DynamicalModelSpec,
  DynamicsMechanismSpec,
  EdgeId,
  Expression,
  IndicatorId,
  MechanismId,
} from "@grassmoth/api-types";
import { modelConstructs, modelEdges } from "@/lib/model-accessors";
import { formatModelDate } from "@/lib/utils/format";
import {
  type ExpressionInputs,
  evaluateExpression,
  finiteValue,
  PlotUnavailable,
  potentialDrift,
  scalarValue,
} from "./expression-evaluation";
import { displayProbability } from "./law-draws";
import { measurementDraw } from "./measurement-draws";
import { type ModelDraws, modelDraws } from "./model-draws";
import { humanize, mechanismName } from "./selection";

export type FlowPlot =
  | {
      readonly kind: "draws";
      readonly label: string;
      readonly note: string;
      readonly times: readonly number[];
      readonly timeOrigin: string;
      readonly rows: readonly (readonly number[])[];
      readonly retained: boolean;
      readonly levels: readonly string[] | null;
    }
  | { readonly kind: "unavailable"; readonly label: string; readonly reason: string };

export interface ModelFlowPlots {
  readonly constructs: ReadonlyMap<ConstructId, FlowPlot>;
  readonly intrinsic: ReadonlyMap<ConstructId, FlowPlot>;
  readonly edges: ReadonlyMap<EdgeId, FlowPlot>;
  /** Each mechanism alone: one term of its effect's drift. */
  readonly mechanisms: ReadonlyMap<MechanismId, FlowPlot>;
  readonly indicators: ReadonlyMap<IndicatorId, FlowPlot>;
}

/** A plot's time coordinates and, for a discrete reading, its category levels. */
interface PlotAxes {
  readonly times?: readonly number[];
  readonly levels?: readonly string[] | null;
}

/** The state operands arriving at a local mechanism, in declaration order. */
export function expressionStates(expressions: readonly Expression[]): readonly ConstructId[] {
  const ids = new Set<ConstructId>();
  const visit = (expression: Expression): void => {
    if (expression.kind === "state") ids.add(expression.construct_id);
    if (expression.kind === "binary") {
      visit(expression.left);
      visit(expression.right);
    }
    if (expression.kind === "call") {
      for (const argument of expression.arguments) visit(argument);
    }
  };
  for (const expression of expressions) visit(expression);
  return [...ids];
}

const cache = new WeakMap<DynamicalModelSpec, ModelFlowPlots>();

function readingTimeNote(draws: ModelDraws): string {
  const firstTime = draws.times[0];
  if (firstTime === undefined) return "";
  if (draws.timeOrigin === "relative") return `At model day ${firstTime}. `;
  return `At ${formatModelDate(firstTime, draws.timeOrigin)}. `;
}

/**
 * Compose local outputs from the model's saved laws, without changing any action result.
 * All plots share one draw context. Missing inputs produce an explicit absence, never
 * an independently reconstructed posterior or a substitute parameter marginal.
 */
export function modelFlowPlots(model: DynamicalModelSpec): ModelFlowPlots {
  const cached = cache.get(model);
  if (cached) return cached;
  const draws = modelDraws(model);
  const readingTime = readingTimeNote(draws);
  const constructs = modelConstructs(model);
  const name = (id: ConstructId) => humanize(model.constructs[id]?.name ?? id);
  const plot = (
    label: string,
    note: string,
    evaluate: (inputs: ExpressionInputs, draw: number, time: number) => number,
    { times = draws.times, levels = null }: PlotAxes = {},
  ): FlowPlot => {
    try {
      return {
        kind: "draws",
        label,
        note,
        times,
        timeOrigin: draws.timeOrigin,
        retained: draws.retained,
        levels,
        rows: Array.from({ length: draws.count }, (_, draw) =>
          times.map((time) => finiteValue(evaluate(draws.inputs(draw, time), draw, time))),
        ),
      };
    } catch (error) {
      if (!(error instanceof PlotUnavailable)) throw error;
      return { kind: "unavailable", label, reason: error.message };
    }
  };
  const mechanismPlot = (
    mechanisms: readonly DynamicsMechanismSpec[],
    target: ConstructId,
    label = `Drift into ${name(target)}`,
  ): FlowPlot => {
    if (mechanisms.length === 0) {
      return { kind: "unavailable", label, reason: "No mechanism assigned." };
    }
    return plot(
      label,
      `Model law ${expressionStates(mechanisms.map((mechanism) => mechanism.expression)).some((id) => !model.constructs[id]?.distribution) ? "at the initial state" : "at the saved state coordinates"}; contribution in ${name(target)} units per day.`,
      (inputs) =>
        mechanisms.reduce(
          (sum, mechanism) =>
            sum +
            (mechanism.kind === "potential"
              ? potentialDrift(mechanism.expression, inputs, target)
              : scalarValue(evaluateExpression(mechanism.expression, inputs))),
          0,
        ),
    );
  };
  const view: ModelFlowPlots = {
    constructs: new Map(
      constructs.map((construct) => [
        construct.id,
        plot(
          `${name(construct.id)}: ${construct.distribution ? "state law" : "initial state law"}`,
          construct.distribution
            ? "State draws from the saved model law."
            : "Initial state distribution; later trajectories come from simulate.",
          (inputs) => inputs.state(construct.id),
          { times: construct.distribution ? draws.times : draws.times.slice(0, 1) },
        ),
      ]),
    ),
    intrinsic: new Map(
      constructs.map((construct) => [
        construct.id,
        mechanismPlot(construct.dynamics, construct.id),
      ]),
    ),
    edges: new Map(
      modelEdges(model).map((edge) => [edge.id, mechanismPlot(edge.mechanisms, edge.effect.id)]),
    ),
    mechanisms: new Map(
      [
        ...constructs.map((construct) => ({ target: construct.id, owned: construct.dynamics })),
        ...modelEdges(model).map((edge) => ({ target: edge.effect.id, owned: edge.mechanisms })),
      ].flatMap(({ target, owned }) =>
        owned.map((mechanism) => [
          mechanism.id,
          mechanismPlot([mechanism], target, `${mechanismName(mechanism.id)} into ${name(target)}`),
        ]),
      ),
    ),
    indicators: new Map(
      constructs.flatMap((construct) =>
        construct.indicators.map((indicator) => {
          const likelihood = indicator.likelihood;
          const label = `${humanize(indicator.observation.name)}: local reading law`;
          const value: FlowPlot = likelihood
            ? plot(
                label,
                `${readingTime}Conditional reading distribution${likelihood.standardized ? " in standardized units" : " in recorded units"}. Windowed readings come from saved simulations.`,
                (_inputs, draw, time) =>
                  measurementDraw(
                    likelihood.law,
                    draws.inputs(draw, time, indicator),
                    displayProbability(`reading:${indicator.observation.id}:${time}`, draw),
                  ),
                {
                  // A local measurement distribution has one state input. It is not a new predictive history.
                  times: draws.times.slice(0, 1),
                  levels:
                    "levels" in indicator.observation.measurement
                      ? indicator.observation.measurement.levels
                      : null,
                },
              )
            : { kind: "unavailable", label, reason: "No observation law assigned." };
          return [indicator.observation.id, value] as const;
        }),
      ),
    ),
  };
  cache.set(model, view);
  return view;
}
