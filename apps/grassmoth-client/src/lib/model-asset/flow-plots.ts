import type {
  ConstructId,
  ConstructSpec,
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
  finiteValue,
  mechanismDrift,
  PlotUnavailable,
} from "./expression-evaluation";
import { displayProbability } from "./law-draws";
import { measurementDraw } from "./measurement-draws";
import { durationDays, type ModelDraws, modelDraws } from "./model-draws";
import { humanize, mechanismName } from "./selection";

const DAY_MS = 86_400_000;

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
  /** Each construct's own law. A known input reads as its authored path, on every version. */
  readonly constructs: ReadonlyMap<ConstructId, FlowPlot>;
  /** Each state as the mechanisms read it: at the times their outputs are drawn. */
  readonly inputs: ReadonlyMap<ConstructId, FlowPlot>;
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

/** The model window's final day, on a time axis with the given origin. */
function windowEnd(model: DynamicalModelSpec, origin: string): number {
  const horizon = durationDays(model.time.window.horizon);
  if (origin === "relative") return horizon;
  return (Date.parse(model.time.window.start) - Date.parse(origin)) / DAY_MS + horizon;
}

/**
 * A held path as the line it is: each value runs from its day to the next change, where the
 * line steps and the day appears twice, and the last value runs to `end`.
 */
function heldLine(
  days: readonly number[],
  values: readonly number[],
  end: number,
): { readonly days: readonly number[]; readonly values: readonly number[] } {
  const points = days.flatMap((day, index) => {
    const value = values[index];
    const previous = values[index - 1];
    if (value === undefined) return [];
    return previous === undefined
      ? [[day, value] as const]
      : [[day, previous] as const, [day, value] as const];
  });
  const last = points.at(-1);
  const line = last && end > last[0] ? [...points, [end, last[1]] as const] : points;
  return { days: line.map(([day]) => day), values: line.map(([, value]) => value) };
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
    // An output drawn at one day names that day: a known input it reads shows its whole path.
    const at = draws.times.length === 1 ? readingTime : "";
    return plot(
      label,
      `${at}Model law ${expressionStates(mechanisms.map((mechanism) => mechanism.expression)).some((id) => !model.constructs[id]?.distribution) ? "at the initial state" : "at the saved state coordinates"}; contribution in ${name(target)} units per day.`,
      (inputs) =>
        mechanisms.reduce((sum, mechanism) => sum + mechanismDrift(mechanism, inputs, target), 0),
    );
  };
  /**
   * A known input's authored path over the model window. The path is the model's own law, so it
   * reads over time on every version, while a state that still waits for a simulation shows only
   * its initial law. A fixed path is one line, not one per draw.
   */
  const knownPath = (construct: ConstructSpec): FlowPlot | null => {
    const layout = construct.distribution ? model.law_layouts[construct.distribution] : undefined;
    if (construct.role !== "exogenous" || !layout || layout.time_points.length === 0) return null;
    const shift =
      draws.timeOrigin === "relative" || layout.time_origin === "relative"
        ? 0
        : (Date.parse(layout.time_origin) - Date.parse(draws.timeOrigin)) / DAY_MS;
    const changes = layout.time_points.map((point) => point + shift);
    const sampled = plot(
      `${name(construct.id)}: state law`,
      "The authored trajectory law: each value holds until the next point, and the last to the end of the model window.",
      (inputs) => inputs.state(construct.id),
      { times: changes },
    );
    if (sampled.kind === "unavailable") return sampled;
    const end = windowEnd(model, draws.timeOrigin);
    const lines = sampled.rows.map((values) => heldLine(changes, values, end));
    const [first] = lines;
    if (!first) return null;
    const fixed = lines.every((line) =>
      line.values.every((value, index) => value === first.values[index]),
    );
    return {
      ...sampled,
      times: first.days,
      rows: (fixed ? [first] : lines).map((line) => line.values),
      retained: false,
    };
  };
  const states = constructs.map((construct) => {
    const read = plot(
      `${name(construct.id)}: ${construct.distribution ? "state law" : "initial state law"}`,
      construct.distribution
        ? "State draws from the saved model law."
        : "Initial state distribution; later trajectories come from simulate.",
      (inputs) => inputs.state(construct.id),
      { times: construct.distribution ? draws.times : draws.times.slice(0, 1) },
    );
    return { id: construct.id, read, own: knownPath(construct) ?? read };
  });
  const view: ModelFlowPlots = {
    constructs: new Map(states.map(({ id, own }) => [id, own])),
    inputs: new Map(states.map(({ id, read }) => [id, read])),
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
