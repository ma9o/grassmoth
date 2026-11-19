import type {
  ActionId,
  ConstructId,
  ConstructSpec,
  DynamicalModelSpec,
  DynamicsMechanismSpec,
  Expression,
  IndicatorSpec,
  ModelSnapshot,
} from "@grassmoth/api-types";
import {
  type CoefficientUse,
  coefficientUses,
  modelConstructs,
  modelEdges,
  presentEntries,
} from "@/lib/model-accessors";
import {
  type ExpressionInputs,
  type ExpressionValue,
  finiteValue,
  mechanismDrift,
  PlotUnavailable,
} from "./expression-evaluation";
import { expressionStates, type FlowPlot, type ModelFlowPlots, modelFlowPlots } from "./flow-plots";
import { type ModelDraws, modelDraws } from "./model-draws";
import { type EntitySelection, humanize, mechanismName } from "./selection";
import { readingArguments, readingLawPlot } from "./spec-reading-law";
import {
  type LawPoint,
  type PlotResult,
  plotGrid,
  projectPlot,
  type Range,
  type SpecLawPlot,
  stateLawPlot,
} from "./spec-state-law";

export interface ResponseAxis {
  readonly id: ConstructId;
  readonly label: string;
  readonly domain: Range;
  readonly law: SpecLawPlot | null;
  readonly given: readonly number[];
}

export interface ResponseSlice {
  readonly input: ConstructId | null;
  readonly plot: PlotResult<{
    readonly axis: ResponseAxis | null;
    readonly points: readonly LawPoint[];
  }>;
}

interface GlyphOwner {
  readonly selection: EntitySelection;
  readonly construct: ConstructId;
  readonly label: string;
  readonly uses: readonly CoefficientUse[];
  readonly output: FlowPlot | undefined;
}

export type SpecGlyph = GlyphOwner &
  (
    | { readonly kind: "distribution"; readonly law: SpecLawPlot }
    | {
        readonly kind: "response";
        readonly slices: readonly ResponseSlice[];
        readonly units: string;
      }
  );

export function glyphKey(selection: EntitySelection): string {
  return "glyph" in selection ? `${selection.id}/${selection.glyph}` : selection.id;
}

export type SpecPlots = ReadonlyMap<string, SpecGlyph>;

/** Both laws belong to the state card. Each response function has its own node. */
export function stateLawGlyphs(plots: SpecPlots, id: ConstructId) {
  return [plots.get(`${id}/initial`), plots.get(`${id}/noise`)].filter(
    (glyph) => glyph?.kind === "distribution",
  );
}

export function plotsForSelection(plots: SpecPlots, selection: EntitySelection) {
  const exact = plots.get(glyphKey(selection));
  if (exact) return [exact];
  if (selection.kind === "construct") return stateLawGlyphs(plots, selection.id);
  return [...plots.values()].filter((glyph) => glyph.selection.id === selection.id);
}

/** A sweep changes only the abscissa. Resolve each coefficient coordinate once per draw. */
function sweepInputs(inputs: ExpressionInputs): ExpressionInputs {
  const parameters = new Map<string, ExpressionValue>();
  return {
    state: inputs.state,
    parameter: (id, role) => {
      const key = `${id}/${role ?? ""}`;
      const cached = parameters.get(key);
      if (cached !== undefined) return cached;
      const value = inputs.parameter(id, role);
      parameters.set(key, value);
      return value;
    },
  };
}

/** A median-parameter reference and the outside of all the aligned display curves. */
function responsePoints(
  axis: ResponseAxis | null,
  reference: ExpressionInputs,
  draws: readonly ExpressionInputs[],
  evaluate: (inputs: ExpressionInputs) => number,
): LawPoint[] {
  return (axis ? plotGrid(axis.domain, 64) : [0]).map((x) => {
    const at = (inputs: ExpressionInputs) =>
      finiteValue(
        evaluate(
          axis
            ? {
                parameter: inputs.parameter,
                state: (id) => (id === axis.id ? x : inputs.state(id)),
              }
            : inputs,
        ),
      );
    const value = at(reference);
    const heights = draws.length > 0 ? draws.map((inputs) => at(inputs)) : [value];
    return { x, value, min: Math.min(...heights), max: Math.max(...heights) };
  });
}

function inputAxis(
  id: ConstructId,
  model: DynamicalModelSpec,
  initial: ReadonlyMap<ConstructId, SpecLawPlot>,
  flows: ModelFlowPlots,
): ResponseAxis {
  const label = humanize(model.constructs[id]?.name ?? id);
  const law = initial.get(id);
  if (law) {
    if (law.shape.kind === "unavailable") throw new PlotUnavailable(law.shape.reason);
    return { id, label, domain: law.shape.value.domain, law, given: [] };
  }
  const schedule = flows.constructs.get(id);
  if (schedule?.kind !== "draws") {
    throw new PlotUnavailable(schedule?.reason ?? "No schedule assigned.");
  }
  const given = flows.inputs.get(id);
  let low = Infinity;
  let high = -Infinity;
  for (const row of schedule.rows) {
    for (const value of row) {
      low = Math.min(low, value);
      high = Math.max(high, value);
    }
  }
  return {
    id,
    label,
    domain: [low, high],
    law: null,
    given: given?.kind === "draws" ? [...new Set(given.rows.flat())] : [],
  };
}

function readingUnits(indicator: IndicatorSpec): string {
  if ("levels" in indicator.observation.measurement) return "Expected level index";
  return indicator.likelihood?.standardized
    ? "Expected reading (standardized)"
    : "Expected reading";
}
function stateGlyphs(model: DynamicalModelSpec, draws: ModelDraws, flows: ModelFlowPlots) {
  const states = new Map<string, SpecGlyph>();
  const initial = new Map<ConstructId, SpecLawPlot>();
  for (const construct of modelConstructs(model).filter((c) => c.role === "endogenous")) {
    const kinds =
      construct.temporal_status === "time_varying"
        ? (["initial", "noise"] as const)
        : (["initial"] as const);
    for (const kind of kinds) {
      const law = stateLawPlot(construct, kind, model, draws);
      if (kind === "initial") initial.set(construct.id, law);
      states.set(`${construct.id}/${kind}`, {
        selection: { kind: "construct", id: construct.id },
        construct: construct.id,
        label: law.label,
        kind: "distribution",
        law,
        uses: law.uses,
        output: kind === "initial" ? flows.constructs.get(construct.id) : undefined,
      });
    }
  }

  return { states, initial };
}

const cache = new WeakMap<DynamicalModelSpec, SpecPlots>();

/** Conditional laws and response curves for an authored model; all glyphs use the same draw identities. */
export function modelSpecPlots(model: DynamicalModelSpec): SpecPlots {
  const cached = cache.get(model);
  if (cached) return cached;
  const flows = modelFlowPlots(model);
  const draws = modelDraws(model);
  const constructs = modelConstructs(model);
  const { states: glyphs, initial } = stateGlyphs(model, draws, flows);
  const add = (glyph: SpecGlyph) => glyphs.set(glyphKey(glyph.selection), glyph);
  const slices = (
    expressions: readonly Expression[],
    evaluate: (inputs: ExpressionInputs) => number,
    indicator?: IndicatorSpec,
  ): ResponseSlice[] => {
    const ids = expressionStates(expressions);
    const variable = coefficientUses(expressions).length > 0 || ids.length > 1;
    const reference = sweepInputs(draws.inputs("median", draws.times[0] ?? 0, indicator));
    const inputs = Array.from({ length: variable ? draws.count : 0 }, (_, i) =>
      sweepInputs(draws.inputs(i, draws.times[0] ?? 0, indicator)),
    );
    return (ids.length > 0 ? ids : [null]).map((input) => ({
      input,
      plot: projectPlot(() => {
        const axis = input === null ? null : inputAxis(input, model, initial, flows);
        return { axis, points: responsePoints(axis, reference, inputs, evaluate) };
      }),
    }));
  };
  const term = (
    mechanism: DynamicsMechanismSpec,
    target: ConstructId,
    owner: Extract<EntitySelection, { kind: "construct" | "edge" }>,
  ) => {
    const expressions = [mechanism.expression];
    add({
      kind: "response",
      selection: { ...owner, glyph: mechanism.id },
      construct: target,
      label: mechanismName(mechanism.id),
      units: `${humanize(model.constructs[target]?.name ?? target)} per day`,
      uses: coefficientUses(expressions),
      output: flows.mechanisms.get(mechanism.id),
      slices: slices(expressions, (inputs) => mechanismDrift(mechanism, inputs, target)),
    });
  };
  const reading = (indicator: IndicatorSpec, construct: ConstructSpec) => {
    const { observation } = indicator;
    const args = readingArguments(indicator);
    const law = readingLawPlot(indicator, model, draws, args);
    add({
      kind: "distribution",
      selection: { kind: "indicator", id: observation.id },
      construct: construct.id,
      label: humanize(observation.name),
      law,
      uses: law.uses,
      output: flows.indicators.get(observation.id),
    });
    add({
      kind: "response",
      selection: { kind: "indicator", id: observation.id, glyph: "response" },
      construct: construct.id,
      label: humanize(observation.name),
      uses: coefficientUses(args.response),
      output: flows.indicators.get(observation.id),
      units: readingUnits(indicator),
      slices: slices(args.response, args.expected, indicator),
    });
  };
  for (const construct of constructs) {
    for (const mechanism of construct.dynamics) {
      term(mechanism, construct.id, { kind: "construct", id: construct.id });
    }
    if (construct.role === "endogenous") {
      for (const indicator of construct.indicators) reading(indicator, construct);
    }
  }
  for (const edge of modelEdges(model)) {
    for (const mechanism of edge.mechanisms) {
      term(mechanism, edge.effect.id, { kind: "edge", id: edge.id });
    }
  }
  cache.set(model, glyphs);
  return glyphs;
}

/** A fitted model may also be carried into a new edit as a retained joint law. */
export function viewedSpecPlots(snapshot: ModelSnapshot, step: ActionId | null): SpecPlots | null {
  const model = snapshot.dynamical_model_spec;
  if (!model || step !== "edit_model" || snapshot.fit || snapshot.simulation) return null;
  if (
    presentEntries(model.constructs).some(
      ([, c]) => c.role === "endogenous" && c.distribution !== null,
    ) ||
    presentEntries(model.law_layouts).some(
      ([id]) => model.distributions[id]?.distribution === "MixtureSameFamily",
    )
  ) {
    return null;
  }
  return modelSpecPlots(model);
}
