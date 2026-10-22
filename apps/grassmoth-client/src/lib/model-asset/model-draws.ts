import type {
  ConstructId,
  DistributionId,
  DynamicalModelSpec,
  Expression,
  IndicatorSpec,
  JointLawLayout,
  NumPyroDistribution,
  ParameterId,
  ParameterTransformSpec,
} from "@grassmoth/api-types";
import { jStat } from "jstat";
import { modelConstructs, presentEntries } from "@/lib/model-accessors";
import { assertNever } from "@/lib/assert-never";
import {
  type ExpressionInputs,
  evaluateExpression,
  finiteValue,
  PlotUnavailable,
  scalarValue,
} from "./expression-evaluation";
import {
  atomIndex,
  displayProbability,
  jointAtoms,
  scalarQuantile,
  type JointAtoms,
} from "./law-draws";
import { nativeObject, nativeScalar } from "./scalar-law";

export interface ModelDraws {
  readonly count: number;
  readonly times: readonly number[];
  readonly timeOrigin: string;
  readonly retained: boolean;
  readonly inputs: (draw: number, time: number, indicator?: IndicatorSpec) => ExpressionInputs;
}

function intervalDays(transform: ParameterTransformSpec, model: DynamicalModelSpec): number {
  if (!("interval_days" in transform)) throw new Error("This transform has no interval.");
  if (transform.interval_days !== "model_clock") return transform.interval_days;
  const clock = model.measurement_clock;
  if (!clock) throw new PlotUnavailable("The model clock has not been assigned.");
  const units: Readonly<Record<string, number>> = {
    s: 1 / 86400,
    m: 1 / 1440,
    h: 1 / 24,
    d: 1,
    w: 7,
  };
  const unit = units[clock.slice(-1)];
  if (unit === undefined) throw new Error("Invalid saved model clock.");
  return Number(clock.slice(0, -1)) * unit;
}

/** The declared correlation transform truncates its authored support, rather than clipping draws. */
function correlationLaw(law: NumPyroDistribution): NumPyroDistribution {
  if (law.distribution === "Normal")
    return {
      distribution: "TwoSidedTruncatedDistribution",
      params: { base_dist: law, low: -1, high: 1 },
    };
  if (law.distribution === "Uniform" || law.distribution === "TwoSidedTruncatedDistribution")
    return {
      ...law,
      params: {
        ...law.params,
        low: Math.max(-1, nativeScalar(law.params.low)),
        high: Math.min(1, nativeScalar(law.params.high)),
      },
    };
  return law;
}

/**
 * Display draws of the saved law. A joint atom is selected once per draw and reused for
 * every parameter and state it owns. Initial Gaussian draws are local input laws, never
 * trajectories; later states must come from a saved trajectory law or simulation.
 */
export function modelDraws(model: DynamicalModelSpec): ModelDraws {
  const constructs = modelConstructs(model);
  const byId = new Map(constructs.map((construct) => [construct.id, construct]));
  const atoms = new Map<DistributionId, JointAtoms>();
  const layoutEntries = presentEntries(model.law_layouts);
  const parameterColumns = new Map(
    layoutEntries.map(([identity, layout]) => {
      let column = 0;
      return [
        identity,
        new Map(
          layout.parameters.map(([id, elements]) => {
            const coordinate = { column, width: elements.length };
            column += elements.length;
            return [id, coordinate] as const;
          }),
        ),
      ] as const;
    }),
  );
  const empirical = layoutEntries.flatMap(([id]) => {
    const law = model.distributions[id];
    if (
      law?.distribution !== "MixtureSameFamily" ||
      nativeObject(law.params.component_distribution)?.distribution !== "Delta"
    )
      return [];
    const value = jointAtoms(law);
    atoms.set(id, value);
    return [value.count];
  });
  const count = empirical.length ? Math.max(...empirical) : 256;
  const trajectory =
    layoutEntries.find(([, layout]) =>
      layout.constructs.some((id) => byId.get(id)?.role === "endogenous"),
    )?.[1] ?? layoutEntries.find(([, layout]) => layout.constructs.length > 0)?.[1];
  const initialOnly = constructs.some(
    (construct) => construct.role === "endogenous" && !construct.distribution,
  );
  const times = initialOnly
    ? [trajectory?.time_points[0] ?? 0]
    : trajectory?.time_points.length
      ? trajectory.time_points
      : [0];
  const timeOrigin = trajectory?.time_origin ?? "relative";
  const jointRows = new Map<DistributionId, readonly number[]>();
  const readJoint = (id: DistributionId, draw: number, column: number): number => {
    let lawAtoms = atoms.get(id);
    if (!lawAtoms) {
      const law = model.distributions[id];
      if (!law) throw new Error("A saved law reference is missing.");
      lawAtoms = jointAtoms(law);
      atoms.set(id, lawAtoms);
    }
    let rows = jointRows.get(id);
    if (!rows) {
      const uniform =
        lawAtoms.count === count &&
        lawAtoms.probabilities.every((p) => Math.abs(p - 1 / count) < 1e-7);
      const source = lawAtoms;
      rows = Array.from({ length: count }, (_, index) =>
        uniform ? index : atomIndex(source, id, index, count),
      );
      jointRows.set(id, rows);
    }
    const row = rows[draw];
    if (row === undefined) throw new Error("Missing saved draw index.");
    const value = lawAtoms.values[row * lawAtoms.width + column];
    if (column < 0 || column >= lawAtoms.width || value === undefined)
      throw new Error("Missing saved joint coordinate.");
    return value;
  };
  const quantiles = new Map<ParameterId, (p: number) => number>();
  const parameter = (id: ParameterId, draw: number, elementKey: string = id): number => {
    const specification = model.parameters[id];
    if (!specification) throw new Error("An expression references a missing parameter.");
    const identity = specification.distribution;
    if (!identity) throw new PlotUnavailable(`${specification.name} has no law yet.`);
    const layout = model.law_layouts[identity];
    let value: number;
    if (layout) {
      const coordinate = parameterColumns.get(identity)?.get(id);
      if (!coordinate) throw new Error("A parameter is absent from its saved law layout.");
      if (coordinate.width !== 1)
        throw new PlotUnavailable(
          "This plot needs named category coordinates for a vector parameter.",
        );
      value = readJoint(identity, draw, coordinate.column);
    } else {
      let quantile = quantiles.get(id);
      if (!quantile) {
        const law = model.distributions[identity];
        if (!law) throw new Error("A saved parameter law is missing.");
        quantile = scalarQuantile(
          specification.transform.kind === "initial_state_correlation" ? correlationLaw(law) : law,
        );
        quantiles.set(id, quantile);
      }
      value = quantile(displayProbability(elementKey, draw));
    }
    switch (specification.transform.kind) {
      case "identity":
      case "initial_state_correlation":
        return finiteValue(value);
      case "dt_persistence_to_ct_decay":
        return finiteValue(-Math.log(value) / intervalDays(specification.transform, model));
      case "dt_effect_to_ct_rate":
        return finiteValue(value / intervalDays(specification.transform, model));
      default:
        return assertNever(specification.transform);
    }
  };

  const initialCache = new Map<string, number>();
  const coefficient = (expression: Expression, draw: number) =>
    scalarValue(
      evaluateExpression(expression, {
        parameter: (id) => parameter(id, draw),
        state: () => {
          throw new Error("An initial coefficient cannot read a state.");
        },
      }),
    );

  const initial = (id: ConstructId, draw: number): number => {
    const cached = initialCache.get(`${id}:${draw}`);
    if (cached !== undefined) return cached;
    const related = new Set<ConstructId>([id]);
    // Initial correlations define the connected block whose Gaussian input draw must stay joint.
    for (const member of related) {
      for (const construct of constructs)
        for (const operand of construct.coefficients) {
          if (operand.role !== "initial_correlation") continue;
          const other = operand.construct_ids[0];
          if (other !== undefined && (construct.id === member || other === member)) {
            related.add(construct.id);
            related.add(other);
          }
        }
    }
    const members = constructs.filter((construct) => related.has(construct.id));
    const lower: number[][] = [];
    const values: Array<readonly [string, number]> = [];
    const noise = members.map((construct) =>
      jStat.normal.inv(displayProbability(`initial:${construct.id}`, draw), 0, 1),
    );
    members.forEach((construct, row) => {
      if (construct.distribution)
        throw new PlotUnavailable(
          "Initial and retained state laws cannot be combined into a new correlated block.",
        );
      const mean = construct.coefficients.find((operand) => operand.role === "initial_mean");
      const scale = construct.coefficients.find((operand) => operand.role === "initial_scale");
      if (!mean || !scale)
        throw new PlotUnavailable(`${construct.name} has no complete initial state law yet.`);
      const current: number[] = [];
      for (let column = 0; column <= row; column++) {
        const other = members[column];
        if (!other) throw new Error("Missing initial state coordinate.");
        const correlation =
          construct.coefficients.find(
            (operand) =>
              operand.role === "initial_correlation" && operand.construct_ids[0] === other.id,
          ) ??
          other.coefficients.find(
            (operand) =>
              operand.role === "initial_correlation" && operand.construct_ids[0] === construct.id,
          );
        let covariance = row === column ? 1 : correlation ? coefficient(correlation, draw) : 0;
        for (let k = 0; k < column; k++)
          covariance -=
            (current[k] ?? 0) * (row === column ? (current[k] ?? 0) : (lower[column]?.[k] ?? 0));
        if (row === column) {
          if (covariance <= 0)
            throw new PlotUnavailable(
              "These initial correlation draws do not define a positive definite input law.",
            );
          current.push(Math.sqrt(covariance));
        } else {
          const diagonal = lower[column]?.[column];
          if (diagonal === undefined) throw new Error("Missing initial covariance coordinate.");
          current.push(covariance / diagonal);
        }
      }
      lower.push(current);
      const normal = current.reduce((sum, value, column) => sum + value * (noise[column] ?? 0), 0);
      values.push([
        `${construct.id}:${draw}`,
        finiteValue(coefficient(mean, draw) + coefficient(scale, draw) * normal),
      ]);
    });
    values.forEach(([key, value]) => initialCache.set(key, value));
    const value = initialCache.get(`${id}:${draw}`);
    if (value === undefined) throw new Error("Missing initial state draw.");
    return value;
  };

  const stateColumns = new Map<string, number>();
  const state = (id: ConstructId, draw: number, time: number): number => {
    const construct = byId.get(id);
    if (!construct) throw new Error("An expression references a missing construct.");
    if (!construct.distribution) {
      if (time !== times[0])
        throw new PlotUnavailable(
          "Generate trajectories with simulate to view this state beyond its initial law.",
        );
      return initial(id, draw);
    }
    const key = `${id}:${time}`;
    const cachedColumn = stateColumns.get(key);
    if (cachedColumn !== undefined) return readJoint(construct.distribution, draw, cachedColumn);
    const layout: JointLawLayout | undefined = model.law_layouts[construct.distribution];
    if (!layout) throw new Error("A trajectory law is missing its coordinates.");
    const localTime =
      time +
      (timeOrigin !== "relative" && layout.time_origin !== "relative"
        ? (Date.parse(timeOrigin) - Date.parse(layout.time_origin)) / 86400000
        : 0);
    const index =
      construct.role === "exogenous"
        ? layout.time_points.findLastIndex((point) => point <= localTime)
        : layout.time_points.indexOf(localTime);
    if (index < 0) throw new PlotUnavailable("No saved state at this time.");
    const member = layout.constructs.indexOf(id);
    if (member < 0) throw new Error("A construct is absent from its trajectory layout.");
    const offset = layout.parameters.reduce((sum, [, elements]) => sum + elements.length, 0);
    const column = offset + member * layout.time_points.length + index;
    stateColumns.set(key, column);
    return readJoint(construct.distribution, draw, column);
  };
  return {
    count,
    times,
    timeOrigin,
    retained: empirical.length > 0,
    inputs: (draw, time, indicator) => ({
      parameter: (id, role) => {
        if (
          role !== "cutpoint_gaps" &&
          role !== "category_intercepts" &&
          role !== "category_slopes"
        )
          return parameter(id, draw);
        if (!indicator)
          throw new PlotUnavailable(
            "Vector coefficients require their indicator's category coordinates.",
          );
        if (role === "category_slopes")
          throw new PlotUnavailable(
            "Category slope plots require the saved anchoring coordinates.",
          );
        const levels =
          role === "cutpoint_gaps"
            ? indicator.observation.ordinal_levels
            : indicator.observation.categorical_levels;
        if (!levels) throw new Error("A discrete observation law is missing its levels.");
        const width = levels.length - (role === "cutpoint_gaps" ? 2 : 1);
        return Array.from({ length: width }, (_, element) =>
          parameter(id, draw, `${id}:${indicator.observation.id}:${element}`),
        );
      },
      state: (id) => state(id, draw, time),
    }),
  };
}
