import type {
  CoefficientExpression,
  ConstructSpec,
  DynamicalModelSpec,
} from "@grassmoth/api-types";
import { jStat } from "jstat";
import { type CoefficientUse, coefficientUses } from "@/lib/model-accessors";
import { formatSignificant } from "@/lib/utils/format";
import {
  evaluateExpression,
  finiteValue,
  PlotUnavailable,
  scalarValue,
} from "./expression-evaluation";
import { durationDays, type ModelDraws } from "./model-draws";

export type Range = readonly [number, number];

export interface LawPoint {
  readonly x: number;
  readonly value: number;
  readonly min: number;
  readonly max: number;
}

export interface StateShape {
  readonly domain: Range;
  readonly curve: readonly LawPoint[];
  /** A mass plot uses rectangles spanning this many integer outcomes. */
  readonly binWidth?: number;
  /** Zero-scale conditional laws are point masses, including an uncertain initial location. */
  readonly atoms: Range | null;
  readonly referenceAtom: number | null;
}

export type PlotResult<T> =
  | { readonly kind: "plot"; readonly value: T }
  | { readonly kind: "unavailable"; readonly reason: string };

export function projectPlot<T>(evaluate: () => T): PlotResult<T> {
  try {
    return { kind: "plot", value: evaluate() };
  } catch (error) {
    if (!(error instanceof PlotUnavailable)) throw error;
    return { kind: "unavailable", reason: error.message };
  }
}

export function plotGrid([low, high]: Range, intervals: number): number[] {
  if (low === high) return [low];
  return Array.from({ length: intervals + 1 }, (_, i) => low + ((high - low) * i) / intervals);
}

export interface SpecLawPlot {
  readonly label: string;
  readonly family: string;
  readonly uses: readonly CoefficientUse[];
  readonly shape: PlotResult<StateShape>;
}

interface Bell {
  readonly loc: number;
  readonly scale: number;
  readonly df: number | null;
}

const normalEnd = jStat.normal.inv(0.99, 0, 1);

function bellRange({ loc, scale, df }: Bell): Range {
  const end = scale * (df === null ? normalEnd : jStat.studentt.inv(0.99, df));
  return [finiteValue(loc - end), finiteValue(loc + end)];
}

/** Each conditional density is in units of its own peak, so the width remains legible. */
function heightAt({ loc, scale, df }: Bell, x: number): number {
  if (scale === 0) return 0;
  const z2 = ((x - loc) / scale) ** 2;
  return df === null ? Math.exp(-0.5 * z2) : (1 + z2 / df) ** (-(df + 1) / 2);
}

function stateShape(reference: Bell, samples: readonly Bell[]): StateShape {
  const bells = [reference, ...samples];
  const ranges = bells.map((bell) => bellRange(bell));
  const domain: Range = [
    Math.min(...ranges.map(([low]) => low)),
    Math.max(...ranges.map(([, high]) => high)),
  ];
  const atoms = bells.filter((bell) => bell.scale === 0).map((bell) => bell.loc);
  const continuous = bells.filter((bell) => bell.scale > 0);
  // Include the actual centres and the reference bell's grid even when one bell is very narrow.
  const xs =
    continuous.length === 0
      ? []
      : [
          ...new Set([
            ...plotGrid(domain, 96),
            ...plotGrid(bellRange(reference), 64),
            ...continuous.map((bell) => bell.loc),
          ]),
        ].toSorted((a, b) => a - b);
  return {
    domain,
    curve: xs.map((x) => {
      const heights = continuous.map((bell) => heightAt(bell, x));
      return {
        x,
        value: heightAt(reference, x),
        min: Math.min(...heights),
        max: Math.max(...heights),
      };
    }),
    atoms: atoms.length > 0 ? [Math.min(...atoms), Math.max(...atoms)] : null,
    referenceAtom: reference.scale === 0 ? reference.loc : null,
  };
}

const operandLabel = (operand: CoefficientExpression | undefined, symbol: string) =>
  typeof operand?.value === "number" ? formatSignificant(operand.value) : symbol;

export function stateLawPlot(
  construct: ConstructSpec,
  kind: "initial" | "noise",
  model: DynamicalModelSpec,
  draws: ModelDraws,
): SpecLawPlot {
  const initial = kind === "initial";
  const mean = initial ? construct.coefficients.find((c) => c.role === "initial_mean") : undefined;
  const scale = construct.coefficients.find(
    (c) => c.role === (initial ? "initial_scale" : "diffusion_scale"),
  );
  const student = !initial && construct.innovation_family === "student_t";
  const df = student
    ? construct.coefficients.find((c) => c.role === "process_degrees_of_freedom")
    : undefined;
  const uses = coefficientUses([mean, scale, df].filter((c) => c !== undefined));
  const step = model.time.step;
  const stepScale = initial ? 1 : Math.sqrt(durationDays(step));
  const widthLabel =
    typeof scale?.value === "number" ? formatSignificant(scale.value * stepScale) : "σ";
  const args = [initial ? operandLabel(mean, "μ") : "0", widthLabel];
  if (student) args.unshift(operandLabel(df, "ν"));
  return {
    label: initial ? "Initial law" : `Step noise · ${step}`,
    family: `${student ? "StudentT" : "Normal"}(${args.join(", ")})`,
    uses,
    shape: projectPlot(() => {
      if (!scale || (initial && !mean) || (student && !df)) {
        throw new PlotUnavailable("Incomplete state law.");
      }
      const bell = (draw: number | "median") => {
        const inputs = draws.inputs(draw, draws.times[0] ?? 0);
        const read = (operand: CoefficientExpression) =>
          finiteValue(scalarValue(evaluateExpression(operand, inputs)));
        const width = read(scale) * stepScale;
        const degrees = df ? read(df) : null;
        if (width < 0 || (degrees !== null && degrees <= 0)) {
          throw new PlotUnavailable("Some inputs lie outside the state law's domain.");
        }
        return { loc: mean ? read(mean) : 0, scale: width, df: degrees };
      };
      return stateShape(
        bell("median"),
        Array.from({ length: uses.length > 0 ? draws.count : 0 }, (_, i) => bell(i)),
      );
    }),
  };
}
