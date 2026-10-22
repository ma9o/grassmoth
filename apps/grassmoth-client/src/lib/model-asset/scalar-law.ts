import type { NumPyroValue, NumPyroDistribution } from "@grassmoth/api-types";
import { readNumericalArray } from "@grassmoth/api-types";
import { jStat } from "jstat";
import { PlotUnavailable } from "./expression-evaluation";

export interface ScalarDensity {
  pdf(value: number): number;
  quantile(probability: number): number;
}

const isArray: (value: NumPyroValue) => value is readonly NumPyroValue[] = Array.isArray;

export function nativeObject(
  value: NumPyroValue | undefined,
): NumPyroDistribution["params"] | null {
  return value !== undefined &&
    value !== null &&
    typeof value === "object" &&
    !isArray(value) &&
    !("npy" in value)
    ? value
    : null;
}

/** Native scalars may be JSON numbers or zero-dimensional array constructors. */
export function nativeScalar(value: NumPyroValue | undefined): number {
  if (typeof value === "number") return value;
  if (
    value !== null &&
    typeof value === "object" &&
    "npy" in value &&
    value.npy instanceof Uint8Array
  ) {
    const array = readNumericalArray({ npy: value.npy });
    if (array.shape.length !== 0)
      throw new PlotUnavailable("This display sampler requires scalar constructor arguments.");
    return Number(array.values[0]);
  }
  const encoded = nativeObject(value);
  if (encoded && "array" in encoded) return nativeScalar(encoded.array);
  if (encoded?.float === "inf") return Infinity;
  if (encoded?.float === "-inf") return -Infinity;
  throw new PlotUnavailable("This display sampler requires scalar constructor arguments.");
}

function twoParameters(
  family: Pick<typeof jStat.normal, "pdf" | "inv">,
  first: number,
  second: number,
): ScalarDensity {
  return {
    pdf: (value) => family.pdf(value, first, second),
    quantile: (probability) => family.inv(probability, first, second),
  };
}

/** Normalize the declared truncation, reflecting right tails before subtracting CDFs. */
function truncatedNormal(loc: number, scale: number, low: number, high: number): ScalarDensity {
  const reflected = low > loc;
  const mean = reflected ? -loc : loc;
  const lower = reflected ? -high : low;
  const upper = reflected ? -low : high;
  const left = lower === -Infinity ? 0 : jStat.normal.cdf(lower, mean, scale);
  const right = upper === Infinity ? 1 : jStat.normal.cdf(upper, mean, scale);
  const mass = right - left;
  return {
    pdf: (value) => (value < low || value > high ? 0 : jStat.normal.pdf(value, loc, scale) / mass),
    quantile: (probability) => {
      const value = jStat.normal.inv(
        left + (reflected ? 1 - probability : probability) * mass,
        mean,
        scale,
      );
      return reflected ? -value : value;
    },
  };
}

/** Interpret the supported scalar constructor shapes without changing the saved law. */
export function scalarDensity(law: NumPyroDistribution): ScalarDensity {
  const p = law.params;
  switch (law.distribution) {
    case "Normal":
      return twoParameters(jStat.normal, nativeScalar(p.loc), nativeScalar(p.scale));
    case "HalfNormal":
      return truncatedNormal(0, nativeScalar(p.scale), 0, Infinity);
    case "Beta":
      return twoParameters(
        jStat.beta,
        nativeScalar(p.concentration1),
        nativeScalar(p.concentration0),
      );
    case "Uniform":
      return twoParameters(jStat.uniform, nativeScalar(p.low), nativeScalar(p.high));
    case "Gamma": {
      const rate = nativeScalar(p.rate);
      return twoParameters(jStat.gamma, nativeScalar(p.concentration), 1 / rate);
    }
    case "LogNormal":
      return twoParameters(jStat.lognormal, nativeScalar(p.loc), nativeScalar(p.scale));
    case "Exponential": {
      const rate = nativeScalar(p.rate);
      return {
        pdf: (value) => jStat.exponential.pdf(value, rate),
        quantile: (probability) => jStat.exponential.inv(probability, rate),
      };
    }
    case "LeftTruncatedDistribution":
    case "RightTruncatedDistribution":
    case "TwoSidedTruncatedDistribution": {
      const base = nativeObject(p.base_dist);
      const parameters = nativeObject(base?.params);
      if (base?.distribution !== "Normal" || parameters === null) {
        throw new PlotUnavailable(
          "Display evaluation of a truncated law requires a Normal base law.",
        );
      }
      return truncatedNormal(
        nativeScalar(parameters.loc),
        nativeScalar(parameters.scale),
        law.distribution === "RightTruncatedDistribution" ? -Infinity : nativeScalar(p.low),
        law.distribution === "LeftTruncatedDistribution" ? Infinity : nativeScalar(p.high),
      );
    }
    default:
      throw new PlotUnavailable(
        `Authored prior plots do not support the ${law.distribution} constructor.`,
      );
  }
}
