import type { NumPyroDistribution, NumPyroValue } from "@grassmoth/api-types";
import { readNumericalArray } from "@grassmoth/api-types";
import { PlotUnavailable } from "./expression-evaluation";
import { nativeObject, nativeScalar, scalarDensity } from "./scalar-law";

/** A stable, open-unit display variate. Independent keys never share an RNG cursor. */
export function displayProbability(key: string, draw: number): number {
  let hash = 2166136261;
  for (const character of `${key}:${draw}`)
    hash = Math.imul(hash ^ character.charCodeAt(0), 16777619);
  hash = Math.imul(hash ^ (hash >>> 16), 0x21f0aaad);
  hash = Math.imul(hash ^ (hash >>> 15), 0x735a2d97);
  return ((hash ^ (hash >>> 15)) >>> 0) / 4294967296 + 0.5 / 4294967296;
}

/** Native constructor arrays are already decoded by the API client. */
export function nativeArray(value: NumPyroValue | undefined): {
  readonly shape: readonly number[];
  readonly values: readonly number[];
} {
  if (
    value !== null &&
    typeof value === "object" &&
    "npy" in value &&
    value.npy instanceof Uint8Array
  ) {
    const array = readNumericalArray({ npy: value.npy });
    return { shape: array.shape, values: Array.from(array.values, Number) };
  }
  if (Array.isArray(value)) {
    const children = value.map((child: NumPyroValue) => nativeArray(child));
    return {
      shape: [value.length, ...(children[0]?.shape ?? [])],
      values: children.flatMap((child) => child.values),
    };
  }
  const object = nativeObject(value);
  if (object && "array" in object) return nativeArray(object.array);
  return { shape: [], values: [nativeScalar(value)] };
}

export interface JointAtoms {
  readonly count: number;
  readonly width: number;
  readonly values: readonly number[];
  readonly probabilities: readonly number[];
}

/** Retained posterior atoms stay joint; their columns are never sampled independently. */
export function jointAtoms(law: NumPyroDistribution): JointAtoms {
  if (law.distribution === "Delta") {
    const array = nativeArray(law.params.v);
    return { count: 1, width: array.values.length, values: array.values, probabilities: [1] };
  }
  const component = nativeObject(law.params.component_distribution);
  const mixing = nativeObject(law.params.mixing_distribution);
  if (
    law.distribution !== "MixtureSameFamily" ||
    component?.distribution !== "Delta" ||
    mixing?.distribution !== "CategoricalProbs"
  )
    throw new PlotUnavailable(
      `Display sampling of the ${law.distribution} joint law is not supported.`,
    );
  const array = nativeArray(nativeObject(component.params)?.v);
  const probabilities = nativeArray(nativeObject(mixing.params)?.probs).values;
  const [count, width] = array.shape;
  if (
    count === undefined ||
    width === undefined ||
    array.shape.length !== 2 ||
    probabilities.length !== count
  )
    throw new Error("Saved joint atoms do not match their mixture coordinates.");
  return { count, width, values: array.values, probabilities };
}

/** Keep every equally weighted retained draw in order when using its complete draw axis. */
export function atomIndex(atoms: JointAtoms, key: string, draw: number, count: number): number {
  if (atoms.count === count && atoms.probabilities.every((p) => Math.abs(p - 1 / count) < 1e-7))
    return draw;
  const probability = displayProbability(key, draw);
  let cumulative = 0;
  for (const [index, mass] of atoms.probabilities.entries()) {
    cumulative += mass;
    if (probability < cumulative) return index;
  }
  throw new PlotUnavailable("The saved mixture probabilities do not sum to one.");
}

/** The same supported scalar laws serve both density plots and display draws. */
export function scalarQuantile(law: NumPyroDistribution): (probability: number) => number {
  if (law.distribution === "Delta") {
    const value = nativeScalar(law.params.v);
    return () => value;
  }
  switch (law.distribution) {
    case "Normal":
    case "HalfNormal":
    case "Beta":
    case "Uniform":
    case "Gamma":
    case "LogNormal":
    case "Exponential":
    case "LeftTruncatedDistribution":
    case "RightTruncatedDistribution":
    case "TwoSidedTruncatedDistribution":
      return scalarDensity(law).quantile;
    default:
      throw new PlotUnavailable(
        `Display sampling of the ${law.distribution} scalar law is not supported.`,
      );
  }
}
