import { presentEntries } from "@/lib/model-accessors";
import {
  readNumericalArray,
  type NumPyroValue,
  type NumPyroDistribution,
} from "@grassmoth/api-types";

const isArray: (value: NumPyroValue) => value is readonly NumPyroValue[] = Array.isArray;

/** Render native constructor arguments; probability calculations stay in Python. */
export function distributionArgumentText(value: NumPyroValue): string {
  if (typeof value === "number") return String(Number(value.toPrecision(4)));
  if (isArray(value)) return `[${value.map(distributionArgumentText).join(", ")}]`;
  if (value === null || typeof value !== "object") return String(value);
  if ("npy" in value && value.npy instanceof Uint8Array) {
    const array = readNumericalArray({ npy: value.npy });
    return array.shape.length === 0 ? String(array.values[0]) : `array[${array.shape.join(" × ")}]`;
  }
  if ("npy" in value) throw new Error("Invalid numerical distribution argument");
  if (value.array !== undefined) return distributionArgumentText(value.array);
  if (value.tuple !== undefined) return distributionArgumentText(value.tuple);
  if ("float" in value) return String(value.float);
  const constructorName = value.distribution ?? value.transform ?? value.constraint;
  const params = value.params;
  if (
    typeof constructorName === "string" &&
    params &&
    typeof params === "object" &&
    !isArray(params) &&
    !("npy" in params)
  ) {
    return `${constructorName}(${presentEntries(params)
      .filter(([name]) => name !== "validate_args")
      .map(([name, argument]) => `${name}=${distributionArgumentText(argument)}`)
      .join(", ")})`;
  }
  return `{${presentEntries(value)
    .map(([name, argument]) => `${name}: ${distributionArgumentText(argument)}`)
    .join(", ")}}`;
}

export function distributionText(prior: NumPyroDistribution): string {
  return `${prior.distribution}(${presentEntries(prior.params)
    .filter(([name]) => name !== "validate_args")
    .map(([name, value]) => `${name}=${distributionArgumentText(value)}`)
    .join(", ")})`;
}
