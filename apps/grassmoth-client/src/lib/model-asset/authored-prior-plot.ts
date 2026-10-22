import type { DensityCurve, NumPyroDistribution } from "@grassmoth/api-types";
import { scalarDensity } from "./scalar-law";

const plots = new WeakMap<NumPyroDistribution, DensityCurve>();

/** Plot the law's central 98%; every authoring family must produce a finite scalar PDF. */
export function authoredPriorPlot(law: NumPyroDistribution): DensityCurve {
  const cached = plots.get(law);
  if (cached) return cached;
  const distribution = scalarDensity(law);
  const low = distribution.quantile(0.01);
  const high = distribution.quantile(0.99);
  const x = Array.from({ length: 129 }, (_, index) => low + ((high - low) * index) / 128);
  const density = x.map((value) => distribution.pdf(value));
  if (
    !(low < high) ||
    !x.every(Number.isFinite) ||
    !density.every((value) => Number.isFinite(value) && value >= 0)
  ) {
    throw new Error(`Cannot plot a finite scalar density for the ${law.distribution} prior.`);
  }
  const result = { x, density };
  plots.set(law, result);
  return result;
}
