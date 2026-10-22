import { scaleLinear, scaleUtc } from "d3-scale";
import { formatSignificant } from "@/lib/utils/format";

/** Display coordinates only: no smoothing, thinning, imputation or statistical reduction. */

export const DAY_MS = 86_400_000;

export type Domain = readonly [number, number];

export interface PlotBox {
  readonly left: number;
  readonly top: number;
  readonly width: number;
  readonly height: number;
}

export interface Tick {
  readonly value: number;
  readonly label: string;
}

export function extentOf(values: Iterable<number | null | undefined>): Domain | null {
  let low = Number.POSITIVE_INFINITY;
  let high = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    if (value == null || !Number.isFinite(value)) continue;
    low = Math.min(low, value);
    high = Math.max(high, value);
  }
  return low <= high ? [low, high] : null;
}

/** Give a domain room at both ends, and width when every value is equal. */
export function padDomain([low, high]: Domain, fraction = 0.05): Domain {
  if (low === high) {
    const half = low === 0 ? 0.5 : Math.abs(low) * 0.05;
    return [low - half, high + half];
  }
  const pad = (high - low) * fraction;
  return [low - pad, high + pad];
}

export function linearScale(domain: Domain, range: Domain): (value: number) => number {
  const span = domain[1] - domain[0];
  return (value) => range[0] + ((value - domain[0]) / span) * (range[1] - range[0]);
}

export function valueTicks(domain: Domain, count: number): Tick[] {
  return scaleLinear()
    .domain(domain)
    .ticks(count)
    .map((value) => ({ value, label: formatSignificant(value) }));
}

/** Calendar ticks on a pinned UTC origin, or plain day numbers without one. */
export function timeTicks(domain: Domain, origin: string | null, count: number): Tick[] {
  if (origin === null) return valueTicks(domain, count);
  const start = Date.parse(origin);
  const scale = scaleUtc().domain([start + domain[0] * DAY_MS, start + domain[1] * DAY_MS]);
  const label = scale.tickFormat(count);
  return scale
    .ticks(count)
    .map((date) => ({ value: (date.getTime() - start) / DAY_MS, label: label(date) }));
}

/** Runs of consecutive finite points; a missing value always breaks the line. */
export function finiteRuns(
  times: readonly number[],
  values: readonly (number | null)[],
): Array<Array<readonly [number, number]>> {
  const runs: Array<Array<readonly [number, number]>> = [];
  let run: Array<readonly [number, number]> = [];
  times.forEach((time, index) => {
    const value = values[index];
    if (value == null || !Number.isFinite(value)) {
      if (run.length) runs.push(run);
      run = [];
      return;
    }
    run.push([time, value]);
  });
  if (run.length) runs.push(run);
  return runs;
}

/** SVG path data for a series; canvas strokes the same data through Path2D. */
export function linePath(
  times: readonly number[],
  values: readonly (number | null)[],
  x: (time: number) => number,
  y: (value: number) => number,
): string {
  return finiteRuns(times, values)
    .map((run) =>
      run
        .map(
          ([time, value], index) =>
            `${index ? "L" : "M"}${x(time).toFixed(2)},${y(value).toFixed(2)}`,
        )
        .join(""),
    )
    .join("");
}

/** A stable vertical offset in [0, 1) per index, so strips of dots never reshuffle. */
export function jitter(index: number): number {
  const s = Math.sin(index * 12.9898 + 78.233) * 43_758.5453;
  return s - Math.floor(s);
}
