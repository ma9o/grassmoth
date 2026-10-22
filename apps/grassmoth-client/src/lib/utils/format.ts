import type { PosteriorMarginal } from "@grassmoth/api-types";

export function formatNumber(n: number, decimals = 3): string {
  if (Number.isNaN(n)) return "NaN";
  if (!Number.isFinite(n)) return n > 0 ? "+Inf" : "-Inf";
  return n.toFixed(decimals);
}

/** Calendar display of a model day using the backend's pinned UTC origin. */
export function formatModelDate(day: number, origin: string): string {
  return new Date(Date.parse(origin) + day * 86_400_000).toLocaleDateString(undefined, {
    timeZone: "UTC",
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

const significantFormatter = new Intl.NumberFormat("en", { maximumSignificantDigits: 3 });

/** Three significant digits, so small rates and large intercepts both stay legible. */
export function formatSignificant(n: number): string {
  return significantFormatter.format(n).replace("-", "−");
}

const posteriorMassFormatter = new Intl.NumberFormat("en", {
  style: "percent",
  maximumFractionDigits: 2,
});

export function formatPosteriorIntervalLabel(
  estimate: Pick<PosteriorMarginal, "interval_kind" | "interval_mass">,
): string {
  const kind = { hdi: "HDI", equal_tail: "equal-tail interval" }[estimate.interval_kind];
  return `${posteriorMassFormatter.format(estimate.interval_mass)} ${kind}`;
}
