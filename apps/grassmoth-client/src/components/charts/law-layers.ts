import type { LawCurve } from "@/lib/model-asset/laws";
import { CHART_COLORS } from "./chart-tokens";
import { type DensityLayer, densityExtent } from "./distribution-chart";
import type { Domain } from "./plot-geometry";

/** The prior outlined, posterior histograms filled after a fit. */
export function lawLayers(curve: LawCurve): DensityLayer[] {
  const fitted = curve.posteriors.length > 0;
  const single = curve.posteriors.length === 1;
  const posterior = CHART_COLORS.posterior;
  const prior: DensityLayer[] =
    curve.prior.x.length > 0
      ? [
          {
            key: "prior",
            label: curve.kind === "fitted" ? "conditioned prior" : "authored prior",
            color: CHART_COLORS.prior,
            curve: curve.prior,
            dashed: fitted,
            fill: fitted ? 0.07 : 0.16,
            weight: 0.9,
          },
        ]
      : [];
  return [
    ...prior,
    ...curve.posteriors.map(
      (marginal): DensityLayer => ({
        key: `posterior-${marginal.subject.element_id}`,
        label: "posterior",
        color: posterior,
        curve: marginal.density_curve,
        histogram: true,
        weight: single ? 1.3 : 0.8,
        ...(single ? { fill: 0.24 } : {}),
      }),
    ),
  ];
}

/** The value range shared by every curve drawn for one law. */
export function lawDomain(curve: LawCurve): Domain | null {
  return densityExtent(lawLayers(curve));
}
