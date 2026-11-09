"use client";

import type { PosteriorMarginal } from "@grassmoth/api-types";
import type { ReactNode } from "react";
import { type LawCurve, lawLabel } from "@/lib/model-asset/laws";
import { humanize } from "@/lib/model-asset/selection";
import { formatPosteriorIntervalLabel, formatSignificant } from "@/lib/utils/format";
import { CHART_COLORS, type ChartColor, cssColor } from "./chart-tokens";
import { DistributionChart } from "./distribution-chart";
import { lawDomain, lawLayers } from "./law-layers";

function Swatch({ color, dashed = false }: { readonly color: string; readonly dashed?: boolean }) {
  return (
    <svg width={14} height={6} aria-hidden="true">
      <line
        x1={0}
        x2={14}
        y1={3}
        y2={3}
        stroke={color}
        strokeWidth={dashed ? 1 : 2}
        strokeDasharray={dashed ? "2.5 1.5" : undefined}
      />
    </svg>
  );
}

/** The posterior mean and interval, or the element count when the parameter has several. */
function PosteriorSummary({
  posterior,
  count,
  tone,
}: {
  readonly posterior: PosteriorMarginal | undefined;
  readonly count: number;
  readonly tone: ChartColor;
}) {
  if (posterior) {
    return (
      <span
        className="flex-none text-right font-mono text-[10px]"
        title={formatPosteriorIntervalLabel(posterior)}
      >
        <span style={{ color: cssColor(tone) }}>{formatSignificant(posterior.mean)}</span>{" "}
        <span className="text-muted-foreground">
          [{formatSignificant(posterior.lower)}, {formatSignificant(posterior.upper)}]
        </span>
      </span>
    );
  }
  if (count > 1) {
    return <span className="flex-none text-[10px] text-muted-foreground">{count} elements</span>;
  }
  return null;
}

/** A law's curves on one axis, with the recorded posterior interval after a fit. */
export function LawChart({
  curve,
  caption,
}: {
  readonly curve: LawCurve;
  readonly caption?: ReactNode;
}) {
  const posterior = curve.posteriors.length === 1 ? curve.posteriors[0] : undefined;
  const tone = CHART_COLORS.posterior;
  return (
    <figure className="m-0 flex min-w-0 flex-col gap-1">
      <figcaption className="flex items-baseline justify-between gap-2 text-[11px]">
        <span className="min-w-0">
          {caption ?? (
            <>
              <span className="font-medium">{lawLabel(curve)}</span>
              <span className="block truncate text-[10px] text-muted-foreground">
                {humanize(curve.parameter.name)}
              </span>
            </>
          )}
        </span>
        <PosteriorSummary posterior={posterior} count={curve.posteriors.length} tone={tone} />
      </figcaption>
      <DistributionChart
        label={`${lawLabel(curve)}: ${curve.posteriors.length > 0 ? "prior and posterior densities" : "prior density"}`}
        height={96}
        densities={lawLayers(curve)}
        frame={lawDomain(curve)}
        interval={
          posterior
            ? {
                lower: posterior.lower,
                upper: posterior.upper,
                center: posterior.mean,
                color: tone,
                label: `mean ${formatSignificant(posterior.mean)} · ${formatPosteriorIntervalLabel(posterior)}`,
              }
            : null
        }
      />
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-muted-foreground">
        {curve.prior.x.length > 0 && (
          <span className="inline-flex items-center gap-1">
            <Swatch color={cssColor(CHART_COLORS.prior)} dashed={curve.posteriors.length > 0} />
            {curve.kind === "fitted" ? "conditioned prior" : "authored prior"}
          </span>
        )}
        {curve.posteriors.length > 0 && (
          <span className="inline-flex items-center gap-1">
            <Swatch color={cssColor(tone)} />
            posterior histogram
          </span>
        )}
        {posterior && <span>mean · {formatPosteriorIntervalLabel(posterior)}</span>}
      </div>
    </figure>
  );
}
