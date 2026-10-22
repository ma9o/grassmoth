"use client";

import type { IndicatorEmpiricalProfile } from "@grassmoth/api-types";
import { formatSignificant } from "@/lib/utils/format";
import { CHART_COLORS, cssColor } from "./chart-tokens";
import { type Domain, extentOf, jitter, linearScale, padDomain } from "./plot-geometry";
import { useChartSize } from "./use-chart-size";

export function profileDomain(
  profile: IndicatorEmpiricalProfile,
  values: readonly (number | null)[],
): Domain | null {
  return extentOf([profile.min, profile.max, ...values]);
}

/**
 * An indicator's prepared observations on one axis: every observation as a dot over the
 * interquartile box, the median as a bar and the mean as a hollow diamond.
 */
export function ProfileMarks({
  profile,
  values,
  x,
  top,
  height,
}: {
  profile: IndicatorEmpiricalProfile;
  values: readonly (number | null)[];
  x: (value: number) => number;
  top: number;
  height: number;
}) {
  const mid = top + height / 2;
  const box = Math.min(16, height - 2);
  const ink = cssColor(CHART_COLORS.observed);
  const { min, q25, q50, q75, max, mean } = profile;
  return (
    <g>
      {min != null && max != null && (
        <line x1={x(min)} x2={x(max)} y1={mid} y2={mid} stroke={cssColor(CHART_COLORS.grid)} />
      )}
      {q25 != null && q75 != null && (
        <rect
          x={x(q25)}
          y={mid - box / 2}
          width={Math.max(1.5, x(q75) - x(q25))}
          height={box}
          fill={cssColor(CHART_COLORS.grid)}
        />
      )}
      {values.map((value, index) =>
        value == null ? null : (
          <circle
            // biome-ignore lint/suspicious/noArrayIndexKey: observations are positional
            key={index}
            cx={x(value)}
            cy={mid - box / 2 + jitter(index) * box}
            r={height > 20 ? 1.9 : 1.2}
            fill={ink}
            fillOpacity={0.55}
          />
        ),
      )}
      {q50 != null && (
        <line
          x1={x(q50)}
          x2={x(q50)}
          y1={mid - box / 2 - 1}
          y2={mid + box / 2 + 1}
          stroke={ink}
          strokeWidth={2.2}
        />
      )}
      {mean != null && (
        <path
          d={`M${x(mean)},${mid - 4}L${x(mean) + 4},${mid}L${x(mean)},${mid + 4}L${x(mean) - 4},${mid}Z`}
          fill="var(--card)"
          stroke={ink}
          strokeWidth={1.2}
        />
      )}
    </g>
  );
}

export function profileSummary(profile: IndicatorEmpiricalProfile): string {
  const parts = [`n ${profile.n_obs.toLocaleString()}`];
  if (profile.q50 != null) parts.push(`median ${formatSignificant(profile.q50)}`);
  if (profile.q25 != null && profile.q75 != null)
    parts.push(`quartiles ${formatSignificant(profile.q25)}–${formatSignificant(profile.q75)}`);
  if (profile.mean != null) parts.push(`mean ${formatSignificant(profile.mean)}`);
  return parts.join(" · ");
}

export function ProfileStrip({
  profile,
  values,
  label,
  height = 44,
}: {
  profile: IndicatorEmpiricalProfile;
  values: readonly (number | null)[];
  label: string;
  height?: number;
}) {
  const [container, size] = useChartSize<HTMLDivElement>();
  const domain = profileDomain(profile, values);
  const summary = profileSummary(profile);
  return (
    <figure className="m-0 flex min-w-0 flex-col gap-1">
      <div
        ref={container}
        className="relative w-full"
        style={{ height }}
        role="img"
        aria-label={`${label}: ${summary}`}
      >
        {size && domain && (
          <ProfileSvg
            profile={profile}
            values={values}
            domain={domain}
            width={size.width}
            height={height}
          />
        )}
      </div>
      <figcaption className="text-[10px] text-muted-foreground">{summary}</figcaption>
    </figure>
  );
}

function ProfileSvg({
  profile,
  values,
  domain,
  width,
  height,
}: {
  profile: IndicatorEmpiricalProfile;
  values: readonly (number | null)[];
  domain: Domain;
  width: number;
  height: number;
}) {
  const padded = padDomain(domain, 0.03);
  const x = linearScale(padded, [6, width - 6]);
  return (
    <svg width={width} height={height} className="overflow-visible" aria-hidden="true">
      <ProfileMarks profile={profile} values={values} x={x} top={0} height={height - 14} />
      <g fontSize={10} fill={cssColor(CHART_COLORS.axis)}>
        <text x={x(domain[0])} y={height - 2} textAnchor="start">
          {formatSignificant(domain[0])}
        </text>
        <text x={x(domain[1])} y={height - 2} textAnchor="end">
          {formatSignificant(domain[1])}
        </text>
      </g>
    </svg>
  );
}
