"use client";

import { ValueGridlines } from "./chart-frame";
import { CHART_COLORS, cssColor } from "./chart-tokens";
import type { DrawsGeometry, ObservedPoints } from "./draws-chart";
import { timeTicks, valueTicks } from "./plot-geometry";

export function ChartAxes({
  geometry,
  timeOrigin,
  levels,
  xLabel,
}: {
  readonly geometry: DrawsGeometry;
  readonly timeOrigin: string | null;
  readonly levels: readonly string[] | null;
  readonly xLabel: string | null;
}) {
  const { box, timeDomain, valueDomain, x, y } = geometry;
  const rows = levels
    ? levels.flatMap((level, index) =>
        index % Math.ceil(levels.length / 6) === 0 ? [{ value: index, label: level }] : [],
      )
    : valueTicks(valueDomain, Math.max(2, Math.round(box.height / 40)));
  const columns = timeTicks(timeDomain, timeOrigin, Math.max(2, Math.round(box.width / 90)));
  const bottom = box.top + box.height;
  return (
    <g fontSize={10} fill={cssColor(CHART_COLORS.axis)}>
      <ValueGridlines box={box} y={y} ticks={rows} />
      {columns.map((tick) => (
        <g key={tick.value}>
          <line
            x1={x(tick.value)}
            x2={x(tick.value)}
            y1={box.top}
            y2={bottom}
            stroke={cssColor(CHART_COLORS.grid)}
            strokeOpacity={0.6}
          />
          <text x={x(tick.value)} y={bottom + 14} textAnchor="middle">
            {tick.label}
          </text>
        </g>
      ))}
      {xLabel && (
        <text x={box.left + box.width} y={bottom + 14} textAnchor="end" fontSize={9}>
          {xLabel}
        </text>
      )}
    </g>
  );
}

export function ObservedMarks({
  observed,
  times,
  geometry,
  compact,
  describe,
}: {
  readonly observed: ObservedPoints;
  readonly times: readonly number[];
  readonly geometry: DrawsGeometry;
  readonly compact: boolean;
  readonly describe: (index: number, value: number) => string;
}) {
  const { x, y } = geometry;
  const color = cssColor(CHART_COLORS.observed);
  return (
    <g>
      {observed.values.map((value, index) => {
        const time = times[index];
        if (value == null || time === undefined || !Number.isFinite(value)) return null;
        const start = observed.supportStart?.[index];
        const end = observed.supportEnd?.[index];
        return (
          // eslint-disable-next-line react/no-array-index-key -- observations are positional on the schedule
          <g key={index}>
            {start != null && end != null && end > start && (
              <line
                x1={x(start)}
                x2={x(end)}
                y1={y(value)}
                y2={y(value)}
                stroke={color}
                strokeOpacity={0.35}
                strokeWidth={2}
              />
            )}
            <circle
              cx={x(time)}
              cy={y(value)}
              r={compact ? 1.6 : 2.6}
              fill={color}
              stroke="var(--card)"
              strokeWidth={compact ? 0.5 : 1}
            >
              {!compact && <title>{describe(index, value)}</title>}
            </circle>
          </g>
        );
      })}
    </g>
  );
}
