"use client";

import { type PointerEvent, useEffect, useMemo, useRef, useState } from "react";
import { formatModelDate, formatSignificant } from "@/lib/utils/format";
import { CHART_COLORS, type ChartColor, cssColor, resolveColor } from "./chart-tokens";
import {
  type Domain,
  type PlotBox,
  extentOf,
  linePath,
  linearScale,
  padDomain,
  timeTicks,
  valueTicks,
} from "./plot-geometry";
import { paintCanvas, useChartSize } from "./use-chart-size";

export interface DrawRow {
  readonly key: string;
  readonly label: string;
  readonly values: readonly (number | null)[];
}

/** Rows that share one role: a simulation arm, a check's replicates or one category. */
export interface DrawLayer {
  readonly key: string;
  readonly label: string;
  readonly color: ChartColor;
  readonly rows: readonly DrawRow[];
  readonly points?: boolean;
  readonly dashed?: boolean;
  /** Each row matters on its own, such as a category probability: drawn opaque. */
  readonly strong?: boolean;
}

export interface ObservedPoints {
  readonly label: string;
  readonly values: readonly (number | null)[];
  readonly supportStart?: readonly (number | null)[];
  readonly supportEnd?: readonly (number | null)[];
}

export interface TimeMarker {
  readonly time: number;
  readonly label: string;
}

export interface Threshold {
  readonly value: number;
  readonly label: string;
  readonly color: ChartColor;
}

export interface DrawsChartProps {
  readonly label: string;
  /** Model days, increasing. */
  readonly times: readonly number[];
  /** Pinned UTC origin of model day zero; without one the axis counts days. */
  readonly timeOrigin: string | null;
  readonly layers: readonly DrawLayer[];
  readonly observed?: ObservedPoints;
  readonly markers?: readonly TimeMarker[];
  readonly thresholds?: readonly Threshold[];
  /** The backend's display range; rows beyond it leave the chart, which counts them. */
  readonly frame?: Domain | null;
  readonly levels?: readonly string[] | null;
  readonly timeWindow?: Domain | null;
  readonly height: number;
  /** Graph cards: marks only, without axes, labels or hover. */
  readonly compact?: boolean;
  /** Extra backing-store scale for a chart inside a zoomed graph. */
  readonly resolution?: number;
  /** Names the x-axis when it is not calendar time. */
  readonly xLabel?: string;
}

const MARGIN = { left: 46, right: 12, top: 16, bottom: 22 } as const;
const COMPACT_MARGIN = { left: 1, right: 1, top: 2, bottom: 2 } as const;

/** Many draws are faint so their density shows; a few rows are drawn plainly. */
function rowAlpha(layer: DrawLayer): number {
  if (layer.strong) return 0.9;
  const rows = layer.rows.length;
  if (rows > 60) return 0.14;
  if (rows > 12) return 0.28;
  return rows > 1 ? 0.55 : 0.85;
}

function* plottedValues(
  layers: readonly DrawLayer[],
  observed: ObservedPoints | undefined,
  thresholds: readonly Threshold[],
): Generator<number | null> {
  for (const layer of layers) for (const row of layer.rows) yield* row.values;
  if (observed) yield* observed.values;
  for (const threshold of thresholds) yield threshold.value;
}

function rowsOutside(layers: readonly DrawLayer[], [low, high]: Domain): number {
  return layers
    .flatMap((layer) => layer.rows)
    .filter((row) => row.values.some((value) => value != null && (value < low || value > high)))
    .length;
}

/** Index of the time nearest to `time` in an increasing schedule. */
function nearestIndex(times: readonly number[], time: number): number {
  let low = 0;
  let high = times.length - 1;
  while (low < high) {
    const middle = (low + high) >> 1;
    if ((times[middle] ?? time) < time) low = middle + 1;
    else high = middle;
  }
  const previous = times[low - 1];
  const current = times[low];
  return previous !== undefined && current !== undefined && time - previous < current - time
    ? low - 1
    : low;
}

interface Hover {
  readonly index: number;
  readonly layer: DrawLayer | null;
  readonly row: DrawRow | null;
}

/**
 * Draws over time: every draw is painted on a canvas so hundreds cost about what a few do.
 * Axes, observations, markers and the hovered draw sit on an SVG layer above.
 */
export function DrawsChart({
  label,
  times,
  timeOrigin,
  layers,
  observed,
  markers = [],
  thresholds = [],
  frame = null,
  levels = null,
  timeWindow = null,
  height,
  compact = false,
  resolution = 1,
  xLabel,
}: DrawsChartProps) {
  const [container, size] = useChartSize<HTMLDivElement>();
  const canvas = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<Hover | null>(null);
  const categorical = levels !== null && levels.length > 0;

  const geometry = useMemo(() => {
    if (!size) return null;
    const margin = compact ? COMPACT_MARGIN : MARGIN;
    const box: PlotBox = {
      left: margin.left,
      top: margin.top,
      width: Math.max(1, size.width - margin.left - margin.right),
      height: Math.max(1, size.height - margin.top - margin.bottom),
    };
    const span =
      timeWindow ??
      extentOf([...times, ...(observed?.supportStart ?? []), ...(observed?.supportEnd ?? [])]) ??
      ([0, 1] as const);
    const timeDomain = span[0] === span[1] ? padDomain(span) : span;
    const valueDomain: Domain = categorical
      ? [-0.5, levels.length - 0.5]
      : padDomain(frame ?? extentOf(plottedValues(layers, observed, thresholds)) ?? [0, 1]);
    return {
      box,
      timeDomain,
      valueDomain,
      x: linearScale(timeDomain, [box.left, box.left + box.width]),
      y: linearScale(valueDomain, [box.top + box.height, box.top]),
    };
  }, [size, compact, timeWindow, times, observed, categorical, levels, frame, layers, thresholds]);

  useEffect(() => {
    const target = canvas.current;
    if (!target || !size || !geometry) return;
    const { box, x, y } = geometry;
    paintCanvas(target, size, resolution, (context) => {
      context.save();
      context.beginPath();
      context.rect(box.left, box.top, box.width, box.height);
      context.clip();
      for (const layer of layers) {
        const color = resolveColor(target, layer.color);
        context.globalAlpha = rowAlpha(layer);
        if (layer.points) {
          context.fillStyle = color;
          for (const row of layer.rows)
            row.values.forEach((value, index) => {
              const time = times[index];
              if (value == null || time === undefined || !Number.isFinite(value)) return;
              context.fillRect(x(time) - 1.1, y(value) - 1.1, 2.2, 2.2);
            });
          continue;
        }
        context.strokeStyle = color;
        context.lineWidth = layer.strong ? 1.5 : 1;
        context.setLineDash(layer.dashed ? [4, 3] : []);
        for (const row of layer.rows) context.stroke(new Path2D(linePath(times, row.values, x, y)));
      }
      context.restore();
    });
  }, [size, geometry, layers, times, resolution]);

  const outside = useMemo(
    () => (frame && !categorical ? rowsOutside(layers, frame) : 0),
    [layers, frame, categorical],
  );
  const total = layers.reduce((count, layer) => count + layer.rows.length, 0);
  const dateLabel = (time: number) =>
    timeOrigin
      ? formatModelDate(time, timeOrigin)
      : `${xLabel ?? "Day"} ${formatSignificant(time)}`;
  const valueLabel = (value: number) =>
    (categorical ? levels[Math.round(value)] : undefined) ?? formatSignificant(value);

  const track = (event: PointerEvent<SVGRectElement>) => {
    if (!geometry || times.length === 0) return;
    const { box, timeDomain, y } = geometry;
    const bounds = event.currentTarget.getBoundingClientRect();
    const fraction = (event.clientX - bounds.left) / Math.max(1, bounds.width);
    const index = nearestIndex(times, timeDomain[0] + fraction * (timeDomain[1] - timeDomain[0]));
    const pointer = box.top + (event.clientY - bounds.top);
    let best: { layer: DrawLayer; row: DrawRow; distance: number } | null = null;
    for (const layer of layers)
      for (const row of layer.rows) {
        const value = row.values[index];
        if (value == null || !Number.isFinite(value)) continue;
        const distance = Math.abs(y(value) - pointer);
        if (distance < 10 && (!best || distance < best.distance)) best = { layer, row, distance };
      }
    setHover({ index, layer: best?.layer ?? null, row: best?.row ?? null });
  };

  const hoveredTime = hover ? times[hover.index] : undefined;
  const hoveredValue = hover?.row?.values[hover.index];
  const observedValue = hover ? observed?.values[hover.index] : undefined;

  return (
    <div
      ref={container}
      className="relative w-full"
      style={{ height }}
      role="img"
      aria-label={label}
    >
      {size && geometry && (
        <>
          <canvas
            ref={canvas}
            className="absolute inset-0"
            style={{ width: size.width, height: size.height }}
          />
          <svg
            className="absolute inset-0 overflow-visible"
            width={size.width}
            height={size.height}
            aria-hidden="true"
          >
            {!compact && (
              <ChartAxes
                geometry={geometry}
                timeOrigin={timeOrigin}
                levels={categorical ? levels : null}
                xLabel={timeOrigin ? null : (xLabel ?? "Day")}
              />
            )}
            {thresholds.map((threshold) => {
              const py = geometry.y(threshold.value);
              return (
                <g key={threshold.label} stroke={cssColor(threshold.color)}>
                  <line
                    x1={geometry.box.left}
                    x2={geometry.box.left + geometry.box.width}
                    y1={py}
                    y2={py}
                    strokeDasharray="4 3"
                  />
                  {!compact && (
                    <text
                      x={geometry.box.left + geometry.box.width - 2}
                      y={py - 3}
                      textAnchor="end"
                      fontSize={9}
                      stroke="none"
                      fill={cssColor(threshold.color)}
                    >
                      {threshold.label}
                    </text>
                  )}
                </g>
              );
            })}
            {markers.map((marker) => {
              const px = geometry.x(marker.time);
              if (px < geometry.box.left || px > geometry.box.left + geometry.box.width)
                return null;
              return (
                <g key={`${marker.time}-${marker.label}`}>
                  <line
                    x1={px}
                    x2={px}
                    y1={geometry.box.top}
                    y2={geometry.box.top + geometry.box.height}
                    stroke={cssColor(CHART_COLORS.intervened)}
                    strokeDasharray="3 3"
                  />
                  {!compact && (
                    <text
                      x={Math.min(px + 3, geometry.box.left + geometry.box.width - 30)}
                      y={geometry.box.top - 4}
                      fontSize={9}
                      fill={cssColor(CHART_COLORS.intervened)}
                    >
                      {marker.label}
                    </text>
                  )}
                </g>
              );
            })}
            {!compact && (
              <rect
                x={geometry.box.left}
                y={geometry.box.top}
                width={geometry.box.width}
                height={geometry.box.height}
                fill="transparent"
                onPointerMove={track}
                onPointerLeave={() => setHover(null)}
              />
            )}
            {observed && (
              <ObservedMarks
                observed={observed}
                times={times}
                geometry={geometry}
                compact={compact}
                describe={(index, value) => {
                  const time = times[index];
                  const start = observed.supportStart?.[index];
                  const end = observed.supportEnd?.[index];
                  return `${observed.label} · ${time === undefined ? "" : dateLabel(time)}: ${valueLabel(value)}${
                    start != null && end != null
                      ? ` · measured ${dateLabel(start)}–${dateLabel(end)}`
                      : ""
                  }`;
                }}
              />
            )}
            {hover?.row && hover.layer && (
              <path
                d={linePath(times, hover.row.values, geometry.x, geometry.y)}
                fill="none"
                stroke={cssColor(hover.layer.color)}
                strokeWidth={2}
                pointerEvents="none"
              />
            )}
            {hoveredTime !== undefined && (
              <line
                x1={geometry.x(hoveredTime)}
                x2={geometry.x(hoveredTime)}
                y1={geometry.box.top}
                y2={geometry.box.top + geometry.box.height}
                stroke={cssColor(CHART_COLORS.axis)}
                strokeOpacity={0.5}
                pointerEvents="none"
              />
            )}
            {!compact && (hoveredTime !== undefined || outside > 0) && (
              <text
                x={geometry.box.left + geometry.box.width}
                y={geometry.box.top - 5}
                textAnchor="end"
                fontSize={9.5}
                fill={cssColor(CHART_COLORS.axis)}
              >
                {hoveredTime !== undefined
                  ? [
                      dateLabel(hoveredTime),
                      hover?.row && hover.layer && hoveredValue != null
                        ? `${hover.layer.label} · ${hover.row.label}: ${valueLabel(hoveredValue)}`
                        : null,
                      observedValue != null && observed
                        ? `${observed.label}: ${valueLabel(observedValue)}`
                        : null,
                    ]
                      .filter((part) => part !== null)
                      .join(" · ")
                  : `${outside} of ${total} draws leave the frame`}
              </text>
            )}
          </svg>
        </>
      )}
    </div>
  );
}

interface Geometry {
  readonly box: PlotBox;
  readonly timeDomain: Domain;
  readonly valueDomain: Domain;
  readonly x: (time: number) => number;
  readonly y: (value: number) => number;
}

function ChartAxes({
  geometry,
  timeOrigin,
  levels,
  xLabel,
}: {
  geometry: Geometry;
  timeOrigin: string | null;
  levels: readonly string[] | null;
  xLabel: string | null;
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
      {rows.map((tick) => (
        <g key={tick.value}>
          <line
            x1={box.left}
            x2={box.left + box.width}
            y1={y(tick.value)}
            y2={y(tick.value)}
            stroke={cssColor(CHART_COLORS.grid)}
          />
          <text x={box.left - 6} y={y(tick.value)} textAnchor="end" dominantBaseline="middle">
            {tick.label}
          </text>
        </g>
      ))}
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

function ObservedMarks({
  observed,
  times,
  geometry,
  compact,
  describe,
}: {
  observed: ObservedPoints;
  times: readonly number[];
  geometry: Geometry;
  compact: boolean;
  describe: (index: number, value: number) => string;
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
          // biome-ignore lint/suspicious/noArrayIndexKey: observations are positional on the schedule
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
