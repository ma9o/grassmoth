"use client";

import { type PointerEvent, useEffect, useMemo, useRef, useState } from "react";
import { formatModelDate, formatSignificant } from "@/lib/utils/format";
import { CanvasChartFrame } from "./chart-frame";
import { CHART_COLORS, type ChartColor, cssColor } from "./chart-tokens";
import { paintDraws } from "./draws-canvas";
import { ChartAxes, ObservedMarks } from "./draws-overlay";
import {
  type Domain,
  extentOf,
  linearScale,
  linePath,
  marginBox,
  type PlotBox,
  padDomain,
} from "./plot-geometry";
import { paintCanvas, useChartSize, useChartVisibility } from "./use-chart-size";

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
const NO_MARKERS: readonly TimeMarker[] = [];
const NO_THRESHOLDS: readonly Threshold[] = [];

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

/** The plot box, axis domains and pixel scales shared by the canvas and the SVG layer. */
export interface DrawsGeometry {
  readonly box: PlotBox;
  readonly timeDomain: Domain;
  readonly valueDomain: Domain;
  readonly x: (time: number) => number;
  readonly y: (value: number) => number;
}

/**
 * Draws over time: every draw is painted on a canvas without a DOM node per trajectory.
 * Axes, observations, markers and the hovered draw sit on an SVG layer above.
 */
export function DrawsChart({
  label,
  times,
  timeOrigin,
  layers,
  observed,
  markers = NO_MARKERS,
  thresholds = NO_THRESHOLDS,
  frame = null,
  levels = null,
  timeWindow = null,
  height,
  compact = false,
  resolution = 1,
  xLabel,
}: DrawsChartProps) {
  const [container, size] = useChartSize<HTMLDivElement>();
  const visible = useChartVisibility(container);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<Hover | null>(null);
  const categorical = levels !== null && levels.length > 0;

  const geometry = useMemo((): DrawsGeometry | null => {
    if (!size || !visible) return null;
    const box = marginBox(size.width, size.height, compact ? COMPACT_MARGIN : MARGIN);
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
  }, [
    size,
    visible,
    compact,
    timeWindow,
    times,
    observed,
    categorical,
    levels,
    frame,
    layers,
    thresholds,
  ]);

  useEffect(() => {
    const target = canvas.current;
    if (!target || !size || !geometry) return;
    paintCanvas(target, size, resolution, (context) => {
      paintDraws(context, layers, times, geometry);
    });
  }, [size, geometry, layers, times, resolution]);

  const outside = useMemo(
    () => (visible && frame && !categorical ? rowsOutside(layers, frame) : 0),
    [visible, layers, frame, categorical],
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
    for (const layer of layers) {
      for (const row of layer.rows) {
        const value = row.values[index];
        if (value == null || !Number.isFinite(value)) continue;
        const distance = Math.abs(y(value) - pointer);
        if (distance < 10 && (!best || distance < best.distance)) best = { layer, row, distance };
      }
    }
    setHover({ index, layer: best?.layer ?? null, row: best?.row ?? null });
  };

  const hoveredTime = hover ? times[hover.index] : undefined;
  const hoveredValue = hover?.row?.values[hover.index];
  const observedValue = hover ? observed?.values[hover.index] : undefined;

  return (
    <CanvasChartFrame
      container={container}
      canvas={canvas}
      size={size}
      height={height}
      label={label}
    >
      {geometry && (
        <>
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
            if (px < geometry.box.left || px > geometry.box.left + geometry.box.width) return null;
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
              {hoveredTime === undefined
                ? `${outside} of ${total} draws leave the frame`
                : [
                    dateLabel(hoveredTime),
                    hover?.row && hover.layer && hoveredValue != null
                      ? `${hover.layer.label} · ${hover.row.label}: ${valueLabel(hoveredValue)}`
                      : null,
                    observedValue != null && observed
                      ? `${observed.label}: ${valueLabel(observedValue)}`
                      : null,
                  ]
                    .filter((part) => part !== null)
                    .join(" · ")}
            </text>
          )}
        </>
      )}
    </CanvasChartFrame>
  );
}
