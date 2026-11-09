"use client";

import type { DensityCurve } from "@grassmoth/api-types";
import { area, curveLinear, curveStep, curveStepAfter, line } from "d3-shape";
import { useEffect, useMemo, useRef } from "react";
import { formatSignificant } from "@/lib/utils/format";
import { CanvasChartFrame, ValueGridlines } from "./chart-frame";
import { CHART_COLORS, type ChartColor, cssColor, resolveColor } from "./chart-tokens";
import {
  type Domain,
  extentOf,
  jitter,
  linearScale,
  type Margin,
  marginBox,
  type PlotBox,
  padDomain,
  valueTicks,
} from "./plot-geometry";
import { paintCanvas, useChartSize } from "./use-chart-size";

/** A prior law's density, or a retained posterior histogram's bin heights. */
export interface DensityLayer {
  readonly key: string;
  readonly label: string;
  readonly color: ChartColor;
  readonly curve: DensityCurve;
  /** Histograms keep their bin heights as steps; no spline invents modes. */
  readonly histogram?: boolean;
  readonly dashed?: boolean;
  /** Opacity of the fill under the curve; none when absent. */
  readonly fill?: number;
  readonly weight?: number;
}

/** Every draw or replicate as one dot, on its own band. */
export interface DotLayer {
  readonly key: string;
  readonly label: string;
  readonly color: ChartColor;
  readonly values: readonly number[];
  /** Flagged draws, such as divergent ones, are drawn last in the flag colour. */
  readonly flagged?: readonly boolean[];
}

export interface ValueMark {
  readonly key: string;
  readonly label: string;
  readonly value: number;
  readonly color: ChartColor;
}

export interface IntervalMark {
  readonly lower: number;
  readonly upper: number;
  readonly center: number;
  readonly color: ChartColor;
  readonly label: string;
}

/** An exact empirical distribution: each jump keeps the count at that value. */
export interface CumulativeLayer {
  readonly key: string;
  readonly label: string;
  readonly color: ChartColor;
  readonly points: readonly { readonly value: number; readonly probability: number }[];
}

type Point = readonly [number, number];

const densityPoints = (curve: DensityCurve): Point[] =>
  curve.x.flatMap((x, index) => {
    const density = curve.density[index];
    return density === undefined ? [] : [[x, density] as const];
  });

export function densityExtent(layers: readonly DensityLayer[]): Domain | null {
  return extentOf(layers.flatMap((layer) => layer.curve.x));
}

export function densityPeak(layers: readonly DensityLayer[]): number {
  const peak = Math.max(0, ...layers.flatMap((layer) => layer.curve.density));
  return peak > 0 ? peak : 1;
}

/** Density curves in a box: shared value and density scales, prior outlined, fills underneath. */
export function DensityMarks({
  layers,
  x,
  box,
  peak,
}: {
  readonly layers: readonly DensityLayer[];
  readonly x: (value: number) => number;
  readonly box: PlotBox;
  readonly peak: number;
}) {
  const y = (density: number) => box.top + box.height - (density / peak) * box.height;
  return (
    <g>
      {layers.map((layer) => {
        const points = densityPoints(layer.curve);
        const curve = layer.histogram ? curveStep : curveLinear;
        const fill = area<Point>()
          .x(([value]) => x(value))
          .y0(box.top + box.height)
          .y1(([, density]) => y(density))
          .curve(curve)(points);
        const outline = line<Point>()
          .x(([value]) => x(value))
          .y(([, density]) => y(density))
          .curve(curve)(points);
        return (
          <g key={layer.key}>
            {layer.fill !== undefined && fill && (
              <path d={fill} fill={cssColor(layer.color)} fillOpacity={layer.fill} />
            )}
            {outline && (
              <path
                d={outline}
                fill="none"
                stroke={cssColor(layer.color)}
                strokeWidth={layer.weight ?? 1.2}
                strokeDasharray={layer.dashed ? "2.5 1.5" : undefined}
              />
            )}
          </g>
        );
      })}
    </g>
  );
}

const MARGIN = { left: 8, right: 8, top: 8, bottom: 20 } as const;
const COMPACT_MARGIN = { left: 3, right: 3, top: 2, bottom: 2 } as const;
const CUMULATIVE_MARGIN = { left: 30, right: 8, top: 8, bottom: 20 } as const;

function chartMargin(compact: boolean, cumulative: boolean): Margin {
  if (compact) return COMPACT_MARGIN;
  return cumulative ? CUMULATIVE_MARGIN : MARGIN;
}

/**
 * Distributions: prior and posterior curves, every draw or replicate as a dot, and observed
 * values as marks. Dots go on a canvas; values beyond the frame pile at its edge and are counted.
 */
export function DistributionChart({
  label,
  height,
  densities = [],
  dots = [],
  marks = [],
  interval = null,
  cumulative = [],
  diagonal = false,
  frame = null,
  compact = false,
}: {
  readonly label: string;
  readonly height: number;
  readonly densities?: readonly DensityLayer[];
  readonly dots?: readonly DotLayer[];
  readonly marks?: readonly ValueMark[];
  readonly interval?: IntervalMark | null;
  readonly cumulative?: readonly CumulativeLayer[];
  /** The uniform reference a calibrated probability integral transform follows. */
  readonly diagonal?: boolean;
  /** The value range shown, from the backend; otherwise every value fits. */
  readonly frame?: Domain | null;
  readonly compact?: boolean;
}) {
  const [container, size] = useChartSize<HTMLDivElement>();
  const canvas = useRef<HTMLCanvasElement>(null);
  const isCumulative = cumulative.length > 0;

  const geometry = useMemo(() => {
    if (!size) return null;
    const box = marginBox(size.width, size.height, chartMargin(compact, isCumulative));
    const domain =
      frame ??
      extentOf([
        ...densities.flatMap((layer) => layer.curve.x),
        ...dots.flatMap((layer) => layer.values),
        ...marks.map((mark) => mark.value),
        ...(interval ? [interval.lower, interval.upper] : []),
        ...cumulative.flatMap((layer) => layer.points.map((point) => point.value)),
      ]) ??
      ([0, 1] as const);
    const valueDomain = frame || densities.length > 0 ? domain : padDomain(domain, 0.03);
    const intervalRow = interval ? 10 : 0;
    const densityHeight =
      densities.length > 0 ? (box.height - intervalRow) * (dots.length > 0 ? 0.6 : 1) : 0;
    return {
      box,
      valueDomain,
      x: linearScale(valueDomain[0] === valueDomain[1] ? padDomain(valueDomain) : valueDomain, [
        box.left,
        box.left + box.width,
      ]),
      densityBox: { ...box, height: densityHeight },
      intervalY: box.top + densityHeight + intervalRow / 2 + 1,
      dotBox: {
        ...box,
        top: box.top + densityHeight + intervalRow + (densities.length > 0 ? 4 : 0),
        height: box.height - densityHeight - intervalRow - (densities.length > 0 ? 4 : 0),
      },
    };
  }, [size, compact, isCumulative, frame, densities, dots, marks, interval, cumulative]);

  useEffect(() => {
    const target = canvas.current;
    if (!target || !size || !geometry || dots.length === 0) return;
    const { x, dotBox, valueDomain } = geometry;
    const band = dotBox.height / dots.length;
    const radius = compact ? 1.3 : 1.6;
    paintCanvas(target, size, 1, (context) => {
      const flagged = resolveColor(target, CHART_COLORS.flagged);
      const draw = (top: number, value: number, index: number) => {
        const clamped = Math.min(Math.max(value, valueDomain[0]), valueDomain[1]);
        context.beginPath();
        context.arc(
          x(clamped),
          top + radius + jitter(index) * Math.max(0, band - 2 * radius),
          radius,
          0,
          Math.PI * 2,
        );
        context.fill();
      };
      for (const [bandIndex, layer] of dots.entries()) {
        const top = dotBox.top + bandIndex * band;
        context.globalAlpha = layer.values.length > 400 ? 0.3 : 0.5;
        context.fillStyle = resolveColor(target, layer.color);
        for (const [index, value] of layer.values.entries()) {
          if (!layer.flagged?.[index]) draw(top, value, index);
        }
        context.globalAlpha = 0.9;
        context.fillStyle = flagged;
        for (const [index, value] of layer.values.entries()) {
          if (layer.flagged?.[index]) draw(top, value, index);
        }
      }
    });
  }, [size, geometry, dots, compact]);

  const beyond = useMemo(
    () =>
      frame
        ? dots.reduce(
            (count, layer) =>
              count + layer.values.filter((value) => value < frame[0] || value > frame[1]).length,
            0,
          )
        : 0,
    [dots, frame],
  );

  return (
    <CanvasChartFrame
      container={container}
      canvas={dots.length > 0 ? canvas : null}
      size={size}
      height={height}
      label={label}
    >
      {geometry && (
        <>
          {isCumulative ? (
            <CumulativeMarks
              layers={cumulative}
              diagonal={diagonal}
              x={geometry.x}
              box={geometry.box}
            />
          ) : (
            densities.length > 0 && (
              <DensityMarks
                layers={densities}
                x={geometry.x}
                box={geometry.densityBox}
                peak={densityPeak(densities)}
              />
            )
          )}
          {interval && (
            <g stroke={cssColor(interval.color)} fill={cssColor(interval.color)}>
              <title>{interval.label}</title>
              <line
                x1={geometry.x(interval.lower)}
                x2={geometry.x(interval.upper)}
                y1={geometry.intervalY}
                y2={geometry.intervalY}
                strokeWidth={2}
              />
              <circle
                cx={geometry.x(interval.center)}
                cy={geometry.intervalY}
                r={2.6}
                stroke="none"
              />
            </g>
          )}
          {marks.map((mark) => (
            <g key={mark.key}>
              <title>{`${mark.label}: ${formatSignificant(mark.value)}`}</title>
              <line
                x1={geometry.x(mark.value)}
                x2={geometry.x(mark.value)}
                y1={geometry.box.top - (compact ? 1 : 2)}
                y2={geometry.box.top + geometry.box.height + (compact ? 1 : 2)}
                stroke={cssColor(mark.color)}
                strokeWidth={compact ? 1.5 : 1.8}
              />
            </g>
          ))}
          {!compact && (
            <g fontSize={10} fill={cssColor(CHART_COLORS.axis)}>
              {valueTicks(geometry.valueDomain, Math.max(2, Math.round(geometry.box.width / 70)))
                .filter(
                  (tick) =>
                    tick.value >= geometry.valueDomain[0] && tick.value <= geometry.valueDomain[1],
                )
                .map((tick) => (
                  <text
                    key={tick.value}
                    x={geometry.x(tick.value)}
                    y={geometry.box.top + geometry.box.height + 14}
                    textAnchor="middle"
                  >
                    {tick.label}
                  </text>
                ))}
              <line
                x1={geometry.box.left}
                x2={geometry.box.left + geometry.box.width}
                y1={geometry.box.top + geometry.box.height + 0.5}
                y2={geometry.box.top + geometry.box.height + 0.5}
                stroke={cssColor(CHART_COLORS.grid)}
              />
              {beyond > 0 && (
                <text
                  x={geometry.box.left + geometry.box.width}
                  y={geometry.box.top}
                  textAnchor="end"
                  dominantBaseline="hanging"
                  fontSize={9}
                >
                  {`${beyond} beyond the frame, piled at its edges`}
                </text>
              )}
            </g>
          )}
        </>
      )}
    </CanvasChartFrame>
  );
}

function CumulativeMarks({
  layers,
  diagonal,
  x,
  box,
}: {
  readonly layers: readonly CumulativeLayer[];
  readonly diagonal: boolean;
  readonly x: (value: number) => number;
  readonly box: PlotBox;
}) {
  const y = linearScale([0, 1], [box.top + box.height, box.top]);
  return (
    <g>
      <g fontSize={10} fill={cssColor(CHART_COLORS.axis)}>
        <ValueGridlines
          box={box}
          y={y}
          ticks={[0, 0.5, 1].map((value) => ({ value, label: String(value) }))}
        />
      </g>
      {diagonal && (
        <line
          x1={x(0)}
          x2={x(1)}
          y1={y(0)}
          y2={y(1)}
          stroke={cssColor(CHART_COLORS.prior)}
          strokeDasharray="4 4"
        />
      )}
      {layers.map((layer) => {
        const first = layer.points[0];
        const points: Point[] = first
          ? [
              [first.value, 0],
              ...layer.points.map((point) => [point.value, point.probability] as const),
            ]
          : [];
        return (
          <path
            key={layer.key}
            d={
              line<Point>()
                .x(([value]) => x(value))
                .y(([, probability]) => y(probability))
                .curve(curveStepAfter)(points) ?? ""
            }
            fill="none"
            stroke={cssColor(layer.color)}
            strokeWidth={1.6}
          >
            <title>{layer.label}</title>
          </path>
        );
      })}
    </g>
  );
}
