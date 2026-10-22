"use client";

import { useEffect, useMemo, useRef } from "react";
import { CHART_COLORS, cssColor, resolveColor } from "./chart-tokens";
import { type PlotBox, extentOf, linearScale, padDomain, valueTicks } from "./plot-geometry";
import { paintCanvas, useChartSize } from "./use-chart-size";

interface Coordinate {
  readonly label: string;
  readonly values: readonly number[];
}

const MARGIN = { left: 46, right: 10, top: 10, bottom: 34 } as const;

/** Two parameters' retained draws, one dot per draw; flagged draws are drawn last in red. */
export function JointDrawsChart({
  x,
  y,
  flagged,
  height,
}: {
  x: Coordinate;
  y: Coordinate;
  flagged: readonly boolean[] | null;
  height: number;
}) {
  const [container, size] = useChartSize<HTMLDivElement>();
  const canvas = useRef<HTMLCanvasElement>(null);
  const geometry = useMemo(() => {
    if (!size) return null;
    const box: PlotBox = {
      left: MARGIN.left,
      top: MARGIN.top,
      width: Math.max(1, size.width - MARGIN.left - MARGIN.right),
      height: Math.max(1, size.height - MARGIN.top - MARGIN.bottom),
    };
    const xDomain = padDomain(extentOf(x.values) ?? [0, 1]);
    const yDomain = padDomain(extentOf(y.values) ?? [0, 1]);
    return {
      box,
      xDomain,
      yDomain,
      sx: linearScale(xDomain, [box.left, box.left + box.width]),
      sy: linearScale(yDomain, [box.top + box.height, box.top]),
    };
  }, [size, x.values, y.values]);

  useEffect(() => {
    const target = canvas.current;
    if (!target || !size || !geometry) return;
    const { sx, sy } = geometry;
    paintCanvas(target, size, 1, (context) => {
      const plot = (index: number) => {
        const xv = x.values[index];
        const yv = y.values[index];
        if (xv === undefined || yv === undefined) return;
        context.beginPath();
        context.arc(sx(xv), sy(yv), 1.7, 0, Math.PI * 2);
        context.fill();
      };
      context.globalAlpha = x.values.length > 2000 ? 0.25 : 0.45;
      context.fillStyle = resolveColor(target, CHART_COLORS.posterior);
      x.values.forEach((_, index) => {
        if (!flagged?.[index]) plot(index);
      });
      context.globalAlpha = 0.9;
      context.fillStyle = resolveColor(target, CHART_COLORS.flagged);
      x.values.forEach((_, index) => {
        if (flagged?.[index]) plot(index);
      });
    });
  }, [size, geometry, x.values, y.values, flagged]);

  return (
    <div
      ref={container}
      className="relative w-full"
      style={{ height }}
      role="img"
      aria-label={`${x.label} against ${y.label}: every retained draw`}
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
            <g fontSize={10} fill={cssColor(CHART_COLORS.axis)}>
              {valueTicks(geometry.yDomain, 4).map((tick) => (
                <g key={tick.value}>
                  <line
                    x1={geometry.box.left}
                    x2={geometry.box.left + geometry.box.width}
                    y1={geometry.sy(tick.value)}
                    y2={geometry.sy(tick.value)}
                    stroke={cssColor(CHART_COLORS.grid)}
                  />
                  <text
                    x={geometry.box.left - 6}
                    y={geometry.sy(tick.value)}
                    textAnchor="end"
                    dominantBaseline="middle"
                  >
                    {tick.label}
                  </text>
                </g>
              ))}
              {valueTicks(geometry.xDomain, Math.max(2, Math.round(geometry.box.width / 80))).map(
                (tick) => (
                  <text
                    key={tick.value}
                    x={geometry.sx(tick.value)}
                    y={geometry.box.top + geometry.box.height + 14}
                    textAnchor="middle"
                  >
                    {tick.label}
                  </text>
                ),
              )}
              <text
                x={geometry.box.left + geometry.box.width}
                y={geometry.box.top + geometry.box.height + 28}
                textAnchor="end"
              >
                {x.label}
              </text>
              <text x={geometry.box.left} y={geometry.box.top - 2} fontSize={9}>
                {y.label}
              </text>
            </g>
          </svg>
        </>
      )}
    </div>
  );
}
