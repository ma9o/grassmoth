"use client";

import type { ReactNode, Ref } from "react";
import { CHART_COLORS, cssColor } from "./chart-tokens";
import type { PlotBox, Tick } from "./plot-geometry";
import type { ChartSize } from "./use-chart-size";

/** A measured chart box: marks painted on a canvas beneath an SVG overlay of axes and labels. */
export function CanvasChartFrame({
  container,
  canvas,
  size,
  height,
  label,
  children,
}: {
  container: Ref<HTMLDivElement>;
  /** Null when the chart has no canvas marks. */
  canvas: Ref<HTMLCanvasElement>;
  size: ChartSize | null;
  height: number;
  label: string;
  children: ReactNode;
}) {
  return (
    <div
      ref={container}
      className="relative w-full"
      style={{ height }}
      role="img"
      aria-label={label}
    >
      {size && (
        <>
          {canvas && (
            <canvas
              ref={canvas}
              className="absolute inset-0"
              style={{ width: size.width, height: size.height }}
            />
          )}
          <svg
            className="absolute inset-0 overflow-visible"
            width={size.width}
            height={size.height}
            aria-hidden="true"
          >
            {children}
          </svg>
        </>
      )}
    </div>
  );
}

/** Horizontal gridlines across the plot box, each labelled at its left edge. */
export function ValueGridlines({
  box,
  y,
  ticks,
}: {
  box: PlotBox;
  y: (value: number) => number;
  ticks: readonly Tick[];
}) {
  return (
    <>
      {ticks.map((tick) => (
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
    </>
  );
}
