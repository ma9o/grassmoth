"use client";

import type { InferenceDetailView } from "@/lib/model-asset/result-values";

import type { RankHistogram } from "@grassmoth/api-types";
import { useEffect, useMemo, useRef } from "react";
import { CHART_COLORS, chainColor, cssColor, resolveColor } from "./chart-tokens";
import {
  type Domain,
  type PlotBox,
  extentOf,
  linearScale,
  padDomain,
  valueTicks,
} from "./plot-geometry";
import { paintCanvas, useChartSize } from "./use-chart-size";

const MARGIN = { left: 40, right: 8, top: 6, bottom: 18 } as const;

function plotBox(width: number, height: number): PlotBox {
  return {
    left: MARGIN.left,
    top: MARGIN.top,
    width: Math.max(1, width - MARGIN.left - MARGIN.right),
    height: Math.max(1, height - MARGIN.top - MARGIN.bottom),
  };
}

function Axes({
  box,
  y,
  yTicks,
  xLabels,
}: {
  box: PlotBox;
  y: (value: number) => number;
  yTicks: readonly { value: number; label: string }[];
  xLabels: readonly [string, string];
}) {
  const bottom = box.top + box.height;
  return (
    <g fontSize={10} fill={cssColor(CHART_COLORS.axis)}>
      {yTicks.map((tick) => (
        <g key={tick.value}>
          <line
            x1={box.left}
            x2={box.left + box.width}
            y1={y(tick.value)}
            y2={y(tick.value)}
            stroke={cssColor(CHART_COLORS.grid)}
          />
          <text x={box.left - 5} y={y(tick.value)} textAnchor="end" dominantBaseline="middle">
            {tick.label}
          </text>
        </g>
      ))}
      <text x={box.left} y={bottom + 13}>
        {xLabels[0]}
      </text>
      <text x={box.left + box.width} y={bottom + 13} textAnchor="end">
        {xLabels[1]}
      </text>
    </g>
  );
}

/** Each chain's retained draws in order, one line per chain on a shared scale. */
export function TraceChart({
  chains,
  label,
  height = 96,
}: {
  chains: readonly (readonly number[])[];
  label: string;
  height?: number;
}) {
  const [container, size] = useChartSize<HTMLDivElement>();
  const canvas = useRef<HTMLCanvasElement>(null);
  const length = Math.max(0, ...chains.map((chain) => chain.length));
  const geometry = useMemo(() => {
    if (!size) return null;
    const box = plotBox(size.width, size.height);
    const domain: Domain = padDomain(extentOf(chains.flat()) ?? [0, 1]);
    return {
      box,
      domain,
      x: linearScale([0, Math.max(1, length - 1)], [box.left, box.left + box.width]),
      y: linearScale(domain, [box.top + box.height, box.top]),
    };
  }, [size, chains, length]);

  useEffect(() => {
    const target = canvas.current;
    if (!target || !size || !geometry) return;
    paintCanvas(target, size, 1, (context) => {
      context.lineWidth = 0.8;
      context.globalAlpha = 0.8;
      chains.forEach((chain, index) => {
        context.strokeStyle = resolveColor(target, chainColor(index));
        context.beginPath();
        chain.forEach((value, draw) => {
          if (draw) context.lineTo(geometry.x(draw), geometry.y(value));
          else context.moveTo(geometry.x(draw), geometry.y(value));
        });
        context.stroke();
      });
    });
  }, [size, geometry, chains]);

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
            <Axes
              box={geometry.box}
              y={geometry.y}
              yTicks={valueTicks(geometry.domain, 3)}
              xLabels={["draw 1", `draw ${length.toLocaleString()}`]}
            />
          </svg>
        </>
      )}
    </div>
  );
}

/** Rank histograms per chain: bins group the chains side by side; the dashed line is even mixing. */
export function RankChart({
  histogram,
  height = 96,
}: {
  histogram: RankHistogram;
  height?: number;
}) {
  const [container, size] = useChartSize<HTMLDivElement>();
  const rows = histogram.chains;
  const bins = Math.max(0, ...rows.map((row) => row.length));
  const peak = Math.max(histogram.expected_per_bin * 2, ...rows.flat());
  return (
    <div
      ref={container}
      className="relative w-full"
      style={{ height }}
      role="img"
      aria-label="Rank histogram by chain"
    >
      {size && bins > 0 && (
        <RankSvg
          rows={rows}
          bins={bins}
          peak={peak}
          expected={histogram.expected_per_bin}
          width={size.width}
          height={height}
        />
      )}
    </div>
  );
}

function RankSvg({
  rows,
  bins,
  peak,
  expected,
  width,
  height,
}: {
  rows: readonly (readonly number[])[];
  bins: number;
  peak: number;
  expected: number;
  width: number;
  height: number;
}) {
  const box = plotBox(width, height);
  const y = linearScale([0, peak], [box.top + box.height, box.top]);
  const binWidth = box.width / bins;
  const barWidth = Math.max(0.6, (binWidth - 1) / Math.max(1, rows.length));
  return (
    <svg width={width} height={height} className="overflow-visible" aria-hidden="true">
      <Axes box={box} y={y} yTicks={valueTicks([0, peak], 2)} xLabels={["low rank", "high rank"]} />
      {rows.map((counts, chain) =>
        counts.map((count, bin) => (
          <rect
            // biome-ignore lint/suspicious/noArrayIndexKey: chains and bins are positional
            key={`${chain}-${bin}`}
            x={box.left + bin * binWidth + chain * barWidth}
            y={y(count)}
            width={barWidth}
            height={Math.max(0, box.top + box.height - y(count))}
            fill={cssColor(chainColor(chain))}
            fillOpacity={0.85}
          />
        )),
      )}
      <line
        x1={box.left}
        x2={box.left + box.width}
        y1={y(expected)}
        y2={y(expected)}
        stroke={cssColor(CHART_COLORS.observed)}
        strokeDasharray="4 3"
      />
    </svg>
  );
}

/** Latent proposal step sizes by time point, one line per chain, on a log scale. */
export function StepSizeChart({
  detail,
  height = 96,
}: {
  detail: InferenceDetailView;
  height?: number;
}) {
  const [container, size] = useChartSize<HTMLDivElement>();
  const final = detail.final_latent_delta ?? [];
  const initial = (detail.initial_latent_delta ?? []).flat();
  const positive = [...final.flat(), ...initial].filter((value) => value > 0);
  const first = initial.at(0);
  const start = first !== undefined && initial.every((value) => value === first) ? first : null;
  const domain = extentOf(positive.map(Math.log10));
  return (
    <div
      ref={container}
      className="relative w-full"
      style={{ height }}
      role="img"
      aria-label="Latent proposal step size by time point and chain"
    >
      {size && domain && (
        <StepSvg
          rows={final}
          start={start}
          domain={padDomain(domain)}
          width={size.width}
          height={height}
        />
      )}
    </div>
  );
}

function StepSvg({
  rows,
  start,
  domain,
  width,
  height,
}: {
  rows: readonly (readonly number[])[];
  start: number | null;
  domain: Domain;
  width: number;
  height: number;
}) {
  const box = plotBox(width, height);
  const length = Math.max(1, ...rows.map((row) => row.length));
  const x = linearScale([0, Math.max(1, length - 1)], [box.left, box.left + box.width]);
  const logY = linearScale(domain, [box.top + box.height, box.top]);
  const y = (value: number) => logY(Math.log10(value));
  const ticks = valueTicks(domain, 3).map((tick) => ({
    value: 10 ** tick.value,
    label: (10 ** tick.value).toPrecision(2),
  }));
  return (
    <svg width={width} height={height} className="overflow-visible" aria-hidden="true">
      <Axes box={box} y={y} yTicks={ticks} xLabels={["first time point", "last"]} />
      {start !== null && start > 0 && (
        <line
          x1={box.left}
          x2={box.left + box.width}
          y1={y(start)}
          y2={y(start)}
          stroke={cssColor(CHART_COLORS.prior)}
          strokeDasharray="3 3"
        />
      )}
      {rows.map((row, chain) => (
        <polyline
          // biome-ignore lint/suspicious/noArrayIndexKey: rows are chains in engine order
          key={chain}
          fill="none"
          stroke={cssColor(chainColor(chain))}
          strokeWidth={0.9}
          points={row
            .flatMap((value, index) => (value > 0 ? [`${x(index)},${y(value)}`] : []))
            .join(" ")}
        />
      ))}
    </svg>
  );
}

export function ChainLegend({ chains }: { chains: number }) {
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-muted-foreground">
      {Array.from({ length: chains }, (_, chain) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: chains are numbered by position
        <span key={chain} className="inline-flex items-center gap-1">
          <span
            className="size-2 rounded-full"
            style={{ background: cssColor(chainColor(chain)) }}
          />
          chain {chain + 1}
        </span>
      ))}
    </div>
  );
}
