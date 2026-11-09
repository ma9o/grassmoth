import type { IndicatorEmpiricalProfile, IndicatorSpec } from "@grassmoth/api-types";
import type { ReactNode } from "react";
import { CHART_COLORS, cssColor } from "@/components/charts/chart-tokens";
import { DrawsChart } from "@/components/charts/draws-chart";
import { extentOf, linearScale, padDomain } from "@/components/charts/plot-geometry";
import { ProfileMarks, profileDomain, profileSummary } from "@/components/charts/profile-strip";
import { pathLayers } from "@/components/charts/series-adapters";
import type { GraphDensity } from "@/lib/dag/build-layered-causal-graph";
import { DAG_COLORS } from "@/lib/dag/palette";
import type { PathSeriesView, SimulationPathsView } from "@/lib/model-asset/result-values";
import { humanize } from "@/lib/model-asset/selection";
import { formatSignificant } from "@/lib/utils/format";
import { truncate } from "./layered-graph-marks";

/** One row per law or indicator in a card's chart strip. */
export const STRIP_ROWS = 3;

/** Where a chart sits inside a card or a chip. */
export interface Box {
  readonly x: number;
  readonly top: number;
  readonly width: number;
  readonly height: number;
}

export interface StateGeometry {
  readonly width: number;
  readonly height: number;
  readonly title: {
    readonly band: number;
    readonly y: number;
    readonly size: number;
    readonly length: number;
  };
  readonly pillY: number;
  readonly strip: Box;
  readonly row: { readonly height: number; readonly value: number; readonly plot: number };
  readonly indicator: {
    readonly top: number;
    readonly height: number;
    readonly chart: number;
    readonly length: number;
  };
}

/** Where a card's own column draws its title, chart strip and indicator rows, at one density. */
export function stateGeometry(density: GraphDensity): StateGeometry {
  const rich = density.key === "rich";
  const { width, title, strip, indicator, height } = density.state;
  const x = rich ? 14 : 12;
  const stripWidth = width - 2 * x;
  return {
    width,
    height,
    title: { band: title, y: rich ? 24 : 19, size: rich ? 13 : 11.5, length: rich ? 29 : 24 },
    pillY: rich ? 8 : 5,
    strip: { x, top: title, width: stripWidth, height: strip },
    row: {
      height: strip / STRIP_ROWS,
      value: Math.round(stripWidth * 0.4865),
      plot: Math.round(stripWidth * 0.5135),
    },
    indicator: {
      top: title + strip + 4,
      height: indicator,
      chart: Math.round(width * 0.52),
      length: rich ? 21 : 17,
    },
  };
}

export const rowTop = (geometry: StateGeometry, index: number) =>
  geometry.strip.top + index * geometry.row.height;

/** One labelled row of a card's chart strip; its chart sits right of the value. */
export function StripRow({
  geometry,
  index,
  title,
  label,
  value,
  valueTone,
  range,
  children,
}: {
  readonly geometry: StateGeometry;
  readonly index: number;
  readonly title: string;
  readonly label: string;
  readonly value: string;
  readonly valueTone: string;
  /** The chart's value range, labelled under its ends. */
  readonly range?: readonly [number, number];
  readonly children: ReactNode;
}) {
  const { strip, row } = geometry;
  const top = rowTop(geometry, index);
  const baseline = top + Math.round(row.height * 0.52);
  const bottom = top + row.height - 0.5;
  return (
    <g role="img" aria-label={title}>
      <title>{title}</title>
      <rect x={strip.x} y={top} width={strip.width} height={row.height} fill="transparent" />
      <text x={strip.x} y={baseline} fontSize={7.5} fontWeight={600} fill={DAG_COLORS.slate}>
        {truncate(label, 17)}
      </text>
      <text
        x={strip.x + row.value}
        y={baseline}
        textAnchor="end"
        fontSize={7.5}
        fontFamily="ui-monospace, monospace"
        fill={valueTone}
      >
        {value}
      </text>
      {children}
      {range ? (
        <>
          <text x={strip.x + row.plot} y={bottom} fontSize={5.5} fill={DAG_COLORS.muted}>
            {formatSignificant(range[0])}
          </text>
          <text
            x={strip.x + strip.width}
            y={bottom}
            textAnchor="end"
            fontSize={5.5}
            fill={DAG_COLORS.muted}
          >
            {formatSignificant(range[1])}
          </text>
        </>
      ) : null}
    </g>
  );
}

/** Rows past the strip stay in the inspector. */
export function StripOverflow({
  geometry,
  hidden,
}: {
  readonly geometry: StateGeometry;
  readonly hidden: number;
}) {
  return hidden > 0 ? (
    <text
      x={geometry.width - 8}
      y={geometry.height - 4}
      textAnchor="end"
      fontSize={6.5}
      fill={DAG_COLORS.muted}
    >
      {`+${hidden} more`}
    </text>
  ) : null;
}

/** A dated event on a chart's time axis. */
export interface CardMarker {
  readonly time: number;
  readonly label: string;
}

/** Dated markers drawn by the card itself, so each keeps its own label at any zoom. */
function CardMarkers({
  box,
  times,
  markers,
}: {
  readonly box: Box;
  readonly times: readonly number[];
  readonly markers: readonly CardMarker[];
}) {
  const domain = extentOf(times);
  if (!domain || markers.length === 0) return null;
  const x = linearScale(domain[0] === domain[1] ? padDomain(domain) : domain, [
    box.x + 1,
    box.x + box.width - 1,
  ]);
  return (
    <g pointerEvents="none">
      {markers.map((marker) => (
        <g key={`${marker.time}-${marker.label}`} role="img" aria-label={marker.label}>
          <title>{marker.label}</title>
          <line
            x1={x(marker.time)}
            x2={x(marker.time)}
            y1={box.top}
            y2={box.top + box.height}
            stroke={cssColor(CHART_COLORS.intervened)}
            strokeDasharray="3 3"
            strokeWidth={0.8}
          />
        </g>
      ))}
    </g>
  );
}

/** A chart's frame inside a card or a chip; the chart takes no pointer events of its own. */
export function ChartFrame({ box, children }: { readonly box: Box; readonly children: ReactNode }) {
  return (
    <foreignObject x={box.x} y={box.top} width={box.width} height={box.height} pointerEvents="none">
      {children}
    </foreignObject>
  );
}

/** A construct's simulated draws, with its dated assignments marked. */
export function SimulatedDraws({
  box,
  paths,
  series,
  markers,
  zoom,
}: {
  readonly box: Box;
  readonly paths: SimulationPathsView;
  readonly series: PathSeriesView;
  readonly markers: readonly CardMarker[];
  readonly zoom: number;
}) {
  return (
    <>
      <CardMarkers box={box} times={paths.times} markers={markers} />
      <ChartFrame box={box}>
        <DrawsChart
          compact
          height={box.height}
          resolution={zoom}
          times={paths.times}
          timeOrigin={paths.time_origin}
          layers={pathLayers(series, "both", false)}
          frame={series.frame}
          levels={series.levels}
          label={`${humanize(series.label)}: ${series.action.length} individual simulation draws`}
        />
      </ChartFrame>
    </>
  );
}

/** A prepared indicator with the empirical profile its data preparation recorded. */
export interface PreparedRow {
  readonly indicator: IndicatorSpec;
  readonly profile: IndicatorEmpiricalProfile;
}

/** A node's prepared indicators: observation counts and their profile strips. */
export function DataStrip({
  geometry,
  rows,
}: {
  readonly geometry: StateGeometry;
  readonly rows: readonly PreparedRow[];
}) {
  const shown = rows.slice(0, STRIP_ROWS);
  return (
    <g>
      {shown.map(({ indicator, profile }, index) => {
        const name = humanize(indicator.observation.name);
        const domain = profileDomain(profile, []);
        return (
          <StripRow
            key={indicator.observation.id}
            geometry={geometry}
            index={index}
            title={`${name} · ${profileSummary(profile)}`}
            label={name}
            value={`n ${formatSignificant(profile.n_obs)}`}
            valueTone={DAG_COLORS.ink}
            {...(domain ? { range: domain } : {})}
          >
            {domain && (
              <ProfileMarks
                profile={profile}
                values={[]}
                x={linearScale(padDomain(domain, 0.02), [
                  geometry.strip.x + geometry.row.plot,
                  geometry.strip.x + geometry.strip.width,
                ])}
                top={rowTop(geometry, index) + 2}
                height={Math.max(6, geometry.row.height - 9)}
              />
            )}
          </StripRow>
        );
      })}
      <StripOverflow geometry={geometry} hidden={rows.length - shown.length} />
    </g>
  );
}
