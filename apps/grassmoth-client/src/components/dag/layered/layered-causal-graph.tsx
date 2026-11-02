"use client";

import type { SimulationPathsView } from "@/lib/model-asset/result-values";
import type { DataComparisonView } from "@/lib/model-asset/data-comparison";

import type {
  ConstructSpec,
  IndicatorEmpiricalProfile,
  IndicatorSpec,
  StateAssignment,
  ActionId,
} from "@grassmoth/api-types";
import type { KeyboardEvent, ReactNode } from "react";
import { DrawsChart } from "@/components/charts/draws-chart";
import { CHART_COLORS, cssColor } from "@/components/charts/chart-tokens";
import { extentOf, linearScale, padDomain } from "@/components/charts/plot-geometry";
import { ProfileMarks, profileDomain, profileSummary } from "@/components/charts/profile-strip";
import { dataComparisonChart, pathLayers } from "@/components/charts/series-adapters";
import type { GraphDensity, MechanismRow } from "@/lib/dag/build-layered-causal-graph";
import type { ConstructStatus } from "@/lib/dag/construct-statuses";
import { BLOCKING, COMPARISON_COLORS, DAG_COLORS, LATENT } from "@/lib/dag/palette";
import {
  CANVAS_PADDING,
  type LayeredGraphOptions,
  useLayeredGraph,
} from "@/lib/dag/use-layered-graph";
import { entityFailures } from "@/lib/model-asset/inspector";
import { mechanismName, type EntitySelection } from "@/lib/model-asset/selection";
import { FlowChart } from "@/components/charts/flow-chart";
import { modelFlowPlots, type FlowPlot } from "@/lib/model-asset/flow-plots";
import { formatModelDate, formatSignificant } from "@/lib/utils/format";
import { DagCanvasFrame, DagSvg } from "../core/dag-canvas";
import { DagEdge } from "../core/dag-edge";
import { DagNodeShell } from "../core/dag-node";
import { DagZoomControls } from "../core/dag-zoom-controls";
import { LayeredComparisonOverlay } from "./layered-comparison-overlay";

export interface LayeredCausalGraphProps extends LayeredGraphOptions {
  simulationPaths?: SimulationPathsView | null;
  dataDiff?: DataComparisonView | null;
  onSelect: (selection: EntitySelection | null) => void;
  /** The action whose version is viewed; a data preparation shows each node's prepared data. */
  step?: ActionId | null;
}

function humanize(value: string): string {
  return value.replaceAll("_", " ");
}

function truncate(value: string, length: number): string {
  return value.length > length ? `${value.slice(0, Math.max(1, length - 1))}…` : value;
}

function activateOnKeyboard(event: KeyboardEvent<SVGGElement>, action: () => void): void {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    action();
  }
}

function statusAccent(status: ConstructStatus | undefined): string | undefined {
  if (status === "blocking") return BLOCKING;
  if (status === "latent") return LATENT;
  return undefined;
}

function statusLabel(status: ConstructStatus | undefined): string | null {
  if (status === "blocking") return "blocking";
  if (status === "latent") return "latent";
  return null;
}

function LayerPill({
  x,
  y,
  label,
  color,
  description,
  minWidth,
}: {
  x: number;
  y: number;
  label: string;
  color: string;
  description?: string;
  minWidth?: number;
}) {
  const width = pillWidth(label, minWidth);
  return (
    <g transform={`translate(${x - width},${y})`} role="img" aria-label={description ?? label}>
      <title>{description ?? label}</title>
      <rect width={width} height={17} rx={8.5} fill={color} fillOpacity={0.11} />
      <text x={width / 2} y={11.5} textAnchor="middle" fontSize={7.5} fontWeight={650} fill={color}>
        {label}
      </text>
    </g>
  );
}

const pillWidth = (label: string, minWidth = 38) => Math.max(minWidth, label.length * 5.3 + 14);

/** A chip has room for the count alone; the dated list stays in its description. */
const CHIP_PILL_WIDTH = 20;

/** The construct's own findings, as one mark. */
function WarningMark({ x, y, failures }: { x: number; y: number; failures: readonly string[] }) {
  if (failures.length === 0) return null;
  return (
    <g role="img" aria-label={failures.join("; ")} transform={`translate(${x},${y})`}>
      <title>{failures.join("\n")}</title>
      <path d="M0 11 L6 0 L12 11 Z" fill="var(--card)" stroke="var(--warning-foreground)" />
      <text x={6} y={9} textAnchor="middle" fontSize={8} fill="var(--warning-foreground)">
        !
      </text>
    </g>
  );
}

/** Follows a time-varying construct's name: its state carries over from one moment to the next. */
function CarryMark({ construct, size }: { construct: ConstructSpec; size: number }) {
  return construct.temporal_status === "time_varying" ? (
    <tspan dx={4} fontSize={size} fontWeight={400} fill={DAG_COLORS.muted}>
      <title>time varying</title>↻
    </tspan>
  ) : null;
}

function assignmentLabel(event: StateAssignment, timeOrigin: string): string {
  return `${formatModelDate(event.time, timeOrigin)} (day ${event.time}): set to ${event.value}`;
}

/** One row per law or indicator in a card's chart strip. */
const STRIP_ROWS = 3;

/** Where a card's own column draws its title, chart strip and indicator rows, at one density. */
function stateGeometry(density: GraphDensity) {
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
type StateGeometry = ReturnType<typeof stateGeometry>;
type Box = StateGeometry["strip"];

const rowTop = (geometry: StateGeometry, index: number) =>
  geometry.strip.top + index * geometry.row.height;

/** One labelled row of a card's chart strip; its chart sits right of the value. */
function StripRow({
  geometry,
  index,
  title,
  label,
  value,
  valueTone,
  range,
  children,
}: {
  geometry: StateGeometry;
  index: number;
  title: string;
  label: string;
  value: string;
  valueTone: string;
  /** The chart's value range, labelled under its ends. */
  range?: readonly [number, number];
  children: ReactNode;
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
function StripOverflow({ geometry, hidden }: { geometry: StateGeometry; hidden: number }) {
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

/** Dated markers drawn by the card itself, so each keeps its own label at any zoom. */
function CardMarkers({
  box,
  times,
  markers,
}: {
  box: Box;
  times: readonly number[];
  markers: readonly { time: number; label: string }[];
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

/** A node's prepared indicators: observation counts and their profile strips. */
function DataStrip({
  geometry,
  rows,
}: {
  geometry: StateGeometry;
  rows: Array<{ indicator: IndicatorSpec; profile: IndicatorEmpiricalProfile }>;
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

function ConstructCard({
  construct,
  width,
  height,
  rowsWidth,
  geometry,
  isOutcome,
  failures,
  status,
  assignments,
  timeOrigin,
  selected,
  dimmed,
  onSelect,
  rows,
  children,
}: {
  construct: ConstructSpec;
  width: number;
  height: number;
  /** Width of the mechanism rows' column, left of the construct's own column. */
  rowsWidth: number;
  geometry: StateGeometry;
  isOutcome: boolean;
  failures: string[];
  status: ConstructStatus | undefined;
  assignments: readonly StateAssignment[];
  timeOrigin: string | null;
  selected: boolean;
  dimmed: boolean;
  onSelect: () => void;
  rows: ReactNode;
  children: ReactNode;
}) {
  const label = statusLabel(status);
  const hasAssignments = assignments.length > 0;
  const accent = hasAssignments ? DAG_COLORS.intervention : statusAccent(status);
  const shellAccent = selected ? "var(--primary)" : accent;
  const badge = hasAssignments
    ? `${assignments.length} assignment${assignments.length === 1 ? "" : "s"}`
    : label;
  const badgeColor = hasAssignments
    ? DAG_COLORS.intervention
    : status === "blocking"
      ? BLOCKING
      : status === "latent"
        ? LATENT
        : DAG_COLORS.slate;
  const title = `${isOutcome ? "★ " : ""}${truncate(humanize(construct.name), geometry.title.length)}`;
  const { band, y, size } = geometry.title;

  return (
    <g
      opacity={dimmed ? 0.18 : status === "latent" ? 0.62 : 1}
      role="button"
      aria-label={humanize(construct.name)}
      tabIndex={0}
      style={{ cursor: "pointer" }}
      onClick={onSelect}
      onKeyDown={(event) => activateOnKeyboard(event, onSelect)}
    >
      <DagNodeShell
        width={width}
        height={height}
        {...(shellAccent === undefined ? {} : { accent: shellAccent })}
        dashed={status === "latent"}
        highlighted={selected}
        outcome={isOutcome}
      />
      {rows}
      <g transform={`translate(${rowsWidth},0)`}>
        <text x={geometry.strip.x} y={y} fontSize={size} fontWeight={650} fill="var(--foreground)">
          {title}
          <CarryMark construct={construct} size={size - 2} />
        </text>
        {badge ? (
          <LayerPill
            x={geometry.width - 8}
            y={geometry.pillY}
            label={badge}
            color={badgeColor}
            {...(hasAssignments && timeOrigin !== null
              ? {
                  description: assignments
                    .map((event) => assignmentLabel(event, timeOrigin))
                    .join("; "),
                }
              : {})}
          />
        ) : null}
        {children}
        <WarningMark x={geometry.width - 15} y={band - 10} failures={failures} />
      </g>
    </g>
  );
}

/** One mechanism as a row of the card it drives: its name, then its output for the viewed state. */
function MechanismRowView({
  row,
  density,
  plot,
  description,
  failures,
  zoom,
  selected,
  dimmed,
  onSelect,
  onHover,
}: {
  row: MechanismRow;
  density: GraphDensity;
  plot: FlowPlot | undefined;
  description: string;
  failures: readonly string[];
  zoom: number;
  selected: boolean;
  dimmed: boolean;
  onSelect: () => void;
  onHover: (hovered: boolean) => void;
}) {
  const rich = density.key === "rich";
  const { width, height } = density.row;
  const name = mechanismName(row.mechanism.id);
  const chartTop = rich ? 17 : 12;
  return (
    <g
      transform={`translate(0,${row.top})`}
      opacity={dimmed ? 0.25 : 1}
      role="button"
      tabIndex={0}
      aria-label={`Mechanism ${name}`}
      data-mechanism-id={row.mechanism.id}
      style={{ cursor: "pointer" }}
      onClick={(event) => {
        event.stopPropagation();
        onSelect();
      }}
      onKeyDown={(event) => {
        event.stopPropagation();
        activateOnKeyboard(event, onSelect);
      }}
      onPointerEnter={() => onHover(true)}
      onPointerLeave={() => onHover(false)}
    >
      <title>{description}</title>
      <rect
        x={4}
        y={1.5}
        width={width - 8}
        height={height - 3}
        rx={6}
        fill={DAG_COLORS.col}
        stroke={selected ? "var(--primary)" : DAG_COLORS.line}
        strokeWidth={selected ? 1.6 : 0.8}
      />
      <text
        x={10}
        y={rich ? 12.5 : 9.5}
        fontSize={rich ? 8 : 7}
        fontWeight={600}
        fill={DAG_COLORS.slate}
      >
        {truncate(name, rich ? 31 : 26)}
      </text>
      <foreignObject
        x={8}
        y={chartTop}
        width={width - 24}
        height={height - chartTop - 4}
        pointerEvents="none"
      >
        {plot && <FlowChart plot={plot} height={height - chartTop - 4} compact resolution={zoom} />}
      </foreignObject>
      <path
        d={`M${width - 9} ${height / 2 - 4} l4 4 l-4 4`}
        fill="none"
        stroke={DAG_COLORS.muted}
        strokeWidth={1.2}
      />
      <WarningMark x={width - 22} y={3} failures={failures} />
    </g>
  );
}

/**
 * The state the selected action left, one part at a glance. Each construct is a card whose
 * mechanism rows feed its own column, and each edge ends on the rows that read its cause.
 * Exogenous constructs are chips with their path. Cards show current laws or posteriors,
 * prepared data or simulated draws, and a single mark when a check on them failed; the details
 * pane holds the depth. A data_diff leaf overlays every saved replicate and the observations on
 * indicator rows, or marks added, removed and revised observations in comparison colours.
 * Latent-only constructs and edges recede.
 */
export function LayeredCausalGraph({
  modelSnapshot,
  entities,
  simulation = null,
  simulationPaths = null,
  dataDiff = null,
  comparison = null,
  selection,
  onSelect,
  step = null,
}: LayeredCausalGraphProps) {
  const {
    visible,
    setHovered,
    zoom,
    setZoom,
    fitToPane,
    paneRef,
    pane,
    outcome,
    topology,
    nodes,
    arrows,
    width,
    isLayouting,
    difference,
    nodeStatuses,
    indicatorsByConstruct,
    simulationResult,
    focused,
    edgeVisual,
  } = useLayeredGraph({ modelSnapshot, entities, simulation, comparison, selection });
  const { density } = topology;
  const geometry = stateGeometry(density);
  const selectedNode = selection?.kind === "construct" ? selection.id : null;
  const flows =
    modelSnapshot.dynamical_model_spec && (visible.has("specification") || visible.has("fit"))
      ? modelFlowPlots(modelSnapshot.dynamical_model_spec)
      : null;
  const name = (id: ConstructSpec["id"]) => humanize(entities.constructById.get(id)?.name ?? id);
  const firstArrow = new Set<string>();

  /** What a construct shows for the viewed state: its data, draws or laws. */
  const compared = (construct: ConstructSpec) =>
    dataDiff !== null &&
    construct.indicators.some((indicator) =>
      dataDiff.variables.some((variable) => variable.indicator_id === indicator.observation.id),
    );
  const preparedRows = (construct: ConstructSpec) =>
    step === "prepare_data"
      ? (indicatorsByConstruct.get(construct.id) ?? []).flatMap((indicator) => {
          const profile = modelSnapshot.profile?.indicators[indicator.observation.id]?.profile;
          return profile ? [{ indicator, profile }] : [];
        })
      : [];
  const assignmentsOf = (construct: ConstructSpec) =>
    simulationResult?.evidence.assignments.filter((event) => event.target === construct.id) ?? [];
  const markersOf = (assignments: readonly StateAssignment[]) =>
    simulationResult
      ? assignments.map((event) => ({
          time: event.time,
          label: assignmentLabel(event, simulationResult.evidence.time_origin),
        }))
      : [];
  const selectIndicator = (id: IndicatorSpec["observation"]["id"]) => () =>
    onSelect({ kind: "indicator", id });

  /** The construct's own column: its prepared data, simulated draws or laws, then its readings. */
  const stateColumn = (construct: ConstructSpec, assignments: readonly StateAssignment[]) => {
    const nodeIndicators = indicatorsByConstruct.get(construct.id) ?? [];
    const series = simulationPaths?.states[construct.id];
    const statePlot = flows?.constructs.get(construct.id);
    const prepared = preparedRows(construct);
    return (
      <>
        {dataDiff ? (
          <>
            {nodeIndicators.slice(0, STRIP_ROWS).map((indicator, index) => {
              const variable = dataDiff.variables.find(
                (item) => item.indicator_id === indicator.observation.id,
              );
              const evaluation = variable?.predictive?.evaluation ?? null;
              const failures = (
                evaluation && !("kind" in evaluation) ? evaluation.findings : []
              ).filter((finding) => finding.kind !== "evaluated" || finding.outcome !== "passed");
              const select = selectIndicator(indicator.observation.id);
              return (
                <g
                  key={indicator.observation.id}
                  role="button"
                  tabIndex={0}
                  aria-label={`${humanize(indicator.observation.name)} comparison`}
                  onClick={(event) => {
                    event.stopPropagation();
                    select();
                  }}
                  onKeyDown={(event) => {
                    event.stopPropagation();
                    activateOnKeyboard(event, select);
                  }}
                >
                  <StripRow
                    geometry={geometry}
                    index={index}
                    title={
                      failures
                        .map((finding) =>
                          finding.kind === "evaluated" ? finding.evidence.note : finding.detail,
                        )
                        .join("; ") || humanize(indicator.observation.name)
                    }
                    label={humanize(indicator.observation.name)}
                    value={failures.length ? "⚠" : ""}
                    valueTone="var(--warning-foreground)"
                  >
                    {variable && (
                      <foreignObject
                        x={geometry.strip.x + geometry.row.plot}
                        y={rowTop(geometry, index)}
                        width={geometry.strip.width - geometry.row.plot}
                        height={geometry.row.height - 2}
                        pointerEvents="none"
                      >
                        <DrawsChart
                          {...dataComparisonChart(variable)}
                          height={geometry.row.height - 2}
                          resolution={zoom}
                          compact
                        />
                      </foreignObject>
                    )}
                  </StripRow>
                </g>
              );
            })}
            <StripOverflow geometry={geometry} hidden={nodeIndicators.length - STRIP_ROWS} />
          </>
        ) : series ? (
          <>
            <CardMarkers
              box={geometry.strip}
              times={simulationPaths.times}
              markers={markersOf(assignments)}
            />
            <foreignObject
              x={geometry.strip.x}
              y={geometry.strip.top}
              width={geometry.strip.width}
              height={geometry.strip.height}
              pointerEvents="none"
            >
              <DrawsChart
                compact
                height={geometry.strip.height}
                resolution={zoom}
                times={simulationPaths.times}
                timeOrigin={simulationPaths.time_origin}
                layers={pathLayers(series, "both", false)}
                frame={series.frame}
                levels={series.levels}
                label={`${humanize(series.label)}: ${series.action.length} individual simulation draws`}
              />
            </foreignObject>
          </>
        ) : prepared.length > 0 ? (
          <DataStrip geometry={geometry} rows={prepared} />
        ) : (
          <foreignObject
            x={geometry.strip.x}
            y={geometry.strip.top}
            width={geometry.strip.width}
            height={geometry.strip.height}
            pointerEvents="none"
          >
            {statePlot && (
              <FlowChart
                plot={statePlot}
                height={geometry.strip.height}
                compact
                resolution={zoom}
              />
            )}
          </foreignObject>
        )}
        {!dataDiff &&
          prepared.length === 0 &&
          nodeIndicators.slice(0, 2).map((indicator, index) => {
            const id = indicator.observation.id;
            const readings = simulationPaths?.indicators[id];
            const plot = flows?.indicators.get(id);
            const select = selectIndicator(id);
            const top = geometry.indicator.top + index * geometry.indicator.height;
            return (
              <g
                key={id}
                role="button"
                tabIndex={0}
                aria-label={`${humanize(indicator.observation.name)} reading law`}
                onClick={(event) => {
                  event.stopPropagation();
                  select();
                }}
                onKeyDown={(event) => {
                  event.stopPropagation();
                  activateOnKeyboard(event, select);
                }}
              >
                <title>
                  {readings
                    ? `${humanize(readings.label)}: saved simulation readings`
                    : plot
                      ? `${plot.label}: ${plot.kind === "draws" ? plot.note : plot.reason}`
                      : indicator.observation.name}
                </title>
                <text
                  x={geometry.strip.x}
                  y={top + Math.round(geometry.indicator.height * 0.52)}
                  fontSize={7}
                  fill={DAG_COLORS.slate}
                >
                  {truncate(humanize(indicator.observation.name), geometry.indicator.length)}
                </text>
                <foreignObject
                  x={geometry.indicator.chart}
                  y={top}
                  width={geometry.width - geometry.strip.x - geometry.indicator.chart}
                  height={geometry.indicator.height - 3}
                  pointerEvents="none"
                >
                  {readings ? (
                    <DrawsChart
                      compact
                      height={geometry.indicator.height - 3}
                      resolution={zoom}
                      times={simulationPaths.times}
                      timeOrigin={simulationPaths.time_origin}
                      layers={pathLayers(readings, "both", true)}
                      levels={readings.levels}
                      label={`${humanize(readings.label)}: saved readings`}
                    />
                  ) : plot ? (
                    <FlowChart
                      plot={plot}
                      height={geometry.indicator.height - 3}
                      compact
                      resolution={zoom}
                    />
                  ) : null}
                </foreignObject>
              </g>
            );
          })}
        {!dataDiff && prepared.length === 0 && (
          <StripOverflow geometry={geometry} hidden={nodeIndicators.length - 2} />
        )}
      </>
    );
  };

  /** An exogenous construct's one chart, chosen as its card's strip would be. */
  const chipChart = (
    construct: ConstructSpec,
    assignments: readonly StateAssignment[],
    box: Box,
  ) => {
    const nodeIndicators = indicatorsByConstruct.get(construct.id) ?? [];
    const series = simulationPaths?.states[construct.id];
    const statePlot = flows?.constructs.get(construct.id);
    const prepared = preparedRows(construct);
    const frame = (content: ReactNode) => (
      <foreignObject
        x={box.x}
        y={box.top}
        width={box.width}
        height={box.height}
        pointerEvents="none"
      >
        {content}
      </foreignObject>
    );
    if (dataDiff) {
      const shown = nodeIndicators.flatMap((indicator) => {
        const variable = dataDiff.variables.find(
          (item) => item.indicator_id === indicator.observation.id,
        );
        return variable ? [{ indicator, variable }] : [];
      });
      const first = shown[0];
      if (!first) return null;
      const select = selectIndicator(first.indicator.observation.id);
      return (
        <g
          role="button"
          tabIndex={0}
          aria-label={`${humanize(first.indicator.observation.name)} comparison`}
          onClick={(event) => {
            event.stopPropagation();
            select();
          }}
          onKeyDown={(event) => {
            event.stopPropagation();
            activateOnKeyboard(event, select);
          }}
        >
          <title>
            {shown.map(({ indicator }) => humanize(indicator.observation.name)).join(", ")}
          </title>
          {frame(
            <DrawsChart
              {...dataComparisonChart(first.variable)}
              height={box.height}
              resolution={zoom}
              compact
            />,
          )}
        </g>
      );
    }
    if (series)
      return (
        <>
          <CardMarkers box={box} times={simulationPaths.times} markers={markersOf(assignments)} />
          {frame(
            <DrawsChart
              compact
              height={box.height}
              resolution={zoom}
              times={simulationPaths.times}
              timeOrigin={simulationPaths.time_origin}
              layers={pathLayers(series, "both", false)}
              frame={series.frame}
              levels={series.levels}
              label={`${humanize(series.label)}: ${series.action.length} individual simulation draws`}
            />,
          )}
        </>
      );
    const first = prepared[0];
    if (first) {
      const domain = profileDomain(first.profile, []);
      return (
        <g role="img" aria-label={humanize(first.indicator.observation.name)}>
          <title>
            {prepared
              .map(
                ({ indicator, profile }) =>
                  `${humanize(indicator.observation.name)} · ${profileSummary(profile)}`,
              )
              .join("\n")}
          </title>
          <rect x={box.x} y={box.top} width={box.width} height={box.height} fill="transparent" />
          {domain && (
            <ProfileMarks
              profile={first.profile}
              values={[]}
              x={linearScale(padDomain(domain, 0.02), [box.x, box.x + box.width])}
              top={box.top}
              height={box.height}
            />
          )}
        </g>
      );
    }
    return statePlot
      ? frame(<FlowChart plot={statePlot} height={box.height} compact resolution={zoom} />)
      : null;
  };

  const canvas = (
    <DagCanvasFrame>
      {isLayouting ? (
        <div className="h-full animate-pulse rounded-xl bg-slate-100" />
      ) : (
        <DagSvg
          contentWidth={difference.width + CANVAS_PADDING * 2}
          contentHeight={difference.height + CANVAS_PADDING * 2}
          zoom={zoom}
          style={{
            marginLeft: Math.max(0, (pane.width - 14 - (width + CANVAS_PADDING * 2) * zoom) / 2),
            overflow: "visible",
          }}
          role="img"
          aria-label="Layered causal graph"
        >
          <g transform={`translate(${CANVAS_PADDING},${CANVAS_PADDING})`}>
            {arrows.map((arrow) => {
              const meta = topology.arrowMeta.get(arrow.id);
              if (!meta) return null;
              const visual = edgeVisual(meta);
              const { edge } = meta;
              const first = !firstArrow.has(edge.id);
              firstArrow.add(edge.id);
              const select = () =>
                onSelect(
                  selection?.kind === "edge" && selection.id === edge.id
                    ? null
                    : { kind: "edge", id: edge.id },
                );
              return (
                <g
                  key={arrow.id}
                  data-edge-id={edge.id}
                  data-change={visual.change}
                  role="button"
                  tabIndex={first ? 0 : -1}
                  aria-label={`${name(meta.cause.id)} → ${name(meta.effect.id)}`}
                  onClick={select}
                  onKeyDown={(event) => activateOnKeyboard(event, select)}
                >
                  <DagEdge
                    points={arrow.points}
                    color={visual.color}
                    width={visual.width}
                    opacity={dataDiff ? 0.15 : visual.opacity}
                    highlighted={visual.highlighted}
                    onHoverChange={(hovered) =>
                      setHovered(hovered ? { kind: "edge", id: edge.id } : null)
                    }
                  />
                </g>
              );
            })}

            {nodes.map((node) => {
              const meta = topology.nodeMeta.get(node.id);
              if (!meta) return null;
              const { construct } = meta;
              const constructChange = difference.constructChanges.get(construct.id);
              const dimmed =
                (focused !== null && !focused.constructs.has(construct.id)) ||
                (dataDiff !== null && !compared(construct));
              const selected = selectedNode === construct.id;
              const select = () =>
                onSelect(selected ? null : { kind: "construct", id: construct.id });
              const status = nodeStatuses.get(construct.id) ?? undefined;
              const assignments = assignmentsOf(construct);
              const failures = dataDiff ? [] : entityFailures(modelSnapshot, construct);
              const outline = constructChange !== undefined && (
                <rect
                  x={-3}
                  y={-3}
                  width={node.width + 6}
                  height={node.height + 6}
                  rx={12}
                  fill="none"
                  stroke={COMPARISON_COLORS[constructChange]}
                  strokeWidth={2 / zoom}
                />
              );

              if (meta.kind === "exogenous") {
                const rich = density.key === "rich";
                const { width: chipWidth, height: chipHeight } = density.chip;
                const size = rich ? 10.5 : 9.5;
                const chart = rich ? 72 : 56;
                const chartX = chipWidth - 9 - (failures.length > 0 ? 15 : 0) - chart;
                const badge = assignments.length > 0 ? String(assignments.length) : null;
                const labelEnd =
                  (badge ? chartX - 6 - pillWidth(badge, CHIP_PILL_WIDTH) : chartX) - 6;
                const carry = construct.temporal_status === "time_varying" ? 14 : 0;
                const label = truncate(
                  humanize(construct.name),
                  Math.max(4, Math.floor((labelEnd - 10 - carry) / (size * 0.56))),
                );
                const accent = selected
                  ? "var(--primary)"
                  : badge
                    ? DAG_COLORS.intervention
                    : (statusAccent(status) ?? "var(--border)");
                return (
                  <g
                    key={node.id}
                    data-node-id={node.id}
                    data-change={constructChange}
                    transform={`translate(${node.x},${node.y})`}
                  >
                    <g
                      opacity={dimmed ? 0.18 : 1}
                      role="button"
                      aria-label={humanize(construct.name)}
                      tabIndex={0}
                      style={{ cursor: "pointer" }}
                      onClick={select}
                      onKeyDown={(event) => activateOnKeyboard(event, select)}
                    >
                      <rect
                        width={chipWidth}
                        height={chipHeight}
                        rx={8}
                        fill={DAG_COLORS.col}
                        stroke={accent}
                        strokeWidth={selected ? 2.4 : 1.2}
                      />
                      <text
                        x={10}
                        y={chipHeight / 2 + size * 0.36}
                        fontSize={size}
                        fontWeight={600}
                        fill="var(--foreground)"
                      >
                        {label}
                        <CarryMark construct={construct} size={size - 1} />
                      </text>
                      {badge && simulationResult && (
                        <LayerPill
                          x={chartX - 6}
                          y={chipHeight / 2 - 8.5}
                          label={badge}
                          color={DAG_COLORS.intervention}
                          minWidth={CHIP_PILL_WIDTH}
                          description={`${badge} assignment${assignments.length === 1 ? "" : "s"}: ${assignments
                            .map((event) =>
                              assignmentLabel(event, simulationResult.evidence.time_origin),
                            )
                            .join("; ")}`}
                        />
                      )}
                      {chipChart(construct, assignments, {
                        x: chartX,
                        top: 5,
                        width: chart,
                        height: chipHeight - 10,
                      })}
                      <WarningMark x={chipWidth - 21} y={chipHeight / 2 - 6} failures={failures} />
                    </g>
                    {outline}
                  </g>
                );
              }

              const rowsWidth = meta.rows.length > 0 ? density.row.width : 0;
              const rows = meta.rows.map((row) => {
                const owner = row.owner;
                const edge = owner.kind === "edge" ? entities.edgeById.get(owner.id) : undefined;
                const plot = flows?.mechanisms.get(row.mechanism.id);
                const rowName = mechanismName(row.mechanism.id);
                const reads = row.reads.map(name);
                const description = [
                  `${rowName} (${row.mechanism.kind}) · ${
                    edge
                      ? `on ${name(edge.cause.id)} → ${name(edge.effect.id)}`
                      : `${name(construct.id)}'s own dynamics`
                  }`,
                  reads.length > 0 ? `Reads ${reads.join(", ")}` : "Reads no state",
                  ...(plot
                    ? [`${plot.label}: ${plot.kind === "draws" ? plot.note : plot.reason}`]
                    : []),
                ].join("\n");
                return (
                  <MechanismRowView
                    key={row.mechanism.id}
                    row={row}
                    density={density}
                    plot={plot}
                    description={description}
                    failures={edge && !dataDiff ? entityFailures(modelSnapshot, edge) : []}
                    zoom={zoom}
                    selected={selection?.kind === "edge" && selection.id === owner.id}
                    dimmed={
                      dataDiff !== null ||
                      (focused !== null && !focused.mechanisms.has(row.mechanism.id))
                    }
                    onSelect={() =>
                      onSelect(
                        selection?.kind === owner.kind && selection.id === owner.id ? null : owner,
                      )
                    }
                    onHover={(hovered) =>
                      setHovered(hovered ? { kind: "mechanism", id: row.mechanism.id } : null)
                    }
                  />
                );
              });
              return (
                <g
                  key={node.id}
                  data-node-id={node.id}
                  data-change={constructChange}
                  transform={`translate(${node.x},${node.y})`}
                >
                  <ConstructCard
                    construct={construct}
                    width={meta.width}
                    height={meta.height}
                    rowsWidth={rowsWidth}
                    geometry={geometry}
                    isOutcome={construct.id === outcome}
                    failures={failures}
                    status={status}
                    assignments={assignments}
                    timeOrigin={simulationResult?.evidence.time_origin ?? null}
                    selected={selected}
                    dimmed={dimmed}
                    onSelect={select}
                    rows={rows}
                  >
                    {stateColumn(construct, assignments)}
                  </ConstructCard>
                  {outline}
                </g>
              );
            })}
            {comparison && <LayeredComparisonOverlay overlay={difference} zoom={zoom} />}
          </g>
        </DagSvg>
      )}
    </DagCanvasFrame>
  );

  return (
    <div ref={paneRef} className="relative flex h-full min-h-0 flex-col">
      {canvas}
      <div className="absolute right-3 bottom-2 flex items-center gap-1.5 rounded-lg bg-white/85 px-1.5 py-0.5">
        <DagZoomControls zoom={zoom} onZoomChange={setZoom} onFit={fitToPane} />
      </div>
    </div>
  );
}
