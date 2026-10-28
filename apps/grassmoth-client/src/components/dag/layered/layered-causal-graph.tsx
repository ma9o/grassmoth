"use client";

import { causalEffect } from "@/lib/simulation-report";

import type { SimulationPathsView } from "@/lib/model-asset/result-values";
import type { DataComparisonView } from "@/lib/model-asset/data-comparison";

import type {
  ConstructSpec,
  IndicatorEmpiricalProfile,
  IndicatorSpec,
  StateAssignment,
  ActionId,
} from "@grassmoth/api-types";
import { Pause, Play } from "lucide-react";
import type { KeyboardEvent, ReactNode } from "react";
import { DrawsChart } from "@/components/charts/draws-chart";
import { CHART_COLORS, cssColor } from "@/components/charts/chart-tokens";
import { extentOf, linearScale, padDomain } from "@/components/charts/plot-geometry";
import { ProfileMarks, profileDomain, profileSummary } from "@/components/charts/profile-strip";
import { dataComparisonChart, pathLayers } from "@/components/charts/series-adapters";
import { Button } from "@/components/ui/button";
import {
  LAYERED_EDGE_SLOT_HEIGHT,
  LAYERED_EDGE_SLOT_WIDTH,
  LAYERED_HISTORY_HEIGHT,
  LAYERED_HISTORY_WIDTH,
  LAYERED_NODE_HEIGHT,
  LAYERED_NODE_WIDTH,
  type LayeredGraphEdgeMeta,
} from "@/lib/dag/build-layered-causal-graph";
import type { ConstructStatus } from "@/lib/dag/construct-statuses";
import { boundsForBand, type CausalGraphLayerId } from "@/lib/dag/layered-model";
import { BLOCKING, COMPARISON_COLORS, DAG_COLORS, LATENT } from "@/lib/dag/palette";
import { type LayeredGraphOptions, useLayeredGraph } from "@/lib/dag/use-layered-graph";
import { entityFailures } from "@/lib/model-asset/inspector";
import type { EntitySelection } from "@/lib/model-asset/selection";
import { FlowChart } from "@/components/charts/flow-chart";
import { modelFlowPlots, type FlowPlot } from "@/lib/model-asset/flow-plots";
import { formatModelDate, formatSignificant } from "@/lib/utils/format";
import { DagCanvasFrame, DagSvg } from "../core/dag-canvas";
import { DagEdge } from "../core/dag-edge";
import { DagNodeShell } from "../core/dag-node";
import { DagZoomControls } from "../core/dag-zoom-controls";
import { LayeredComparisonOverlay } from "./layered-comparison-overlay";

const CANVAS_PADDING = 36;

const LAYER_LABELS: Record<CausalGraphLayerId, string> = {
  structure: "Structure",
  measurement: "Measurement",
  design: "Design",
  specification: "Specification",
  fit: "Fit",
  simulation: "Simulation",
};

export type LayeredCausalGraphVariant = "workbench" | "asset";

export interface LayeredCausalGraphProps extends LayeredGraphOptions {
  simulationPaths?: SimulationPathsView | null;
  dataDiff?: DataComparisonView | null;
  onSelect: (selection: EntitySelection | null) => void;
  /** The action whose version is viewed; a data preparation shows each node's prepared data. */
  step?: ActionId | null;
  /**
   * `workbench` keeps the layer toggles, simulation timeline and legend around the canvas;
   * `asset` renders the canvas alone, filling its container, with only the zoom control.
   */
  variant?: LayeredCausalGraphVariant;
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
  label,
  color,
  description,
}: {
  x: number;
  label: string;
  color: string;
  description?: string;
}) {
  const width = Math.max(38, label.length * 5.3 + 14);
  return (
    <g transform={`translate(${x - width},8)`} role="img" aria-label={description ?? label}>
      <title>{description ?? label}</title>
      <rect width={width} height={17} rx={8.5} fill={color} fillOpacity={0.11} />
      <text x={width / 2} y={11.5} textAnchor="middle" fontSize={7.5} fontWeight={650} fill={color}>
        {label}
      </text>
    </g>
  );
}

function assignmentLabel(event: StateAssignment, timeOrigin: string): string {
  return `${formatModelDate(event.time, timeOrigin)} (day ${event.time}): set to ${event.value}`;
}

/** The card's chart strip, below its title. */
const STRIP = { x: 14, top: 36, width: LAYERED_NODE_WIDTH - 28, height: 64 } as const;
/** One row per law or indicator: label, value, then its chart. */
const STRIP_ROW = { height: 21, value: 108, plot: 114 } as const;
const STRIP_ROWS = 3;
const rowTop = (index: number) => STRIP.top + index * STRIP_ROW.height;

/** One labelled row of a card's chart strip; its chart sits right of the value. */
function StripRow({
  index,
  title,
  label,
  value,
  valueTone,
  range,
  children,
}: {
  index: number;
  title: string;
  label: string;
  value: string;
  valueTone: string;
  /** The chart's value range, labelled under its ends. */
  range?: readonly [number, number];
  children: ReactNode;
}) {
  const top = rowTop(index);
  const bottom = top + STRIP_ROW.height - 0.5;
  return (
    <g role="img" aria-label={title}>
      <title>{title}</title>
      <rect x={STRIP.x} y={top} width={STRIP.width} height={STRIP_ROW.height} fill="transparent" />
      <text x={STRIP.x} y={top + 11} fontSize={7.5} fontWeight={600} fill={DAG_COLORS.slate}>
        {truncate(label, 17)}
      </text>
      <text
        x={STRIP.x + STRIP_ROW.value}
        y={top + 11}
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
          <text x={STRIP.x + STRIP_ROW.plot} y={bottom} fontSize={5.5} fill={DAG_COLORS.muted}>
            {formatSignificant(range[0])}
          </text>
          <text
            x={STRIP.x + STRIP.width}
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
function StripOverflow({ hidden }: { hidden: number }) {
  return hidden > 0 ? (
    <text
      x={LAYERED_NODE_WIDTH - 8}
      y={LAYERED_NODE_HEIGHT - 4}
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
  times,
  markers,
}: {
  times: readonly number[];
  markers: readonly { time: number; label: string }[];
}) {
  const domain = extentOf(times);
  if (!domain || markers.length === 0) return null;
  const x = linearScale(domain[0] === domain[1] ? padDomain(domain) : domain, [
    STRIP.x + 1,
    STRIP.x + STRIP.width - 1,
  ]);
  return (
    <g pointerEvents="none">
      {markers.map((marker) => (
        <g key={`${marker.time}-${marker.label}`} role="img" aria-label={marker.label}>
          <title>{marker.label}</title>
          <line
            x1={x(marker.time)}
            x2={x(marker.time)}
            y1={STRIP.top}
            y2={STRIP.top + STRIP.height}
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
  rows,
}: {
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
                  STRIP.x + STRIP_ROW.plot,
                  STRIP.x + STRIP.width,
                ])}
                top={rowTop(index) + 2}
                height={12}
              />
            )}
          </StripRow>
        );
      })}
      <StripOverflow hidden={rows.length - shown.length} />
    </g>
  );
}

function ConstructCard({
  construct,
  isOutcome,
  failures,
  status,
  children,
  assignments,
  timeOrigin,
  selected,
  dimmed,
  onSelect,
}: {
  construct: ConstructSpec;
  isOutcome: boolean;
  failures: string[];
  status: ConstructStatus | undefined;
  children: ReactNode;
  assignments: readonly StateAssignment[];
  timeOrigin: string | null;
  selected: boolean;
  dimmed: boolean;
  onSelect: () => void;
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
        width={LAYERED_NODE_WIDTH}
        height={LAYERED_NODE_HEIGHT}
        title={`${isOutcome ? "★ " : ""}${truncate(humanize(construct.name), 29)}`}
        {...(shellAccent === undefined ? {} : { accent: shellAccent })}
        dashed={status === "latent"}
        highlighted={selected}
        outcome={isOutcome}
      >
        {badge ? (
          <LayerPill
            x={LAYERED_NODE_WIDTH - 8}
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
        {failures.length > 0 ? (
          <g role="img" aria-label={failures.join("; ")}>
            <title>{failures.join("\n")}</title>
            <path
              d="M235 37 L241 26 L247 37 Z"
              fill="none"
              stroke="var(--warning-foreground)"
              strokeWidth={1.2}
            />
            <text x={241} y={35} textAnchor="middle" fontSize={8} fill="var(--warning-foreground)">
              !
            </text>
          </g>
        ) : null}
      </DagNodeShell>
    </g>
  );
}

function HistoryCard({
  construct,
  status,
  dimmed,
  selected,
  onSelect,
}: {
  construct: ConstructSpec;
  status: ConstructStatus | undefined;
  dimmed: boolean;
  selected: boolean;
  onSelect: () => void;
}) {
  const accent = selected ? "var(--primary)" : statusAccent(status);
  return (
    <g
      opacity={dimmed ? 0.13 : 0.45}
      role="button"
      aria-label={`${humanize(construct.name)} at the previous time`}
      tabIndex={0}
      style={{ cursor: "pointer" }}
      onClick={onSelect}
      onKeyDown={(event) => activateOnKeyboard(event, onSelect)}
    >
      <DagNodeShell
        width={LAYERED_HISTORY_WIDTH}
        height={LAYERED_HISTORY_HEIGHT}
        title={truncate(humanize(construct.name), 20)}
        {...(accent === undefined ? {} : { accent })}
        dashed
        highlighted={selected}
      />
    </g>
  );
}

function EdgeSlot({
  meta,
  plot,
  color,
  dimmed,
  simulation,
}: {
  meta: LayeredGraphEdgeMeta;
  plot: FlowPlot | undefined;
  color: string;
  dimmed: boolean;
  simulation: boolean;
}) {
  if (!plot)
    return (
      <line
        x1={0}
        x2={LAYERED_EDGE_SLOT_WIDTH}
        y1={LAYERED_EDGE_SLOT_HEIGHT / 2}
        y2={LAYERED_EDGE_SLOT_HEIGHT / 2}
        stroke={color}
        strokeWidth={1.2}
        opacity={dimmed ? 0.12 : 1}
      />
    );
  return (
    <g opacity={dimmed ? 0.12 : 1}>
      <title>{`${plot.label}: ${plot.kind === "draws" ? plot.note : plot.reason}`}</title>
      <rect
        width={LAYERED_EDGE_SLOT_WIDTH}
        height={LAYERED_EDGE_SLOT_HEIGHT}
        rx={8}
        fill="var(--card)"
        stroke={color}
        strokeOpacity={0.55}
      />
      <foreignObject
        x={6}
        y={4}
        width={LAYERED_EDGE_SLOT_WIDTH - 12}
        height={LAYERED_EDGE_SLOT_HEIGHT - 18}
        pointerEvents="none"
      >
        <FlowChart plot={plot} height={LAYERED_EDGE_SLOT_HEIGHT - 18} compact />
      </foreignObject>
      <text
        x={LAYERED_EDGE_SLOT_WIDTH / 2}
        y={LAYERED_EDGE_SLOT_HEIGHT - 5}
        textAnchor="middle"
        fontSize={7}
        fill={DAG_COLORS.slate}
      >
        {simulation ? "Model drift" : meta.isSelf ? "Intrinsic drift" : "Drift contribution"}
      </text>
    </g>
  );
}

function LayerControls({
  available,
  visible,
  onToggle,
}: {
  available: CausalGraphLayerId[];
  visible: ReadonlySet<CausalGraphLayerId>;
  onToggle: (layer: CausalGraphLayerId) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-1.5" aria-label="Causal graph layers">
      {available.map((layer, index) => {
        const active = visible.has(layer);
        return (
          <button
            key={layer}
            type="button"
            disabled={layer === "structure"}
            aria-pressed={active}
            onClick={() => onToggle(layer)}
            className={`rounded-full border px-2.5 py-1 text-[10px] font-medium transition-colors ${
              active
                ? "border-slate-500 bg-slate-800 text-white"
                : "border-slate-200 bg-white text-slate-400"
            } disabled:cursor-default disabled:opacity-100`}
          >
            {`${index + 1} · ${LAYER_LABELS[layer]}`}
          </button>
        );
      })}
    </div>
  );
}

/**
 * The state the selected action left, one part at a glance. Each construct and edge shows its
 * current laws or posteriors, its prepared data or its simulated draws, and a single mark when a
 * check on it failed. The details pane holds the depth. A data_diff leaf overlays every saved
 * replicate and the observations on indicator rows, or marks added, removed and revised
 * observations in comparison colours. Latent-only constructs and edges recede.
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
  variant = "workbench",
}: LayeredCausalGraphProps) {
  const {
    available,
    visible,
    hoveredEdge,
    setHoveredEdge,
    zoom,
    setZoom,
    paneRef,
    paneWidth,
    topology,
    nodes,
    routedSegments,
    width,
    isLayouting,
    difference,
    designVisible,
    fitVisible,
    simulationVisible,
    nodeStatuses,
    indicatorsByConstruct,
    simulationResult,
    days,
    clampedDayIndex,
    setDayIndex,
    playing,
    setPlaying,
    currentDay,
    selectedNeighborhood,
    graphBands,
    toggleLayer,
    edgeVisual,
  } = useLayeredGraph({ modelSnapshot, entities, simulation, comparison, selection });
  const selectedNode = selection?.kind === "construct" ? selection.id : null;
  const flows =
    modelSnapshot.dynamical_model_spec && (visible.has("specification") || visible.has("fit"))
      ? modelFlowPlots(modelSnapshot.dynamical_model_spec)
      : null;

  const canvas = (
    <DagCanvasFrame fill={variant === "asset"}>
      {isLayouting ? (
        <div
          className={`animate-pulse rounded-xl bg-slate-100 ${variant === "asset" ? "h-full" : "h-[560px]"}`}
        />
      ) : (
        <DagSvg
          contentWidth={difference.width + CANVAS_PADDING * 2}
          contentHeight={difference.height + CANVAS_PADDING * 2}
          zoom={zoom}
          style={
            variant === "asset"
              ? {
                  marginLeft: Math.max(
                    0,
                    (paneWidth - 14 - (width + CANVAS_PADDING * 2) * zoom) / 2,
                  ),
                  overflow: "visible",
                }
              : undefined
          }
          role="img"
          aria-label="Layered causal graph"
        >
          <g transform={`translate(${CANVAS_PADDING},${CANVAS_PADDING})`}>
            {graphBands.map((band) => {
              const bounds = boundsForBand(band);
              if (!bounds) return null;
              return (
                <g key={band.key}>
                  <rect
                    {...bounds}
                    rx={14}
                    fill={band.key === "present" ? "#f8fafc" : "#fbfcfd"}
                    stroke="#e7ebef"
                    strokeDasharray={band.key === "history" ? "5,4" : undefined}
                  />
                  <text
                    x={bounds.x + 10}
                    y={bounds.y + 16}
                    fontSize={8}
                    fontWeight={700}
                    letterSpacing={0.7}
                    fill={DAG_COLORS.muted}
                  >
                    {band.label.toUpperCase()}
                  </text>
                </g>
              );
            })}

            {routedSegments.map((segment) => {
              const segmentMeta = topology.segmentMeta.get(segment.id);
              if (!segmentMeta) return null;
              const meta = topology.edgeMeta.get(segmentMeta.edgeId);
              if (!meta) return null;
              const visual = edgeVisual(meta);
              return (
                <g key={segment.id} data-edge-id={meta.id} data-change={visual.change}>
                  <DagEdge
                    points={segment.points}
                    color={visual.color}
                    width={visual.width}
                    opacity={dataDiff ? 0.15 : visual.opacity}
                    markerEnd={segmentMeta.markerEnd}
                    highlighted={hoveredEdge === meta.id}
                    onHoverChange={(hovered) => setHoveredEdge(hovered ? meta.id : null)}
                  />
                </g>
              );
            })}

            {nodes.map((node) => {
              const meta = topology.nodeMeta.get(node.id);
              if (!meta) return null;
              if (meta.kind === "edge_slot") {
                const edge = meta.edge;
                const visual = edgeVisual(edge);
                const owner = entities.edges.find((item) => item.id === edge.id);
                const target: EntitySelection = edge.isSelf
                  ? { kind: "construct", id: edge.cause.id }
                  : { kind: "edge", id: edge.id };
                const select = () => onSelect(selection?.id === target.id ? null : target);
                const failures = owner && !dataDiff ? entityFailures(modelSnapshot, owner) : [];
                return (
                  <g
                    key={node.id}
                    data-node-id={node.id}
                    transform={`translate(${node.x},${node.y})`}
                    role="button"
                    aria-label={
                      edge.isSelf
                        ? `${humanize(edge.cause.name)} intrinsic dynamics`
                        : `${humanize(edge.cause.name)} → ${humanize(edge.effect.name)}`
                    }
                    tabIndex={0}
                    style={{ cursor: "pointer" }}
                    onClick={select}
                    onKeyDown={(event) => activateOnKeyboard(event, select)}
                    onPointerEnter={() => setHoveredEdge(edge.id)}
                    onPointerLeave={() => setHoveredEdge(null)}
                  >
                    <rect
                      width={LAYERED_EDGE_SLOT_WIDTH}
                      height={LAYERED_EDGE_SLOT_HEIGHT}
                      fill="transparent"
                    />
                    <EdgeSlot
                      meta={edge}
                      plot={
                        edge.isSelf
                          ? flows?.intrinsic.get(edge.cause.id)
                          : flows?.edges.get(edge.id)
                      }
                      color={visual.color}
                      dimmed={dataDiff !== null || visual.dimmed}
                      simulation={step === "simulate"}
                    />
                    {failures.length > 0 && (
                      <g role="img" aria-label={failures.join("; ")}>
                        <title>{failures.join("\n")}</title>
                        <path
                          d="M79 13 L85 2 L91 13 Z"
                          fill="var(--card)"
                          stroke="var(--warning-foreground)"
                        />
                        <text
                          x={85}
                          y={11}
                          textAnchor="middle"
                          fontSize={8}
                          fill="var(--warning-foreground)"
                        >
                          !
                        </text>
                      </g>
                    )}
                  </g>
                );
              }

              const construct = meta.construct;
              const constructChange = difference.constructChanges.get(construct.id);
              const dimmed =
                (selectedNeighborhood != null && !selectedNeighborhood.has(construct.id)) ||
                (dataDiff !== null &&
                  (meta.kind === "history" ||
                    !construct.indicators.some((indicator) =>
                      dataDiff.variables.some(
                        (variable) => variable.indicator_id === indicator.observation.id,
                      ),
                    )));
              const selected = selectedNode === construct.id;
              const select = () =>
                onSelect(
                  selectedNode === construct.id ? null : { kind: "construct", id: construct.id },
                );
              if (meta.kind === "history") {
                return (
                  <g
                    key={node.id}
                    data-node-id={node.id}
                    transform={`translate(${node.x},${node.y})`}
                  >
                    <HistoryCard
                      construct={construct}
                      status={nodeStatuses.get(construct.id) ?? undefined}
                      dimmed={dimmed}
                      selected={selected}
                      onSelect={select}
                    />
                  </g>
                );
              }

              const nodeIndicators = indicatorsByConstruct.get(construct.id) ?? [];
              const series = simulationPaths?.states[construct.id];
              const statePlot = flows?.constructs.get(construct.id);
              const prepared =
                step === "prepare_data"
                  ? nodeIndicators.flatMap((indicator) => {
                      const profile =
                        modelSnapshot.profile?.indicators[indicator.observation.id]?.profile;
                      return profile ? [{ indicator, profile }] : [];
                    })
                  : [];
              const assignments =
                simulationResult?.evidence.assignments.filter(
                  (event) => event.target === construct.id,
                ) ?? [];
              return (
                <g
                  key={node.id}
                  data-node-id={node.id}
                  data-change={difference.constructChanges.get(construct.id)}
                  transform={`translate(${node.x},${node.y})`}
                >
                  <ConstructCard
                    construct={construct}
                    isOutcome={
                      construct.id ===
                      (causalEffect(simulation)?.outcome ?? modelSnapshot.question?.outcome)
                    }
                    failures={dataDiff ? [] : entityFailures(modelSnapshot, construct)}
                    status={nodeStatuses.get(construct.id) ?? undefined}
                    assignments={assignments}
                    timeOrigin={simulationResult?.evidence.time_origin ?? null}
                    selected={selected}
                    dimmed={dimmed}
                    onSelect={select}
                  >
                    {dataDiff ? (
                      <>
                        {nodeIndicators.slice(0, STRIP_ROWS).map((indicator, index) => {
                          const variable = dataDiff.variables.find(
                            (item) => item.indicator_id === indicator.observation.id,
                          );
                          const evaluation = variable?.predictive?.evaluation ?? null;
                          const failures = (
                            evaluation && !("kind" in evaluation) ? evaluation.findings : []
                          ).filter(
                            (finding) =>
                              finding.kind !== "evaluated" || finding.outcome !== "passed",
                          );
                          const selectIndicator = () =>
                            onSelect({ kind: "indicator", id: indicator.observation.id });
                          return (
                            <g
                              key={indicator.observation.id}
                              role="button"
                              tabIndex={0}
                              aria-label={`${humanize(indicator.observation.name)} comparison`}
                              onClick={(event) => {
                                event.stopPropagation();
                                selectIndicator();
                              }}
                              onKeyDown={(event) => {
                                event.stopPropagation();
                                activateOnKeyboard(event, selectIndicator);
                              }}
                            >
                              <StripRow
                                index={index}
                                title={
                                  failures
                                    .map((finding) =>
                                      finding.kind === "evaluated"
                                        ? finding.evidence.note
                                        : finding.detail,
                                    )
                                    .join("; ") || humanize(indicator.observation.name)
                                }
                                label={humanize(indicator.observation.name)}
                                value={failures.length ? "⚠" : ""}
                                valueTone="var(--warning-foreground)"
                              >
                                {variable && (
                                  <foreignObject
                                    x={STRIP.x + STRIP_ROW.plot}
                                    y={rowTop(index)}
                                    width={STRIP.width - STRIP_ROW.plot}
                                    height={STRIP_ROW.height - 2}
                                    pointerEvents="none"
                                  >
                                    <DrawsChart
                                      {...dataComparisonChart(variable)}
                                      height={STRIP_ROW.height - 2}
                                      resolution={zoom}
                                      compact
                                    />
                                  </foreignObject>
                                )}
                              </StripRow>
                            </g>
                          );
                        })}
                        <StripOverflow hidden={nodeIndicators.length - STRIP_ROWS} />
                      </>
                    ) : series ? (
                      <>
                        <CardMarkers
                          times={simulationPaths.times}
                          markers={[
                            ...(simulationResult
                              ? assignments.map((event) => ({
                                  time: event.time,
                                  label: assignmentLabel(
                                    event,
                                    simulationResult.evidence.time_origin,
                                  ),
                                }))
                              : []),
                            ...(variant === "workbench" && currentDay !== undefined
                              ? [{ time: currentDay, label: `Viewed day ${currentDay}` }]
                              : []),
                          ]}
                        />
                        <foreignObject
                          x={STRIP.x}
                          y={STRIP.top}
                          width={STRIP.width}
                          height={STRIP.height}
                          pointerEvents="none"
                        >
                          <DrawsChart
                            compact
                            height={STRIP.height}
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
                      <DataStrip rows={prepared} />
                    ) : (
                      <foreignObject
                        x={STRIP.x}
                        y={STRIP.top}
                        width={STRIP.width}
                        height={STRIP.height}
                        pointerEvents="none"
                      >
                        {statePlot && (
                          <FlowChart
                            plot={statePlot}
                            height={STRIP.height}
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
                        const selectIndicator = () => onSelect({ kind: "indicator", id });
                        return (
                          <g
                            key={id}
                            role="button"
                            tabIndex={0}
                            aria-label={`${humanize(indicator.observation.name)} reading law`}
                            onClick={(event) => {
                              event.stopPropagation();
                              selectIndicator();
                            }}
                            onKeyDown={(event) => {
                              event.stopPropagation();
                              activateOnKeyboard(event, selectIndicator);
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
                              x={STRIP.x}
                              y={115 + index * 21}
                              fontSize={7}
                              fill={DAG_COLORS.slate}
                            >
                              {truncate(humanize(indicator.observation.name), 21)}
                            </text>
                            <foreignObject
                              x={130}
                              y={104 + index * 21}
                              width={106}
                              height={18}
                              pointerEvents="none"
                            >
                              {readings ? (
                                <DrawsChart
                                  compact
                                  height={18}
                                  resolution={zoom}
                                  times={simulationPaths.times}
                                  timeOrigin={simulationPaths.time_origin}
                                  layers={pathLayers(readings, "both", true)}
                                  levels={readings.levels}
                                  label={`${humanize(readings.label)}: saved readings`}
                                />
                              ) : plot ? (
                                <FlowChart plot={plot} height={18} compact resolution={zoom} />
                              ) : null}
                            </foreignObject>
                          </g>
                        );
                      })}
                    {!dataDiff && prepared.length === 0 && (
                      <StripOverflow hidden={nodeIndicators.length - 2} />
                    )}
                  </ConstructCard>
                  {constructChange !== undefined && (
                    <rect
                      x={-3}
                      y={-3}
                      width={LAYERED_NODE_WIDTH + 6}
                      height={LAYERED_NODE_HEIGHT + 6}
                      rx={12}
                      fill="none"
                      stroke={COMPARISON_COLORS[constructChange]}
                      strokeWidth={2 / zoom}
                    />
                  )}
                </g>
              );
            })}
            {comparison && <LayeredComparisonOverlay overlay={difference} zoom={zoom} />}
          </g>
        </DagSvg>
      )}
    </DagCanvasFrame>
  );

  if (variant === "asset") {
    return (
      <div ref={paneRef} className="relative flex h-full min-h-0 flex-col">
        {canvas}
        <div className="absolute right-3 bottom-2 flex items-center gap-1.5 rounded-lg bg-white/85 px-1.5 py-0.5">
          <DagZoomControls zoom={zoom} onZoomChange={setZoom} />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-0 space-y-3 bg-slate-50/60 p-3">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border bg-white px-3 py-2.5">
        <div>
          <div className="text-xs font-semibold text-slate-800">Causal model</div>
          <div className="text-[10px] text-muted-foreground">
            One structural topology; each materialized artifact adds one visual layer.
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <LayerControls available={available} visible={visible} onToggle={toggleLayer} />
          <DagZoomControls zoom={zoom} onZoomChange={setZoom} />
        </div>
      </div>

      {canvas}

      {simulationVisible && days.length > 0 ? (
        <div className="flex flex-wrap items-center gap-3 rounded-xl border bg-white px-3 py-2">
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="h-7 w-7 p-0"
            onClick={() => setPlaying((current) => !current)}
            aria-label={playing ? "Pause simulation timeline" : "Play simulation timeline"}
          >
            {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
          </Button>
          <input
            type="range"
            min={0}
            max={days.length - 1}
            value={clampedDayIndex}
            onChange={(event) => setDayIndex(Number(event.target.value))}
            className="min-w-52 flex-1 accent-blue-600"
            aria-label="Simulation day"
          />
          <span className="min-w-16 text-right font-mono text-[10px] text-slate-600">
            day {currentDay}
          </span>
          <span className="text-[10px] text-muted-foreground">
            gray reference · blue intervention
          </span>
        </div>
      ) : null}

      <div className="flex flex-wrap gap-x-4 gap-y-1 px-1 text-[9px] text-muted-foreground">
        <span>Dynamic causes connect successive state slices.</span>
        {simulationVisible ? <span>Dashed blue markers show dated assignments.</span> : null}
        {designVisible ? <span>Dashed edge slots are projected by design.</span> : null}
        {fitVisible ? <span>State and mechanism plots preserve joint draws.</span> : null}
      </div>
    </div>
  );
}
