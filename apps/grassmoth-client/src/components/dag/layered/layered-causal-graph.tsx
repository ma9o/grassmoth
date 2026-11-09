"use client";

import type { ActionId, ConstructSpec, IndicatorSpec, ModelSnapshot } from "@grassmoth/api-types";
import type { MechanismRow } from "@/lib/dag/build-layered-causal-graph";
import { COMPARISON_COLORS } from "@/lib/dag/palette";
import {
  CANVAS_PADDING,
  type LayeredGraphOptions,
  useLayeredGraph,
} from "@/lib/dag/use-layered-graph";
import type { DataComparisonView } from "@/lib/model-asset/data-comparison";
import { type FlowPlot, modelFlowPlots } from "@/lib/model-asset/flow-plots";
import { entityFailures } from "@/lib/model-asset/inspector";
import type { SimulationPathsView } from "@/lib/model-asset/result-values";
import { type EntitySelection, humanize, mechanismName } from "@/lib/model-asset/selection";
import { DagCanvasFrame, DagSvg } from "../core/dag-canvas";
import { DagEdge } from "../core/dag-edge";
import { DagZoomControls } from "../core/dag-zoom-controls";
import { type PreparedRow, stateGeometry } from "./layered-card-strip";
import { LayeredComparisonOverlay } from "./layered-comparison-overlay";
import { ConstructCard, MechanismRowView } from "./layered-construct-card";
import { ExogenousChip } from "./layered-exogenous-chip";
import { activateOnKeyboard, assignmentLabel, plotTitle } from "./layered-graph-marks";
import { type ConstructState, StateColumn, type StateView } from "./layered-state-column";

export interface LayeredCausalGraphProps extends LayeredGraphOptions {
  readonly simulationPaths?: SimulationPathsView | null;
  readonly dataDiff?: DataComparisonView | null;
  readonly onSelect: (selection: EntitySelection | null) => void;
  /** The action whose version is viewed; a data preparation shows each node's prepared data. */
  readonly step?: ActionId | null;
}

/** The indicators a data preparation profiled, each with its profile. */
function preparedRows(
  modelSnapshot: ModelSnapshot,
  indicators: readonly IndicatorSpec[],
): readonly PreparedRow[] {
  return indicators.flatMap((indicator) => {
    const profile = modelSnapshot.profile?.indicators[indicator.observation.id]?.profile;
    return profile ? [{ indicator, profile }] : [];
  });
}

/** A mechanism row's hover text: whose term it is, the states it reads and its chart. */
function mechanismDescription({
  row,
  owner,
  reads,
  plot,
}: {
  readonly row: MechanismRow;
  readonly owner: string;
  readonly reads: readonly string[];
  readonly plot: FlowPlot | undefined;
}): string {
  return [
    `${mechanismName(row.mechanism.id)} (${row.mechanism.kind}) · ${owner}`,
    reads.length > 0 ? `Reads ${reads.join(", ")}` : "Reads no state",
    ...(plot ? [plotTitle(plot)] : []),
  ].join("\n");
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
  const selectedNode = selection?.kind === "construct" ? selection.id : null;
  const flows =
    modelSnapshot.dynamical_model_spec && (visible.has("specification") || visible.has("fit"))
      ? modelFlowPlots(modelSnapshot.dynamical_model_spec)
      : null;
  const name = (id: ConstructSpec["id"]) => humanize(entities.constructById.get(id)?.name ?? id);
  const firstArrow = new Set<string>();
  const timeOrigin = simulationResult?.evidence.time_origin ?? null;
  const view: StateView = {
    geometry: stateGeometry(density),
    zoom,
    flows,
    simulationPaths,
    dataDiff,
    selectIndicator: (id) => onSelect({ kind: "indicator", id }),
  };

  const compared = (construct: ConstructSpec) =>
    dataDiff !== null &&
    construct.indicators.some((indicator) =>
      dataDiff.variables.some((variable) => variable.indicator_id === indicator.observation.id),
    );
  /** What a construct shows for the viewed state: its data, draws or laws. */
  const stateOf = (construct: ConstructSpec): ConstructState => {
    const indicators = indicatorsByConstruct.get(construct.id) ?? [];
    const assignments =
      simulationResult?.evidence.assignments.filter((event) => event.target === construct.id) ?? [];
    return {
      construct,
      indicators,
      prepared: step === "prepare_data" ? preparedRows(modelSnapshot, indicators) : [],
      assignments,
      markers:
        timeOrigin === null
          ? []
          : assignments.map((event) => ({
              time: event.time,
              label: assignmentLabel(event, timeOrigin),
            })),
    };
  };

  /** One mechanism as a row of its effect's card; selecting it selects its owner. */
  const mechanismRow = (construct: ConstructSpec, row: MechanismRow) => {
    const owner = row.owner;
    const edge = owner.kind === "edge" ? entities.edgeById.get(owner.id) : undefined;
    const plot = flows?.mechanisms.get(row.mechanism.id);
    return (
      <MechanismRowView
        key={row.mechanism.id}
        row={row}
        density={density}
        plot={plot}
        description={mechanismDescription({
          row,
          owner: edge
            ? `on ${name(edge.cause.id)} → ${name(edge.effect.id)}`
            : `${name(construct.id)}'s own dynamics`,
          reads: row.reads.map((id) => name(id)),
          plot,
        })}
        failures={edge && !dataDiff ? entityFailures(modelSnapshot, edge) : []}
        zoom={zoom}
        selected={selection?.kind === "edge" && selection.id === owner.id}
        dimmed={
          dataDiff !== null || (focused !== null && !focused.mechanisms.has(row.mechanism.id))
        }
        onSelect={() =>
          onSelect(selection?.kind === owner.kind && selection.id === owner.id ? null : owner)
        }
        onHover={(hovered) =>
          setHovered(hovered ? { kind: "mechanism", id: row.mechanism.id } : null)
        }
      />
    );
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
              const state = stateOf(construct);
              const failures = dataDiff ? [] : entityFailures(modelSnapshot, construct);
              return (
                <g
                  key={node.id}
                  data-node-id={node.id}
                  data-change={constructChange}
                  transform={`translate(${node.x},${node.y})`}
                >
                  {meta.kind === "exogenous" ? (
                    <ExogenousChip
                      density={density}
                      view={view}
                      state={state}
                      status={status}
                      failures={failures}
                      timeOrigin={timeOrigin}
                      selected={selected}
                      dimmed={dimmed}
                      onSelect={select}
                    />
                  ) : (
                    <ConstructCard
                      construct={construct}
                      width={meta.width}
                      height={meta.height}
                      rowsWidth={meta.rows.length > 0 ? density.row.width : 0}
                      geometry={view.geometry}
                      isOutcome={construct.id === outcome}
                      failures={failures}
                      status={status}
                      assignments={state.assignments}
                      timeOrigin={timeOrigin}
                      selected={selected}
                      dimmed={dimmed}
                      onSelect={select}
                      rows={meta.rows.map((row) => mechanismRow(construct, row))}
                    >
                      <StateColumn view={view} state={state} />
                    </ConstructCard>
                  )}
                  {constructChange !== undefined && (
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

  return (
    <div ref={paneRef} className="relative flex h-full min-h-0 flex-col">
      {canvas}
      <div className="absolute right-3 bottom-2 flex items-center gap-1.5 rounded-lg bg-white/85 px-1.5 py-0.5">
        <DagZoomControls zoom={zoom} onZoomChange={setZoom} onFit={fitToPane} />
      </div>
    </div>
  );
}
