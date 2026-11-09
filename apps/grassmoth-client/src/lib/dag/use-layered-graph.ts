"use client";

import type {
  ConstructId,
  EdgeId,
  IndicatorSpec,
  ModelSnapshot,
  PosteriorMarginal,
  SimulationReport,
} from "@grassmoth/api-types";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ModelEntities } from "@/lib/model-asset/entities";
import { useDagLayout } from "@/lib/hooks/use-dag-layout";
import { ownLawUses } from "@/lib/model-asset/laws";
import type { EntitySelection } from "@/lib/model-asset/selection";
import { causalEffect } from "@/lib/simulation-report";
import {
  buildLayeredCausalGraph,
  GRAPH_DENSITIES,
  type LayeredGraphArrowMeta,
} from "./build-layered-causal-graph";
import { placeComparisonOverlay, type ModelComparison } from "./comparison-overlay";
import { availableGraphLayers, graphEntities, graphStatus } from "./layered-model";
import { BLOCKING, COMPARISON_COLORS, DAG_COLORS, LATENT } from "./palette";
import { focusedParts, type GraphFocus } from "./selection";

export interface LayeredGraphOptions {
  modelSnapshot: ModelSnapshot;
  entities: ModelEntities;
  /** A simulation of the viewed model revision; its node histories replace the law charts. */
  simulation?: SimulationReport | null;
  comparison?: ModelComparison | null;
  selection: EntitySelection | null;
}

/** Space around the laid-out graph inside the canvas, in layout units. */
export const CANVAS_PADDING = 36;
/** The canvas frame's border and padding, in pixels. */
const FRAME_INSET = 14;
/** Cards stay rich while the whole rich layout fits its pane at this zoom or larger. */
const RICH_FIT = 0.6;
const MIN_ZOOM = 0.08;
const MAX_ZOOM = 1.8;

/** Derive graph display state from recorded model findings and user interaction. */
export function useLayeredGraph({
  modelSnapshot,
  entities: indexed,
  simulation = null,
  comparison = null,
  selection,
}: LayeredGraphOptions) {
  const visible = useMemo(
    () => new Set(availableGraphLayers(modelSnapshot, simulation)),
    [modelSnapshot, simulation],
  );
  const [manualZoom, setManualZoom] = useState<number | null>(null);
  const [hovered, setHovered] = useState<GraphFocus | null>(null);
  const paneRef = useRef<HTMLDivElement>(null);
  const [pane, setPane] = useState({ width: 0, height: 0 });

  const entities = useMemo(() => graphEntities(indexed), [indexed]);
  const outcome =
    causalEffect(simulation)?.outcome ?? modelSnapshot.dynamical_model_spec?.outcome ?? null;
  const topologies = useMemo(
    () => ({
      rich: buildLayeredCausalGraph(
        entities.constructs,
        entities.edges,
        outcome,
        GRAPH_DENSITIES.rich,
      ),
      compact: buildLayeredCausalGraph(
        entities.constructs,
        entities.edges,
        outcome,
        GRAPH_DENSITIES.compact,
      ),
    }),
    [entities, outcome],
  );
  const rich = useDagLayout(topologies.rich.graph);
  const compact = useDagLayout(topologies.compact.graph);
  useEffect(() => {
    const element = paneRef.current;
    if (!element) return;
    const measure = () => {
      const box = element.getBoundingClientRect();
      setPane({ width: box.width, height: box.height });
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const richFits =
    pane.width === 0 ||
    rich.isLayouting ||
    Math.min(
      (pane.width - FRAME_INSET) / (rich.width + 2 * CANVAS_PADDING),
      (pane.height - FRAME_INSET) / (rich.height + 2 * CANVAS_PADDING),
    ) >= RICH_FIT;
  const topology = richFits ? topologies.rich : topologies.compact;
  const { nodes, edges: arrows, width, height, isLayouting } = richFits ? rich : compact;
  const difference = useMemo(
    () =>
      placeComparisonOverlay(isLayouting ? null : comparison, topology, {
        nodes,
        edges: arrows,
        width,
        height,
      }),
    [comparison, topology, nodes, arrows, width, height, isLayouting],
  );
  // The graph opens fitted to its pane, never above full size, until the viewer zooms.
  const fitZoom =
    pane.width === 0 || isLayouting
      ? 1
      : Math.max(
          MIN_ZOOM,
          Math.min(
            1,
            (pane.width - FRAME_INSET) / (difference.width + 2 * CANVAS_PADDING),
            (pane.height - FRAME_INSET) / (difference.height + 2 * CANVAS_PADDING),
          ),
        );
  const zoom = manualZoom ?? fitZoom;
  const setZoom = (value: number) => setManualZoom(Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, value)));
  const fitToPane = () => setManualZoom(null);

  const designVisible = visible.has("design");
  const fitVisible = visible.has("fit");
  const simulationVisible = visible.has("simulation");
  const nodeStatuses = new Map(
    entities.constructs.map((entity) => [
      entity.id,
      designVisible ? graphStatus(modelSnapshot, entity.id) : null,
    ]),
  );
  const indicatorsByConstruct = new Map<ConstructId, readonly IndicatorSpec[]>(
    visible.has("measurement")
      ? entities.constructs.map((construct) => [construct.id, construct.indicators])
      : [],
  );
  const edgePosteriors = new Map<EdgeId, PosteriorMarginal>(
    fitVisible
      ? entities.edges.flatMap((edge) => {
          const parameters = new Set(
            ownLawUses(edge)
              .filter((use) => use.role === "weight")
              .map((use) => use.parameterId),
          );
          const rows =
            modelSnapshot.fit?.posterior_marginals.filter((row) =>
              parameters.has(row.subject.parameter_id),
            ) ?? [];
          const posterior = rows.length === 1 ? rows[0] : undefined;
          return posterior ? [[edge.id, posterior] as const] : [];
        })
      : [],
  );

  const simulationResult = simulationVisible ? simulation : null;
  const focus = hovered ?? selection;
  const focused = useMemo(() => (focus ? focusedParts(focus, topology) : null), [focus, topology]);

  const edgeVisual = ({ edge, cause, effect }: LayeredGraphArrowMeta) => {
    const blocking =
      nodeStatuses.get(cause.id) === "blocking" || nodeStatuses.get(effect.id) === "blocking";
    const latent =
      nodeStatuses.get(cause.id) === "latent" || nodeStatuses.get(effect.id) === "latent";
    const color = blocking ? BLOCKING : latent ? LATENT : DAG_COLORS.slate;
    // A coefficient's mean is not the strength or sign of a nonlinear state-dependent effect.
    const width = 1.7;
    const lit = focused?.edges.has(edge.id) ?? false;
    const change = difference.edgeChanges.get(edge.id);
    return {
      color:
        selection?.kind === "edge" && selection.id === edge.id
          ? "var(--primary)"
          : change
            ? COMPARISON_COLORS[change]
            : color,
      width: change ? Math.max(width, 3 / zoom) : width,
      opacity: change
        ? 1
        : focused && !lit
          ? 0.1
          : latent
            ? 0.38
            : edgePosteriors.has(edge.id)
              ? 0.92
              : 0.78,
      highlighted: lit,
      change,
    };
  };

  return {
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
  };
}
