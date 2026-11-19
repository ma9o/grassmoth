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
import { useDagLayout } from "@/lib/hooks/use-dag-layout";
import type { ModelEntities } from "@/lib/model-asset/entities";
import { ownLawUses } from "@/lib/model-asset/laws";
import type { EntitySelection } from "@/lib/model-asset/selection";
import type { SpecPlots } from "@/lib/model-asset/spec-plots";
import { causalEffect } from "@/lib/simulation-report";
import {
  buildLayeredCausalGraph,
  GRAPH_DENSITIES,
  type GraphDensity,
  type LayeredGraphArrowMeta,
  type LayeredGraphCarryMeta,
} from "./build-layered-causal-graph";
import { type ModelComparison, placeComparisonOverlay } from "./comparison-overlay";
import { availableGraphLayers, graphEntities, graphStatus } from "./layered-model";
import { BLOCKING, COMPARISON_COLORS, DAG_COLORS, LATENT } from "./palette";
import { focusedParts, type GraphFocus } from "./selection";
import { specLayout } from "./spec-layout";

export interface LayeredGraphOptions {
  readonly modelSnapshot: ModelSnapshot;
  readonly entities: ModelEntities;
  /** A simulation of the viewed model revision; its node histories replace the law charts. */
  readonly simulation?: SimulationReport | null;
  readonly comparison?: ModelComparison | null;
  readonly selection: EntitySelection | null;
}

/** Space around the laid-out graph inside the canvas, in layout units. */
export const CANVAS_PADDING = 36;
/** The canvas frame's border and padding, in pixels. */
const FRAME_INSET = 14;
/** Cards stay rich while the whole rich layout fits its pane at this zoom or larger. */
const SPEC_DENSITY = {
  ...GRAPH_DENSITIES.rich,
  state: { ...GRAPH_DENSITIES.rich.state, width: 278, height: 112 },
};

const RICH_FIT = 0.6;
const MIN_ZOOM = 0.08;
const MAX_ZOOM = 1.8;

function edgeColor({
  selected,
  change,
  blocking,
  latent,
}: {
  readonly selected: boolean;
  readonly change: keyof typeof COMPARISON_COLORS | undefined;
  readonly blocking: boolean;
  readonly latent: boolean;
}): string {
  if (selected) return "var(--primary)";
  if (change !== undefined) return COMPARISON_COLORS[change];
  if (blocking) return BLOCKING;
  if (latent) return LATENT;
  return DAG_COLORS.slate;
}

function edgeOpacity({
  changed,
  dimmed,
  latent,
  hasPosterior,
}: {
  readonly changed: boolean;
  readonly dimmed: boolean;
  readonly latent: boolean;
  readonly hasPosterior: boolean;
}): number {
  if (changed) return 1;
  if (dimmed) return 0.1;
  if (latent) return 0.38;
  if (hasPosterior) return 0.92;
  return 0.78;
}

/** Derive graph display state from recorded model findings and user interaction. */
export function useLayeredGraph(
  {
    modelSnapshot,
    entities: indexed,
    simulation = null,
    comparison = null,
    selection,
  }: LayeredGraphOptions,
  plots: SpecPlots | null,
) {
  const visible = useMemo(
    () => new Set(availableGraphLayers(modelSnapshot, simulation)),
    [modelSnapshot, simulation],
  );
  const [manualZoom, setManualZoom] = useState<number | "fit" | null>(null);
  const [hovered, setHovered] = useState<GraphFocus | null>(null);
  const paneRef = useRef<HTMLDivElement>(null);
  const [pane, setPane] = useState({ width: 0, height: 0 });

  const entities = useMemo(() => graphEntities(indexed), [indexed]);
  const outcome =
    causalEffect(simulation)?.outcome ?? modelSnapshot.dynamical_model_spec?.outcome ?? null;
  const topologies = useMemo(() => {
    const build = (density: GraphDensity) =>
      buildLayeredCausalGraph(entities.constructs, entities.edges, outcome, density);
    if (plots) {
      const spec = build(SPEC_DENSITY);
      return { rich: spec, compact: spec };
    }
    return { rich: build(GRAPH_DENSITIES.rich), compact: build(GRAPH_DENSITIES.compact) };
  }, [entities, outcome, plots]);
  const authored = useMemo(
    () => (plots ? specLayout(topologies.rich, plots, entities.edges) : null),
    [plots, topologies.rich, entities.edges],
  );
  const rich = useDagLayout(authored ? null : topologies.rich.graph);
  const compact = useDagLayout(authored ? null : topologies.compact.graph);
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
    plots !== null ||
    pane.width === 0 ||
    rich.isLayouting ||
    Math.min(
      (pane.width - FRAME_INSET) / (rich.width + 2 * CANVAS_PADDING),
      (pane.height - FRAME_INSET) / (rich.height + 2 * CANVAS_PADDING),
    ) >= RICH_FIT;
  const topology = richFits ? topologies.rich : topologies.compact;
  const {
    nodes,
    edges: arrows,
    width,
    height,
    isLayouting,
  } = authored ?? (richFits ? rich : compact);
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
  // Authored functions open at reading size. Fit remains available when the graph needs scrolling.
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
  const zoom = manualZoom === "fit" ? fitZoom : (manualZoom ?? (plots ? 1 : fitZoom));
  const setZoom = (value: number) => setManualZoom(Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, value)));
  const fitToPane = () => setManualZoom("fit");

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
    // A coefficient's mean is not the strength or sign of a nonlinear state-dependent effect.
    const lineWidth = 1.7;
    const lit = focused?.edges.has(edge.id) ?? false;
    const change = difference.edgeChanges.get(edge.id);
    const selected = selection?.kind === "edge" && selection.id === edge.id;
    return {
      color: edgeColor({ selected, change, blocking, latent }),
      width: change ? Math.max(lineWidth, 3 / zoom) : lineWidth,
      opacity: edgeOpacity({
        changed: change !== undefined,
        dimmed: focused !== null && !lit,
        latent,
        hasPosterior: edgePosteriors.has(edge.id),
      }),
      highlighted: lit,
      change,
    };
  };

  /** A carry arrow belongs to its construct: it lights, recedes and is selected with its node. */
  const carryVisual = ({ construct }: LayeredGraphCarryMeta) => {
    const status = nodeStatuses.get(construct.id);
    const selected = selection?.kind === "construct" && selection.id === construct.id;
    return {
      color: edgeColor({
        selected,
        change: undefined,
        blocking: status === "blocking",
        latent: status === "latent",
      }),
      opacity: edgeOpacity({
        changed: false,
        dimmed: focused !== null && !focused.constructs.has(construct.id),
        latent: status === "latent",
        hasPosterior: false,
      }),
      highlighted: selected,
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
    authored,
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
    carryVisual,
  };
}
