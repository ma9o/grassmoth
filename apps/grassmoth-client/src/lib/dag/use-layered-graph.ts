"use client";

import type {
  ConstructId,
  EdgeId,
  PosteriorMarginal,
  IndicatorSpec,
  ModelSnapshot,
  SimulationReport,
} from "@grassmoth/api-types";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ModelEntities } from "@/lib/model-asset/entities";
import { useDagLayout } from "@/lib/hooks/use-dag-layout";
import { ownLawUses } from "@/lib/model-asset/laws";
import type { DagLayoutNode } from "@/lib/utils/dag-graph-layout";
import { buildLayeredCausalGraph, type LayeredGraphEdgeMeta } from "./build-layered-causal-graph";
import { placeComparisonOverlay, type ModelComparison } from "./comparison-overlay";
import {
  availableGraphLayers,
  type CausalGraphLayerId,
  type GraphBand,
  graphEntities,
  graphStatus,
} from "./layered-model";
import { BLOCKING, COMPARISON_COLORS, DAG_COLORS, LATENT } from "./palette";
import { selectedNeighbors } from "./selection";
import { useGraphControls, usePlayback } from "./use-graph-controls";

export interface LayeredGraphOptions {
  modelSnapshot: ModelSnapshot;
  entities: ModelEntities;
  /** A simulation of the viewed model revision; its node histories replace the law charts. */
  simulation?: SimulationReport | null;
  comparison?: ModelComparison | null;
  selection: import("@/lib/model-asset/selection").EntitySelection | null;
}

/** Derive graph display state from recorded model findings and user interaction. */
export function useLayeredGraph({
  modelSnapshot,
  entities: indexed,
  simulation = null,
  comparison = null,
  selection,
}: LayeredGraphOptions) {
  const selectedNode = selection?.kind === "construct" ? selection.id : null;
  const available = useMemo(
    () => availableGraphLayers(modelSnapshot, simulation),
    [modelSnapshot, simulation],
  );
  const [hiddenLayers, setHiddenLayers] = useState<Set<CausalGraphLayerId>>(() => new Set());
  const { zoom, setZoom, hoveredEdge, setHoveredEdge } = useGraphControls(1, 0.08, 1.8);
  const paneRef = useRef<HTMLDivElement>(null);
  const [paneWidth, setPaneWidth] = useState(0);
  const visible = useMemo(
    () => new Set(available.filter((layer) => !hiddenLayers.has(layer))),
    [available, hiddenLayers],
  );

  const entities = useMemo(() => graphEntities(indexed), [indexed]);
  const topology = useMemo(
    () =>
      buildLayeredCausalGraph(entities.constructs, entities.edges, entities.dynamicConstructIds),
    [entities],
  );
  const { nodes, edges: routedSegments, width, height, isLayouting } = useDagLayout(topology.graph);
  const difference = useMemo(
    () => placeComparisonOverlay(isLayouting ? null : comparison, topology, nodes, width, height),
    [comparison, topology, nodes, width, height, isLayouting],
  );
  useEffect(() => {
    const pane = paneRef.current;
    if (!pane || isLayouting) return;
    const measure = () => setPaneWidth(pane.getBoundingClientRect().width);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(pane);
    return () => observer.disconnect();
  }, [isLayouting]);

  const designVisible = visible.has("design");
  const specificationVisible = visible.has("specification");
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
  const likelihoodByVariable = new Map(
    specificationVisible
      ? entities.indicators.flatMap((indicator) =>
          indicator.likelihood ? [[indicator.observation.id, indicator.likelihood] as const] : [],
        )
      : [],
  );
  const marginal = (uses: ReturnType<typeof ownLawUses>, role: "weight" | "decay") => {
    const parameters = new Set(
      uses.filter((use) => use.role === role).map((use) => use.parameterId),
    );
    const rows =
      modelSnapshot.fit?.posterior_marginals.filter((row) =>
        parameters.has(row.subject.parameter_id),
      ) ?? [];
    return rows.length === 1 ? rows[0] : undefined;
  };
  const edgePosteriors: Partial<Record<EdgeId, PosteriorMarginal | undefined>> = fitVisible
    ? Object.fromEntries(
        entities.edges.map((edge) => [edge.id, marginal(ownLawUses(edge), "weight")]),
      )
    : {};
  const persistencePosteriors: Partial<Record<ConstructId, PosteriorMarginal | undefined>> =
    fitVisible
      ? Object.fromEntries(
          entities.constructs.map((construct) => [
            construct.id,
            marginal(ownLawUses(construct), "decay"),
          ]),
        )
      : {};

  const simulationResult = simulationVisible ? simulation : null;
  const days = useMemo(() => simulationResult?.evidence.times ?? [], [simulationResult]);
  const {
    index: clampedDayIndex,
    setIndex: setDayIndex,
    playing,
    setPlaying,
  } = usePlayback(days.length, 180);
  const currentDay = days[clampedDayIndex];
  const selectedNeighborhood = useMemo(
    () =>
      selectedNeighbors(
        selectedNode,
        entities.edges.map((edge) => [edge.cause.id, edge.effect.id] as const),
      ),
    [entities.edges, selectedNode],
  );

  const graphBands = useMemo<GraphBand[]>(() => {
    const dynamicIds = new Set(entities.dynamicConstructIds);
    const staticNodes: DagLayoutNode[] = [];
    const historyNodes: DagLayoutNode[] = [];
    const presentNodes: DagLayoutNode[] = [];
    for (const node of nodes) {
      const meta = topology.nodeMeta.get(node.id);
      if (meta?.kind === "history") historyNodes.push(node);
      if (meta?.kind === "construct") {
        if (!dynamicIds.has(meta.construct.id)) staticNodes.push(node);
        else presentNodes.push(node);
      }
    }
    return [
      { key: "static", label: "stable context", nodes: staticNodes },
      { key: "history", label: "t−1", nodes: historyNodes },
      { key: "present", label: "t", nodes: presentNodes },
    ];
  }, [nodes, topology.nodeMeta, entities.dynamicConstructIds]);

  const toggleLayer = (layer: CausalGraphLayerId) => {
    if (layer === "structure") return;
    setHiddenLayers((current) => {
      const next = new Set(current);
      if (next.has(layer)) next.delete(layer);
      else next.add(layer);
      return next;
    });
  };

  const edgeVisual = (meta: LayeredGraphEdgeMeta) => {
    const posterior = meta.isSelf
      ? persistencePosteriors[meta.cause.id]
      : Object.entries(edgePosteriors).find(([id]) => id === meta.id)?.[1];
    const blocking =
      nodeStatuses.get(meta.cause.id) === "blocking" ||
      nodeStatuses.get(meta.effect.id) === "blocking";
    const latent =
      nodeStatuses.get(meta.cause.id) === "latent" || nodeStatuses.get(meta.effect.id) === "latent";
    const color = blocking
      ? BLOCKING
      : latent
        ? LATENT
        : meta.crossSlice
          ? DAG_COLORS.crossSlice
          : DAG_COLORS.contemporaneous;
    // A coefficient's mean is not the strength or sign of a nonlinear state-dependent effect.
    const width = 1.7;
    const selectedEdge =
      selectedNode == null || meta.cause.id === selectedNode || meta.effect.id === selectedNode;
    const dimmed = !selectedEdge || (hoveredEdge != null && hoveredEdge !== meta.id);
    const opacity = dimmed ? 0.1 : latent ? 0.38 : posterior ? 0.92 : 0.78;
    const change = difference.edgeChanges.get(meta.id);
    return {
      posterior,
      color:
        selection?.kind === "edge" && selection.id === meta.id
          ? "var(--primary)"
          : change
            ? COMPARISON_COLORS[change]
            : color,
      width: change ? Math.max(width, 3 / zoom) : width,
      opacity: change ? 1 : opacity,
      dimmed: change ? false : dimmed,
      change,
    };
  };

  return {
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
    specificationVisible,
    fitVisible,
    simulationVisible,
    nodeStatuses,
    indicatorsByConstruct,
    likelihoodByVariable,
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
  };
}
