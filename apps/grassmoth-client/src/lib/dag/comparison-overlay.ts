import { modelEdges, presentEntries } from "@/lib/model-accessors";
import type {
  ConstructId,
  ConstructSpec,
  DynamicalModelSpec,
  ModelDiffOutput,
} from "@grassmoth/api-types";
import type { DagLayoutNode, Point } from "@/lib/utils/dag-graph-layout";
import { modelConstructs } from "@/lib/model-accessors";
import { humanize } from "@/lib/model-asset/selection";
import { ghostId, isGhost } from "@/lib/dag/unroll";
import {
  LAYERED_NODE_HEIGHT,
  LAYERED_NODE_WIDTH,
  LAYERED_HISTORY_HEIGHT,
  LAYERED_HISTORY_WIDTH,
  type LayeredGraphBundle,
} from "@/lib/dag/build-layered-causal-graph";

/** Saved spec changes joined to their original model values for presentation. */
export type ModelComparison = ModelDiffOutput & {
  beforeDynamicalModelSpec: DynamicalModelSpec | null;
  afterDynamicalModelSpec: DynamicalModelSpec | null;
};

type Change = "added" | "removed" | "revised";

interface DifferenceMark {
  id: string;
  change: Change;
  x: number;
  y: number;
  title: string;
  detail: string;
}

/** Place additions around the selected layout. Existing coordinates are never recomputed. */
export function placeComparisonOverlay(
  comparison: ModelComparison | null,
  topology: LayeredGraphBundle,
  nodes: DagLayoutNode[],
  width: number,
  height: number,
) {
  const positions = new Map(nodes.map((node) => [node.id, node]));
  const beforeConstructs = new Map(
    (comparison ? modelConstructs(comparison.beforeDynamicalModelSpec) : []).map((item) => [
      item.id,
      item,
    ]),
  );
  const constructs = new Map(
    (comparison ? modelConstructs(comparison.afterDynamicalModelSpec) : []).map((item) => [
      item.id,
      item,
    ]),
  );
  const beforeEdges = new Map(
    modelEdges(comparison?.beforeDynamicalModelSpec).map((item) => [item.id, item]),
  );
  const afterEdges = new Map(
    modelEdges(comparison?.afterDynamicalModelSpec).map((item) => [item.id, item]),
  );
  const dynamicIds = new Set(
    [...constructs.values()]
      .filter((item) => item.temporal_status === "time_varying")
      .map((item) => item.id),
  );
  const previousDynamicIds = new Set(
    [...beforeConstructs.values()]
      .filter((item) => item.temporal_status === "time_varying")
      .map((item) => item.id),
  );
  const edgeChanges = new Map<string, Change>();
  const constructChanges = new Map<ConstructId, Change>();
  const addedNodes: Array<{ node: DagLayoutNode; construct: ConstructSpec; history: boolean }> = [];
  const addedEdges: Array<{ id: string; points: Point[]; crossSlice: boolean }> = [];
  const marks: DifferenceMark[] = [];
  let overlayWidth = width,
    overlayHeight = height;

  const addNode = (construct: ConstructSpec, history = false) => {
    const id = history ? ghostId(construct.id) : construct.id;
    const existing = positions.get(id);
    if (existing) return existing;
    const node = {
      id,
      x: width + 60,
      y: 20 + addedNodes.length * (LAYERED_NODE_HEIGHT + 40),
      width: history ? LAYERED_HISTORY_WIDTH : LAYERED_NODE_WIDTH,
      height: history ? LAYERED_HISTORY_HEIGHT : LAYERED_NODE_HEIGHT,
    };
    positions.set(id, node);
    addedNodes.push({ node, construct, history });
    overlayWidth = Math.max(overlayWidth, node.x + node.width);
    overlayHeight = Math.max(overlayHeight, node.y + node.height);
    return node;
  };
  const addEdge = (
    id: string,
    source: DagLayoutNode,
    target: DagLayoutNode,
    crossSlice: boolean,
  ) => {
    const start = { x: source.x + source.width, y: source.y + source.height / 2 };
    const end = { x: target.x, y: target.y + target.height / 2 };
    const middle = (start.x + end.x) / 2;
    addedEdges.push({
      id,
      crossSlice,
      points: [start, { x: middle, y: start.y }, { x: middle, y: end.y }, end],
    });
    return { id, x: middle, y: (start.y + end.y) / 2, width: 0, height: 0 };
  };
  for (const [id, patch] of presentEntries(comparison?.changes.constructs ?? {})) {
    const change = patch === null ? "removed" : beforeConstructs.has(id) ? "revised" : "added";
    constructChanges.set(id, change);
    const definition = (change === "removed" ? beforeConstructs : constructs).get(id);
    if (!definition) throw new Error(`Compared construct ${id} is missing from its model`);
    const node = change === "added" ? addNode(definition) : positions.get(id);
    if (!node) continue;
    if (change !== "removed") {
      if (dynamicIds.has(id) && !previousDynamicIds.has(id)) {
        const source = addNode(definition, true);
        addEdge(`self:${id}`, source, node, true);
      }
    }
    if (previousDynamicIds.has(id) && !dynamicIds.has(id)) {
      edgeChanges.set(`self:${id}`, "removed");
    }
    marks.push({
      id: id,
      x: node.x + node.width,
      y: node.y,
      change: change,
      title: `${humanize(change)} construct`,
      detail: humanize(definition.name),
    });
  }
  for (const [id, patch] of presentEntries(comparison?.changes.edges ?? {})) {
    const change = patch === null ? "removed" : beforeEdges.has(id) ? "revised" : "added";
    edgeChanges.set(id, change);
    const definition = (change === "removed" ? beforeEdges : afterEdges).get(id);
    if (!definition) throw new Error(`Compared edge ${id} is missing from its model`);
    const existing = topology.edgeMeta.get(id);
    let anchor = existing ? positions.get(existing.slotId) : undefined;
    if (change !== "removed") {
      const cause = constructs.get(definition.cause.id);
      const target = positions.get(definition.effect.id);
      const source = cause
        ? dynamicIds.has(cause.id)
          ? addNode(cause, true)
          : positions.get(cause.id)
        : undefined;
      if (
        source &&
        target &&
        (!existing || existing.source !== source.id || existing.target !== target.id)
      ) {
        if (existing) edgeChanges.set(id, "removed");
        anchor = addEdge(id, source, target, isGhost(source.id));
      }
    }
    if (anchor)
      marks.push({
        id: id,
        x: anchor.x + anchor.width / 2,
        y: anchor.y + anchor.height / 2,
        change: change,
        title: `${humanize(change)} connection`,
        detail: definition.description,
      });
  }
  return {
    width: overlayWidth,
    height: overlayHeight,
    addedNodes,
    addedEdges,
    edgeChanges,
    constructChanges,
    marks,
  };
}
