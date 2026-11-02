import { modelConstructs, modelEdges, presentEntries } from "@/lib/model-accessors";
import type {
  ConstructId,
  ConstructSpec,
  DynamicalModelSpec,
  EdgeId,
  ModelDiffOutput,
} from "@grassmoth/api-types";
import type { DagLayoutEdge, DagLayoutNode, Point } from "@/lib/utils/dag-graph-layout";
import { humanize } from "@/lib/model-asset/selection";
import type { LayeredGraphBundle } from "@/lib/dag/build-layered-causal-graph";

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

/** The point halfway along a routed polyline. */
function halfway(points: readonly Point[]): Point | undefined {
  const segments = points.flatMap((to, index) => {
    const from = points[index - 1];
    return from ? [{ from, to, length: Math.hypot(to.x - from.x, to.y - from.y) }] : [];
  });
  let remaining = segments.reduce((sum, segment) => sum + segment.length, 0) / 2;
  for (const { from, to, length } of segments) {
    if (remaining <= length) {
      const t = length === 0 ? 0 : remaining / length;
      return { x: from.x + (to.x - from.x) * t, y: from.y + (to.y - from.y) * t };
    }
    remaining -= length;
  }
  return points.at(-1);
}

/** Place additions around the selected layout. Existing coordinates are never recomputed. */
export function placeComparisonOverlay(
  comparison: ModelComparison | null,
  topology: LayeredGraphBundle,
  layout: { nodes: DagLayoutNode[]; edges: DagLayoutEdge[]; width: number; height: number },
) {
  const positions = new Map(layout.nodes.map((node) => [node.id, node]));
  const drawn = new Map<EdgeId, { points: Point[]; cause: ConstructId; effect: ConstructId }>();
  for (const arrow of layout.edges) {
    const meta = topology.arrowMeta.get(arrow.id);
    if (meta && !drawn.has(meta.edge.id))
      drawn.set(meta.edge.id, {
        points: arrow.points,
        cause: meta.cause.id,
        effect: meta.effect.id,
      });
  }
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
  const edgeChanges = new Map<EdgeId, Change>();
  const constructChanges = new Map<ConstructId, Change>();
  const addedNodes: Array<{ node: DagLayoutNode; construct: ConstructSpec }> = [];
  const addedEdges: Array<{ id: EdgeId; points: Point[] }> = [];
  const marks: DifferenceMark[] = [];
  let overlayWidth = layout.width,
    overlayHeight = layout.height,
    nextY = 20;

  const addNode = (construct: ConstructSpec) => {
    const existing = positions.get(construct.id);
    if (existing) return existing;
    const size = topology.density.state;
    const node = {
      id: construct.id,
      x: layout.width + 60,
      y: nextY,
      width: size.width,
      height: size.height,
    };
    nextY += size.height + 40;
    positions.set(construct.id, node);
    addedNodes.push({ node, construct });
    overlayWidth = Math.max(overlayWidth, node.x + node.width);
    overlayHeight = Math.max(overlayHeight, node.y + node.height);
    return node;
  };
  const addEdge = (id: EdgeId, source: DagLayoutNode, target: DagLayoutNode) => {
    const start = { x: source.x + source.width, y: source.y + source.height / 2 };
    const end = { x: target.x, y: target.y + target.height / 2 };
    const middle = (start.x + end.x) / 2;
    addedEdges.push({
      id,
      points: [start, { x: middle, y: start.y }, { x: middle, y: end.y }, end],
    });
    return { x: middle, y: (start.y + end.y) / 2 };
  };
  for (const [id, patch] of presentEntries(comparison?.changes.constructs ?? {})) {
    const change = patch === null ? "removed" : beforeConstructs.has(id) ? "revised" : "added";
    constructChanges.set(id, change);
    const definition = (change === "removed" ? beforeConstructs : constructs).get(id);
    if (!definition) throw new Error(`Compared construct ${id} is missing from its model`);
    const node = change === "added" ? addNode(definition) : positions.get(id);
    if (!node) continue;
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
    const arrow = drawn.get(id);
    let anchor = arrow ? halfway(arrow.points) : undefined;
    const moved =
      arrow !== undefined &&
      (arrow.cause !== definition.cause.id || arrow.effect !== definition.effect.id);
    if (change !== "removed" && (!arrow || moved)) {
      const source = positions.get(definition.cause.id);
      const target = positions.get(definition.effect.id);
      if (source && target) {
        if (moved) edgeChanges.set(id, "removed");
        anchor = addEdge(id, source, target);
      }
    }
    if (anchor)
      marks.push({
        id: id,
        x: anchor.x,
        y: anchor.y,
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
