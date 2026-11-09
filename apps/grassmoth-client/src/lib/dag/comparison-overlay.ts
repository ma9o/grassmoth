import type {
  CausalEdgeSpec,
  ConstructId,
  ConstructSpec,
  DynamicalModelSpec,
  EdgeId,
  ModelDiffOutput,
  ReasoningSpec,
} from "@grassmoth/api-types";
import type { GraphDensity, LayeredGraphBundle } from "@/lib/dag/build-layered-causal-graph";
import { modelConstructs, modelEdges, presentEntries } from "@/lib/model-accessors";
import { humanize } from "@/lib/model-asset/selection";
import type {
  DagLayoutEdge,
  DagLayoutNode,
  DagLayoutResult,
  Point,
} from "@/lib/utils/dag-graph-layout";

/** Saved spec changes joined to their original model values for presentation. */
export type ModelComparison = ModelDiffOutput & {
  readonly beforeDynamicalModelSpec: DynamicalModelSpec | null;
  readonly afterDynamicalModelSpec: DynamicalModelSpec | null;
};

type Change = "added" | "removed" | "revised";

interface DifferenceMark {
  readonly id: string;
  readonly change: Change;
  readonly x: number;
  readonly y: number;
  readonly title: string;
  readonly detail: string | ReasoningSpec;
}

interface AddedNode {
  readonly node: DagLayoutNode;
  readonly construct: ConstructSpec;
}

interface AddedEdge {
  readonly id: EdgeId;
  readonly points: readonly Point[];
}

interface ComparisonOverlay {
  readonly width: number;
  readonly height: number;
  readonly addedNodes: readonly AddedNode[];
  readonly addedEdges: readonly AddedEdge[];
  readonly edgeChanges: ReadonlyMap<EdgeId, Change>;
  readonly constructChanges: ReadonlyMap<ConstructId, Change>;
  readonly marks: readonly DifferenceMark[];
}

/** The first arrow drawn for a causal edge, and the constructs the layout drew it between. */
interface DrawnEdge {
  readonly points: readonly Point[];
  readonly cause: ConstructId;
  readonly effect: ConstructId;
}

interface ChangedEdge {
  readonly id: EdgeId;
  readonly change: Change;
  readonly definition: CausalEdgeSpec;
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

function changeOf(patch: object | null, existedBefore: boolean): Change {
  if (patch === null) return "removed";
  return existedBefore ? "revised" : "added";
}

/** Each causal edge's first drawn arrow. */
function drawnEdges(
  arrows: readonly DagLayoutEdge[],
  topology: LayeredGraphBundle,
): ReadonlyMap<EdgeId, DrawnEdge> {
  const drawn = new Map<EdgeId, DrawnEdge>();
  for (const arrow of arrows) {
    const meta = topology.arrowMeta.get(arrow.id);
    if (meta && !drawn.has(meta.edge.id)) {
      drawn.set(meta.edge.id, {
        points: arrow.points,
        cause: meta.cause.id,
        effect: meta.effect.id,
      });
    }
  }
  return drawn;
}

/**
 * Mark each changed construct on its card. Added constructs the layout has no card for stack in a
 * column to its right, which is the only way the overlay grows.
 */
function placeConstructChanges(
  comparison: ModelComparison | null,
  size: GraphDensity["state"],
  layout: DagLayoutResult,
) {
  const before = new Map(
    modelConstructs(comparison?.beforeDynamicalModelSpec).map((item) => [item.id, item]),
  );
  const after = new Map(
    modelConstructs(comparison?.afterDynamicalModelSpec).map((item) => [item.id, item]),
  );
  const changed = presentEntries(comparison?.changes.constructs ?? {}).map(([id, patch]) => {
    const change = changeOf(patch, before.has(id));
    const definition = (change === "removed" ? before : after).get(id);
    if (!definition) throw new Error(`Compared construct ${id} is missing from its model`);
    return { id, change, definition };
  });
  const laidOut = new Map(layout.nodes.map((node) => [node.id, node]));
  const addedNodes: readonly AddedNode[] = changed
    .filter(({ id, change }) => change === "added" && !laidOut.has(id))
    .map(({ definition }, index) => ({
      node: {
        id: definition.id,
        x: layout.width + 60,
        y: 20 + index * (size.height + 40),
        width: size.width,
        height: size.height,
      },
      construct: definition,
    }));
  const positions: ReadonlyMap<string, DagLayoutNode> = new Map([
    ...laidOut,
    ...addedNodes.map(({ node }) => [node.id, node] as const),
  ]);
  const marks = changed.flatMap(({ id, change, definition }): DifferenceMark[] => {
    const node = positions.get(id);
    if (!node) return [];
    return [
      {
        id,
        x: node.x + node.width,
        y: node.y,
        change,
        title: `${humanize(change)} construct`,
        detail: humanize(definition.name),
      },
    ];
  });
  return {
    width: Math.max(layout.width, ...addedNodes.map(({ node }) => node.x + node.width)),
    height: Math.max(layout.height, ...addedNodes.map(({ node }) => node.y + node.height)),
    positions,
    addedNodes,
    changes: new Map(changed.map(({ id, change }) => [id, change])),
    marks,
  };
}

/** A right-angled route from one card's east side to another's west side, and its midpoint. */
function routeBetween(source: DagLayoutNode, target: DagLayoutNode) {
  const start = { x: source.x + source.width, y: source.y + source.height / 2 };
  const end = { x: target.x, y: target.y + target.height / 2 };
  const middle = (start.x + end.x) / 2;
  return {
    points: [start, { x: middle, y: start.y }, { x: middle, y: end.y }, end],
    anchor: { x: middle, y: (start.y + end.y) / 2 },
  };
}

/**
 * A changed edge's mark sits halfway along its drawn arrow. A present edge the layout has no arrow
 * for, or drew between other constructs, gets a new arrow between its endpoints' cards when both
 * are placed; the mark moves onto it, and a moved edge's old arrow reads as removed.
 */
function placeEdgeChange(
  { id, change, definition }: ChangedEdge,
  drawn: ReadonlyMap<EdgeId, DrawnEdge>,
  positions: ReadonlyMap<string, DagLayoutNode>,
) {
  const arrow = drawn.get(id);
  const moved =
    arrow !== undefined &&
    (arrow.cause !== definition.cause.id || arrow.effect !== definition.effect.id);
  const source = positions.get(definition.cause.id);
  const target = positions.get(definition.effect.id);
  if (change !== "removed" && (!arrow || moved) && source && target) {
    const { points, anchor } = routeBetween(source, target);
    const shown: Change = moved ? "removed" : change;
    return { shown, added: { id, points }, anchor };
  }
  return { shown: change, added: null, anchor: arrow ? halfway(arrow.points) : undefined };
}

function placeEdgeChanges(
  comparison: ModelComparison | null,
  drawn: ReadonlyMap<EdgeId, DrawnEdge>,
  positions: ReadonlyMap<string, DagLayoutNode>,
) {
  const before = new Map(
    modelEdges(comparison?.beforeDynamicalModelSpec).map((item) => [item.id, item]),
  );
  const after = new Map(
    modelEdges(comparison?.afterDynamicalModelSpec).map((item) => [item.id, item]),
  );
  const placed = presentEntries(comparison?.changes.edges ?? {}).map(([id, patch]) => {
    const change = changeOf(patch, before.has(id));
    const definition = (change === "removed" ? before : after).get(id);
    if (!definition) throw new Error(`Compared edge ${id} is missing from its model`);
    return {
      id,
      change,
      definition,
      ...placeEdgeChange({ id, change, definition }, drawn, positions),
    };
  });
  return {
    addedEdges: placed.flatMap(({ added }) => (added ? [added] : [])),
    changes: new Map(placed.map(({ id, shown }) => [id, shown])),
    marks: placed.flatMap(({ id, change, definition, anchor }): DifferenceMark[] => {
      if (!anchor) return [];
      return [
        {
          id,
          x: anchor.x,
          y: anchor.y,
          change,
          title: `${humanize(change)} connection`,
          detail: definition.reasoning,
        },
      ];
    }),
  };
}

/** Place additions around the selected layout. Existing coordinates are never recomputed. */
export function placeComparisonOverlay(
  comparison: ModelComparison | null,
  topology: LayeredGraphBundle,
  layout: DagLayoutResult,
): ComparisonOverlay {
  const constructs = placeConstructChanges(comparison, topology.density.state, layout);
  const edges = placeEdgeChanges(
    comparison,
    drawnEdges(layout.edges, topology),
    constructs.positions,
  );
  return {
    width: constructs.width,
    height: constructs.height,
    addedNodes: constructs.addedNodes,
    addedEdges: edges.addedEdges,
    edgeChanges: edges.changes,
    constructChanges: constructs.changes,
    marks: [...constructs.marks, ...edges.marks],
  };
}
