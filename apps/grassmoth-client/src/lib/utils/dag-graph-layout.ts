import ELK, { type ElkNode } from "elkjs/lib/elk.bundled.js";

/**
 * Generic ELK layout for the bespoke DAG renderer (shared by the static
 * structure DAG and the interactive simulation DAG).
 *
 * Returns ELK's ORTHOGONAL edge routing (start + bend points + end) so the
 * renderer can draw the lines itself. Input is a plain `{nodes, edges}` graph,
 * not the construct/edge domain types, so callers build whatever graph they
 * need (structure graph, or the unrolled / glyph-split simulation graph) and
 * feed it through the same layout + render path.
 */

export interface Point {
  readonly x: number;
  readonly y: number;
}

export type DagDirection = "RIGHT" | "DOWN" | "LEFT" | "UP";

/** A fixed point on its node's west side where edges end, measured from the node's top. */
export interface DagLayoutPortInput {
  readonly id: string;
  readonly y: number;
}

export interface DagLayoutNodeInput {
  readonly id: string;
  readonly width: number;
  readonly height: number;
  /** Per-node ELK options (e.g. partitions for the interactive glyph columns). */
  readonly layoutOptions?: Readonly<Record<string, string>>;
  /** Fixed west-side ports; edges may target a port id instead of the node. */
  readonly ports?: readonly DagLayoutPortInput[];
}

export interface DagLayoutEdgeInput {
  readonly id: string;
  readonly source: string;
  /** A node id, or a port id of the target node. */
  readonly target: string;
}

export interface DagGraphInput {
  readonly nodes: readonly DagLayoutNodeInput[];
  readonly edges: readonly DagLayoutEdgeInput[];
  /** Layout flow direction. Defaults to RIGHT (cause → effect, left to right). */
  readonly direction?: DagDirection;
  /** Graph-level ELK option overrides, merged over the defaults. */
  readonly layoutOptions?: Readonly<Record<string, string>>;
}

export interface DagLayoutNode {
  readonly id: string;
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

export interface DagLayoutEdge {
  readonly id: string;
  readonly source: string;
  readonly target: string;
  /** Routed polyline: [startPoint, ...bendPoints, endPoint]. */
  readonly points: readonly Point[];
}

export interface DagLayoutResult {
  readonly width: number;
  readonly height: number;
  readonly nodes: readonly DagLayoutNode[];
  readonly edges: readonly DagLayoutEdge[];
}

export const EMPTY_LAYOUT: DagLayoutResult = {
  width: 0,
  height: 0,
  nodes: [],
  edges: [],
};

const DEFAULT_OPTIONS: Record<string, string> = {
  "elk.algorithm": "layered",
  "elk.edgeRouting": "ORTHOGONAL",
  "elk.layered.layering.strategy": "NETWORK_SIMPLEX",
  "elk.layered.spacing.nodeNodeBetweenLayers": "64",
  "elk.spacing.nodeNode": "36",
  "elk.spacing.edgeNode": "28",
  "elk.spacing.edgeEdge": "16",
  "elk.layered.spacing.edgeNodeBetweenLayers": "28",
  "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
  "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
};

/** A node's own ELK options, with its fixed west-side ports when it has any. */
function elkNodeOptions(node: DagLayoutNodeInput): Pick<ElkNode, "layoutOptions" | "ports"> {
  if (node.ports) {
    return {
      layoutOptions: { ...node.layoutOptions, "elk.portConstraints": "FIXED_POS" },
      ports: node.ports.map((port) => ({
        id: port.id,
        x: 0,
        y: port.y,
        width: 0,
        height: 0,
        layoutOptions: { "elk.port.side": "WEST" },
      })),
    };
  }
  return node.layoutOptions ? { layoutOptions: node.layoutOptions } : {};
}

/** Build the ELK graph from a generic graph input. Pure — no layout is run. */
export function buildElkGraph(input: DagGraphInput): ElkNode {
  return {
    id: "root",
    layoutOptions: {
      ...DEFAULT_OPTIONS,
      "elk.direction": input.direction ?? "RIGHT",
      ...input.layoutOptions,
    },
    children: input.nodes.map((n) => ({
      id: n.id,
      width: n.width,
      height: n.height,
      ...elkNodeOptions(n),
    })),
    edges: input.edges.map((e) => ({
      id: e.id,
      sources: [e.source],
      targets: [e.target],
    })),
  };
}

/** Read a laid-out ELK graph into the renderer's flat geometry. Pure. */
export function readElkLayout(root: ElkNode): DagLayoutResult {
  const nodes: DagLayoutNode[] = (root.children ?? []).map((c) => ({
    id: c.id,
    x: c.x ?? 0,
    y: c.y ?? 0,
    width: c.width ?? 0,
    height: c.height ?? 0,
  }));

  const edges: DagLayoutEdge[] = (root.edges ?? []).map((e) => {
    const section = e.sections?.[0];
    const points: Point[] = section
      ? [section.startPoint, ...(section.bendPoints ?? []), section.endPoint].map((p) => ({
          x: p.x,
          y: p.y,
        }))
      : [];
    return {
      id: e.id,
      source: e.sources.at(0) ?? "",
      target: e.targets.at(0) ?? "",
      points,
    };
  });

  return {
    width: root.width ?? 0,
    height: root.height ?? 0,
    nodes,
    edges,
  };
}

const elk = new ELK();

/** Run ELK layout and return the renderer's flat, routed geometry. */
export async function runDagLayout(input: DagGraphInput): Promise<DagLayoutResult> {
  const laidOut = await elk.layout(buildElkGraph(input));
  return readElkLayout(laidOut);
}
