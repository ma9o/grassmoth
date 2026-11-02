import type {
  CausalEdgeSpec,
  ConstructId,
  ConstructSpec,
  DynamicsMechanismSpec,
  MechanismId,
} from "@grassmoth/api-types";
import { expressionStates } from "@/lib/model-asset/flow-plots";
import type { EntitySelection } from "@/lib/model-asset/selection";
import type { DagGraphInput, DagLayoutPortInput } from "@/lib/utils/dag-graph-layout";

/** Card sizes at one density; the layout and the drawing read the same numbers. */
export interface GraphDensity {
  readonly key: "rich" | "compact";
  /** A construct's own column: title band, chart strip, then indicator rows. */
  readonly state: {
    readonly width: number;
    readonly title: number;
    readonly strip: number;
    readonly indicator: number;
    readonly height: number;
  };
  /** One mechanism row in the card of the construct it drives. */
  readonly row: { readonly width: number; readonly height: number };
  /** An exogenous construct: its name and its path on one line. */
  readonly chip: { readonly width: number; readonly height: number };
  readonly spacing: { readonly layers: number; readonly nodes: number };
}

/** Rich cards while the whole graph fits its pane; compact ones keep a large model at a glance. */
export const GRAPH_DENSITIES = {
  rich: {
    key: "rich",
    state: { width: 250, title: 36, strip: 64, indicator: 21, height: 158 },
    row: { width: 172, height: 46 },
    chip: { width: 236, height: 32 },
    spacing: { layers: 40, nodes: 14 },
  },
  compact: {
    key: "compact",
    state: { width: 200, title: 28, strip: 40, indicator: 16, height: 112 },
    row: { width: 132, height: 30 },
    chip: { width: 196, height: 26 },
    spacing: { layers: 30, nodes: 9 },
  },
} as const satisfies Record<GraphDensity["key"], GraphDensity>;

/** Space above the first mechanism row and below the last. */
const ROW_INSET = 4;
/** Room under the rows for arrows whose cause no mechanism reads. */
const CARD_PORT_ROOM = 14;

/** One mechanism, drawn as a row of the card of the construct it drives. */
export interface MechanismRow {
  readonly mechanism: DynamicsMechanismSpec;
  /** The construct or edge that declares it; selecting the row opens its owner. */
  readonly owner: Extract<EntitySelection, { kind: "construct" | "edge" }>;
  /** The states its expression reads, in declaration order. */
  readonly reads: readonly ConstructId[];
  /** Top of the row inside its card. */
  readonly top: number;
}

export type LayeredGraphNodeMeta =
  | {
      readonly kind: "construct";
      readonly construct: ConstructSpec;
      readonly rows: readonly MechanismRow[];
      readonly width: number;
      readonly height: number;
    }
  | { readonly kind: "exogenous"; readonly construct: ConstructSpec };

/** One drawn arrow: an edge into the mechanism row that reads its cause, or into the effect's card. */
export interface LayeredGraphArrowMeta {
  readonly edge: CausalEdgeSpec;
  readonly cause: ConstructSpec;
  readonly effect: ConstructSpec;
  readonly row: MechanismRow | null;
}

export interface LayeredGraphBundle {
  readonly density: GraphDensity;
  readonly graph: DagGraphInput;
  readonly nodeMeta: ReadonlyMap<string, LayeredGraphNodeMeta>;
  readonly arrowMeta: ReadonlyMap<string, LayeredGraphArrowMeta>;
  /** Each mechanism's row and the construct whose card holds it. */
  readonly rows: ReadonlyMap<
    MechanismId,
    { readonly row: MechanismRow; readonly effect: ConstructId }
  >;
}

const portOf = (row: MechanismRow) => `${row.mechanism.id}#in`;
const cardPortOf = (id: ConstructId) => `${id}#card`;

/**
 * The layout's model order: each state follows the exogenous inputs it is the first to read,
 * in causal order with feedback set aside. States that reach the outcome come last, so the
 * question's block reads last.
 */
function readingOrder(
  constructs: readonly ConstructSpec[],
  edges: readonly CausalEdgeSpec[],
  outcome: ConstructId | null,
): ConstructSpec[] {
  const byId = new Map(constructs.map((construct) => [construct.id, construct] as const));
  const causes = new Map<ConstructId, ConstructId[]>();
  const effects = new Map<ConstructId, ConstructId[]>();
  for (const edge of edges) {
    if (edge.cause.id === edge.effect.id) continue;
    causes.set(edge.effect.id, [...(causes.get(edge.effect.id) ?? []), edge.cause.id]);
    effects.set(edge.cause.id, [...(effects.get(edge.cause.id) ?? []), edge.effect.id]);
  }
  const visited = new Set<ConstructId>();
  const finished: ConstructId[] = [];
  const visit = (id: ConstructId): void => {
    visited.add(id);
    for (const next of effects.get(id) ?? []) if (!visited.has(next)) visit(next);
    finished.push(id);
  };
  const exogenous = (id: ConstructId) => byId.get(id)?.role === "exogenous";
  for (const construct of [...constructs].sort(
    (a, b) => Number(exogenous(b.id)) - Number(exogenous(a.id)),
  ))
    if (!visited.has(construct.id)) visit(construct.id);
  const states = finished.reverse().filter((id) => !exogenous(id));
  const ancestors = new Set<ConstructId>();
  const pending = outcome === null ? [] : [outcome];
  for (let id = pending.pop(); id !== undefined; id = pending.pop()) {
    if (ancestors.has(id)) continue;
    ancestors.add(id);
    pending.push(...(causes.get(id) ?? []));
  }
  const placed = new Set<ConstructId>();
  const order: ConstructSpec[] = [];
  const place = (id: ConstructId) => {
    const construct = byId.get(id);
    if (!construct || placed.has(id)) return;
    placed.add(id);
    order.push(construct);
  };
  for (const id of [
    ...states.filter((state) => !ancestors.has(state)),
    ...states.filter((state) => ancestors.has(state)),
  ]) {
    for (const cause of causes.get(id) ?? []) if (exogenous(cause)) place(cause);
    place(id);
  }
  for (const construct of constructs) place(construct.id);
  return order;
}

/**
 * Lay out one card per construct. A mechanism is a term of the construct it drives, so it is a
 * row of that card, never a node: the construct's own dynamics first, then the mechanisms its
 * incoming edges declare. Each edge ends on the rows that read its cause. Exogenous constructs
 * are chips. Comparisons decorate this layout without moving its nodes.
 */
export function buildLayeredCausalGraph(
  constructs: readonly ConstructSpec[],
  edges: readonly CausalEdgeSpec[],
  outcome: ConstructId | null,
  density: GraphDensity,
): LayeredGraphBundle {
  const order = readingOrder(constructs, edges, outcome);
  const rank = new Map(order.map((construct, index) => [construct.id, index] as const));
  const byId = new Map(constructs.map((construct) => [construct.id, construct] as const));
  const endpoint = (id: ConstructId) => {
    const construct = byId.get(id);
    if (!construct) throw new Error(`Edge endpoint '${id}' is not a construct of this model.`);
    return construct;
  };
  const links = edges
    .filter((edge) => edge.cause.id !== edge.effect.id)
    .map((edge) => ({ edge, cause: endpoint(edge.cause.id), effect: endpoint(edge.effect.id) }));

  const rowsOf = (construct: ConstructSpec): MechanismRow[] =>
    [
      ...construct.dynamics.map((mechanism) => ({
        mechanism,
        owner: { kind: "construct", id: construct.id } as const,
      })),
      ...edges
        .filter((edge) => edge.effect.id === construct.id)
        .sort((a, b) => (rank.get(a.cause.id) ?? 0) - (rank.get(b.cause.id) ?? 0))
        .flatMap((edge) =>
          edge.mechanisms.map((mechanism) => ({
            mechanism,
            owner: { kind: "edge", id: edge.id } as const,
          })),
        ),
    ].map(({ mechanism, owner }, index) => ({
      mechanism,
      owner,
      reads: expressionStates([mechanism.expression]),
      top: ROW_INSET + index * density.row.height,
    }));

  const nodeMeta = new Map<string, LayeredGraphNodeMeta>();
  const rows = new Map<MechanismId, { row: MechanismRow; effect: ConstructId }>();
  const arrows: Array<{ id: string; meta: LayeredGraphArrowMeta; target: string }> = [];
  const nodes: DagGraphInput["nodes"] = [];
  for (const construct of order) {
    if (construct.role === "exogenous") {
      nodeMeta.set(construct.id, { kind: "exogenous", construct });
      nodes.push({ id: construct.id, width: density.chip.width, height: density.chip.height });
      for (const link of links.filter((item) => item.effect.id === construct.id))
        arrows.push({ id: link.edge.id, meta: { ...link, row: null }, target: construct.id });
      continue;
    }
    const cardRows = rowsOf(construct);
    for (const row of cardRows) rows.set(row.mechanism.id, { row, effect: construct.id });
    const incoming = links
      .filter((link) => link.effect.id === construct.id)
      .map((link) => {
        const readers = cardRows.filter((row) => row.reads.includes(link.cause.id));
        const owned = cardRows.filter(
          (row) => row.owner.kind === "edge" && row.owner.id === link.edge.id,
        );
        return { link, ends: readers.length > 0 ? readers : owned };
      });
    const cardPort = cardRows.length > 0 && incoming.some(({ ends }) => ends.length === 0);
    const rowsHeight = 2 * ROW_INSET + cardRows.length * density.row.height;
    const width = (cardRows.length > 0 ? density.row.width : 0) + density.state.width;
    const height = Math.max(density.state.height, rowsHeight + (cardPort ? CARD_PORT_ROOM : 0));
    nodeMeta.set(construct.id, { kind: "construct", construct, rows: cardRows, width, height });
    const ports: DagLayoutPortInput[] = [
      ...cardRows.map((row) => ({ id: portOf(row), y: row.top + density.row.height / 2 })),
      ...(cardPort ? [{ id: cardPortOf(construct.id), y: height - CARD_PORT_ROOM / 2 }] : []),
    ];
    nodes.push({
      id: construct.id,
      width,
      height,
      ...(ports.length > 0 ? { ports } : {}),
    });
    for (const { link, ends } of incoming) {
      if (ends.length === 0)
        arrows.push({
          id: `${link.edge.id}>card`,
          meta: { ...link, row: null },
          target: cardRows.length > 0 ? cardPortOf(construct.id) : construct.id,
        });
      for (const row of ends)
        arrows.push({
          id: `${link.edge.id}>${row.mechanism.id}`,
          meta: { ...link, row },
          target: portOf(row),
        });
    }
  }

  return {
    density,
    graph: {
      nodes,
      edges: arrows.map((arrow) => ({
        id: arrow.id,
        source: arrow.meta.cause.id,
        target: arrow.target,
      })),
      direction: "RIGHT",
      layoutOptions: {
        "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
        "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
        "elk.layered.cycleBreaking.strategy": "MODEL_ORDER",
        "elk.layered.spacing.nodeNodeBetweenLayers": String(density.spacing.layers),
        "elk.spacing.nodeNode": String(density.spacing.nodes),
        "elk.spacing.edgeNode": "10",
        "elk.spacing.edgeEdge": "5",
        "elk.layered.spacing.edgeNodeBetweenLayers": "10",
        "elk.layered.spacing.edgeEdgeBetweenLayers": "5",
      },
    },
    nodeMeta,
    arrowMeta: new Map(arrows.map((arrow) => [arrow.id, arrow.meta])),
    rows,
  };
}
