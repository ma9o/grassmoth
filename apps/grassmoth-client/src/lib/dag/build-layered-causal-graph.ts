import type {
  CausalEdgeSpec,
  ConstructId,
  ConstructSpec,
  DynamicsMechanismSpec,
  MechanismId,
} from "@grassmoth/api-types";
import { expressionStates } from "@/lib/model-asset/flow-plots";
import type { EntitySelection } from "@/lib/model-asset/selection";
import type {
  DagGraphInput,
  DagLayoutNodeInput,
  DagLayoutPortInput,
} from "@/lib/utils/dag-graph-layout";

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

/** One mechanism, read as a term of the construct it drives. */
export interface MechanismTerm {
  readonly mechanism: DynamicsMechanismSpec;
  /** The construct or edge that declares it; selecting the term opens its owner. */
  readonly owner: Extract<EntitySelection, { kind: "construct" | "edge" }>;
  /** The states its expression reads, in declaration order. */
  readonly reads: readonly ConstructId[];
}

/** A term drawn as a row of the card of the construct it drives. */
export interface MechanismRow extends MechanismTerm {
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

/**
 * A time-varying construct's arrow to itself: its state leaves the current time slice for the
 * next. It ends on a row of its own card that reads the state, or on the node when none does.
 * Every other arrow is instantaneous: a mechanism reads its cause's current state.
 */
export interface LayeredGraphCarryMeta {
  readonly construct: ConstructSpec;
  readonly row: MechanismRow | null;
}

export interface LayeredGraphBundle {
  readonly density: GraphDensity;
  readonly graph: DagGraphInput;
  readonly nodeMeta: ReadonlyMap<string, LayeredGraphNodeMeta>;
  readonly arrowMeta: ReadonlyMap<string, LayeredGraphArrowMeta>;
  readonly carryMeta: ReadonlyMap<string, LayeredGraphCarryMeta>;
  /** Each mechanism's row and the construct whose card holds it. */
  readonly rows: ReadonlyMap<
    MechanismId,
    { readonly row: MechanismRow; readonly effect: ConstructId }
  >;
}

const portOf = (row: MechanismRow) => `${row.mechanism.id}#in`;
const cardPortOf = (id: ConstructId) => `${id}#card`;
const chipOutletOf = (id: ConstructId) => `${id}#out`;
const carryStartOf = (id: ConstructId) => `${id}#carry`;
const carryEndOf = (id: ConstructId) => `${id}#carried`;

/** A carry arrow leaves the top of its node this far along, and returns this much nearer. */
const CARRY_START = 40;
const CARRY_SPAN = 28;

/** A time-varying construct's state leaves each time slice for the next. */
const carriesOver = (construct: ConstructSpec) => construct.temporal_status === "time_varying";

/** Where a construct's arrows leave it: a chip's ports are fixed, so from the one at its middle. */
const outletOf = (construct: ConstructSpec) =>
  construct.role === "exogenous" ? chipOutletOf(construct.id) : construct.id;

/** Each construct's direct causes, or its direct effects. */
type Neighbours = ReadonlyMap<ConstructId, readonly ConstructId[]>;

/** Direct causes and effects of each construct, with self-loops set aside. */
function adjacency(edges: readonly CausalEdgeSpec[]): {
  readonly causes: Neighbours;
  readonly effects: Neighbours;
} {
  const causes = new Map<ConstructId, ConstructId[]>();
  const effects = new Map<ConstructId, ConstructId[]>();
  for (const edge of edges) {
    if (edge.cause.id === edge.effect.id) continue;
    causes.set(edge.effect.id, [...(causes.get(edge.effect.id) ?? []), edge.cause.id]);
    effects.set(edge.cause.id, [...(effects.get(edge.cause.id) ?? []), edge.effect.id]);
  }
  return { causes, effects };
}

/** States in causal order: a depth-first finish order from the exogenous inputs, reversed. */
function causalStates(
  constructs: readonly ConstructSpec[],
  effects: Neighbours,
  exogenous: (id: ConstructId) => boolean,
): ConstructId[] {
  const visited = new Set<ConstructId>();
  const finished: ConstructId[] = [];
  const visit = (id: ConstructId): void => {
    visited.add(id);
    for (const next of effects.get(id) ?? []) if (!visited.has(next)) visit(next);
    finished.push(id);
  };
  for (const construct of constructs.toSorted(
    (a, b) => Number(exogenous(b.id)) - Number(exogenous(a.id)),
  )) {
    if (!visited.has(construct.id)) visit(construct.id);
  }
  return finished.toReversed().filter((id) => !exogenous(id));
}

/** The outcome and every construct with a causal path into it. */
function ancestorsOf(outcome: ConstructId | null, causes: Neighbours): ReadonlySet<ConstructId> {
  const ancestors = new Set<ConstructId>();
  const pending = outcome === null ? [] : [outcome];
  for (let id = pending.pop(); id !== undefined; id = pending.pop()) {
    if (ancestors.has(id)) continue;
    ancestors.add(id);
    pending.push(...(causes.get(id) ?? []));
  }
  return ancestors;
}

/**
 * The layout's model order: each state follows the exogenous inputs it is the first to read,
 * in causal order with feedback set aside. States that reach the outcome come last, so the
 * question's block reads last.
 */
export function readingOrder(
  constructs: readonly ConstructSpec[],
  edges: readonly CausalEdgeSpec[],
  outcome: ConstructId | null,
): ConstructSpec[] {
  const byId = new Map(constructs.map((construct) => [construct.id, construct] as const));
  const { causes, effects } = adjacency(edges);
  const exogenous = (id: ConstructId) => byId.get(id)?.role === "exogenous";
  const states = causalStates(constructs, effects, exogenous);
  const ancestors = ancestorsOf(outcome, causes);
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

/** An edge between two drawn constructs, before it is matched to the rows it ends on. */
type Link = Omit<LayeredGraphArrowMeta, "row">;

/** One arrow of the layout graph and what it draws. */
interface PlacedArrow {
  readonly id: string;
  readonly meta: LayeredGraphArrowMeta;
  readonly target: string;
}

/** One carry arrow of the layout graph: where it leaves its node and where it returns. */
interface PlacedCarry {
  readonly id: string;
  readonly meta: LayeredGraphCarryMeta;
  readonly source: string;
  readonly target: string;
}

/** One construct's node, its card rows, and the arrows that end on it. */
interface PlacedNode {
  readonly meta: LayeredGraphNodeMeta;
  readonly node: DagLayoutNodeInput;
  readonly rows: readonly MechanismRow[];
  readonly arrows: readonly PlacedArrow[];
  readonly carries: readonly PlacedCarry[];
}

/**
 * A time-varying construct's carry arrows and the ports on its top edge they use: one arrow into
 * each row that reads the construct's own state, or a short loop on the node when none does.
 */
function placeCarry(
  construct: ConstructSpec,
  rows: readonly MechanismRow[],
  left: number,
): { readonly ports: readonly DagLayoutPortInput[]; readonly carries: readonly PlacedCarry[] } {
  if (!carriesOver(construct)) return { ports: [], carries: [] };
  const start = carryStartOf(construct.id);
  const end = carryEndOf(construct.id);
  const readers = rows.filter((row) => row.reads.includes(construct.id));
  const leave: DagLayoutPortInput = { id: start, side: "NORTH", x: left + CARRY_START };
  if (readers.length === 0) {
    const loop = { id: `${construct.id}>carry`, meta: { construct, row: null } };
    const back: DagLayoutPortInput = {
      id: end,
      side: "NORTH",
      x: left + CARRY_START - CARRY_SPAN,
    };
    return { ports: [leave, back], carries: [{ ...loop, source: start, target: end }] };
  }
  return {
    ports: [leave],
    carries: readers.map((row) => ({
      id: `${construct.id}>carry>${row.mechanism.id}`,
      meta: { construct, row },
      source: start,
      target: portOf(row),
    })),
  };
}

/** An exogenous construct is a chip, and the arrows into it end on the chip itself. */
function placeChip(
  construct: ConstructSpec,
  links: readonly Link[],
  density: GraphDensity,
): PlacedNode {
  const { width, height } = density.chip;
  const carry = placeCarry(construct, [], 0);
  return {
    meta: { kind: "exogenous", construct },
    node: {
      id: construct.id,
      width,
      height,
      ports: [{ id: chipOutletOf(construct.id), side: "EAST", y: height / 2 }, ...carry.ports],
    },
    rows: [],
    arrows: links.map((link) => ({
      id: link.edge.id,
      meta: { ...link, row: null },
      target: construct.id,
    })),
    carries: carry.carries,
  };
}

/**
 * The terms that drive a construct: its own dynamics first, then the mechanisms its incoming
 * edges declare, by their causes' place in the reading order.
 */
export function mechanismTerms(
  construct: ConstructSpec,
  edges: readonly CausalEdgeSpec[],
  rank: ReadonlyMap<ConstructId, number>,
): MechanismTerm[] {
  return [
    ...construct.dynamics.map((mechanism) => ({
      mechanism,
      owner: { kind: "construct", id: construct.id } as const,
    })),
    ...edges
      .filter((edge) => edge.effect.id === construct.id)
      .toSorted((a, b) => (rank.get(a.cause.id) ?? 0) - (rank.get(b.cause.id) ?? 0))
      .flatMap((edge) =>
        edge.mechanisms.map((mechanism) => ({
          mechanism,
          owner: { kind: "edge", id: edge.id } as const,
        })),
      ),
  ].map(({ mechanism, owner }) => ({
    mechanism,
    owner,
    reads: expressionStates([mechanism.expression]),
  }));
}

/** The terms an edge ends on: those that read its cause, else the terms the edge declares. */
export function endsOf<Term extends MechanismTerm>(
  edge: CausalEdgeSpec,
  terms: readonly Term[],
): readonly Term[] {
  const readers = terms.filter((term) => term.reads.includes(edge.cause.id));
  const owned = terms.filter((term) => term.owner.kind === "edge" && term.owner.id === edge.id);
  return readers.length > 0 ? readers : owned;
}

/** An edge's arrows: one into each row it ends on, or one into the card when it ends on none. */
function linkArrows(link: Link, ends: readonly MechanismRow[], cardTarget: string): PlacedArrow[] {
  if (ends.length === 0) {
    return [{ id: `${link.edge.id}>card`, meta: { ...link, row: null }, target: cardTarget }];
  }
  return ends.map((row) => ({
    id: `${link.edge.id}>${row.mechanism.id}`,
    meta: { ...link, row },
    target: portOf(row),
  }));
}

/**
 * A card: a port per mechanism row, one for arrows that end on no row, and the carry arrow's on
 * its top edge. A card with any port has all of them fixed, so only a card with none lets an
 * arrow end on the card itself.
 */
function placeCard(
  construct: ConstructSpec,
  rows: readonly MechanismRow[],
  links: readonly Link[],
  density: GraphDensity,
): PlacedNode {
  const incoming = links.map((link) => ({ link, ends: endsOf(link.edge, rows) }));
  const rowsWidth = rows.length > 0 ? density.row.width : 0;
  const carry = placeCarry(construct, rows, rowsWidth);
  const fixed = rows.length > 0 || carry.ports.length > 0;
  const cardPort = fixed && incoming.some(({ ends }) => ends.length === 0);
  const rowsHeight = 2 * ROW_INSET + rows.length * density.row.height;
  const width = rowsWidth + density.state.width;
  const height = Math.max(
    density.state.height,
    rowsHeight + (cardPort && rows.length > 0 ? CARD_PORT_ROOM : 0),
  );
  // Under the rows when the card has any, else at the middle of its west side.
  const cardPortY = rows.length > 0 ? height - CARD_PORT_ROOM / 2 : height / 2;
  const ports: DagLayoutPortInput[] = [
    ...rows.map((row) => ({
      id: portOf(row),
      side: "WEST" as const,
      y: row.top + density.row.height / 2,
    })),
    ...(cardPort ? [{ id: cardPortOf(construct.id), side: "WEST" as const, y: cardPortY }] : []),
    ...carry.ports,
  ];
  const cardTarget = fixed ? cardPortOf(construct.id) : construct.id;
  return {
    meta: { kind: "construct", construct, rows, width, height },
    node: { id: construct.id, width, height, ...(ports.length > 0 ? { ports } : {}) },
    rows,
    arrows: incoming.flatMap(({ link, ends }) => linkArrows(link, ends, cardTarget)),
    carries: carry.carries,
  };
}

/**
 * Group terms by the construct they drive. The authored view places each term in a separate
 * function box along its arrows. Result and comparison views use rows of the effect card. Each edge ends on the rows that read its cause, and each time-varying
 * construct sends a carry arrow to itself. Exogenous constructs are chips. Comparisons decorate
 * this layout without moving its nodes.
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
    mechanismTerms(construct, edges, rank).map((term, index) => ({
      ...term,
      top: ROW_INSET + index * density.row.height,
    }));

  const placed = order.map((construct) => {
    const incoming = links.filter((link) => link.effect.id === construct.id);
    return construct.role === "exogenous"
      ? placeChip(construct, incoming, density)
      : placeCard(construct, rowsOf(construct), incoming, density);
  });
  const arrows = placed.flatMap((item) => item.arrows);
  const carries = placed.flatMap((item) => item.carries);

  return {
    density,
    graph: {
      nodes: placed.map((item) => item.node),
      // Carry arrows follow the model's edges, so they never reorder them.
      edges: [
        ...arrows.map((arrow) => ({
          id: arrow.id,
          source: outletOf(arrow.meta.cause),
          target: arrow.target,
        })),
        ...carries.map(({ id, source, target }) => ({ id, source, target })),
      ],
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
    nodeMeta: new Map(placed.map((item) => [item.meta.construct.id, item.meta])),
    arrowMeta: new Map(arrows.map((arrow) => [arrow.id, arrow.meta])),
    carryMeta: new Map(carries.map((carry) => [carry.id, carry.meta])),
    rows: new Map(
      placed.flatMap((item) =>
        item.rows.map(
          (row) => [row.mechanism.id, { row, effect: item.meta.construct.id }] as const,
        ),
      ),
    ),
  };
}
