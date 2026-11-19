import type { CausalEdgeSpec, ConstructId, ConstructSpec, MechanismId } from "@grassmoth/api-types";
import { type EntitySelection, humanize } from "@/lib/model-asset/selection";
import { glyphKey, type SpecGlyph, type SpecPlots } from "@/lib/model-asset/spec-plots";
import type { DagLayoutNode, Point } from "@/lib/utils/dag-graph-layout";
import type { LayeredGraphBundle } from "./build-layered-causal-graph";
import { functionHeight, type IndicatorBox, indicatorStack, wrapLabel } from "./indicator-layout";

export type ResponseGlyph = Extract<SpecGlyph, { kind: "response" }>;
export interface FunctionBox extends DagLayoutNode {
  readonly glyph: ResponseGlyph;
  readonly name: readonly string[];
  readonly mechanism: MechanismId | null;
}
export interface SpecLink {
  readonly mechanism: MechanismId | null;
  readonly source: string;
  readonly target: string;
  readonly points: readonly Point[];
  readonly carry: boolean;
  readonly selection: EntitySelection;
}

const GAP = 12;
const TERM_WIDTH = 195;
const MEASUREMENT_WIDTH = 154;
const BRANCH_START = 12;
const BRANCH_FUNCTION = 34;

function functionSize(
  glyph: ResponseGlyph,
  mechanism: MechanismId | null,
): Omit<FunctionBox, "x" | "y"> {
  const width = mechanism ? TERM_WIDTH : MEASUREMENT_WIDTH;
  const name = wrapLabel(
    mechanism ? glyph.label : "Expected reading",
    Math.floor((width - 16) / 6.8),
  );
  const height = functionHeight(name.length, glyph.slices.length);
  return { id: mechanism ?? glyphKey(glyph.selection), glyph, name, mechanism, width, height };
}

interface Column {
  readonly state: ConstructSpec;
  readonly functions: readonly ReturnType<typeof functionSize>[];
  readonly indicators: readonly IndicatorBox[];
}

function columnsFor(
  bundle: LayeredGraphBundle,
  plots: SpecPlots,
  edges: readonly CausalEdgeSpec[],
) {
  const entries = [...bundle.nodeMeta.values()];
  const columns: Column[] = entries.flatMap((meta) => {
    if (meta.kind !== "construct") return [];
    const functions = meta.rows.flatMap((row) => {
      const glyph = plots.get(glyphKey({ ...row.owner, glyph: row.mechanism.id }));
      return glyph?.kind === "response" ? [functionSize(glyph, row.mechanism.id)] : [];
    });
    return [
      {
        state: meta.construct,
        functions,
        indicators: indicatorStack(meta.construct, bundle.density.state.height, plots),
      },
    ];
  });
  const chips = new Map<ConstructId, ConstructSpec[]>();
  const loose: ConstructSpec[] = [];
  for (const { construct } of entries.filter((meta) => meta.kind === "exogenous")) {
    const feeds = (glyph: ResponseGlyph) => glyph.slices.some((s) => s.input === construct.id);
    const first = columns.find(
      (col) =>
        col.functions.some((f) => feeds(f.glyph)) ||
        col.indicators.some((i) => feeds(i.response)) ||
        edges.some((e) => e.cause.id === construct.id && e.effect.id === col.state.id),
    );
    if (first) chips.set(first.state.id, [...(chips.get(first.state.id) ?? []), construct]);
    else loose.push(construct);
  }
  return { columns: columns.map((c) => ({ ...c, known: chips.get(c.state.id) ?? [] })), loose };
}

const centerY = (box: DagLayoutNode) => box.y + box.height / 2;
const centerX = (box: DagLayoutNode) => box.x + box.width / 2;
const stackHeight = (boxes: readonly { readonly height: number }[]) =>
  boxes.reduce((h, b) => h + GAP + b.height, -GAP);

interface Placement {
  readonly node: DagLayoutNode;
  readonly column: number;
  readonly known: boolean;
}
interface Target {
  readonly node: DagLayoutNode;
  readonly state: Placement;
  readonly branch: boolean;
  readonly rank: number;
  readonly count: number;
}

function outputLink(box: FunctionBox, target: DagLayoutNode, port: number): SpecLink {
  const start = { x: box.x + box.width, y: centerY(box) };
  const end = { x: target.x, y: target.y + target.height * port };
  const bend = (start.x + end.x) / 2;
  return {
    mechanism: box.mechanism,
    source: box.id,
    target: target.id,
    selection: box.glyph.selection,
    carry: false,
    points: [start, { x: bend, y: start.y }, { x: bend, y: end.y }, end],
  };
}

/** Long and feedback connections run above the states. Each source has its own horizontal track. */
function inputPath(source: Placement, target: Target, trackY: number, port: number): Point[] {
  const { node: from } = source;
  const { node: to, state, branch, rank, count } = target;
  const end = { x: to.x, y: to.y + to.height * port };
  if (branch && from.id === state.node.id) {
    const x = from.x + BRANCH_START;
    return [{ x, y: from.y + from.height }, { x, y: end.y }, end];
  }
  if (!branch && from.id === state.node.id && rank === 0) {
    const start = { x: from.x + 32, y: from.y };
    const top = Math.min(start.y, to.y) - 14;
    const x = centerX(to);
    return [start, { x: start.x, y: top }, { x, y: top }, { x, y: to.y }];
  }
  if (to === state.node) {
    const x = centerX(from);
    const start = { x, y: source.known ? from.y + from.height : from.y };
    const targetX = to.x + to.width * 0.75;
    return [start, { x, y: trackY }, { x: targetX, y: trackY }, { x: targetX, y: to.y }];
  }
  const approach = to.x - 10 - 18 * ((rank + port) / (count + 1));
  if (!source.known && !branch && source.column + 1 === state.column) {
    const start = { x: from.x + from.width, y: centerY(from) };
    return [start, { x: approach, y: start.y }, { x: approach, y: end.y }, end];
  }
  const x = source.known ? from.x - 12 : centerX(from);
  const start = source.known ? { x: from.x, y: centerY(from) } : { x, y: from.y };
  // The approach to a reading passes just outside the state card before turning under it.
  const approachX = branch ? state.node.x - 10 - rank * 2 : approach;
  return [
    start,
    { x, y: start.y },
    { x, y: trackY },
    { x: approachX, y: trackY },
    { x: approachX, y: end.y },
    end,
  ];
}

function functionLinks(box: FunctionBox, target: Target, context: RouteContext): SpecLink[] {
  const { state } = target;
  const group = box.glyph.selection;
  const inputs = new Set(box.glyph.slices.flatMap((s) => (s.input === null ? [] : [s.input])));
  // The owning state's carry travels through its dynamics even for a constant drift.
  if (group.kind === "construct" || target.branch) inputs.add(box.glyph.construct);
  return [...inputs].flatMap((id, i) => {
    const source = context.variables.get(id);
    if (!source) return [];
    const carry = id === state.node.id && !target.branch && !source.known;
    const edge = context.edges.find((e) => e.cause.id === id && e.effect.id === state.node.id);
    const selection: EntitySelection =
      edge && !target.branch && !carry ? { kind: "edge", id: edge.id } : group;
    return [
      {
        mechanism: box.mechanism,
        source: id,
        target: box.id,
        selection,
        carry,
        points: inputPath(source, target, context.track(id), (i + 1) / (inputs.size + 1)),
      },
    ];
  });
}

interface RouteContext {
  readonly variables: ReadonlyMap<ConstructId, Placement>;
  readonly track: (id: ConstructId) => number;
  readonly edges: readonly CausalEdgeSpec[];
}

function structuralLinks(context: RouteContext, functions: readonly FunctionBox[]): SpecLink[] {
  return context.edges.flatMap((edge) => {
    const from = context.variables.get(edge.cause.id);
    const to = context.variables.get(edge.effect.id);
    if (!from || !to || from === to || edge.mechanisms.length > 0) return [];
    if (
      functions.some(
        (f) =>
          f.mechanism &&
          f.glyph.construct === edge.effect.id &&
          f.glyph.slices.some((s) => s.input === edge.cause.id),
      )
    ) {
      return [];
    }
    const target = { node: to.node, state: to, branch: false, rank: 0, count: 1 };
    return [
      {
        mechanism: null,
        source: edge.cause.id,
        target: edge.effect.id,
        selection: { kind: "edge" as const, id: edge.id },
        carry: false,
        points: inputPath(from, target, context.track(edge.cause.id), 0.5),
      },
    ];
  });
}

function carryLoop(construct: ConstructSpec, node: DagLayoutNode): SpecLink {
  const start = { x: node.x + 54, y: node.y };
  const end = { x: node.x + 24, y: node.y };
  return {
    mechanism: null,
    source: node.id,
    target: node.id,
    selection: { kind: "construct", id: construct.id },
    carry: true,
    points: [start, { x: start.x, y: start.y - 12 }, { x: end.x, y: end.y - 12 }, end],
  };
}

function unconnectedCarries(
  bundle: LayeredGraphBundle,
  variables: ReadonlyMap<ConstructId, Placement>,
  links: readonly SpecLink[],
) {
  return [...bundle.nodeMeta.values()].flatMap(({ construct }) => {
    const placed = variables.get(construct.id);
    return placed &&
      construct.temporal_status === "time_varying" &&
      !links.some((link) => link.carry && link.source === construct.id)
      ? [carryLoop(construct, placed.node)]
      : [];
  });
}

/** Fixed variable lanes preserve causal reading order. Terms occupy the gap before their effect;
 * measurement functions and distributions form an indented stack below their state. */
export function specLayout(
  bundle: LayeredGraphBundle,
  plots: SpecPlots,
  edges: readonly CausalEdgeSpec[],
) {
  const { columns, loose } = columnsFor(bundle, plots, edges);
  const { state, chip } = bundle.density;
  const chipRows = Math.max(0, ...columns.map((c) => c.known.length), Number(loose.length > 0));
  const railTop = chipRows * (chip.height + 18) + 16;
  const index = new Map([...bundle.nodeMeta.keys()].map((id, i) => [id, i]));
  const rail = (id: ConstructId) => railTop + (index.get(id) ?? 0) * 7;
  const termsHeight = Math.max(state.height, ...columns.map((c) => stackHeight(c.functions)));
  const middle = railTop + index.size * 7 + 14 + termsHeight / 2;
  const variables = new Map<ConstructId, Placement>();
  const functions: FunctionBox[] = [];
  const indicators = new Map<ConstructId, readonly IndicatorBox[]>();
  const targets = new Map<string, Target>();
  const links: SpecLink[] = [];
  const addChip = (construct: ConstructSpec, x: number, row: number, col: number) => {
    const node = { id: construct.id, x, y: 12 + row * (chip.height + 18), ...chip };
    variables.set(construct.id, { node, column: col, known: true });
  };
  let x = 34;
  let bottom = railTop;
  for (const [colIndex, col] of columns.entries()) {
    const stateX = x + (col.functions.length > 0 ? TERM_WIDTH + 45 : 0);
    const node = {
      id: col.state.id,
      x: stateX,
      y: middle - state.height / 2,
      width: state.width,
      height: state.height,
    };
    const placement = { node, column: colIndex, known: false };
    variables.set(col.state.id, placement);
    for (const [i, known] of col.known.entries()) addChip(known, stateX, i, colIndex);
    let y = middle - stackHeight(col.functions) / 2;
    for (const [rank, term] of col.functions.entries()) {
      const box = { ...term, x, y };
      functions.push(box);
      targets.set(box.id, {
        node: box,
        state: placement,
        branch: false,
        rank,
        count: col.functions.length,
      });
      links.push(outputLink(box, node, (rank + 1) / (col.functions.length + 1)));
      y += box.height + GAP;
    }
    indicators.set(col.state.id, col.indicators);
    for (const [rank, indicator] of col.indicators.entries()) {
      const size = functionSize(indicator.response, null);
      const dest = {
        id: indicator.distribution.selection.id,
        width: indicator.width,
        height: indicator.height,
        x: stateX + indicator.x,
        y: node.y + indicator.y,
      };
      const box = { ...size, x: stateX + BRANCH_FUNCTION, y: centerY(dest) - size.height / 2 };
      functions.push(box);
      targets.set(box.id, {
        node: box,
        state: placement,
        branch: true,
        rank,
        count: col.indicators.length,
      });
      links.push(outputLink(box, dest, 0.5));
      bottom = Math.max(bottom, dest.y + dest.height, box.y + box.height);
    }
    const span = Math.max(state.width, ...col.indicators.map((i) => i.x + i.width));
    x = stateX + span + 48;
    bottom = Math.max(bottom, node.y + node.height, y - GAP);
  }
  for (const [i, known] of loose.entries()) {
    addChip(known, x, 0, columns.length + i);
    x += chip.width + 48;
  }
  const context = { variables, edges, track: rail };
  for (const box of functions) {
    const target = targets.get(box.id);
    if (target) links.push(...functionLinks(box, target, context));
  }
  links.push(
    ...structuralLinks(context, functions),
    ...unconnectedCarries(bundle, variables, links),
  );
  return {
    nodes: [...variables.values()].map((v) => v.node),
    functions,
    indicators,
    labels: new Map([
      ...[...bundle.nodeMeta].map(([id, meta]) => [id, humanize(meta.construct.name)] as const),
      ...functions.map(
        (f) =>
          [f.id, f.mechanism ? f.glyph.label : `Expected reading for ${f.glyph.label}`] as const,
      ),
      ...[...indicators.values()]
        .flat()
        .map((box) => [box.distribution.selection.id, box.distribution.label] as const),
    ]),
    links,
    width: Math.max(0, x - 48),
    height: bottom,
    edges: [],
    isLayouting: false,
  };
}
