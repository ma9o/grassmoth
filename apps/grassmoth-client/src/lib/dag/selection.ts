import type { ConstructId, EdgeId, IndicatorId, MechanismId } from "@grassmoth/api-types";
import type { EntitySelection } from "@/lib/model-asset/selection";
import type {
  LayeredGraphArrowMeta,
  LayeredGraphBundle,
  LayeredGraphNodeMeta,
} from "./build-layered-causal-graph";

/** What the pointer or the selection is on: an entity, or one mechanism row of a card. */
export type GraphFocus = EntitySelection | { readonly kind: "mechanism"; readonly id: MechanismId };

export interface FocusedParts {
  readonly constructs: ReadonlySet<ConstructId>;
  readonly mechanisms: ReadonlySet<MechanismId>;
  readonly edges: ReadonlySet<EdgeId>;
}

/** Touched parts in the order a focus reaches them; duplicates collapse when they are joined. */
interface TouchedParts {
  readonly constructs: readonly ConstructId[];
  readonly mechanisms: readonly MechanismId[];
  readonly edges: readonly EdgeId[];
}

function joinParts(...touched: readonly TouchedParts[]): FocusedParts {
  return {
    constructs: new Set(touched.flatMap((parts) => parts.constructs)),
    mechanisms: new Set(touched.flatMap((parts) => parts.mechanisms)),
    edges: new Set(touched.flatMap((parts) => parts.edges)),
  };
}

/** Each drawn arrow that matches: its edge, both endpoints and the row it ends on. */
function arrowParts(
  topology: LayeredGraphBundle,
  matches: (arrow: LayeredGraphArrowMeta) => boolean,
): TouchedParts {
  const arrows = [...topology.arrowMeta.values()].filter((arrow) => matches(arrow));
  return {
    constructs: arrows.flatMap((arrow) => [arrow.cause.id, arrow.effect.id]),
    mechanisms: arrows.flatMap((arrow) => (arrow.row ? [arrow.row.mechanism.id] : [])),
    edges: arrows.map((arrow) => arrow.edge.id),
  };
}

/** The mechanism rows of a construct's card; an exogenous chip has none. */
function cardMechanisms(meta: LayeredGraphNodeMeta | undefined): MechanismId[] {
  return meta?.kind === "construct" ? meta.rows.map((row) => row.mechanism.id) : [];
}

function constructParts(id: ConstructId, topology: LayeredGraphBundle): FocusedParts {
  return joinParts(
    { constructs: [id], mechanisms: cardMechanisms(topology.nodeMeta.get(id)), edges: [] },
    arrowParts(topology, (arrow) => arrow.cause.id === id || arrow.effect.id === id),
  );
}

function edgeParts(id: EdgeId, topology: LayeredGraphBundle): FocusedParts {
  const owned = [...topology.rows.values()].filter(
    ({ row }) => row.owner.kind === "edge" && row.owner.id === id,
  );
  return joinParts(
    arrowParts(topology, (arrow) => arrow.edge.id === id),
    {
      constructs: owned.map(({ effect }) => effect),
      mechanisms: owned.map(({ row }) => row.mechanism.id),
      edges: [],
    },
  );
}

function mechanismParts(id: MechanismId, topology: LayeredGraphBundle): FocusedParts {
  const entry = topology.rows.get(id);
  return joinParts(
    { constructs: entry ? [entry.effect] : [], mechanisms: [id], edges: [] },
    arrowParts(topology, (arrow) => arrow.row?.mechanism.id === id),
  );
}

function indicatorParts(id: IndicatorId, topology: LayeredGraphBundle): FocusedParts {
  const owners = [...topology.nodeMeta.values()].filter((meta) =>
    meta.construct.indicators.some((indicator) => indicator.observation.id === id),
  );
  return joinParts({
    constructs: owners.map((meta) => meta.construct.id),
    mechanisms: owners.flatMap((meta) => cardMechanisms(meta)),
    edges: [],
  });
}

/** The parts a focus touches, used only for highlighting; everything else recedes. */
export function focusedParts(focus: GraphFocus, topology: LayeredGraphBundle): FocusedParts {
  switch (focus.kind) {
    case "construct": {
      return constructParts(focus.id, topology);
    }
    case "edge": {
      return edgeParts(focus.id, topology);
    }
    case "mechanism": {
      return mechanismParts(focus.id, topology);
    }
    case "indicator": {
      return indicatorParts(focus.id, topology);
    }
  }
}
