import type { ConstructId, EdgeId, MechanismId } from "@grassmoth/api-types";
import type { EntitySelection } from "@/lib/model-asset/selection";
import type { LayeredGraphArrowMeta, LayeredGraphBundle } from "./build-layered-causal-graph";

/** What the pointer or the selection is on: an entity, or one mechanism row of a card. */
export type GraphFocus = EntitySelection | { kind: "mechanism"; id: MechanismId };

export interface FocusedParts {
  readonly constructs: ReadonlySet<ConstructId>;
  readonly mechanisms: ReadonlySet<MechanismId>;
  readonly edges: ReadonlySet<EdgeId>;
}

/** The parts a focus touches, used only for highlighting; everything else recedes. */
export function focusedParts(focus: GraphFocus, topology: LayeredGraphBundle): FocusedParts {
  const constructs = new Set<ConstructId>();
  const mechanisms = new Set<MechanismId>();
  const edges = new Set<EdgeId>();
  const arrows = [...topology.arrowMeta.values()];
  const add = (arrow: LayeredGraphArrowMeta) => {
    edges.add(arrow.edge.id);
    constructs.add(arrow.cause.id);
    constructs.add(arrow.effect.id);
    if (arrow.row) mechanisms.add(arrow.row.mechanism.id);
  };
  switch (focus.kind) {
    case "construct": {
      constructs.add(focus.id);
      const meta = topology.nodeMeta.get(focus.id);
      if (meta?.kind === "construct") for (const row of meta.rows) mechanisms.add(row.mechanism.id);
      for (const arrow of arrows)
        if (arrow.cause.id === focus.id || arrow.effect.id === focus.id) add(arrow);
      break;
    }
    case "edge":
      for (const arrow of arrows) if (arrow.edge.id === focus.id) add(arrow);
      for (const { row, effect } of topology.rows.values())
        if (row.owner.kind === "edge" && row.owner.id === focus.id) {
          mechanisms.add(row.mechanism.id);
          constructs.add(effect);
        }
      break;
    case "mechanism": {
      const entry = topology.rows.get(focus.id);
      if (entry) constructs.add(entry.effect);
      mechanisms.add(focus.id);
      for (const arrow of arrows) if (arrow.row?.mechanism.id === focus.id) add(arrow);
      break;
    }
    case "indicator":
      for (const meta of topology.nodeMeta.values())
        if (meta.construct.indicators.some((indicator) => indicator.observation.id === focus.id)) {
          constructs.add(meta.construct.id);
          if (meta.kind === "construct")
            for (const row of meta.rows) mechanisms.add(row.mechanism.id);
        }
      break;
  }
  return { constructs, mechanisms, edges };
}
