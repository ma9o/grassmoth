import type {
  CoefficientRole,
  ConstructSpec,
  CausalEdgeSpec,
  ParameterSpec,
  Expression,
  DynamicalModelSpec,
  ParameterId,
} from "@grassmoth/api-types";
import { assertNever } from "./assert-never";

/** Enumerate present entries without losing the trusted API's named key type. */
export function presentEntries<Key extends string, Value>(
  record: Readonly<Partial<Record<Key, Value | undefined>>>,
): [Key, Value][] {
  return Object.entries<Value | undefined>(record).flatMap(([key, value]) => {
    if (value === undefined) return [];
    // eslint-disable-next-line @typescript-eslint/no-unsafe-type-assertion -- Object.entries erases the key type of this trusted API-owned sparse map.
    return [[key as Key, value] satisfies [Key, Value]];
  });
}

/** Expand the document's keyed definitions into entities used by the viewer. */
export function modelConstructs(
  dynamicalModelSpec: DynamicalModelSpec | null | undefined,
): ConstructSpec[] {
  return presentEntries(dynamicalModelSpec?.constructs ?? {}).map(([id, value]) => ({
    ...value,
    id,
    dynamics: presentEntries(value.dynamics).map(([id, mechanism]) => ({ ...mechanism, id })),
    indicators: presentEntries(value.indicators).map(([id, indicator]) => ({
      ...indicator,
      observation: { ...indicator.observation, id },
    })),
  }));
}

/** Edge endpoints identify the construct definitions held by the same document. */
export function modelEdges(
  dynamicalModelSpec: DynamicalModelSpec | null | undefined,
): CausalEdgeSpec[] {
  return presentEntries(dynamicalModelSpec?.edges ?? {}).map(([id, edge]) => ({
    ...edge,
    id,
    cause: { kind: "construct", id: edge.cause },
    effect: { kind: "construct", id: edge.effect },
    mechanisms: presentEntries(edge.mechanisms).map(([id, mechanism]) => ({ ...mechanism, id })),
  }));
}

/** Restore each parameter's map identity for display and entity selection. */
export function modelParameters(
  dynamicalModelSpec: DynamicalModelSpec | null | undefined,
): ParameterSpec[] {
  return presentEntries(dynamicalModelSpec?.parameters ?? {}).map(([id, value]) => ({
    ...value,
    id,
  }));
}

/** A coefficient operand that names a parameter, with the quantity its authored role declares. */
export interface CoefficientUse {
  role: CoefficientRole;
  parameterId: ParameterId;
}

/** Follow serialized coefficient references in declaration order, keeping each parameter's first role. */
export function coefficientUses(expressions: readonly Expression[]): CoefficientUse[] {
  const uses = new Map<ParameterId, CoefficientUse>();
  const visit = (operand: Expression): void => {
    switch (operand.kind) {
      case "coefficient":
        if (typeof operand.value === "string" && !uses.has(operand.value))
          uses.set(operand.value, { role: operand.role, parameterId: operand.value });
        return;
      case "binary":
        visit(operand.left);
        visit(operand.right);
        return;
      case "call":
        operand.arguments.forEach(visit);
        return;
      case "literal":
      case "state":
        return;
      default:
        assertNever(operand);
    }
  };
  expressions.forEach(visit);
  return [...uses.values()];
}
