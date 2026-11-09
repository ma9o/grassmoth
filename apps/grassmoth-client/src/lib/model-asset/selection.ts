import type {
  ComputedExtractionSpec,
  ConstructId,
  EdgeId,
  IndicatorId,
  MechanismId,
} from "@grassmoth/api-types";

/** The graph entity shown in the details pane. */
export type EntitySelection =
  | { readonly kind: "construct"; readonly id: ConstructId }
  | { readonly kind: "edge"; readonly id: EdgeId }
  | { readonly kind: "indicator"; readonly id: IndicatorId };

export function humanize(value: string): string {
  return value.replaceAll("_", " ");
}

/** A mechanism's id is its only name in the API. */
export function mechanismName(id: MechanismId): string {
  return humanize(id.slice("mechanism:".length));
}

export function formatFillNull(
  extraction: Pick<ComputedExtractionSpec, "fill_null" | "fill_null_limit">,
): string {
  if (extraction.fill_null == null) return "None";
  const method = String(extraction.fill_null);
  return extraction.fill_null_limit == null
    ? method
    : `${method} (limit ${extraction.fill_null_limit})`;
}

export function formatPlain(value: number, digits = 2): string {
  return `${value < 0 ? "−" : ""}${Math.abs(value).toFixed(digits)}`;
}
