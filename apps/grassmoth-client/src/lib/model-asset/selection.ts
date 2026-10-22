import {
  type ConstructId,
  type EdgeId,
  type IndicatorId,
  type ComputedExtractionSpec,
} from "@grassmoth/api-types";

/** The graph entity shown in the details pane. */
export type EntitySelection =
  | { kind: "construct"; id: ConstructId }
  | { kind: "edge"; id: EdgeId }
  | { kind: "indicator"; id: IndicatorId };

export function humanize(value: string): string {
  return value.replaceAll("_", " ");
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
