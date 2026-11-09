import type { TimelineRevision } from "@grassmoth/api-types";

type TimelineSelection =
  | { readonly status: "ready"; readonly seq: number | null }
  | { readonly status: "invalid"; readonly message: string };

/** Parse a pinned action only after the study journal has loaded. */
export function timelineSelection(
  params: URLSearchParams,
  attempts: readonly TimelineRevision[],
): TimelineSelection {
  const values = params.getAll("seq");
  if (values.length === 0) return { status: "ready", seq: null };
  const value = values[0];
  if (
    values.length !== 1 ||
    value === undefined ||
    !/^(0|[1-9]\d*)$/.test(value) ||
    !Number.isSafeInteger(Number(value))
  ) {
    return {
      status: "invalid",
      message: "The timeline URL must contain one non-negative integer seq.",
    };
  }
  const seq = Number(value);
  return attempts.some((entry) => entry.record.seq === seq)
    ? { status: "ready", seq }
    : { status: "invalid", message: `Action ${seq} is not in this study's timeline.` };
}

export function timelineHref(
  pathname: string,
  params: URLSearchParams,
  seq: number | null,
): string {
  const next = new URLSearchParams(params);
  if (seq === null) next.delete("seq");
  else next.set("seq", String(seq));
  const query = next.toString();
  return query ? `${pathname}?${query}` : pathname;
}

export interface TimelineNavigation {
  readonly seq: number | null;
  readonly hrefAt: (seq: number | null) => string;
  readonly viewAt: (seq: number | null) => void;
}
