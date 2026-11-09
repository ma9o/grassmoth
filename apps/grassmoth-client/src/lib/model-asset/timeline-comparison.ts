import type { ActionId, TimelineRevision } from "@grassmoth/api-types";

type ComparableAction = Exclude<ActionId, "data_diff" | "model_diff">;
type TimelineAttempt = TimelineRevision["record"]["attempt"];
type ComparableRevision<Action extends ComparableAction> = TimelineRevision & {
  readonly record: {
    readonly attempt: {
      readonly action: Action;
      readonly outcome: Extract<TimelineAttempt["outcome"], { status: "applied" }>;
    };
  };
};

/** A union of same-action pairs, never a pair of independently chosen action unions. */
export type TimelineComparison = {
  [Action in ComparableAction]: {
    readonly action: Action;
    readonly before: ComparableRevision<Action>;
    readonly after: ComparableRevision<Action>;
  };
}[ComparableAction];

function isComparableRevision<Action extends ComparableAction>(
  revision: TimelineRevision,
  action: Action,
): revision is ComparableRevision<Action> {
  return (
    revision.record.attempt.action === action &&
    revision.record.attempt.outcome.status === "applied"
  );
}

function pairForAction<Action extends ComparableAction>(
  action: Action,
  before: TimelineRevision,
  after: TimelineRevision,
) {
  return isComparableRevision(before, action) && isComparableRevision(after, action)
    ? { action, before, after }
    : null;
}

/** Parse timeline selections once; all preview consumers receive the proven pair. */
export function timelineComparison(
  before: TimelineRevision | undefined,
  after: TimelineRevision | undefined,
): TimelineComparison | null {
  if (!before || !after || before.record.seq === after.record.seq) return null;
  switch (before.record.attempt.action) {
    case "edit_model":
      return pairForAction("edit_model", before, after);
    case "prepare_data":
      return pairForAction("prepare_data", before, after);
    case "fit":
      return pairForAction("fit", before, after);
    case "simulate":
      return pairForAction("simulate", before, after);
    case "data_diff":
    case "model_diff":
      return null;
  }
}
