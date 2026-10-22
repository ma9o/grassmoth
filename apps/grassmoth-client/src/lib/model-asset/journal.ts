import type { TimelineRevision } from "@grassmoth/api-types";

export function attemptError(
  outcome: TimelineRevision["record"]["attempt"]["outcome"],
): string | null {
  switch (outcome.status) {
    case "applied":
      return null;
    case "rejected":
      return outcome.detail;
    case "raised":
      return `${outcome.error_type}: ${outcome.error_message}`;
  }
}

export function latestSeq(attempts: readonly TimelineRevision[]): number {
  return attempts.reduce(
    (max, revision) =>
      revision.record.attempt.outcome.status === "applied" &&
      revision.record.attempt.request !== null &&
      revision.record.attempt.action !== "data_diff" &&
      revision.record.attempt.action !== "model_diff"
        ? Math.max(max, revision.record.seq)
        : max,
    0,
  );
}
