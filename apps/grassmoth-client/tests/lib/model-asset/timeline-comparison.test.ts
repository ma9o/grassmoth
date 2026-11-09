import { describe, expect, it } from "vitest";
import { comparisonRevisions, failedFit, rejectedFit } from "@/lib/__fixtures__/timeline";
import { timelineComparison, type TimelineComparison } from "@/lib/model-asset/timeline-comparison";

type Expect<T extends true> = T;
type FitComparison = Extract<TimelineComparison, { action: "fit" }>;
type EditComparison = Extract<TimelineComparison, { action: "edit_model" }>;

export type MixedActionsAreNotAComparison = Expect<
  Omit<FitComparison, "after"> & {
    readonly after: EditComparison["after"];
  } extends TimelineComparison
    ? false
    : true
>;
export type IndependentActionUnionsAreNotAComparison = Expect<
  {
    readonly action: TimelineComparison["action"];
    readonly before: TimelineComparison["before"];
    readonly after: TimelineComparison["after"];
  } extends TimelineComparison
    ? false
    : true
>;

describe("timeline comparison selection", () => {
  it("allows only matching successful action types and excludes diff actions", () => {
    for (const left of comparisonRevisions) {
      for (const right of comparisonRevisions) {
        const pair = timelineComparison(left.before, right.after);
        const eligible =
          left.action === right.action &&
          left.action !== "data_diff" &&
          left.action !== "model_diff";
        expect(pair, `${left.action} → ${right.action}`).toEqual(
          eligible ? { action: left.action, before: left.before, after: right.after } : null,
        );
      }
    }
  });

  it("excludes failed, rejected, missing and identical selections on either side", () => {
    for (const { before } of comparisonRevisions) {
      expect(timelineComparison(before, before)).toBeNull();
      for (const unavailable of [undefined, failedFit, rejectedFit]) {
        expect(timelineComparison(before, unavailable)).toBeNull();
        expect(timelineComparison(unavailable, before)).toBeNull();
      }
    }
  });
});
