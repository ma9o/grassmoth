import { describe, expect, it } from "vitest";
import { applied, comparison, failedFit, revision, simulation } from "@/lib/__fixtures__/timeline";
import { attemptError, latestSeq } from "@/lib/model-asset/journal";

describe("owned journal outcomes", () => {
  it("presents rejection and execution failure details from their respective variants", () => {
    expect(
      attemptError({
        status: "rejected",
        code: "INPUT_UNAVAILABLE",
        subject: "inputs",
        detail: "inputs missing",
      }),
    ).toBe("inputs missing");
    expect(attemptError(failedFit.record.attempt.outcome)).toBe("FitError: Fit failed.");
    expect(attemptError(applied)).toBeNull();
  });
  it("excludes failed attempts and read-only comparison leaves from scientific revisions", () => {
    const rejected = revision(7, {
      ...simulation.record.attempt,
      outcome: {
        status: "rejected",
        code: "INPUT_UNAVAILABLE",
        subject: "inputs",
        detail: "inputs missing",
      },
    });
    const unknown = revision(8, { action: "edit_model", request: null, outcome: applied });
    const modelComparison = revision(9, {
      action: "model_diff",
      request: {
        action: "model_diff",
        input: { before_ref: "1".repeat(40), after_ref: "3".repeat(40) },
        reasoning: null,
      },
      outcome: applied,
    });
    const raised = revision(10, {
      ...simulation.record.attempt,
      outcome: failedFit.record.attempt.outcome,
    });
    expect(latestSeq([simulation, comparison, rejected, unknown, modelComparison, raised])).toBe(5);
  });
  it("keeps a successful simulation checkpoint without an artifact output", () => {
    const outcome = simulation.record.attempt.outcome;
    expect(outcome.status).toBe("applied");
    if (outcome.status !== "applied") throw new Error("Expected a successful fixture simulation");
    expect(outcome.effects.produced).toEqual([]);
    expect(latestSeq([failedFit, simulation])).toBe(5);
  });
});
