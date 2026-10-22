import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import type { TimelineRevision } from "@grassmoth/api-types";
import { QueryClient, skipToken, useQuery } from "@tanstack/react-query";
import { beforeEach, expect, it, vi } from "vitest";
import { readActionResult } from "@/lib/api/endpoints";
import { useModelDiff } from "@/lib/hooks/use-model-diff";

vi.mock("@tanstack/react-query", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@tanstack/react-query")>()),
  useQuery: vi.fn(),
}));
vi.mock("@/lib/api/endpoints", () => ({ readActionResult: vi.fn() }));

const comparison: TimelineRevision = {
  call_id: `call:${"3".repeat(64)}`,
  commit_id: "saved-comparison",
  parent_ids: [],
  record: {
    seq: 1,
    ts: "2026-10-06T12:00:00Z",
    messages: [],
    trace_ids: [],
    attempt: {
      action: "model_diff",
      request: {
        action: "model_diff",
        input: { before_ref: "before", after_ref: "after" },
        reasoning: "Compare the recorded model revisions.",
      },
      outcome: {
        status: "applied",
        result: null,
        effects: { produced: [], retracted: [], reports: {} },
      },
    },
  },
};

beforeEach(() => vi.clearAllMocks());

it("replays only the saved comparison with its original arguments", async () => {
  useModelDiff("STUDY", "before", "after", [comparison]);
  const options = fixtureValue(vi.mocked(useQuery).mock.calls.at(-1))[0];
  const queryFn = options.queryFn;
  if (typeof queryFn !== "function") throw new Error("Expected a saved comparison query");
  const signal = new AbortController().signal;
  await queryFn({ queryKey: options.queryKey, signal, client: new QueryClient(), meta: undefined });
  expect(readActionResult).toHaveBeenCalledExactlyOnceWith("STUDY", comparison, signal);
});

it("never submits missing, reversed, failed or unknown comparison calls", () => {
  const failed: TimelineRevision = {
    ...comparison,
    record: {
      ...comparison.record,
      attempt: {
        ...comparison.record.attempt,
        outcome: {
          status: "rejected",
          code: "INPUT_UNAVAILABLE",
          subject: "inputs",
          detail: "Missing model",
        },
      },
    },
  };
  const unknown: TimelineRevision = {
    ...comparison,
    record: {
      ...comparison.record,
      attempt: { ...comparison.record.attempt, request: null },
    },
  };
  useModelDiff("STUDY", "before", "after", []);
  useModelDiff("STUDY", "after", "before", [comparison]);
  useModelDiff("STUDY", "before", "after", [failed, unknown]);
  for (const [options] of vi.mocked(useQuery).mock.calls) expect(options.queryFn).toBe(skipToken);
  expect(readActionResult).not.toHaveBeenCalled();
});
