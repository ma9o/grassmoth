import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import type { ProgressEvent } from "@grassmoth/api-types";
import { describe, expect, it } from "vitest";
import { applyProgressEvents, EMPTY_ATTEMPT_PROGRESS } from "@/lib/attempt-progress";

const attempt_id = "0f17a770-5d1e-4c2b-9a3f-6b8e2d4c1a90";
const events: ProgressEvent[] = [
  {
    attempt_id,
    cursor: "01-a.json",
    event: "grassmoth.step",
    status: "completed",
    error: null,
  },
  {
    attempt_id,
    cursor: "02-b.json",
    event: "grassmoth.extraction.plan",
    total_workers: 2,
    max_concurrent_workers: 2,
  },
  {
    attempt_id,
    cursor: "03-c.json",
    event: "grassmoth.extraction.worker",
    worker_id: 0,
    state: "running",
    n_extractions: null,
    n_llm_calls: null,
    error: null,
    n_windows: 3,
  },
  {
    attempt_id,
    cursor: "04-d.json",
    event: "grassmoth.extraction.worker",
    worker_id: 0,
    state: "completed",
    error: null,
    n_windows: 3,
    n_extractions: 9,
    n_llm_calls: 2,
  },
  {
    attempt_id,
    cursor: "05-e.json",
    event: "grassmoth.extraction.snapshot",
    total_workers: 2,
    pending_workers: 0,
    running_workers: 1,
    completed_workers: 1,
    failed_workers: 0,
  },
];

describe("attempt progress", () => {
  it("keeps the latest value each event carries and pages from the last cursor", () => {
    const view = applyProgressEvents(EMPTY_ATTEMPT_PROGRESS, events);
    expect(view.cursor).toBe("05-e.json");
    expect(view.extraction?.status).toBe("completed");
    expect(fixtureValue(view.workers[0]).state).toBe("completed");
    expect(view.snapshot?.completed_workers).toBe(1);
  });

  it("adds nothing when a page is read twice", () => {
    const once = applyProgressEvents(EMPTY_ATTEMPT_PROGRESS, events);
    expect(applyProgressEvents(once, events)).toEqual(once);
  });

  it("reports nothing for an empty or pruned stream", () => {
    expect(applyProgressEvents(EMPTY_ATTEMPT_PROGRESS, [])).toBe(EMPTY_ATTEMPT_PROGRESS);
    const pruned = applyProgressEvents(EMPTY_ATTEMPT_PROGRESS, events.slice(4));
    expect(pruned.workers).toEqual({});
    expect(pruned.snapshot?.running_workers).toBe(1);
  });
});
