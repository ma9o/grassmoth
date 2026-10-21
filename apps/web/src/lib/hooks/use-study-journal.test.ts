import { describe, expect, it, vi } from "vitest";
import { failedFit, simulation } from "@/lib/__fixtures__/timeline";
import { journalPollIntervalMs, readStudyJournal } from "./use-study-journal";

const { get } = vi.hoisted(() => ({ get: vi.fn() }));
vi.mock("@nof1-causal-lab/api-types", async (original) => ({
  ...(await original<typeof import("@nof1-causal-lab/api-types")>()),
  createModelClient: () => ({ GET: get }),
}));

describe("journalPollIntervalMs", () => {
  it("keeps polling idle viewers so externally started runs are discovered", () => {
    expect(journalPollIntervalMs(false)).toBe(10_000);
    expect(journalPollIntervalMs(true)).toBe(500);
  });
});

it("retains an earlier call that completes after a later sequence", async () => {
  get.mockResolvedValueOnce({ data: { attempts: [simulation], running: [], dependencies: [] } });
  const held = await readStudyJournal("study");
  get.mockResolvedValueOnce({
    data: { attempts: [failedFit, simulation], running: [], dependencies: [] },
  });
  const updated = await readStudyJournal("study", held);
  expect(updated.seq).toBe(held.seq);
  expect(updated.attempts).toEqual([failedFit, simulation]);
});
