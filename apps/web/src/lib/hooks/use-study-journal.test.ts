import { describe, expect, it } from "vitest";
import { journalPollIntervalMs } from "./use-study-journal";

describe("journalPollIntervalMs", () => {
  it("keeps polling idle viewers so externally started runs are discovered", () => {
    expect(journalPollIntervalMs(false)).toBe(10_000);
    expect(journalPollIntervalMs(true)).toBe(500);
  });
});
