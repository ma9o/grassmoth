import type { TimelineRevision } from "@grassmoth/api-types";
import { expect, it } from "vitest";
import { entry, journal, oid } from "@/lib/__fixtures__/recorded-calls";
import { modelReference } from "@/lib/model-asset/call-dependencies";

it("addresses models by their outputs or recorded inputs, including failed calls", () => {
  expect(modelReference(entry(1), journal)).toBe(oid(101));
  expect(modelReference(entry(4), journal)).toBe(oid(104));
  expect(modelReference(entry(5), journal)).toBe(oid(4));
  expect(modelReference(entry(9), journal)).toBe(oid(8));
  expect(modelReference(entry(13), journal)).toBe(oid(8));
  const failed: TimelineRevision = {
    ...entry(10),
    parent_ids: [oid(999)],
    record: {
      ...entry(10).record,
      attempt: {
        ...entry(10).record.attempt,
        outcome: { status: "raised", error_type: "FitError", error_message: "Failed", details: [] },
      },
    },
  };
  expect(modelReference(failed, journal)).toBe(oid(7));
});
