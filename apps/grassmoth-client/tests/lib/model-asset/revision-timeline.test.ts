import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import type { TimelineRevision } from "@grassmoth/api-types";
import { describe, expect, it } from "vitest";
import { latestSeq } from "@/lib/model-asset/journal";
import { revisionTimeline } from "@/lib/model-asset/revision-timeline";
import {
  timelineLayout,
  timelinePoint,
  timelineTickLabel,
} from "@/lib/model-asset/timeline-presentation";

import { applied, comparison, failedFit, revision, simulation } from "@/lib/__fixtures__/timeline";

describe("argument timeline", () => {
  it("orders actions by execution, lanes them by what they produce and links served dependencies", () => {
    const records: TimelineRevision[] = [
      revision(1, { action: "edit_model", request: null, outcome: applied }),
      revision(2, { action: "prepare_data", request: null, outcome: applied }),
      revision(3, { action: "fit", request: null, outcome: applied }),
      failedFit,
      simulation,
      comparison,
    ];
    const timeline = revisionTimeline(records, [
      { seq: 3, source_seq: 1, argument: "dynamical_model_spec" },
      { seq: 3, source_seq: 2, argument: "panel" },
      { seq: 4, source_seq: 1, argument: "dynamical_model_spec" },
      { seq: 5, source_seq: 3, argument: "dynamical_model_spec" },
      { seq: 6, source_seq: 2, argument: "left" },
      { seq: 6, source_seq: 5, argument: "right" },
      { seq: 9, source_seq: 2, argument: "panel" },
    ]);
    expect(timeline.nodes.map((node) => [node.tick.record.seq, node.column, node.lane])).toEqual([
      [1, 0, 2],
      [2, 1, 1],
      [3, 2, 2],
      [4, 3, 2],
      [5, 4, 1],
      [6, 5, 0],
    ]);
    // Records outside the journal draw nothing; commit parents never become links.
    expect(
      timeline.links.map((link) => [link.from.tick.record.seq, link.to.tick.record.seq]),
    ).toEqual([
      [1, 3],
      [2, 3],
      [1, 4],
      [3, 5],
      [2, 6],
      [5, 6],
    ]);
    const layout = timelineLayout(timeline);
    const point = (index: number) => timelinePoint(layout, fixtureValue(timeline.nodes[index]));
    expect(point(5).y).toBeLessThan(point(4).y);
    expect(point(5).x).toBeGreaterThan(point(4).x);
    // The failed fit leaves the model track for a side track toward the data lane its inputs
    // come from, so the track later model actions use runs past it.
    expect(point(3).y).toBeLessThan(point(2).y);
    expect(point(3).y).toBeGreaterThan(point(4).y);
    expect(timelineTickLabel(fixtureValue(timeline.nodes[3]).tick)).toBe("fit · 4000000");
    expect(latestSeq(records)).toBe(5);
  });
});
