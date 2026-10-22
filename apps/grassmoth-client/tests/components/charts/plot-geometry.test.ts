import { describe, expect, it } from "vitest";
import { finiteRuns, linePath, linearScale } from "@/components/charts/plot-geometry";

describe("recorded history geometry", () => {
  it("places observations by elapsed time rather than row order", () => {
    const x = linearScale([0, 10], [0, 100]);
    const points = finiteRuns([0, 1, 10], [1, 2, 3]).flat();
    expect(points.map(([time]) => x(time))).toEqual([0, 10, 100]);
  });

  it("breaks trajectories at missing values without fabricating a bridge", () => {
    const path = linePath(
      [0, 1, 2, 10, 11],
      [1, 2, null, 4, 3],
      (time) => time,
      (value) => value,
    );
    expect(path.match(/M/g)).toHaveLength(2);
    expect(path.match(/L/g)).toHaveLength(2);
    expect(path).not.toMatch(/[CQ]/);
  });
});
