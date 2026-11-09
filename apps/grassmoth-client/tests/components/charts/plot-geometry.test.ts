import { describe, expect, it } from "vitest";
import { finiteRuns, linearScale, linePath, traceLine } from "@/components/charts/plot-geometry";

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

  it("traces canvas coordinates at their actual times and leaves missing values as gaps", () => {
    const commands: (readonly [string, number, number])[] = [];
    traceLine(
      {
        moveTo: (x, y) => commands.push(["move", x, y]),
        lineTo: (x, y) => commands.push(["line", x, y]),
      },
      [0, 1, 2, 10, 11, 12, 20, 21],
      [1, 2, null, 4, 3, Number.POSITIVE_INFINITY, 5],
      { x: (time) => time * 5, y: linearScale([0, 5], [50, 0]) },
    );
    expect(commands).toEqual([
      ["move", 0, 40],
      ["line", 5, 30],
      ["move", 50, 10],
      ["line", 55, 20],
      ["move", 100, 0],
    ]);
  });
});
