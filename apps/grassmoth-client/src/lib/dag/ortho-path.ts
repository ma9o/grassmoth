import type { Point } from "@/lib/utils/dag-graph-layout";

const unit = (from: Point, to: Point): Point => {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const len = Math.hypot(dx, dy) || 1;
  return { x: dx / len, y: dy / len };
};

/**
 * Follow ELK's routed polyline, softening each right-angle corner into a small
 * quadratic fillet. `points` is `[start, ...bends, end]` from the layout.
 * Returns an SVG path `d` string (empty when there are fewer than two points).
 */
export function orthoPath(points: Point[], radius = 12): string {
  const [start, ...rest] = points;
  const end = rest.at(-1);
  if (start === undefined || end === undefined) return "";
  let d = `M${start.x},${start.y}`;
  let a = start;
  for (const [i, p] of rest.entries()) {
    const b = rest[i + 1];
    if (b === undefined) break;
    const r = Math.min(
      radius,
      Math.hypot(p.x - a.x, p.y - a.y) / 2,
      Math.hypot(b.x - p.x, b.y - p.y) / 2,
    );
    const u = unit(p, a);
    const v = unit(p, b);
    d += `L${(p.x + u.x * r).toFixed(1)},${(p.y + u.y * r).toFixed(1)}Q${p.x},${p.y} ${(p.x + v.x * r).toFixed(1)},${(p.y + v.y * r).toFixed(1)}`;
    a = p;
  }
  d += `L${end.x},${end.y}`;
  return d;
}
