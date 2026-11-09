import { resolveColor } from "./chart-tokens";
import type { DrawLayer, DrawsGeometry } from "./draws-chart";
import { type PlotScales, traceLine } from "./plot-geometry";

/** Many draws are faint so their density shows; a few rows are drawn plainly. */
function rowAlpha(layer: DrawLayer): number {
  if (layer.strong) return 0.9;
  const rows = layer.rows.length;
  if (rows > 60) return 0.14;
  if (rows > 12) return 0.28;
  return rows > 1 ? 0.55 : 0.85;
}

function paintPoints(
  context: CanvasRenderingContext2D,
  layer: DrawLayer,
  times: readonly number[],
  { x, y }: PlotScales,
): void {
  for (const row of layer.rows) {
    for (const [index, value] of row.values.entries()) {
      const time = times[index];
      if (value == null || time === undefined || !Number.isFinite(value)) continue;
      context.fillRect(x(time) - 1.1, y(value) - 1.1, 2.2, 2.2);
    }
  }
}

function paintLines(
  context: CanvasRenderingContext2D,
  layer: DrawLayer,
  times: readonly number[],
  scales: PlotScales,
): void {
  context.lineWidth = layer.strong ? 1.5 : 1;
  context.setLineDash(layer.dashed ? [4, 3] : []);
  for (const row of layer.rows) {
    context.beginPath();
    traceLine(context, times, row.values, scales);
    context.stroke();
  }
}

/** Every layer's draws on the chart's canvas, clipped to the plot box. */
export function paintDraws(
  context: CanvasRenderingContext2D,
  layers: readonly DrawLayer[],
  times: readonly number[],
  geometry: DrawsGeometry,
): void {
  const { box } = geometry;
  context.save();
  context.beginPath();
  context.rect(box.left, box.top, box.width, box.height);
  context.clip();
  for (const layer of layers) {
    const color = resolveColor(context.canvas, layer.color);
    context.globalAlpha = rowAlpha(layer);
    if (layer.points) {
      context.fillStyle = color;
      paintPoints(context, layer, times, geometry);
      continue;
    }
    context.strokeStyle = color;
    paintLines(context, layer, times, geometry);
  }
  context.restore();
}
