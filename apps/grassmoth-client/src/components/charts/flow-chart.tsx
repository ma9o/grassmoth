import type { FlowPlot } from "@/lib/model-asset/flow-plots";
import { CHART_COLORS } from "./chart-tokens";
import { DistributionChart } from "./distribution-chart";
import { DrawsChart } from "./draws-chart";

/** The same output distribution at full size and inside a compact graph row. */
export function FlowChart({
  plot,
  height,
  compact = false,
  resolution = 1,
}: {
  plot: FlowPlot;
  height: number;
  compact?: boolean;
  resolution?: number;
}) {
  if (plot.kind === "unavailable")
    return (
      <p
        role="status"
        title={plot.reason}
        className={
          compact ? "truncate text-[7px] text-muted-foreground" : "text-xs text-muted-foreground"
        }
      >
        {plot.reason}
      </p>
    );
  const color = plot.retained ? CHART_COLORS.posterior : CHART_COLORS.prior;
  if (plot.times.length === 1) {
    const values = plot.rows.flatMap((row) => row);
    const point = values[0];
    const fixed = point !== undefined && values.every((value) => value === point);
    return (
      <DistributionChart
        label={plot.label}
        height={height}
        compact={compact}
        dots={fixed ? [] : [{ key: "outputs", label: plot.label, color, values }]}
        marks={fixed ? [{ key: "fixed", label: plot.label, value: point, color }] : []}
      />
    );
  }
  return (
    <DrawsChart
      label={plot.label}
      times={plot.times}
      timeOrigin={plot.timeOrigin === "relative" ? null : plot.timeOrigin}
      height={height}
      compact={compact}
      resolution={resolution}
      levels={plot.levels}
      layers={[
        {
          key: "outputs",
          label: plot.label,
          color,
          rows: plot.rows.map((values, index) => ({
            key: String(index),
            label: `Draw ${index + 1}`,
            values,
          })),
        },
      ]}
    />
  );
}
