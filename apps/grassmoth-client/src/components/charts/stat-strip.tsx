"use client";

import type { PPCTestStat } from "@grassmoth/api-types";
import { formatNumber, formatSignificant } from "@/lib/utils/format";
import { CHART_COLORS } from "./chart-tokens";
import { DistributionChart } from "./distribution-chart";

/** One check statistic: every replicated value as a dot against the observed value. */
export function StatStrip({ stat, height = 30 }: { stat: PPCTestStat; height?: number }) {
  return (
    <div className="min-w-0 space-y-0.5">
      <DistributionChart
        compact
        label={`${stat.stat_name}: ${stat.rep_values.length} replicated values against the observed ${formatSignificant(stat.observed_value)}`}
        height={height}
        dots={[
          {
            key: "replicated",
            label: "Replicated",
            color: CHART_COLORS.replicate,
            values: stat.rep_values,
          },
        ]}
        marks={[
          {
            key: "observed",
            label: "Observed",
            value: stat.observed_value,
            color: CHART_COLORS.observed,
          },
        ]}
        frame={stat.frame}
      />
      <p className="m-0 font-mono text-[10px] text-muted-foreground">
        observed {formatSignificant(stat.observed_value)} · p{" "}
        {stat.p_value == null ? "—" : formatNumber(stat.p_value, 2)}
      </p>
    </div>
  );
}
