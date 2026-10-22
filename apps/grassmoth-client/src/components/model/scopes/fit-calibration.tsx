import type { LOODiagnostics, ParetoKPoint } from "@grassmoth/api-types";
import { ChartFigure } from "@/components/charts/chart-figure";
import { CHART_COLORS } from "@/components/charts/chart-tokens";
import { DistributionChart } from "@/components/charts/distribution-chart";
import { DrawsChart } from "@/components/charts/draws-chart";
import { useInferenceReport } from "@/lib/hooks/use-inference-report";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { formatNumber } from "@/lib/utils/format";
import { KeyValue, Section } from "../scope-primitives";

/** Every held-out row's Pareto k at its time step, against the engine's two limits. */
function ParetoK({ loo, points }: { loo: LOODiagnostics; points: readonly ParetoKPoint[] }) {
  const ordered = [...points].sort((a, b) => a.timestep - b.timestep);
  return (
    <ChartFigure
      title="Pareto k by time step"
      legend={`${loo.n_bad_k ?? "Unavailable"} > ${loo.pareto_failure_limit} · ${loo.n_warn_k ?? "Unavailable"} > ${loo.pareto_warning_limit} · n = ${points.length}`}
      height={150}
      note={`Rows with k above ${loo.pareto_failure_limit} are highly influential, and the leave-one-out estimate may be unreliable there.`}
    >
      {(view) => (
        <DrawsChart
          label="Pareto k of every held-out row by time step"
          times={ordered.map((point) => point.timestep)}
          timeOrigin={null}
          xLabel="Time step"
          height={view.height}
          timeWindow={view.timeWindow}
          layers={[
            {
              key: "pareto-k",
              label: "Pareto k",
              color: CHART_COLORS.posterior,
              rows: [
                {
                  key: "k",
                  label: "held-out row",
                  values: ordered.map((point) => (typeof point.k === "number" ? point.k : null)),
                },
              ],
              points: true,
              strong: true,
            },
          ]}
          thresholds={[
            {
              value: loo.pareto_warning_limit,
              label: `k = ${loo.pareto_warning_limit}`,
              color: CHART_COLORS.warning,
            },
            {
              value: loo.pareto_failure_limit,
              label: `k = ${loo.pareto_failure_limit}`,
              color: CHART_COLORS.flagged,
            },
          ]}
        />
      )}
    </ChartFigure>
  );
}

/** The fit's leave-one-out predictive checks, as the engine recorded them. */
export function FitCalibration({ context }: { context: ScopeContext }) {
  const detail = useInferenceReport(context.modelSnapshot).data?.detail;
  const fit = context.modelSnapshot.fit;
  const loo = fit?.loo_diagnostics;
  if (!fit || !loo) return null;
  return (
    <Section title="Leave-one-out calibration" wide>
      <KeyValue
        rows={[
          ["ELPD", `${formatNumber(loo.elpd_loo, 1)} ± ${formatNumber(loo.se, 1)}`],
          ["Effective parameters", formatNumber(loo.p_loo, 1)],
          ["Held-out rows", loo.n_data_points.toLocaleString()],
          ...(loo.n_bad_k != null
            ? [["Unreliable rows", loo.n_bad_k.toLocaleString()] as [string, string]]
            : []),
        ]}
      />
      {detail && detail.loo_pit.length > 0 && (
        <ChartFigure
          title="LOO-PIT calibration"
          height={150}
          note="A calibrated model's cumulative PIT follows the dashed diagonal; deviations indicate miscalibration."
        >
          {(view) => (
            <DistributionChart
              label="Leave-one-out probability integral transform against the uniform"
              height={view.height}
              frame={[0, 1]}
              diagonal
              cumulative={[
                {
                  key: "loo-pit",
                  label: "LOO-PIT ECDF",
                  color: CHART_COLORS.posterior,
                  points: detail.loo_pit.map((point) => ({
                    value: point.pit,
                    probability: point.ecdf,
                  })),
                },
              ]}
            />
          )}
        </ChartFigure>
      )}
      {detail && detail.pareto_k.length > 0 && <ParetoK loo={loo} points={detail.pareto_k} />}
    </Section>
  );
}
