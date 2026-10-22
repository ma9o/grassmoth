import type { PathSeriesView } from "@/lib/model-asset/result-values";
import type {
  EmpiricalPoint,
  IndicatorEmpiricalProfile,
  IndicatorId,
  ModelSnapshot,
} from "@grassmoth/api-types";
import { ChartFigure } from "@/components/charts/chart-figure";
import { CHART_COLORS, chainColor, cssColor } from "@/components/charts/chart-tokens";
import { DistributionChart } from "@/components/charts/distribution-chart";
import { type DrawLayer, DrawsChart } from "@/components/charts/draws-chart";
import { type Domain, extentOf } from "@/components/charts/plot-geometry";
import { ProfileStrip } from "@/components/charts/profile-strip";
import { type ArmChoice, effectLayers, pathLayers } from "@/components/charts/series-adapters";
import { presentEntries } from "@/lib/model-accessors";
import {
  ALL_DRAWS,
  type DrawSelection,
  useObservationHistory,
  useSimulationPaths,
} from "@/lib/hooks/use-visuals";
import { Hint } from "../scope-primitives";

function ArmLegend({ series, arms }: { series: PathSeriesView; arms: ArmChoice }) {
  if (series.reference.length === 0) return null;
  return (
    <span className="flex items-center gap-2">
      {arms !== "intervened" && (
        <span className="inline-flex items-center gap-1">
          <span className="h-0.5 w-3" style={{ background: cssColor(CHART_COLORS.reference) }} />
          reference
        </span>
      )}
      {arms !== "reference" && (
        <span className="inline-flex items-center gap-1">
          <span className="h-0.5 w-3" style={{ background: cssColor(CHART_COLORS.intervened) }} />
          intervened
        </span>
      )}
    </span>
  );
}

const timeline = (times: readonly number[], origin: string | null) => {
  const domain: Domain | null = extentOf(times);
  return domain ? { domain, origin } : undefined;
};

/** One saved state, indicator or paired effect, every selected draw in the chosen arms. */
export function SimulationHistory({
  modelSnapshot,
  id,
  kind,
  title,
  selection = ALL_DRAWS,
  arms = "both",
  height = 150,
}: {
  modelSnapshot: ModelSnapshot;
  id: string;
  kind: "states" | "indicators" | "effect";
  title: string;
  selection?: DrawSelection;
  arms?: ArmChoice;
  height?: number;
}) {
  const paths = useSimulationPaths(modelSnapshot, selection);
  if (paths.error) return <Hint issue>{paths.error.message}</Hint>;
  if (!paths.data)
    return (
      <Hint>
        {paths.isLoading
          ? "Loading recorded paths…"
          : "No saved paths of this model revision's simulation."}
      </Hint>
    );
  const data = paths.data;
  const series =
    kind === "effect" ? data.effect : presentEntries(data[kind]).find(([key]) => key === id)?.[1];
  if (!series) return <Hint>No recorded series for this entity.</Hint>;
  const probabilities =
    kind === "indicators"
      ? presentEntries(data.action_category_probabilities).find(([key]) => key === id)?.[1]
      : undefined;
  const referenceProbabilities =
    kind === "indicators"
      ? presentEntries(data.reference_category_probabilities).find(([key]) => key === id)?.[1]
      : undefined;
  const assignments = modelSnapshot.simulation?.evidence.assignments ?? [];
  const markers = assignments
    .filter((event) => kind === "effect" || event.target === id)
    .map((event) => ({ time: event.time, label: `set ${event.value}` }));
  const layers: DrawLayer[] =
    kind === "effect" ? effectLayers(series) : pathLayers(series, arms, kind === "indicators");
  const span = timeline(data.times, data.time_origin);
  return (
    <>
      {probabilities && (
        <ChartFigure
          title={`${title}: category probabilities`}
          height={height}
          note="Backend probabilities over every retained draw; gaps have no observed emissions. Reference arms are dashed."
          {...(span ? { timeline: span } : {})}
        >
          {(view) => (
            <DrawsChart
              label={`${series.label}: full category distribution`}
              times={data.times}
              timeOrigin={data.time_origin}
              height={view.height}
              timeWindow={view.timeWindow}
              frame={[0, 1]}
              layers={[
                ...presentEntries(probabilities.probabilities).map(
                  ([level, values], index): DrawLayer => ({
                    key: `action-${level}`,
                    label: level,
                    color: chainColor(index),
                    rows: [{ key: level, label: "action", values }],
                    strong: true,
                  }),
                ),
                ...presentEntries(referenceProbabilities?.probabilities ?? {}).map(
                  ([level, values], index): DrawLayer => ({
                    key: `reference-${level}`,
                    label: level,
                    color: chainColor(index),
                    rows: [{ key: level, label: "reference", values }],
                    strong: true,
                    dashed: true,
                  }),
                ),
              ]}
            />
          )}
        </ChartFigure>
      )}
      <ChartFigure
        title={title}
        legend={kind === "effect" ? null : <ArmLegend series={series} arms={arms} />}
        height={height}
        {...(span ? { timeline: span } : {})}
      >
        {(view) => (
          <DrawsChart
            label={`${series.label}: recorded ${kind === "effect" ? "paired effects" : "draws"}`}
            times={data.times}
            timeOrigin={data.time_origin}
            layers={layers}
            markers={markers}
            frame={series.frame}
            levels={series.levels}
            height={view.height}
            timeWindow={view.timeWindow}
          />
        )}
      </ChartFigure>
    </>
  );
}

export function ObservationPlots({
  modelSnapshot,
  id,
}: {
  modelSnapshot: ModelSnapshot;
  id: IndicatorId;
}) {
  const history = useObservationHistory(modelSnapshot, id);
  if (history.error) return <Hint issue>{history.error.message}</Hint>;
  if (!history.data)
    return (
      <Hint>
        {history.isLoading ? "Loading observations…" : "No prepared observations at this revision."}
      </Hint>
    );
  const data = history.data;
  const span = timeline(
    [
      ...data.times,
      ...data.support_start.flatMap((value) => (value === null ? [] : [value])),
      ...data.support_end.flatMap((value) => (value === null ? [] : [value])),
    ],
    data.time_origin,
  );
  return (
    <>
      <ChartFigure
        title="Prepared observations"
        height={150}
        note="Every prepared observation at its actual time. Horizontal marks show measurement windows, not persistence between observations."
        {...(span ? { timeline: span } : {})}
      >
        {(view) => (
          <DrawsChart
            label={`${data.label}: prepared observations`}
            times={data.times}
            timeOrigin={data.time_origin}
            layers={[]}
            observed={{
              label: data.label,
              values: data.values,
              supportStart: data.support_start,
              supportEnd: data.support_end,
            }}
            levels={data.levels}
            height={view.height}
            timeWindow={view.timeWindow}
          />
        )}
      </ChartFigure>
      {data.empirical.length > 0 && (
        <EmpiricalPlot points={data.empirical} label={data.label} xLabel="Observed value" />
      )}
    </>
  );
}

/** The prepared values' profile, with every observation once the history has loaded. */
export function ObservedProfile({
  modelSnapshot,
  id,
  profile,
}: {
  modelSnapshot: ModelSnapshot;
  id: IndicatorId;
  profile: IndicatorEmpiricalProfile;
}) {
  const history = useObservationHistory(modelSnapshot, id);
  return (
    <ProfileStrip
      profile={profile}
      values={history.data?.values ?? []}
      label={history.data?.label ?? "Prepared values"}
    />
  );
}

/** The exact empirical distribution: each jump keeps the count at that value, no binning. */
export function EmpiricalPlot({
  points,
  label,
  xLabel,
}: {
  points: readonly EmpiricalPoint[];
  label: string;
  xLabel: string;
}) {
  return (
    <ChartFigure title="Empirical distribution" legend={xLabel} height={120}>
      {(view) => (
        <DistributionChart
          label={`${label}: empirical distribution`}
          height={view.height}
          cumulative={[
            {
              key: "empirical",
              label: "Empirical cumulative probability",
              color: CHART_COLORS.observed,
              points,
            },
          ]}
        />
      )}
    </ChartFigure>
  );
}
