import type { ConstructSpec, IndicatorSpec, StateAssignment } from "@grassmoth/api-types";
import { DrawsChart } from "@/components/charts/draws-chart";
import { FlowChart } from "@/components/charts/flow-chart";
import { dataComparisonChart, pathLayers } from "@/components/charts/series-adapters";
import { DAG_COLORS } from "@/lib/dag/palette";
import type { DataComparisonView } from "@/lib/model-asset/data-comparison";
import type { ModelFlowPlots } from "@/lib/model-asset/flow-plots";
import type { SimulationPathsView } from "@/lib/model-asset/result-values";
import { humanize } from "@/lib/model-asset/selection";
import {
  type CardMarker,
  ChartFrame,
  DataStrip,
  type PreparedRow,
  rowTop,
  SimulatedDraws,
  STRIP_ROWS,
  type StateGeometry,
  StripOverflow,
  StripRow,
} from "./layered-card-strip";
import { activateOnKeyboard, plotTitle, truncate } from "./layered-graph-marks";

/** What every card and chip shows of the viewed state. */
export interface StateView {
  readonly geometry: StateGeometry;
  readonly zoom: number;
  readonly flows: ModelFlowPlots | null;
  readonly simulationPaths: SimulationPathsView | null;
  readonly dataDiff: DataComparisonView | null;
  readonly selectIndicator: (id: IndicatorSpec["observation"]["id"]) => void;
}

/** One construct's part of the viewed state. */
export interface ConstructState {
  readonly construct: ConstructSpec;
  readonly indicators: readonly IndicatorSpec[];
  readonly prepared: readonly PreparedRow[];
  readonly assignments: readonly StateAssignment[];
  readonly markers: readonly CardMarker[];
}

/** An indicator's data against the compared data; a failed check marks its value. */
function ComparisonRow({
  view,
  dataDiff,
  indicator,
  index,
}: {
  readonly view: StateView;
  readonly dataDiff: DataComparisonView;
  readonly indicator: IndicatorSpec;
  readonly index: number;
}) {
  const { geometry } = view;
  const variable = dataDiff.variables.find(
    (item) => item.indicator_id === indicator.observation.id,
  );
  const evaluation = variable?.predictive?.evaluation ?? null;
  const failures = (evaluation && !("kind" in evaluation) ? evaluation.findings : []).filter(
    (finding) => finding.kind !== "evaluated" || finding.outcome !== "passed",
  );
  const select = () => view.selectIndicator(indicator.observation.id);
  return (
    <g
      role="button"
      tabIndex={0}
      aria-label={`${humanize(indicator.observation.name)} comparison`}
      onClick={(event) => {
        event.stopPropagation();
        select();
      }}
      onKeyDown={(event) => {
        event.stopPropagation();
        activateOnKeyboard(event, select);
      }}
    >
      <StripRow
        geometry={geometry}
        index={index}
        title={
          failures
            .map((finding) =>
              finding.kind === "evaluated" ? finding.evidence.note : finding.detail,
            )
            .join("; ") || humanize(indicator.observation.name)
        }
        label={humanize(indicator.observation.name)}
        value={failures.length > 0 ? "⚠" : ""}
        valueTone="var(--warning-foreground)"
      >
        {variable && (
          <foreignObject
            x={geometry.strip.x + geometry.row.plot}
            y={rowTop(geometry, index)}
            width={geometry.strip.width - geometry.row.plot}
            height={geometry.row.height - 2}
            pointerEvents="none"
          >
            <DrawsChart
              {...dataComparisonChart(variable)}
              height={geometry.row.height - 2}
              resolution={view.zoom}
              compact
            />
          </foreignObject>
        )}
      </StripRow>
    </g>
  );
}

/** The construct's chart strip: its compared data, simulated draws, prepared data or laws. */
function StateStrip({ view, state }: { readonly view: StateView; readonly state: ConstructState }) {
  const { geometry, zoom, dataDiff, simulationPaths } = view;
  const { construct, indicators, prepared } = state;
  const series = simulationPaths?.states[construct.id];
  const statePlot = view.flows?.constructs.get(construct.id);
  if (dataDiff) {
    return (
      <>
        {indicators.slice(0, STRIP_ROWS).map((indicator, index) => (
          <ComparisonRow
            key={indicator.observation.id}
            view={view}
            dataDiff={dataDiff}
            indicator={indicator}
            index={index}
          />
        ))}
        <StripOverflow geometry={geometry} hidden={indicators.length - STRIP_ROWS} />
      </>
    );
  }
  if (series) {
    return (
      <SimulatedDraws
        box={geometry.strip}
        paths={simulationPaths}
        series={series}
        markers={state.markers}
        zoom={zoom}
      />
    );
  }
  if (prepared.length > 0) return <DataStrip geometry={geometry} rows={prepared} />;
  return (
    <ChartFrame box={geometry.strip}>
      {statePlot && (
        <FlowChart plot={statePlot} height={geometry.strip.height} compact resolution={zoom} />
      )}
    </ChartFrame>
  );
}

/** What an indicator row shows: its saved simulation readings, else its reading law. */
function readingView(view: StateView, indicator: IndicatorSpec) {
  const id = indicator.observation.id;
  const height = view.geometry.indicator.height - 3;
  const paths = view.simulationPaths;
  const readings = paths?.indicators[id];
  if (readings) {
    return {
      title: `${humanize(readings.label)}: saved simulation readings`,
      chart: (
        <DrawsChart
          compact
          height={height}
          resolution={view.zoom}
          times={paths.times}
          timeOrigin={paths.time_origin}
          layers={pathLayers(readings, "both", true)}
          levels={readings.levels}
          label={`${humanize(readings.label)}: saved readings`}
        />
      ),
    };
  }
  const plot = view.flows?.indicators.get(id);
  if (plot) {
    return {
      title: plotTitle(plot),
      chart: <FlowChart plot={plot} height={height} compact resolution={view.zoom} />,
    };
  }
  return { title: indicator.observation.name, chart: null };
}

/** One of the indicator rows under the construct's chart strip. */
function ReadingRow({
  view,
  indicator,
  index,
}: {
  readonly view: StateView;
  readonly indicator: IndicatorSpec;
  readonly index: number;
}) {
  const { geometry } = view;
  const reading = readingView(view, indicator);
  const select = () => view.selectIndicator(indicator.observation.id);
  const top = geometry.indicator.top + index * geometry.indicator.height;
  return (
    <g
      role="button"
      tabIndex={0}
      aria-label={`${humanize(indicator.observation.name)} reading law`}
      onClick={(event) => {
        event.stopPropagation();
        select();
      }}
      onKeyDown={(event) => {
        event.stopPropagation();
        activateOnKeyboard(event, select);
      }}
    >
      <title>{reading.title}</title>
      <text
        x={geometry.strip.x}
        y={top + Math.round(geometry.indicator.height * 0.52)}
        fontSize={7}
        fill={DAG_COLORS.slate}
      >
        {truncate(humanize(indicator.observation.name), geometry.indicator.length)}
      </text>
      <foreignObject
        x={geometry.indicator.chart}
        y={top}
        width={geometry.width - geometry.strip.x - geometry.indicator.chart}
        height={geometry.indicator.height - 3}
        pointerEvents="none"
      >
        {reading.chart}
      </foreignObject>
    </g>
  );
}

/** The construct's own column: its prepared data, simulated draws or laws, then its readings. */
export function StateColumn({
  view,
  state,
}: {
  readonly view: StateView;
  readonly state: ConstructState;
}) {
  const { indicators, prepared } = state;
  return (
    <>
      <StateStrip view={view} state={state} />
      {!view.dataDiff &&
        prepared.length === 0 &&
        indicators
          .slice(0, 2)
          .map((indicator, index) => (
            <ReadingRow
              key={indicator.observation.id}
              view={view}
              indicator={indicator}
              index={index}
            />
          ))}
      {!view.dataDiff && prepared.length === 0 && (
        <StripOverflow geometry={view.geometry} hidden={indicators.length - 2} />
      )}
    </>
  );
}
