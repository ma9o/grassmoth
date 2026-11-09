import { DrawsChart } from "@/components/charts/draws-chart";
import { FlowChart } from "@/components/charts/flow-chart";
import { linearScale, padDomain } from "@/components/charts/plot-geometry";
import { ProfileMarks, profileDomain, profileSummary } from "@/components/charts/profile-strip";
import { dataComparisonChart } from "@/components/charts/series-adapters";
import type { GraphDensity } from "@/lib/dag/build-layered-causal-graph";
import type { ConstructStatus } from "@/lib/dag/construct-statuses";
import { DAG_COLORS } from "@/lib/dag/palette";
import { humanize } from "@/lib/model-asset/selection";
import { type Box, ChartFrame, SimulatedDraws } from "./layered-card-strip";
import {
  activateOnKeyboard,
  assignmentLabel,
  CarryMark,
  CHIP_PILL_WIDTH,
  LayerPill,
  pillWidth,
  statusAccent,
  truncate,
  WarningMark,
} from "./layered-graph-marks";
import type { ConstructState, StateView } from "./layered-state-column";

/** An exogenous construct's one chart, chosen as its card's strip would be. */
function ChipChart({
  view,
  state,
  box,
}: {
  readonly view: StateView;
  readonly state: ConstructState;
  readonly box: Box;
}) {
  const { zoom, dataDiff, simulationPaths } = view;
  const { construct, indicators, prepared } = state;
  const series = simulationPaths?.states[construct.id];
  const statePlot = view.flows?.constructs.get(construct.id);
  if (dataDiff) {
    const shown = indicators.flatMap((indicator) => {
      const variable = dataDiff.variables.find(
        (item) => item.indicator_id === indicator.observation.id,
      );
      return variable ? [{ indicator, variable }] : [];
    });
    const first = shown[0];
    if (!first) return null;
    const select = () => view.selectIndicator(first.indicator.observation.id);
    return (
      <g
        role="button"
        tabIndex={0}
        aria-label={`${humanize(first.indicator.observation.name)} comparison`}
        onClick={(event) => {
          event.stopPropagation();
          select();
        }}
        onKeyDown={(event) => {
          event.stopPropagation();
          activateOnKeyboard(event, select);
        }}
      >
        <title>
          {shown.map(({ indicator }) => humanize(indicator.observation.name)).join(", ")}
        </title>
        <ChartFrame box={box}>
          <DrawsChart
            {...dataComparisonChart(first.variable)}
            height={box.height}
            resolution={zoom}
            compact
          />
        </ChartFrame>
      </g>
    );
  }
  if (series) {
    return (
      <SimulatedDraws
        box={box}
        paths={simulationPaths}
        series={series}
        markers={state.markers}
        zoom={zoom}
      />
    );
  }
  const first = prepared[0];
  if (first) {
    const domain = profileDomain(first.profile, []);
    return (
      <g role="img" aria-label={humanize(first.indicator.observation.name)}>
        <title>
          {prepared
            .map(
              ({ indicator, profile }) =>
                `${humanize(indicator.observation.name)} · ${profileSummary(profile)}`,
            )
            .join("\n")}
        </title>
        <rect x={box.x} y={box.top} width={box.width} height={box.height} fill="transparent" />
        {domain && (
          <ProfileMarks
            profile={first.profile}
            values={[]}
            x={linearScale(padDomain(domain, 0.02), [box.x, box.x + box.width])}
            top={box.top}
            height={box.height}
          />
        )}
      </g>
    );
  }
  return statePlot ? (
    <ChartFrame box={box}>
      <FlowChart plot={statePlot} height={box.height} compact resolution={zoom} />
    </ChartFrame>
  ) : null;
}

/** An exogenous construct as a chip: its name and its path on one line. */
export function ExogenousChip({
  density,
  view,
  state,
  status,
  failures,
  timeOrigin,
  selected,
  dimmed,
  onSelect,
}: {
  readonly density: GraphDensity;
  readonly view: StateView;
  readonly state: ConstructState;
  readonly status: ConstructStatus | undefined;
  readonly failures: readonly string[];
  readonly timeOrigin: string | null;
  readonly selected: boolean;
  readonly dimmed: boolean;
  readonly onSelect: () => void;
}) {
  const { construct, assignments } = state;
  const rich = density.key === "rich";
  const { width: chipWidth, height: chipHeight } = density.chip;
  const size = rich ? 10.5 : 9.5;
  const chart = rich ? 72 : 56;
  const chartX = chipWidth - 9 - (failures.length > 0 ? 15 : 0) - chart;
  const badge = assignments.length > 0 ? String(assignments.length) : null;
  const labelEnd = (badge ? chartX - 6 - pillWidth(badge, CHIP_PILL_WIDTH) : chartX) - 6;
  const carry = construct.temporal_status === "time_varying" ? 14 : 0;
  const label = truncate(
    humanize(construct.name),
    Math.max(4, Math.floor((labelEnd - 10 - carry) / (size * 0.56))),
  );
  const accent = badge ? DAG_COLORS.intervention : (statusAccent(status) ?? "var(--border)");
  return (
    <g
      opacity={dimmed ? 0.18 : 1}
      role="button"
      aria-label={humanize(construct.name)}
      tabIndex={0}
      style={{ cursor: "pointer" }}
      onClick={onSelect}
      onKeyDown={(event) => activateOnKeyboard(event, onSelect)}
    >
      <rect
        width={chipWidth}
        height={chipHeight}
        rx={8}
        fill={DAG_COLORS.col}
        stroke={selected ? "var(--primary)" : accent}
        strokeWidth={selected ? 2.4 : 1.2}
      />
      <text
        x={10}
        y={chipHeight / 2 + size * 0.36}
        fontSize={size}
        fontWeight={600}
        fill="var(--foreground)"
      >
        {label}
        <CarryMark construct={construct} size={size - 1} />
      </text>
      {badge && timeOrigin !== null && (
        <LayerPill
          x={chartX - 6}
          y={chipHeight / 2 - 8.5}
          label={badge}
          color={DAG_COLORS.intervention}
          minWidth={CHIP_PILL_WIDTH}
          description={`${badge} assignment${assignments.length === 1 ? "" : "s"}: ${assignments
            .map((event) => assignmentLabel(event, timeOrigin))
            .join("; ")}`}
        />
      )}
      <ChipChart
        view={view}
        state={state}
        box={{ x: chartX, top: 5, width: chart, height: chipHeight - 10 }}
      />
      <WarningMark x={chipWidth - 21} y={chipHeight / 2 - 6} failures={failures} />
    </g>
  );
}
