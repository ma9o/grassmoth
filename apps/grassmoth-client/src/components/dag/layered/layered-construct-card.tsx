import type { ConstructSpec, StateAssignment } from "@grassmoth/api-types";
import type { ReactNode } from "react";
import { FlowChart } from "@/components/charts/flow-chart";
import type { GraphDensity, MechanismRow } from "@/lib/dag/build-layered-causal-graph";
import type { ConstructStatus } from "@/lib/dag/construct-statuses";
import { DAG_COLORS } from "@/lib/dag/palette";
import type { FlowPlot } from "@/lib/model-asset/flow-plots";
import { humanize, mechanismName } from "@/lib/model-asset/selection";
import { DagNodeShell } from "../core/dag-node";
import type { StateGeometry } from "./layered-card-strip";
import {
  assignmentLabel,
  graphButton,
  LayerPill,
  statusAccent,
  truncate,
  WarningMark,
} from "./layered-graph-marks";

function statusLabel(status: ConstructStatus | undefined): string | null {
  if (status === "blocking") return "blocking";
  if (status === "latent") return "latent";
  return null;
}

function cardOpacity(dimmed: boolean, status: ConstructStatus | undefined): number {
  if (dimmed) return 0.18;
  if (status === "latent") return 0.62;
  return 1;
}

export function ConstructCard({
  construct,
  width,
  height,
  rowsWidth,
  geometry,
  isOutcome,
  failures,
  status,
  assignments,
  timeOrigin,
  selected,
  dimmed,
  onSelect,
  rows,
  children,
}: {
  readonly construct: ConstructSpec;
  readonly width: number;
  readonly height: number;
  /** Width of the mechanism rows' column, left of the construct's own column. */
  readonly rowsWidth: number;
  readonly geometry: StateGeometry;
  readonly isOutcome: boolean;
  readonly failures: readonly string[];
  readonly status: ConstructStatus | undefined;
  readonly assignments: readonly StateAssignment[];
  readonly timeOrigin: string | null;
  readonly selected: boolean;
  readonly dimmed: boolean;
  readonly onSelect: () => void;
  readonly rows: ReactNode;
  readonly children: ReactNode;
}) {
  const label = statusLabel(status);
  const hasAssignments = assignments.length > 0;
  const accent = hasAssignments ? DAG_COLORS.intervention : statusAccent(status);
  const shellAccent = selected ? "var(--primary)" : accent;
  const badge = hasAssignments
    ? `${assignments.length} assignment${assignments.length === 1 ? "" : "s"}`
    : label;
  const badgeColor = accent ?? DAG_COLORS.slate;
  const title = `${isOutcome ? "★ " : ""}${truncate(humanize(construct.name), geometry.title.length)}`;
  const { band, y, size } = geometry.title;

  return (
    <g
      opacity={cardOpacity(dimmed, status)}
      {...graphButton(onSelect)}
      aria-pressed={selected}
      aria-label={humanize(construct.name)}
    >
      <DagNodeShell
        width={width}
        height={height}
        {...(shellAccent === undefined ? {} : { accent: shellAccent })}
        dashed={status === "latent"}
        highlighted={selected}
        outcome={isOutcome}
      />
      {rows}
      <g transform={`translate(${rowsWidth},0)`}>
        <text x={geometry.strip.x} y={y} fontSize={size} fontWeight={650} fill="var(--foreground)">
          {title}
        </text>
        {badge ? (
          <LayerPill
            x={geometry.width - 8}
            y={geometry.pillY}
            label={badge}
            color={badgeColor}
            {...(hasAssignments && timeOrigin !== null
              ? {
                  description: assignments
                    .map((event) => assignmentLabel(event, timeOrigin))
                    .join("; "),
                }
              : {})}
          />
        ) : null}
        {children}
        <WarningMark x={geometry.width - 15} y={band - 10} failures={failures} />
      </g>
    </g>
  );
}

function MechanismRowChart({
  density,
  name,
  plot,
  zoom,
}: {
  readonly density: GraphDensity;
  readonly name: string;
  readonly plot: FlowPlot | undefined;
  readonly zoom: number;
}) {
  const { width, height } = density.row;
  const rich = density.key === "rich";
  const label = rich ? { y: 12.5, size: 8, characters: 31 } : { y: 9.5, size: 7, characters: 26 };
  const chartTop = rich ? 17 : 12;
  return (
    <>
      <text x={10} y={label.y} fontSize={label.size} fontWeight={600} fill={DAG_COLORS.slate}>
        {truncate(name, label.characters)}
      </text>
      <foreignObject
        x={8}
        y={chartTop}
        width={width - 24}
        height={height - chartTop - 4}
        pointerEvents="none"
      >
        {plot && <FlowChart plot={plot} height={height - chartTop - 4} compact resolution={zoom} />}
      </foreignObject>
    </>
  );
}

/** One mechanism as a row of the card it drives: its name, then its output for the viewed state. */
export function MechanismRowView({
  row,
  density,
  plot,
  description,
  failures,
  zoom,
  selected,
  dimmed,
  onSelect,
  onHover,
}: {
  readonly row: MechanismRow;
  readonly density: GraphDensity;
  readonly plot: FlowPlot | undefined;
  readonly description: string;
  readonly failures: readonly string[];
  readonly zoom: number;
  readonly selected: boolean;
  readonly dimmed: boolean;
  readonly onSelect: (() => void) | null;
  readonly onHover: (hovered: boolean) => void;
}) {
  const { width, height } = density.row;
  const name = mechanismName(row.mechanism.id);
  const highlighted = selected;
  return (
    <g
      transform={`translate(0,${row.top})`}
      opacity={dimmed ? 0.25 : 1}
      {...graphButton(onSelect)}
      aria-label={`Mechanism ${name}`}
      data-mechanism-id={row.mechanism.id}
      onPointerEnter={onSelect ? () => onHover(true) : undefined}
      onPointerLeave={onSelect ? () => onHover(false) : undefined}
    >
      <title>{description}</title>
      <rect
        x={4}
        y={1.5}
        width={width - 8}
        height={height - 3}
        rx={6}
        fill={DAG_COLORS.col}
        stroke={highlighted ? "var(--primary)" : DAG_COLORS.line}
        strokeWidth={highlighted ? 1.6 : 0.8}
      />
      <MechanismRowChart density={density} name={name} plot={plot} zoom={zoom} />
      <path
        d={`M${width - 9} ${height / 2 - 4} l4 4 l-4 4`}
        fill="none"
        stroke={DAG_COLORS.muted}
        strokeWidth={1.2}
      />
      <WarningMark x={width - 22} y={3} failures={failures} />
    </g>
  );
}
