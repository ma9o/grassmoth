import type { ConstructSpec, StateAssignment } from "@grassmoth/api-types";
import type { KeyboardEvent } from "react";
import type { ConstructStatus } from "@/lib/dag/construct-statuses";
import { BLOCKING, DAG_COLORS, LATENT } from "@/lib/dag/palette";
import type { FlowPlot } from "@/lib/model-asset/flow-plots";
import { formatModelDate } from "@/lib/utils/format";

export function truncate(value: string, length: number): string {
  return value.length > length ? `${value.slice(0, Math.max(1, length - 1))}…` : value;
}

export function activateOnKeyboard(event: KeyboardEvent<SVGGElement>, action: () => void): void {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    action();
  }
}

export function statusAccent(status: ConstructStatus | undefined): string | undefined {
  if (status === "blocking") return BLOCKING;
  if (status === "latent") return LATENT;
  return undefined;
}

export function assignmentLabel(event: StateAssignment, timeOrigin: string): string {
  return `${formatModelDate(event.time, timeOrigin)} (day ${event.time}): set to ${event.value}`;
}

/** A law chart's label with its note, or with the reason it has no draws. */
export function plotTitle(plot: FlowPlot): string {
  return `${plot.label}: ${plot.kind === "draws" ? plot.note : plot.reason}`;
}

export function LayerPill({
  x,
  y,
  label,
  color,
  description,
  minWidth,
}: {
  readonly x: number;
  readonly y: number;
  readonly label: string;
  readonly color: string;
  readonly description?: string;
  readonly minWidth?: number;
}) {
  const width = pillWidth(label, minWidth);
  return (
    <g transform={`translate(${x - width},${y})`} role="img" aria-label={description ?? label}>
      <title>{description ?? label}</title>
      <rect width={width} height={17} rx={8.5} fill={color} fillOpacity={0.11} />
      <text x={width / 2} y={11.5} textAnchor="middle" fontSize={7.5} fontWeight={650} fill={color}>
        {label}
      </text>
    </g>
  );
}

export const pillWidth = (label: string, minWidth = 38) =>
  Math.max(minWidth, label.length * 5.3 + 14);

/** A chip has room for the count alone; the dated list stays in its description. */
export const CHIP_PILL_WIDTH = 20;

/** The construct's own findings, as one mark. */
export function WarningMark({
  x,
  y,
  failures,
}: {
  readonly x: number;
  readonly y: number;
  readonly failures: readonly string[];
}) {
  if (failures.length === 0) return null;
  return (
    <g role="img" aria-label={failures.join("; ")} transform={`translate(${x},${y})`}>
      <title>{failures.join("\n")}</title>
      <path d="M0 11 L6 0 L12 11 Z" fill="var(--card)" stroke="var(--warning-foreground)" />
      <text x={6} y={9} textAnchor="middle" fontSize={8} fill="var(--warning-foreground)">
        !
      </text>
    </g>
  );
}

/** Follows a time-varying construct's name: its state carries over from one moment to the next. */
export function CarryMark({
  construct,
  size,
}: {
  readonly construct: ConstructSpec;
  readonly size: number;
}) {
  return construct.temporal_status === "time_varying" ? (
    <tspan dx={4} fontSize={size} fontWeight={400} fill={DAG_COLORS.muted}>
      <title>time varying</title>↻
    </tspan>
  ) : null;
}
