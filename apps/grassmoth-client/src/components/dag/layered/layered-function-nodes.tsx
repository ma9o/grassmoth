import { ResponseChart } from "@/components/charts/spec-chart";
import { FUNCTION_CHART_HEIGHT } from "@/lib/dag/indicator-layout";
import { DAG_COLORS } from "@/lib/dag/palette";
import type { FocusedParts, GraphFocus } from "@/lib/dag/selection";
import type { FunctionBox, SpecLink } from "@/lib/dag/spec-layout";
import type { EntitySelection } from "@/lib/model-asset/selection";
import { glyphKey } from "@/lib/model-asset/spec-plots";
import { DagEdge } from "../core/dag-edge";
import { WarningMark } from "./layered-graph-marks";
import { glyphControls, type SpecView } from "./layered-spec-glyphs";

export function FunctionNode({
  box,
  spec,
  focused,
  onFocus,
  failures,
}: {
  readonly box: FunctionBox;
  readonly spec: SpecView;
  readonly focused: FocusedParts | null;
  readonly onFocus: (focus: GraphFocus | null) => void;
  readonly failures: readonly string[];
}) {
  const { glyph, mechanism } = box;
  const controls = glyphControls(spec, glyph);
  const selected = controls["aria-pressed"];
  const several = glyph.slices.length > 1;
  const dimmed =
    focused &&
    !(mechanism ? focused.mechanisms.has(mechanism) : focused.constructs.has(glyph.construct));
  const focus: GraphFocus = mechanism ? { kind: "mechanism", id: mechanism } : glyph.selection;
  const hover = (active: boolean) => onFocus(active ? focus : null);
  return (
    <g
      {...controls}
      onPointerEnter={() => hover(true)}
      onPointerLeave={() => hover(false)}
      onFocus={() => hover(true)}
      onBlur={() => hover(false)}
      data-mechanism-id={mechanism ?? undefined}
      data-function-id={box.id}
      data-indicator-arrow={mechanism ? undefined : glyph.selection.id}
      transform={`translate(${box.x},${box.y})`}
      opacity={dimmed && !selected ? 0.25 : 1}
      aria-label={mechanism ? `Mechanism ${glyph.label}` : `Expected reading for ${glyph.label}`}
    >
      <title>{`${glyph.label} · ${glyph.units}`}</title>
      <rect
        width={box.width}
        height={box.height}
        rx={5}
        fill={DAG_COLORS.col}
        stroke={selected ? "var(--primary)" : DAG_COLORS.line}
        strokeWidth={selected ? 1.8 : 1}
      />
      {box.name.map((line, i) => (
        <text
          // eslint-disable-next-line react/no-array-index-key -- A position identifies a line in this wrapped name.
          key={i}
          x={8}
          y={15 + 14 * i}
          fontSize={11}
          fontWeight={600}
          fill={DAG_COLORS.slate}
        >
          {line}
        </text>
      ))}
      {glyph.slices.map((slice, i) => {
        const top = box.name.length * 14 + 6 + i * (FUNCTION_CHART_HEIGHT + (several ? 14 : 0));
        return (
          <g key={slice.input ?? "constant"} transform={`translate(7,${top})`}>
            {several && (
              <text y={10} fontSize={10} fill={DAG_COLORS.slate}>
                {slice.plot.kind === "plot" ? slice.plot.value.axis?.label : slice.input}
              </text>
            )}
            <foreignObject
              y={several ? 14 : 0}
              width={box.width - 14}
              height={FUNCTION_CHART_HEIGHT}
              pointerEvents="none"
            >
              <ResponseChart
                slice={slice}
                width={box.width - 14}
                height={FUNCTION_CHART_HEIGHT}
                reading={mechanism === null}
              />
            </foreignObject>
          </g>
        );
      })}
      <WarningMark x={box.width - 15} y={3} failures={failures} />
    </g>
  );
}

export function SpecConnection({
  link,
  labels,
  focused,
  selection,
}: {
  readonly link: SpecLink;
  readonly labels: ReadonlyMap<string, string>;
  readonly focused: FocusedParts | null;
  readonly selection: EntitySelection | null;
}) {
  const key = glyphKey(link.selection);
  const active = selection !== null && glyphKey(selection) === key;
  const lit =
    active ||
    (focused !== null &&
      (link.mechanism
        ? focused.mechanisms.has(link.mechanism)
        : link.selection.kind === "edge" && focused.edges.has(link.selection.id)));
  const label = `${labels.get(link.source)} → ${labels.get(link.target)}${link.carry ? " (next time slice)" : ""}`;
  return (
    <g
      data-source={link.source}
      data-target={link.target}
      data-edge-id={link.selection.kind === "edge" ? link.selection.id : undefined}
      data-carry-id={link.carry ? link.source : undefined}
      aria-label={label}
      role="img"
    >
      <title>{label}</title>
      <DagEdge
        points={link.points}
        color={active ? "var(--primary)" : DAG_COLORS.slate}
        dashed={link.carry}
        width={1.5}
        highlighted={active}
        opacity={focused && !lit ? 0.22 : 0.95}
      />
    </g>
  );
}
