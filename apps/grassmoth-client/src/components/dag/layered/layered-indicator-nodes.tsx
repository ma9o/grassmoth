import { SpecLawChart } from "@/components/charts/spec-chart";
import {
  INDICATOR_LAW_HEIGHT,
  INDICATOR_LINE_HEIGHT,
  type IndicatorBox,
} from "@/lib/dag/indicator-layout";
import { DAG_COLORS } from "@/lib/dag/palette";
import { glyphControls, type SpecView } from "./layered-spec-glyphs";

function IndicatorNode({
  box,
  spec,
  dimmed,
}: {
  readonly box: IndicatorBox;
  readonly spec: SpecView;
  readonly dimmed: boolean;
}) {
  const glyph = box.distribution;
  const controls = glyphControls(spec, glyph);
  const selected = controls["aria-pressed"];
  const lines = [...box.name, ...box.caption];
  return (
    <g
      {...controls}
      opacity={dimmed && !selected ? 0.25 : 1}
      data-node-id={glyph.selection.id}
      transform={`translate(${box.x},${box.y})`}
      aria-label={`${glyph.label} reading distribution`}
    >
      <title>{[glyph.label, glyph.law.family, glyph.law.label].join(" · ")}</title>
      <rect
        width={box.width}
        height={box.height}
        rx={7}
        stroke={selected ? "var(--primary)" : DAG_COLORS.line}
        fill="var(--card)"
        strokeWidth={selected ? 1.8 : 1.2}
      />
      <text x={10} y={16} fontFamily="ui-monospace, monospace">
        {lines.map((line, i) => (
          <tspan
            // eslint-disable-next-line react/no-array-index-key -- Positions identify wrapped lines in one SVG text run.
            key={i}
            x={10}
            dy={i === 0 ? 0 : INDICATOR_LINE_HEIGHT}
            fontSize={i < box.name.length ? 12 : 10}
            fontWeight={i < box.name.length ? 600 : 400}
            fill={i < box.name.length ? "var(--foreground)" : DAG_COLORS.slate}
          >
            {line}
          </tspan>
        ))}
      </text>
      <foreignObject
        x={8}
        y={box.height - INDICATOR_LAW_HEIGHT - 4}
        width={box.width - 16}
        height={INDICATOR_LAW_HEIGHT}
        pointerEvents="none"
      >
        <SpecLawChart
          law={glyph.law}
          width={box.width - 16}
          height={INDICATOR_LAW_HEIGHT}
          captions={false}
        />
      </foreignObject>
    </g>
  );
}

export function IndicatorNodes({
  boxes,
  spec,
  dimmed,
}: {
  readonly boxes: readonly IndicatorBox[];
  readonly spec: SpecView;
  readonly dimmed: boolean;
}) {
  return (
    <>
      {boxes.map((box) => (
        <IndicatorNode key={box.distribution.selection.id} box={box} spec={spec} dimmed={dimmed} />
      ))}
    </>
  );
}
