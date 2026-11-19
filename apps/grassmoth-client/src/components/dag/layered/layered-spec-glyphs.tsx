import type { ConstructId } from "@grassmoth/api-types";
import { SpecLawChart } from "@/components/charts/spec-chart";
import type { EntitySelection } from "@/lib/model-asset/selection";
import {
  glyphKey,
  type SpecGlyph,
  type SpecPlots,
  stateLawGlyphs,
} from "@/lib/model-asset/spec-plots";
import type { Box } from "./layered-card-strip";
import { graphButton } from "./layered-graph-marks";

export interface SpecView {
  readonly plots: SpecPlots;
  readonly selected: string | null;
  readonly select: (node: EntitySelection) => void;
}

export function StateLawStrip({
  spec,
  id,
  box,
}: {
  readonly spec: SpecView;
  readonly id: ConstructId;
  readonly box: Box;
}) {
  const laws = stateLawGlyphs(spec.plots, id);
  const width = box.width / laws.length;
  return (
    <g transform={`translate(${box.x},${box.top})`}>
      {laws.map((glyph, i) => (
        <g
          key={glyph.label}
          transform={`translate(${i * width},0)`}
          role="img"
          aria-label={`${glyph.label} · ${glyph.law.family}`}
        >
          <foreignObject x={2} y={0} width={width - 4} height={box.height} pointerEvents="none">
            <SpecLawChart law={glyph.law} width={width - 4} height={box.height} />
          </foreignObject>
        </g>
      ))}
    </g>
  );
}

export function glyphControls(spec: SpecView, glyph: SpecGlyph) {
  const key = glyphKey(glyph.selection);
  const selected = spec.selected === key;
  return {
    ...graphButton(() => spec.select(glyph.selection)),
    "data-glyph": key,
    "aria-pressed": selected,
  };
}
