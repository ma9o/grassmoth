import type { ConstructId } from "@grassmoth/api-types";
import { useState } from "react";
import { SpecLawChart } from "@/components/charts/spec-chart";
import type { EntitySelection } from "@/lib/model-asset/selection";
import {
  glyphKey,
  plotsForSelection,
  type SpecGlyph,
  type SpecPlots,
  stateLawGlyphs,
} from "@/lib/model-asset/spec-plots";
import type { Box } from "./layered-card-strip";
import { graphButton } from "./layered-graph-marks";

export interface SpecView {
  readonly plots: SpecPlots;
  readonly selected: string | null;
  readonly highlighted: ReadonlySet<ConstructId>;
  readonly lit: (glyph: SpecGlyph) => boolean;
  readonly hover: (node: EntitySelection | null) => void;
  readonly select: (glyph: SpecGlyph) => void;
}

export function useSpecGlyphs(
  plots: SpecPlots | null,
  selection: EntitySelection | null,
  onSelect: (selection: EntitySelection) => void,
): SpecView | null {
  const [hovered, setHovered] = useState<EntitySelection | null>(null);
  if (!plots) return null;
  const selected = selection ? glyphKey(selection) : null;
  const node = hovered ?? selection;
  const active = node ? plotsForSelection(plots, node) : [];
  const shared = new Set(active.flatMap((glyph) => glyph.uses.map((use) => use.parameterId)));
  const lit = (glyph: SpecGlyph) =>
    active.includes(glyph) || glyph.uses.some((use) => shared.has(use.parameterId));
  const highlighted = new Set(
    [...plots.values()].filter((glyph) => lit(glyph)).map((glyph) => glyph.construct),
  );
  return {
    plots,
    selected,
    highlighted,
    lit,
    hover: setHovered,
    select: (glyph: SpecGlyph) => onSelect(glyph.selection),
  };
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
          <rect
            x={1}
            y={0.5}
            width={width - 2}
            height={box.height - 1}
            rx={3}
            stroke={spec.lit(glyph) ? "var(--primary)" : "none"}
            fill="transparent"
          />
          <foreignObject x={2} y={0} width={width - 4} height={box.height} pointerEvents="none">
            <SpecLawChart law={glyph.law} width={width - 4} height={box.height} />
          </foreignObject>
        </g>
      ))}
    </g>
  );
}

export function glyphControls(spec: SpecView, glyph: SpecGlyph) {
  const select = () => spec.select(glyph);
  return {
    ...graphButton(select),
    "data-glyph": glyphKey(glyph.selection),
    "aria-pressed": spec.selected === glyphKey(glyph.selection),
    "data-highlighted": spec.lit(glyph) || undefined,
    onPointerEnter: () => spec.hover(glyph.selection),
    onPointerLeave: () => spec.hover(null),
    onFocus: () => spec.hover(glyph.selection),
    onBlur: () => spec.hover(null),
  };
}
