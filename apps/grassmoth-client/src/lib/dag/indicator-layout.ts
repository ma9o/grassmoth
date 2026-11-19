import type { ConstructSpec } from "@grassmoth/api-types";
import type { SpecGlyph, SpecPlots } from "@/lib/model-asset/spec-plots";

export interface IndicatorBox {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
  readonly name: readonly string[];
  readonly caption: readonly string[];
  readonly distribution: Extract<SpecGlyph, { kind: "distribution" }>;
  readonly response: Extract<SpecGlyph, { kind: "response" }>;
}

/** Break long words too. The renderer uses the same fixed-width type for these lines. */
export function wrapLabel(text: string, columns: number): string[] {
  const lines: string[] = [];
  let line = "";
  for (let word of text.split(/\s+/u)) {
    if (line && line.length + 1 + word.length > columns) {
      lines.push(line);
      line = "";
    }
    while (word.length > columns) {
      lines.push(word.slice(0, columns));
      word = word.slice(columns);
    }
    line = [line, word].filter(Boolean).join(" ");
  }
  if (line) lines.push(line);
  return lines;
}

export const FUNCTION_CHART_HEIGHT = 46;

export function functionHeight(lines: number, panels: number): number {
  return 10 + lines * 14 + panels * (FUNCTION_CHART_HEIGHT + (panels > 1 ? 14 : 0));
}

export const INDICATOR_LAW_HEIGHT = 40;
export const INDICATOR_LINE_HEIGHT = 14;

export function indicatorStack(
  construct: ConstructSpec,
  cardHeight: number,
  plots: SpecPlots,
): IndicatorBox[] {
  const x = 212;
  const width = 204;
  let y = cardHeight + 16;
  return construct.indicators.flatMap(({ observation }) => {
    const distribution = plots.get(observation.id);
    const response = plots.get(`${observation.id}/response`);
    if (distribution?.kind !== "distribution" || response?.kind !== "response") return [];
    const name = wrapLabel(distribution.label, Math.floor((width - 20) / 7.3));
    const caption = [distribution.law.family, distribution.law.label].flatMap((s) =>
      wrapLabel(s, Math.floor((width - 20) / 6.2)),
    );
    const height =
      12 + (name.length + caption.length) * INDICATOR_LINE_HEIGHT + INDICATOR_LAW_HEIGHT;
    const rowHeight = Math.max(height, functionHeight(1, response.slices.length));
    const box = {
      x,
      y: y + (rowHeight - height) / 2,
      width,
      height,
      name,
      caption,
      distribution,
      response,
    };
    y += rowHeight + 7;
    return [box];
  });
}
