import { ResponseChart, SpecLawChart } from "@/components/charts/spec-chart";
import type { ScopeContext } from "@/lib/model-asset/scope";
import type { EntitySelection } from "@/lib/model-asset/selection";
import {
  glyphKey,
  plotsForSelection,
  type SpecGlyph,
  type SpecPlots,
} from "@/lib/model-asset/spec-plots";
import { Hint, Section } from "../scope-primitives";
import { FlowSection } from "./flow-sections";
import { ParameterLawSections } from "./law-sections";

function GlyphSection({ glyph }: { readonly glyph: SpecGlyph }) {
  if (glyph.kind === "distribution") {
    return (
      <Section title={glyph.label} wide>
        <SpecLawChart law={glyph.law} width={368} height={112} detail />
        {glyph.selection.kind === "indicator" ? (
          <Hint>
            The horizontal axis is in reading units. Density curves share a peak height so their
            widths can be compared. Discrete bars share a probability scale.
          </Hint>
        ) : (
          <Hint>Conditional law on its value axis. Each bell is scaled to its own peak.</Hint>
        )}
        {glyph.uses.length > 0 && (
          <Hint>
            The fill uses the saved parameter medians. The outline spans the conditional laws across
            aligned parameter draws.
          </Hint>
        )}
      </Section>
    );
  }
  const several = glyph.slices.length > 1;
  const reading = glyph.selection.kind === "indicator";
  return (
    <Section title={glyph.label} wide>
      {glyph.slices.map((slice) => {
        const axis = slice.plot.kind === "plot" ? slice.plot.value.axis : null;
        return (
          <div key={slice.input ?? "constant"} className="space-y-1">
            <p>
              {glyph.units}
              {axis && ` vs. ${axis.label}`}
            </p>
            <ResponseChart
              slice={slice}
              width={368}
              height={158}
              output={glyph.output}
              detail
              reading={reading}
            />
            {axis && (
              <Hint>
                {axis.label} · {axis.law ? "initial law" : "first-day input on the schedule range"}
              </Hint>
            )}
          </div>
        );
      })}
      {reading && <Hint>The line shows the expected reading as the state changes.</Hint>}
      {(glyph.uses.length > 0 || several) && (
        <Hint>
          The solid line uses the saved parameter medians. Outlines trace the outside of the aligned
          curves.
          {several &&
            " The reference holds other states at their initial centres. Each outlined curve holds them at its own first-day values."}
        </Hint>
      )}
      {glyph.output?.kind === "draws" && (
        <Hint>Dots at right show the first-day outputs. {glyph.output.note}</Hint>
      )}
      {glyph.output?.kind === "unavailable" && <Hint>{glyph.output.reason}</Hint>}
    </Section>
  );
}

export function SpecSections({
  context,
  selection,
  plots,
}: {
  readonly context: ScopeContext;
  readonly selection: EntitySelection;
  readonly plots: SpecPlots;
}) {
  const glyphs = plotsForSelection(plots, selection);
  const uses = glyphs.flatMap((glyph) => glyph.uses);
  const selected = selection.glyph ? glyphs[0] : undefined;
  return (
    <>
      {glyphs.map((glyph) => (
        <GlyphSection
          key={glyph.kind === "distribution" ? glyph.label : glyphKey(glyph.selection)}
          glyph={glyph}
        />
      ))}
      {uses.length > 0 && <ParameterLawSections context={context} uses={uses} />}
      {selected?.kind === "response" && selected.selection.kind !== "indicator" && (
        <FlowSection title="First-day mechanism output" plot={selected.output} />
      )}
    </>
  );
}
