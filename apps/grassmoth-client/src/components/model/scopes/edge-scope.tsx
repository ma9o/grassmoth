import type { EdgeId } from "@grassmoth/api-types";
import { ownLawUses } from "@/lib/model-asset/laws";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { Prose, Section } from "../scope-primitives";
import { LawSections } from "./law-sections";
import { modelFlowPlots } from "@/lib/model-asset/flow-plots";
import { FlowInputs, FlowSection } from "./flow-sections";
import { edgeEquation } from "@/lib/model-asset/equations";
import { Katex } from "@/components/analysis-widgets/statistical-model-spec/ssm-equation-display";

export function EdgeScope({ context, id }: { context: ScopeContext; id: EdgeId }) {
  const edge = context.entities.edgeById.get(id);
  if (!edge) return null;
  const model = context.modelSnapshot.dynamical_model_spec;
  const equation = edgeEquation(edge, context.entities);
  return (
    <>
      <Section title="Relationship">
        <Prose>{edge.description}</Prose>
        {edge.sources.length > 0 && (
          <details>
            <summary className="cursor-pointer text-muted-foreground">Sources</summary>
            <ul className="mt-2 space-y-2">
              {edge.sources.map((source) => (
                <li key={source.title}>
                  {source.url ? (
                    <a
                      href={source.url}
                      target="_blank"
                      rel="noreferrer"
                      className="underline underline-offset-2"
                    >
                      {source.title}
                    </a>
                  ) : (
                    source.title
                  )}
                </li>
              ))}
            </ul>
          </details>
        )}
      </Section>
      <FlowInputs
        context={context}
        expressions={edge.mechanisms.map((mechanism) => mechanism.expression)}
        target={edge.effect.id}
      />
      {equation && (
        <Section title="Mechanism equation" wide>
          <Katex latex={equation} />
        </Section>
      )}
      <FlowSection
        title="Mechanism output"
        plot={model ? modelFlowPlots(model).edges.get(id) : undefined}
      />
      <LawSections context={context} uses={ownLawUses(edge)} />
    </>
  );
}
