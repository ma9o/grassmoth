import type { EdgeId } from "@grassmoth/api-types";
import { Katex } from "@/components/analysis-widgets/statistical-model-spec/ssm-equation-display";
import { edgeEquation } from "@/lib/model-asset/equations";
import { modelFlowPlots } from "@/lib/model-asset/flow-plots";
import { ownLawUses } from "@/lib/model-asset/laws";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { Section } from "../scope-primitives";
import { FlowInputs, FlowSection } from "./flow-sections";
import { LawSections } from "./law-sections";
import { Reasoning } from "./reasoning";

export function EdgeScope({
  context,
  id,
}: {
  readonly context: ScopeContext;
  readonly id: EdgeId;
}) {
  const edge = context.entities.edgeById.get(id);
  if (!edge) return null;
  const model = context.modelSnapshot.dynamical_model_spec;
  const equation = edgeEquation(edge, context.entities);
  return (
    <>
      <Section title="Relationship">
        <Reasoning value={edge.reasoning} />
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
