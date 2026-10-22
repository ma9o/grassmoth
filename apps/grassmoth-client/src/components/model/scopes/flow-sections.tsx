import type { ConstructId, Expression } from "@grassmoth/api-types";
import { FlowChart } from "@/components/charts/flow-chart";
import { expressionStates, modelFlowPlots, type FlowPlot } from "@/lib/model-asset/flow-plots";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { humanize } from "@/lib/model-asset/selection";
import { Hint, OwnerLink, Section } from "../scope-primitives";

export function FlowSection({ title, plot }: { title: string; plot: FlowPlot | undefined }) {
  if (!plot) return null;
  return (
    <Section title={title} wide>
      <FlowChart plot={plot} height={150} />
      {plot.kind === "draws" && <Hint>{plot.note}</Hint>}
      {plot.kind === "draws" && plot.levels?.some((level, index) => level !== String(index)) && (
        <Hint>{plot.levels.map((level, index) => `${index}: ${level}`).join(" · ")}</Hint>
      )}
    </Section>
  );
}

/** The state distributions entering the selected expression, on the same draw axis as its output. */
export function FlowInputs({
  context,
  expressions,
  target,
}: {
  context: ScopeContext;
  expressions: readonly Expression[];
  target: ConstructId;
}) {
  const model = context.modelSnapshot.dynamical_model_spec;
  if (!model) return null;
  const flows = modelFlowPlots(model);
  const ids = expressionStates(expressions);
  return (
    <Section title="State inputs" wide>
      {ids.map((id) => {
        const plot = flows.constructs.get(id);
        const construct = context.entities.constructById.get(id);
        return (
          <div key={id} className="space-y-1">
            <OwnerLink onClick={() => context.select({ kind: "construct", id })}>
              {humanize(construct?.name ?? id)}
            </OwnerLink>
            {plot && <FlowChart plot={plot} height={70} />}
          </div>
        );
      })}
      <Hint>
        {ids.length > 0
          ? "These state draws and their parameter draws enter the saved expression together."
          : "This mechanism has no state input."}
      </Hint>
      <OwnerLink onClick={() => context.select({ kind: "construct", id: target })}>
        → {humanize(context.entities.constructById.get(target)?.name ?? target)}
      </OwnerLink>
    </Section>
  );
}
