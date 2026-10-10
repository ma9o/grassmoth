import type { ConstructId, ConstructSpec } from "@nof1-causal-lab/api-types";
import { constructPresentation } from "@/lib/model-asset/inspector";
import { constructEquation } from "@/lib/model-asset/equations";
import { ownLawUses } from "@/lib/model-asset/laws";
import { humanize } from "@/lib/model-asset/selection";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { Callout, KeyValue, OwnerLink, Prose, Section, StatusIcon } from "../scope-primitives";
import { LawSections, SimulatedHistory } from "./law-sections";
import { FlowSection } from "./flow-sections";
import { modelFlowPlots } from "@/lib/model-asset/flow-plots";
import { Katex } from "@/components/analysis-widgets/statistical-model-spec/ssm-equation-display";

export function ConstructScope({ context, id }: { context: ScopeContext; id: ConstructId }) {
  const scope = constructPresentation(context, id);
  if (!scope) return null;
  const { modelSnapshot, construct, indicators } = scope;
  const equation = constructEquation(construct, context.entities);
  const latent = modelSnapshot.fit?.inference_diagnostics?.per_latent?.find(
    (item) => item.subject.id === id,
  );
  const flows = modelSnapshot.dynamical_model_spec
    ? modelFlowPlots(modelSnapshot.dynamical_model_spec)
    : null;
  return (
    <>
      <Section title="Structure">
        <Prose>{construct.description}</Prose>
        <details className="text-xs">
          <summary className="cursor-pointer text-muted-foreground">Properties</summary>
          <div className="mt-2">
            <KeyValue
              rows={[
                ["Role", construct.role],
                ["Time", humanize(construct.temporal_status)],
              ]}
            />
          </div>
        </details>
      </Section>
      {indicators.length > 0 && (
        <Section title="Indicators">
          <ul className="m-0 flex list-none flex-col gap-2 p-0">
            {indicators.map((indicator) => (
              <li key={indicator.observation.id}>
                <OwnerLink
                  onClick={() =>
                    context.select({ kind: "indicator", id: indicator.observation.id })
                  }
                >
                  {humanize(indicator.observation.name)}
                </OwnerLink>
              </li>
            ))}
          </ul>
        </Section>
      )}
      {equation && (
        <Section title={equation.title} wide>
          <Katex latex={equation.latex} />
        </Section>
      )}
      <FlowSection title="State distribution" plot={flows?.constructs.get(id)} />
      {construct.dynamics.length > 0 && (
        <FlowSection title="Intrinsic drift" plot={flows?.intrinsic.get(id)} />
      )}
      <LawSections context={context} uses={ownLawUses(construct)} />
      {latent && (
        <Section title="Latent convergence">
          <Prose>
            Worst chain diagnostics across {latent.coordinates} independent, unobserved time
            coordinates of {latent.label}.
          </Prose>
          <KeyValue
            rows={[
              ["Maximum R-hat", latent.r_hat?.toPrecision(4) ?? "Unavailable"],
              ["Minimum bulk ESS", latent.ess_bulk?.toPrecision(4) ?? "Unavailable"],
              ["Minimum tail ESS", latent.ess_tail?.toPrecision(4) ?? "Unavailable"],
              ["Maximum mean MCSE", latent.mcse_mean?.toPrecision(4) ?? "Unavailable"],
            ]}
          />
        </Section>
      )}
      <SimulatedHistory context={context} id={id} kind="states" />

      {modelSnapshot.identification?.treatments[id] && (
        <Section title="Identification">
          <IdentificationFinding context={context} construct={construct} />
        </Section>
      )}
    </>
  );
}

export function IdentificationFinding({
  context,
  construct,
}: {
  context: ScopeContext;
  construct: ConstructSpec;
}) {
  const finding = context.modelSnapshot.identification?.treatments[construct.id];
  const identified = finding?.status === "identified" ? finding : null;
  const notIdentified = finding?.status === "not_identified" ? finding : null;
  const namesFor = (ids: readonly ConstructId[]) =>
    context.entities.constructs
      .filter((entity) => ids.includes(entity.id))
      .map((entity) => humanize(entity.name))
      .join(", ");
  return (
    <>
      {identified && (
        <Callout tone="ok">
          <div className="flex items-start gap-2">
            <StatusIcon status="passed" label="Identified" />
            <span>Do-calculus</span>
          </div>
          {identified.marginalized_confounders.length > 0 && (
            <p className="mt-2">
              Marginalized confounders: {namesFor(identified.marginalized_confounders)}.
            </p>
          )}
          <p className="mt-2 break-words font-mono">{identified.estimand}</p>
        </Callout>
      )}
      {notIdentified && (
        <Callout tone="bad">
          <div className="flex items-start gap-2">
            <StatusIcon status="failed" label="Not identified" />
            <span>
              {namesFor(notIdentified.confounders)} confound this treatment under the current
              design.{notIdentified.notes ? ` ${notIdentified.notes}` : ""}
            </span>
          </div>
        </Callout>
      )}
    </>
  );
}
