import type { ConstructId, ConstructSpec } from "@grassmoth/api-types";
import { type ReactNode, useState } from "react";
import { modelFlowPlots } from "@/lib/model-asset/flow-plots";
import { constructPresentation } from "@/lib/model-asset/inspector";
import { ownLawUses } from "@/lib/model-asset/laws";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { humanize, mechanismName } from "@/lib/model-asset/selection";
import { Callout, KeyValue, OwnerLink, Prose, Section, StatusIcon } from "../scope-primitives";
import { EdgeScope } from "./edge-scope";
import { FlowSection } from "./flow-sections";
import { IndicatorScope } from "./indicator-scope";
import { LawSections, SimulatedHistory } from "./law-sections";
import { Reasoning } from "./reasoning";

/** Open a component in its owning node's details without changing the graph selection. */
function ComponentDetails({
  name,
  children,
}: {
  readonly name: string;
  readonly children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  return (
    <details onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary className="cursor-pointer font-medium">{name}</summary>
      {open && <div className="mt-2 space-y-3">{children}</div>}
    </details>
  );
}

export function ConstructScope({
  context,
  id,
  editReadings,
}: {
  readonly context: ScopeContext;
  readonly id: ConstructId;
  readonly editReadings: "nodes" | "rows" | null;
}) {
  const scope = constructPresentation(context, id);
  if (!scope) return null;
  const { modelSnapshot, construct, indicators } = scope;
  const embedded =
    editReadings === "rows" || (editReadings === "nodes" && construct.role === "exogenous");
  const incoming = editReadings
    ? context.entities.edges.filter((edge) => edge.effect.id === id)
    : [];
  const latent = modelSnapshot.fit?.inference_diagnostics?.per_latent?.find(
    (item) => item.subject.id === id,
  );
  const flows = modelSnapshot.dynamical_model_spec
    ? modelFlowPlots(modelSnapshot.dynamical_model_spec)
    : null;
  return (
    <>
      <Section title="Structure">
        <Reasoning value={construct.reasoning} />
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
        <Section title="Indicators" wide={embedded}>
          <ul className="m-0 flex list-none flex-col gap-2 p-0">
            {indicators.map((indicator) => (
              <li key={indicator.observation.id}>
                {embedded ? (
                  <ComponentDetails name={humanize(indicator.observation.name)}>
                    <IndicatorScope context={context} id={indicator.observation.id} />
                  </ComponentDetails>
                ) : (
                  <OwnerLink
                    onClick={() =>
                      context.select({ kind: "indicator", id: indicator.observation.id })
                    }
                  >
                    {humanize(indicator.observation.name)}
                  </OwnerLink>
                )}
              </li>
            ))}
          </ul>
        </Section>
      )}
      {incoming.length > 0 && (
        <Section title="Incoming relationships" wide>
          {incoming.map((edge) => (
            <ComponentDetails
              key={edge.id}
              name={`${humanize(context.entities.constructById.get(edge.cause.id)?.name ?? edge.cause.id)} → ${humanize(construct.name)}`}
            >
              <EdgeScope context={context} id={edge.id} />
            </ComponentDetails>
          ))}
        </Section>
      )}
      <FlowSection title="State distribution" plot={flows?.constructs.get(id)} />
      {editReadings === "rows" && flows && (
        <>
          {[...construct.dynamics, ...incoming.flatMap((edge) => edge.mechanisms)].map((term) => (
            <FlowSection
              key={term.id}
              title={mechanismName(term.id)}
              plot={flows.mechanisms.get(term.id)}
            />
          ))}
          {indicators.map(({ observation }) => (
            <FlowSection
              key={observation.id}
              title={humanize(observation.name)}
              plot={flows.indicators.get(observation.id)}
            />
          ))}
        </>
      )}
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
  readonly context: ScopeContext;
  readonly construct: ConstructSpec;
}) {
  const finding = context.modelSnapshot.identification?.treatments[construct.id];
  const identified = finding?.status === "identified" ? finding : null;
  const notIdentified = finding?.status === "not_identified" ? finding : null;
  const explanations = context.modelSnapshot.question_checks?.findings.filter(
    (item) =>
      item.code === "identification" &&
      "target" in item.subject &&
      item.subject.target.id === construct.id,
  );
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
              design.
            </span>
          </div>
          {explanations?.map((item, index) => (
            // eslint-disable-next-line react/no-array-index-key -- findings on one target share code and subject, so position is their identity
            <p key={index} className="mt-2">
              {item.kind === "evaluated" ? item.evidence : item.detail}
            </p>
          ))}
        </Callout>
      )}
    </>
  );
}
