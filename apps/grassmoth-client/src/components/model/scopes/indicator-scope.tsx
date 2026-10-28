import type { IndicatorId } from "@grassmoth/api-types";
import { indicatorPresentation } from "@/lib/model-asset/inspector";
import { observationEquation } from "@/lib/model-asset/equations";
import { ownLawUses } from "@/lib/model-asset/laws";
import { formatFillNull, humanize } from "@/lib/model-asset/selection";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { Hint, KeyValue, Section, StatusIcon } from "../scope-primitives";
import { LawSections, SimulatedHistory } from "./law-sections";
import { Katex } from "@/components/analysis-widgets/statistical-model-spec/ssm-equation-display";
import { ObservationPlots, ObservedProfile } from "./recorded-history";
import { FlowSection } from "./flow-sections";
import { modelFlowPlots } from "@/lib/model-asset/flow-plots";

export function IndicatorScope({ context, id }: { context: ScopeContext; id: IndicatorId }) {
  const scope = indicatorPresentation(context, id);
  if (!scope) return null;
  const { indicator, audit, likelihood, findings } = scope;
  const metadata = context.modelSnapshot.metadata;
  const preparation = metadata?.preparation.variables.find(
    (variable) => variable.observation.id === id,
  );
  const equation = observationEquation(indicator, context.entities);
  const empirical = context.modelSnapshot.profile?.indicators[id]?.profile;
  const model = context.modelSnapshot.dynamical_model_spec;
  return (
    <>
      <Section title="Measurement">
        <details>
          <summary className="cursor-pointer text-muted-foreground">Measurement settings</summary>
          <div className="mt-2">
            <KeyValue
              rows={[
                ["Type", indicator.observation.measurement.dtype],
                ["Aggregation", indicator.observation.aggregation],
                [
                  "Window",
                  indicator.observation.observation_window ??
                    context.modelSnapshot.dynamical_model_spec?.measurement_clock,
                ],
                ["Polarity", indicator.construct_polarity],
                ...(likelihood
                  ? ([
                      ["Likelihood", likelihood.law.distribution],
                      ["Standardized", likelihood.standardized ? "Yes" : "No"],
                    ] as Array<[string, string]>)
                  : []),
              ]}
            />
          </div>
        </details>
      </Section>
      {equation && (
        <Section title="Observation equation" wide>
          <Katex latex={equation} />
        </Section>
      )}
      <FlowSection
        title="Reading distribution"
        plot={model ? modelFlowPlots(model).indicators.get(id) : undefined}
      />
      <LawSections context={context} uses={ownLawUses(indicator)} />
      {preparation && (
        <Section title="Data preparation">
          <Hint>{preparation.extraction.how_to_measure}</Hint>
          <KeyValue
            rows={[
              ["Extraction", preparation.extraction.kind],
              [
                "Source columns",
                (preparation.extraction.kind === "recorded"
                  ? [
                      preparation.extraction.value_column,
                      preparation.extraction.anchor_column,
                      preparation.extraction.support_start_column,
                      preparation.extraction.support_end_column,
                    ]
                  : preparation.extraction.source_columns
                ).join(", "),
              ],
              ...(preparation.extraction.kind === "computed"
                ? [["Null filling", formatFillNull(preparation.extraction)] as [string, string]]
                : []),
            ]}
          />
        </Section>
      )}

      {preparation && (
        <Section title="Observations" wide>
          <ObservationPlots modelSnapshot={context.modelSnapshot} id={id} />
          {empirical && (
            <>
              <ObservedProfile modelSnapshot={context.modelSnapshot} id={id} profile={empirical} />
              <Hint>
                Every prepared value as a dot over the interquartile box; the bar is the median and
                the diamond the mean.
              </Hint>
            </>
          )}
        </Section>
      )}
      {audit && (
        <Section title="Validation">
          {findings.map((finding) => (
            <div key={finding.code} className="flex items-start gap-2">
              <StatusIcon
                status={finding.kind === "evaluated" ? finding.outcome : "not_evaluated"}
              />
              <Hint issue>{finding.kind === "evaluated" ? finding.evidence : finding.detail}</Hint>
            </div>
          ))}
          <details>
            <summary className="cursor-pointer text-muted-foreground">All checks</summary>
            <ul className="mt-2 space-y-2">
              {audit.findings.map((finding) => (
                <li key={finding.code} className="flex items-center gap-2">
                  <StatusIcon
                    status={finding.kind === "evaluated" ? finding.outcome : "not_evaluated"}
                  />
                  <span>{humanize(finding.code)}</span>
                </li>
              ))}
            </ul>
          </details>
        </Section>
      )}
      <SimulatedHistory context={context} id={id} kind="indicators" />
    </>
  );
}
