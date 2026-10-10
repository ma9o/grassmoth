import { useInferenceReport } from "@/lib/hooks/use-inference-report";
import { distributionText } from "@/lib/utils/distribution-format";
import { ChainLegend, RankChart, TraceChart } from "@/components/charts/chain-charts";
import { JointDrawsChart } from "@/components/charts/joint-draws-chart";
import { LawChart } from "@/components/charts/law-chart";
import { useParameterDraws } from "@/lib/hooks/use-visuals";
import { useState } from "react";
import type { CoefficientUse } from "@/lib/model-accessors";
import { type LawCurve, lawCurves, lawLabel } from "@/lib/model-asset/laws";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { humanize } from "@/lib/model-asset/selection";
import { Hint, KeyValue, Section, StatusIcon } from "../scope-primitives";
import { EmpiricalPlot, SimulationHistory } from "./recorded-history";

function interval(days: number | "model_clock"): string {
  return days === "model_clock" ? "model clock tick" : `${days} d`;
}

/** Only scale changes and missing priors need words; the legend names the curves. */
function lawHint(curve: LawCurve): string | null {
  if (curve.kind === "fitted")
    return curve.prior.x.length > 0
      ? null
      : "The fit's input model has no prior curve on the fitted scale.";
  const transform = curve.parameter.transform;
  if (transform.kind === "dt_persistence_to_ct_decay")
    return `Authored as persistence per ${interval(transform.interval_days)}; a fit reports decay rates.`;
  if (transform.kind === "dt_effect_to_ct_rate")
    return `Authored as the effect over ${interval(transform.interval_days)}; a fit reports rates per day.`;
  return null;
}

/** One section per law an entity's own terms name, with its prior and any posterior. */
function ParameterLawSections({
  context,
  uses,
}: {
  context: ScopeContext;
  uses: readonly CoefficientUse[];
}) {
  const fit = context.modelSnapshot.fit;
  const detail = useInferenceReport(context.modelSnapshot);
  const chains = detail.data?.detail;
  const mcmc = fit?.inference_diagnostics;
  const rows = mcmc?.per_parameter ?? [];
  const curves = lawCurves(context.modelSnapshot, uses);
  return context.entities.parameters
    .filter((parameter) => uses.some((use) => use.parameterId === parameter.id))
    .map((parameter) => {
      const law = parameter.distribution
        ? context.modelSnapshot.dynamical_model_spec?.distributions[parameter.distribution]
        : null;
      const curve = curves.find((item) => item.parameter.id === parameter.id);
      const label = curve
        ? lawLabel(curve)
        : uses
            .filter((use) => use.parameterId === parameter.id)
            .map((use) => humanize(use.role))
            .join(", ");
      const diagnostics = rows.filter((row) => row.subject.parameter_id === parameter.id);
      return (
        <Section key={parameter.id} title={label.charAt(0).toUpperCase() + label.slice(1)}>
          <Hint>{humanize(parameter.description)}</Hint>
          {curve ? (
            <LawChart curve={curve} caption={humanize(parameter.name)} />
          ) : (
            <p>{humanize(parameter.name)}</p>
          )}
          {(!curve || law?.distribution === "Delta") &&
            (parameter.distribution &&
            context.modelSnapshot.dynamical_model_spec?.law_layouts[parameter.distribution] ? (
              <Hint>
                This parameter belongs to the joint law used in the composed output plots.
              </Hint>
            ) : law ? (
              <p className="break-words font-mono">{distributionText(law)}</p>
            ) : (
              <Hint>No law assigned.</Hint>
            ))}
          {curve && lawHint(curve) && <Hint>{lawHint(curve)}</Hint>}
          {diagnostics.map((row) => {
            const subject = row.subject;
            const findings =
              fit?.convergence.findings.filter(
                (item) =>
                  typeof item.subject !== "string" &&
                  "parameter" in item.subject &&
                  item.subject.parameter.element_id === subject.element_id,
              ) ?? [];
            const trace = chains?.trace_data.find(
              (item) => item.subject.element_id === subject.element_id,
            );
            const ranks = chains?.rank_histograms.find(
              (item) => item.subject.element_id === subject.element_id,
            );
            return (
              <div key={String(subject.element_id)} className="space-y-2 border-t pt-2">
                <span>{humanize(row.parameter)}</span>
                {findings.map((item) => (
                  <div key={item.code} className="flex items-center gap-2">
                    <StatusIcon
                      status={item.kind === "evaluated" ? item.outcome : "not_evaluated"}
                    />
                    <Hint>
                      {item.kind === "evaluated" ? humanize(item.evidence.criterion) : item.detail}
                    </Hint>
                  </div>
                ))}
                <KeyValue
                  rows={[
                    ["R-hat", row.r_hat],
                    ["ESS bulk", row.ess_bulk],
                    ["ESS tail", row.ess_tail],
                    ["MCSE mean", row.mcse_mean],
                  ].map(([name, value]) => [
                    String(name),
                    typeof value === "number"
                      ? value.toLocaleString(undefined, { maximumSignificantDigits: 5 })
                      : "Unavailable",
                  ])}
                />
                {trace && (
                  <TraceChart
                    chains={trace.chains}
                    label={`${humanize(row.parameter)}: draws by chain`}
                    height={72}
                  />
                )}
                {ranks && <RankChart histogram={ranks} height={72} />}
              </div>
            );
          })}
          {diagnostics.length > 0 &&
            chains &&
            (chains.trace_data.length > 0 || chains.rank_histograms.length > 0) &&
            typeof mcmc?.num_chains === "number" && <ChainLegend chains={mcmc.num_chains} />}
        </Section>
      );
    });
}

/** Parameter-level detail sits behind the composed quantity passed to the next node. */
export function LawSections({
  context,
  uses,
}: {
  context: ScopeContext;
  uses: readonly CoefficientUse[];
}) {
  const [open, setOpen] = useState(false);
  if (uses.length === 0) return null;
  return (
    <Section title="Parameter laws">
      <details onToggle={(event) => setOpen(event.currentTarget.open)}>
        <summary className="cursor-pointer text-muted-foreground">
          Input laws and diagnostics
        </summary>
        {open && (
          <div className="mt-3 space-y-3">
            <ParameterLawSections context={context} uses={uses} />
          </div>
        )}
      </details>
    </Section>
  );
}

/** The saved simulation's history of one state or indicator, when it was simulated. */
export function SimulatedHistory({
  context,
  id,
  kind,
}: {
  context: ScopeContext;
  id: string;
  kind: "states" | "indicators";
}) {
  const simulation = context.modelSnapshot.simulation;
  const included =
    simulation &&
    (kind === "states"
      ? simulation.evidence.state_ids.some((state) => state === id)
      : simulation.evidence.observation_layout.variables.some((variable) => variable.id === id));
  if (!simulation || !included) return null;
  return (
    <Section title="Simulated history" wide>
      <SimulationHistory
        modelSnapshot={context.modelSnapshot}
        id={id}
        kind={kind}
        title="Every saved draw"
      />
    </Section>
  );
}

/** All retained coordinates and their empirical marginal, without a preselected report subset. */
export function PosteriorPairs({ context }: { context: ScopeContext }) {
  const fit = context.modelSnapshot.fit;
  const draws = useParameterDraws(context.modelSnapshot);
  const report = useInferenceReport(context.modelSnapshot);
  const [xId, setX] = useState<string | null>(null);
  const [yId, setY] = useState<string | null>(null);
  const columns = draws.data ?? [];
  const x = columns.find((column) => column.subject.element_id === xId) ?? columns.at(0);
  const y =
    columns.find((column) => column.subject.element_id === yId) ??
    columns.find((column) => column.subject.element_id !== x?.subject.element_id);
  if (!fit) return null;
  return (
    <Section title="Joint posterior" wide>
      {draws.error ? (
        <Hint issue>{draws.error.message}</Hint>
      ) : draws.isLoading ? (
        <Hint>Loading retained draws…</Hint>
      ) : x ? (
        <>
          {(
            [
              ["X coordinate", x, setX],
              ["Y coordinate", y, setY],
            ] as const
          ).map(
            ([label, column, set]) =>
              column && (
                <label key={label} className="flex items-center gap-2 text-[10px]">
                  {label}
                  <select
                    className="min-w-0 flex-1 rounded border bg-background p-1"
                    value={column.subject.element_id}
                    onChange={(event) => set(event.target.value)}
                  >
                    {columns.map((choice) => (
                      <option key={choice.subject.element_id} value={choice.subject.element_id}>
                        {choice.label}
                      </option>
                    ))}
                  </select>
                </label>
              ),
          )}
          <EmpiricalPlot points={x.empirical} label={x.label} xLabel={humanize(x.label)} />
          {y && (
            <JointDrawsChart
              x={{ label: humanize(x.label), values: x.values }}
              y={{ label: humanize(y.label), values: y.values }}
              flagged={report.data?.detail.divergent ?? null}
              height={240}
            />
          )}
          <Hint>
            All {x.values.length.toLocaleString()} aligned retained draws. Choose any of{" "}
            {columns.length} parameter coordinates; no thinning or preselection.
          </Hint>
        </>
      ) : (
        <Hint>No retained parameter coordinates.</Hint>
      )}
    </Section>
  );
}
