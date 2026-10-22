import { DefinitionContext } from "./definition-context";
import { Button } from "@/components/ui/button";
import { constructEquation, observationEquation } from "@/lib/model-asset/equations";
import { resolveEntity } from "@/lib/model-asset/entities";
import type { EntitySelection } from "@/lib/model-asset/selection";
import { Hint, OwnerLink, Section } from "./scope-primitives";
import { ConstructScope, IdentificationFinding } from "./scopes/construct-scope";
import { EdgeScope } from "./scopes/edge-scope";
import { IndicatorScope } from "./scopes/indicator-scope";
import type { ScopeContext } from "@/lib/model-asset/scope";
import type { TimelineRevision } from "@grassmoth/api-types";
import { humanize } from "@/lib/model-asset/selection";
import { Katex } from "@/components/analysis-widgets/statistical-model-spec/ssm-equation-display";
import { DataComparisonEvidence, PreparedObservations } from "./scopes/data-details";
import { FitDetails } from "./scopes/fit-details";
import { QuestionChecks, QuestionDetails } from "./scopes/question-details";
import { SimulationEvidence } from "./simulation-evidence";
import { ModelComparisonDetails } from "./scopes/edit-details";

function ModelScope({
  context,
  tick,
}: {
  context: ScopeContext;
  tick: TimelineRevision | undefined;
}) {
  if (
    !tick ||
    tick.record.attempt.outcome.status !== "applied" ||
    tick.record.attempt.request === null
  )
    return <Hint>No new model-wide state was produced.</Hint>;
  if (tick.record.attempt.action === "edit_question")
    return context.modelSnapshot.question ? (
      <QuestionDetails context={context} question={context.modelSnapshot.question} />
    ) : (
      <Hint>No question recorded at this version.</Hint>
    );
  if (tick.record.attempt.action === "fit") return <FitDetails context={context} />;
  if (tick.record.attempt.action === "simulate") return <SimulationEvidence context={context} />;
  if (tick.record.attempt.action === "prepare_data")
    return <PreparedObservations context={context} />;
  if (tick.record.attempt.action === "data_diff")
    return context.dataDiff ? (
      <DataComparisonEvidence context={context} report={context.dataDiff} selection={null} />
    ) : (
      <Hint>Reading the saved comparison…</Hint>
    );
  if (tick.record.attempt.action === "model_diff")
    return <ModelComparisonDetails context={context} />;
  const identification = context.modelSnapshot.identification;
  const equations = [
    ...context.entities.constructs.flatMap((construct) => {
      const equation = constructEquation(construct, context.entities);
      return equation
        ? [
            {
              id: construct.id,
              label: `${construct.name} · ${equation.title}`,
              latex: equation.latex,
            },
          ]
        : [];
    }),
    ...context.entities.indicators.flatMap((indicator) => {
      const latex = observationEquation(indicator, context.entities);
      return latex
        ? [{ id: indicator.observation.id, label: indicator.observation.name, latex }]
        : [];
    }),
  ];

  return (
    <>
      <QuestionChecks context={context} />
      {identification && (
        <Section title="Identification" wide>
          {Object.keys(identification.treatments).length === 0 && (
            <Hint>No treatment findings recorded.</Hint>
          )}
          {context.entities.constructs
            .filter((construct) => construct.id === identification.outcome)
            .map((construct) => (
              <Hint key={construct.id}>Outcome: {humanize(construct.name)}</Hint>
            ))}
          {context.entities.constructs.flatMap((construct) => {
            const finding = identification.treatments[construct.id];
            return finding
              ? [
                  <details key={construct.id}>
                    <summary className="cursor-pointer">
                      <OwnerLink
                        onClick={() => context.select({ kind: "construct", id: construct.id })}
                      >
                        {humanize(construct.name)}
                      </OwnerLink>
                      {" · "}
                      {humanize(finding.status)}
                    </summary>
                    <IdentificationFinding context={context} construct={construct} />
                  </details>,
                ]
              : [];
          })}
        </Section>
      )}
      {equations.length > 0 && (
        <Section title="Authored equations" wide>
          {equations.map(({ id, label, latex }) => (
            <div key={id} className="min-w-0 space-y-2 border-b pb-2">
              <Hint>{humanize(label)}</Hint>
              <div className="overflow-x-auto pb-2">
                <Katex latex={latex} />
              </div>
            </div>
          ))}
        </Section>
      )}
      {!identification && equations.length === 0 && (
        <Hint>No identification report or authored equations available.</Hint>
      )}
    </>
  );
}

/**
 * The state the selected action left, in depth. With a graph part selected it shows that part.
 * With nothing selected it shows the model-wide state that action produced, and only that:
 * - edit_question: the question, meaning its words, its outcome and each query's dated contrast.
 * - edit_model: the model as specified, meaning its checks against the question, how the question
 *   is identified, the equations, and the evidence of its predictive checks for every indicator
 *   under the laws the model holds.
 * - prepare_data: the panel as a whole, meaning every variable's observations over time and the
 *   dataset-level issues.
 * - fit: the fitted model as a whole, meaning predictive calibration, latent mixing and the
 *   joint posterior. Predictive comparisons of saved observations belong to data_diff.
 * - simulate: the simulation as a whole, meaning its design, the paired draws of every state and
 *   indicator, the effect's paired draws or why it is withheld, and its checks.
 * - data_diff: every variable's observations and all replicates, test-statistic distributions,
 *   each side's statistic histograms, point changes and comparison limitations. Selecting an
 *   indicator scopes that evidence to it; selecting a construct scopes it to its indicators.
 * Evidence stays at full resolution: distributions, individual draws and chains, observations
 * over time. Counts, means, intervals and verdicts may label a chart but never replace it, because
 * reading a Bayesian model depends on seeing the whole distribution.
 */
export function DetailsPane({
  selection,
  context,
  loading,
  tick,
}: {
  selection: EntitySelection | null;
  context: ScopeContext;
  loading: boolean;
  tick: TimelineRevision | undefined;
}) {
  const entity = selection ? resolveEntity(context.entities, selection) : null;
  return (
    <section
      aria-label="Model details"
      className="flex h-[460px] min-h-0 min-w-0 flex-none flex-col gap-3 overflow-hidden rounded-2xl border bg-card px-3 py-3 md:max-h-[52%]"
    >
      <div className="flex flex-none items-center gap-2 text-xs">
        <h2 className="min-w-0 flex-1 font-semibold">
          {selection ? (entity?.label ?? "Absent entity") : "Model state"}
        </h2>
        {selection && (
          <Button
            type="button"
            size="icon-sm"
            variant="ghost"
            aria-label="Show model state"
            onClick={() => context.select(null)}
          >
            ×
          </Button>
        )}
      </div>
      {entity && <DefinitionContext entity={entity} onSelect={context.select} />}
      <div
        key={selection?.id ?? tick?.record.seq}
        className="flex min-h-0 flex-1 flex-col flex-wrap content-start items-start gap-x-[18px] gap-y-3 overflow-x-auto pb-2 [&>section]:max-h-full [&>section]:w-[280px] [&>section[data-wide]]:w-[400px] [&>section>div:last-child]:overflow-auto"
      >
        {loading ? (
          <p role="status" className="text-xs">
            Loading version…
          </p>
        ) : context.dataDiff ? (
          <DataComparisonEvidence
            context={context}
            report={context.dataDiff}
            selection={selection}
          />
        ) : !selection ? (
          <ModelScope context={context} tick={tick} />
        ) : !entity ? (
          <Hint>This {selection.kind} is absent from this revision.</Hint>
        ) : selection.kind === "construct" ? (
          <ConstructScope context={context} id={selection.id} />
        ) : selection.kind === "edge" ? (
          <EdgeScope context={context} id={selection.id} />
        ) : (
          <IndicatorScope context={context} id={selection.id} />
        )}
      </div>
    </section>
  );
}
