import type { TimelineRevision } from "@grassmoth/api-types";
import { Button } from "@/components/ui/button";
import { type EntityPresentation, resolveEntity } from "@/lib/model-asset/entities";
import type { ScopeContext } from "@/lib/model-asset/scope";
import type { EntitySelection } from "@/lib/model-asset/selection";
import { humanize, mechanismName } from "@/lib/model-asset/selection";
import { glyphKey, viewedSpecPlots } from "@/lib/model-asset/spec-plots";
import { cn } from "@/lib/utils";
import { AuthoredEquations } from "./authored-equations";
import { DefinitionContext } from "./definition-context";
import { Hint, OwnerLink, Section } from "./scope-primitives";
import { ConstructScope, IdentificationFinding } from "./scopes/construct-scope";
import { DataComparisonEvidence, PreparedObservations } from "./scopes/data-details";
import { EdgeScope } from "./scopes/edge-scope";
import { ModelComparisonDetails } from "./scopes/edit-details";
import { FitDetails } from "./scopes/fit-details";
import { IndicatorScope } from "./scopes/indicator-scope";
import { QuestionChecks, QuestionDetails } from "./scopes/question-details";
import { SpecSections } from "./scopes/spec-sections";
import { SimulationEvidence } from "./simulation-evidence";
import { PaneToggle } from "./workbench-panes";

function ModelScope({
  context,
  tick,
}: {
  readonly context: ScopeContext;
  readonly tick: TimelineRevision | undefined;
}) {
  if (tick?.record.attempt.outcome.status !== "applied" || tick.record.attempt.request === null) {
    return <Hint>No new model-wide state was produced.</Hint>;
  }
  if (tick.record.attempt.action === "fit") return <FitDetails context={context} />;
  if (tick.record.attempt.action === "simulate") return <SimulationEvidence context={context} />;
  if (tick.record.attempt.action === "prepare_data") {
    return <PreparedObservations context={context} />;
  }
  if (tick.record.attempt.action === "data_diff") {
    return context.dataDiff ? (
      <DataComparisonEvidence context={context} report={context.dataDiff} selection={null} />
    ) : (
      <Hint>Reading the saved comparison…</Hint>
    );
  }
  if (tick.record.attempt.action === "model_diff") {
    return <ModelComparisonDetails context={context} />;
  }
  const identification = context.modelSnapshot.identification;

  return (
    <>
      {context.modelSnapshot.dynamical_model_spec && (
        <QuestionDetails context={context} model={context.modelSnapshot.dynamical_model_spec} />
      )}
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
      {!identification && context.entities.constructs.length === 0 && (
        <Hint>No identification report or authored equations available.</Hint>
      )}
    </>
  );
}

/** A data comparison, the selected part's scope, or the model-wide state with no selection. */
function DetailsContent({
  selection,
  entity,
  context,
  loading,
  tick,
  showCardRows,
}: {
  readonly selection: EntitySelection | null;
  readonly entity: EntityPresentation | null | undefined;
  readonly context: ScopeContext;
  readonly loading: boolean;
  readonly tick: TimelineRevision | undefined;
  readonly showCardRows: boolean;
}) {
  if (loading) {
    return (
      <p role="status" className="text-xs">
        Loading version…
      </p>
    );
  }
  if (context.dataDiff) {
    return (
      <DataComparisonEvidence context={context} report={context.dataDiff} selection={selection} />
    );
  }
  if (!selection) return <ModelScope context={context} tick={tick} />;
  if (!entity) return <Hint>This {selection.kind} is absent from this revision.</Hint>;
  if (selection.glyph) {
    // The function is a node of its own. Its full relationship or measurement is one click away.
    return (
      <Section title={selection.kind === "indicator" ? "Measurement" : "Defined on"}>
        <OwnerLink onClick={() => context.select(entity.selection)}>{entity.label}</OwnerLink>
      </Section>
    );
  }
  if (selection.kind === "construct") {
    return <ConstructScope context={context} id={selection.id} showCardRows={showCardRows} />;
  }
  if (selection.kind === "edge") return <EdgeScope context={context} id={selection.id} />;
  return <IndicatorScope context={context} id={selection.id} />;
}

function detailsTitle(
  selection: EntitySelection | null,
  owner: EntityPresentation | null | undefined,
) {
  if (!selection) return "Model state";
  if (!owner) return "Absent entity";
  if (selection.glyph === "response") return `Expected reading for ${owner.label}`;
  return selection.glyph ? mechanismName(selection.glyph) : owner.label;
}

/**
 * The scope's sections flow down the pane, then across it. The box itself flows vertically and
 * its sections flow back: only then is it as wide as the columns it holds, where a wrapping
 * column flexbox keeps the width of one. The equations take the width that leaves.
 */
const SCOPE =
  "relative flex min-h-0 flex-row flex-wrap content-start items-start gap-x-3 gap-y-[18px] overflow-x-auto pb-2 [writing-mode:vertical-lr] [&>*]:[writing-mode:horizontal-tb] [&>section]:max-h-full [&>section]:w-[280px] [&>section[data-wide]]:w-[400px] [&>section>div:last-child]:overflow-auto";

/**
 * The state the selected action left, in depth. It leads with the model in text: the authored
 * equations stay beside every scope, take all the width the scope leaves, mark the selected part
 * and navigate as the graph does. With a graph part selected the rest shows that part. With
 * nothing selected it shows the model-wide state that action produced, and only that:
 * - edit_model: the model as specified, meaning its checks against the question, how the question
 *   is identified, and the evidence of its predictive checks for every indicator under the laws
 *   the model holds.
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
  readonly selection: EntitySelection | null;
  readonly context: ScopeContext;
  readonly loading: boolean;
  readonly tick: TimelineRevision | undefined;
}) {
  const entity = selection ? resolveEntity(context.entities, selection) : null;
  const step = tick?.record.attempt.action ?? null;
  const plots = viewedSpecPlots(context.modelSnapshot, step);
  const model = context.modelSnapshot.dynamical_model_spec;
  const text = model && context.entities.constructs.length > 0 ? model : null;
  const related = entity && entity.relationships.length > 0 ? entity : null;
  return (
    <section
      aria-label="Model details"
      className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden rounded-2xl border bg-card"
    >
      <div className="flex min-h-11 flex-none items-center gap-2 border-b px-3 py-1.5 text-xs">
        <h2 className="min-w-0 flex-1 font-semibold">{detailsTitle(selection, entity)}</h2>
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
        <PaneToggle title="model state" />
      </div>
      <div className="flex min-h-0 flex-1 gap-3 p-3">
        {text && (
          <div className="flex min-h-0 min-w-[440px] flex-1 border-r pr-3 pb-2 [&>section]:max-h-full [&>section]:w-full [&>section>div:last-child]:overflow-y-auto">
            <AuthoredEquations model={text} context={context} selection={selection} plots={plots} />
          </div>
        )}
        <div
          key={selection ? glyphKey(selection) : tick?.record.seq}
          className={cn(
            SCOPE,
            text ? "max-w-[calc(100%-464px)] flex-none" : "flex-1",
            related && "pt-7",
          )}
        >
          {related && (
            <div className="absolute top-0 left-0">
              <DefinitionContext entity={related} onSelect={context.select} />
            </div>
          )}
          {!loading && plots && selection && entity && (
            <SpecSections context={context} selection={selection} plots={plots} />
          )}
          <DetailsContent
            selection={selection}
            entity={entity}
            context={context}
            loading={loading}
            tick={tick}
            showCardRows={step === "edit_model" && !plots}
          />
        </div>
      </div>
    </section>
  );
}
