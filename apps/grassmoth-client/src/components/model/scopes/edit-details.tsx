import type { ModelDiffOutput, DynamicalModelSpec, TimelineRevision } from "@grassmoth/api-types";
import {
  modelConstructs,
  modelEdges,
  modelParameters,
  presentEntries,
} from "@/lib/model-accessors";
import { useModelDiff } from "@/lib/hooks/use-model-diff";
import { useModelSnapshot } from "@/lib/hooks/use-model-snapshot";
import { producingCall } from "@/lib/model-asset/call-dependencies";
import { parameterOwner } from "@/lib/model-asset/entities";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { type EntitySelection, humanize } from "@/lib/model-asset/selection";
import { Hint, OwnerLink, Section } from "../scope-primitives";

export function EditDetails({ context, tick }: { context: ScopeContext; tick: TimelineRevision }) {
  const workspaceId = context.modelSnapshot.workspace_id;
  const outcome = tick.record.attempt.outcome;
  const modelRecord =
    outcome.status === "applied"
      ? outcome.effects.produced.find((artifact) => artifact.artifact_id === "model")
      : undefined;
  const base = modelRecord?.derived_from.model ?? null;
  const hasModel = base !== null;
  const diff = useModelDiff(
    workspaceId,
    base,
    hasModel ? (modelRecord?.revision ?? null) : null,
    context.ticks,
  );
  const before = useModelSnapshot(workspaceId, base ?? undefined, hasModel && Boolean(diff.data));
  const error = diff.error ?? before.error;
  const pruning = context.result?.action === "edit_model" ? context.result.body.pruning : null;
  const removed = pruning ? Object.entries(pruning).filter(([, ids]) => ids.length > 0) : [];
  return (
    <Section title="Model changes" wide>
      {error ? (
        <p role="alert" className="text-destructive">
          Unable to read model changes: {error.message}
        </p>
      ) : !hasModel ? (
        <p className="font-medium">Model created</p>
      ) : diff.isLoading || (diff.data && !before.data) ? (
        <p role="status">Reading model changes…</p>
      ) : diff.data ? (
        <ModelChanges
          context={context}
          report={diff.data}
          before={before.data?.dynamical_model_spec ?? null}
        />
      ) : (
        <Hint>No saved comparison for these model versions.</Hint>
      )}
      {removed.length > 0 && (
        <details className="mt-3 text-sm">
          <summary>Removed outside the question’s outcome ancestry</summary>
          {removed.map(([kind, ids]) => (
            <p key={kind} className="mt-1 break-all text-muted-foreground">
              {humanize(kind)}: {ids.join(", ")}
            </p>
          ))}
        </details>
      )}
    </Section>
  );
}

function DefinitionChange({
  context,
  title,
  label,
  selection,
  value,
}: {
  context: ScopeContext;
  title: string;
  label: string;
  selection: EntitySelection | null;
  value: unknown;
}) {
  return (
    <div>
      <p>
        {title} ·{" "}
        {selection ? (
          <OwnerLink onClick={() => context.select(selection)}>{humanize(label)}</OwnerLink>
        ) : (
          humanize(label)
        )}
      </p>
      {value !== null && (
        <details className="mt-1 text-xs">
          <summary className="cursor-pointer text-muted-foreground">Changed definition</summary>
          <pre className="mt-2 overflow-auto whitespace-pre-wrap rounded bg-muted p-2">
            {JSON.stringify(value, null, 2)}
          </pre>
        </details>
      )}
    </div>
  );
}

function ModelChanges({
  context,
  report,
  before,
}: {
  context: ScopeContext;
  report: ModelDiffOutput;
  before: DynamicalModelSpec | null;
}) {
  const { changes } = report;
  const after = context.modelSnapshot.dynamical_model_spec;
  const status = (removed: boolean, existed: boolean) =>
    removed ? "Removed" : existed ? "Updated" : "Added";
  if (Object.keys(changes).length === 0) return <Hint>No spec changes.</Hint>;
  return (
    <>
      {presentEntries(changes.constructs ?? {}).map(([id, patch]) => {
        const dynamicalModelSpec = patch === null ? before : after;
        const definition = modelConstructs(dynamicalModelSpec).find((item) => item.id === id);
        return (
          <DefinitionChange
            key={id}
            context={context}
            title={`${status(patch === null, before?.constructs[id] !== undefined)} construct`}
            label={definition?.name ?? id}
            selection={patch === null ? null : { kind: "construct", id }}
            value={patch}
          />
        );
      })}
      {presentEntries(changes.edges ?? {}).map(([id, patch]) => {
        const dynamicalModelSpec = patch === null ? before : after;
        const edge = modelEdges(dynamicalModelSpec).find((item) => item.id === id);
        const name = (identity: string) =>
          modelConstructs(dynamicalModelSpec).find((item) => item.id === identity)?.name ??
          identity;
        return (
          <DefinitionChange
            key={id}
            context={context}
            title={`${status(patch === null, before?.edges[id] !== undefined)} edge`}
            label={edge ? `${name(edge.cause.id)} → ${name(edge.effect.id)}` : id}
            selection={patch === null ? null : { kind: "edge", id }}
            value={patch}
          />
        );
      })}
      {presentEntries(changes.parameters ?? {}).map(([id, patch]) => {
        const dynamicalModelSpec = patch === null ? before : after;
        const parameter = modelParameters(dynamicalModelSpec).find((item) => item.id === id);
        const owner = parameterOwner(context.entities, id);
        return (
          <DefinitionChange
            key={id}
            context={context}
            title={`${status(patch === null, before?.parameters[id] !== undefined)} parameter`}
            label={parameter?.name ?? id}
            selection={patch === null ? null : (owner?.selection ?? null)}
            value={patch}
          />
        );
      })}
      {presentEntries(changes.distributions ?? {}).map(([id, patch]) => {
        const dynamicalModelSpec = patch === null ? before : after;
        const owners = [
          ...modelParameters(dynamicalModelSpec),
          ...modelConstructs(dynamicalModelSpec),
        ].filter((item) => item.distribution === id);
        return (
          <DefinitionChange
            key={id}
            context={context}
            title={`${status(patch === null, before?.distributions[id] !== undefined)} law`}
            label={owners.map((item) => item.name).join(", ") || id}
            selection={null}
            value={patch}
          />
        );
      })}
      {presentEntries(changes.law_layouts ?? {}).map(([id, patch]) => (
        <DefinitionChange
          key={id}
          context={context}
          title={`${status(patch === null, before?.law_layouts[id] !== undefined)} law coordinates`}
          label={id}
          selection={null}
          value={patch}
        />
      ))}
      {changes.measurement_clock !== undefined && (
        <DefinitionChange
          context={context}
          title="Updated"
          label="Measurement clock"
          selection={null}
          value={{ measurement_clock: changes.measurement_clock }}
        />
      )}
    </>
  );
}

export function ModelComparisonDetails({ context }: { context: ScopeContext }) {
  const report = context.result?.action === "model_diff" ? context.result.body : null;
  const request = producingCall(context.ticks, context.result?.commit_id)?.record.attempt.request;
  const before = useModelSnapshot(
    context.modelSnapshot.workspace_id,
    request?.action === "model_diff" ? request.input.before_ref : undefined,
    report !== null,
  );
  return (
    <Section title="Model comparison" wide>
      {before.error ? (
        <p role="alert" className="text-destructive">
          Unable to read the comparison base: {before.error.message}
        </p>
      ) : report && before.data ? (
        <ModelChanges context={context} report={report} before={before.data.dynamical_model_spec} />
      ) : (
        <Hint>Reading the saved comparison…</Hint>
      )}
    </Section>
  );
}
