import type { DynamicalModelSpec, ModelDiffOutput, ModelSnapshot } from "@grassmoth/api-types";
import { useModelSnapshot } from "@/lib/hooks/use-model-snapshot";
import {
  modelConstructs,
  modelEdges,
  modelParameters,
  presentEntries,
} from "@/lib/model-accessors";
import { producingCall } from "@/lib/model-asset/call-dependencies";
import { parameterOwner } from "@/lib/model-asset/entities";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { type EntitySelection, humanize } from "@/lib/model-asset/selection";
import { Hint, OwnerLink, Section } from "../scope-primitives";

function DefinitionChange({
  context,
  title,
  label,
  selection,
  value,
}: {
  readonly context: ScopeContext;
  readonly title: string;
  readonly label: string;
  readonly selection: EntitySelection | null;
  readonly value: unknown;
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

function status(removed: boolean, existed: boolean) {
  if (removed) return "Removed";
  if (existed) return "Updated";
  return "Added";
}

function ModelChanges({
  context,
  report,
  before,
}: {
  readonly context: ScopeContext;
  readonly report: ModelDiffOutput;
  readonly before: DynamicalModelSpec | null;
}) {
  const { changes } = report;
  const after = context.modelSnapshot.dynamical_model_spec;
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
      {(["question", "outcome", "time", "core_query"] as const).map(
        (field) =>
          changes[field] !== undefined && (
            <DefinitionChange
              key={field}
              context={context}
              title="Updated"
              label={field}
              selection={null}
              value={{ [field]: changes[field] }}
            />
          ),
      )}
    </>
  );
}

/** The saved patch against its recorded base, once both are read. */
function ModelComparisonContent({
  context,
  report,
  error,
  base,
}: {
  readonly context: ScopeContext;
  readonly report: ModelDiffOutput | null;
  readonly error: Readonly<Error> | null;
  readonly base: ModelSnapshot | undefined;
}) {
  if (error) {
    return (
      <p role="alert" className="text-destructive">
        Unable to read the comparison base: {error.message}
      </p>
    );
  }
  if (report && base) {
    return <ModelChanges context={context} report={report} before={base.dynamical_model_spec} />;
  }
  return <Hint>Reading the saved comparison…</Hint>;
}

export function ModelComparisonDetails({ context }: { readonly context: ScopeContext }) {
  const report = context.result?.action === "model_diff" ? context.result.body : null;
  const request = producingCall(context.ticks, context.result?.commit_id)?.record.attempt.request;
  const before = useModelSnapshot(
    context.modelSnapshot.workspace_id,
    request?.action === "model_diff" ? request.input.before_ref : undefined,
    report !== null,
  );
  return (
    <Section title="Model comparison" wide>
      <ModelComparisonContent
        context={context}
        report={report}
        error={before.error}
        base={before.data}
      />
    </Section>
  );
}
