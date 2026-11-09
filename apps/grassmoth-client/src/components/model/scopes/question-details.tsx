import type { ConstructId, DynamicalModelSpec } from "@grassmoth/api-types";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { humanize } from "@/lib/model-asset/selection";
import { Hint, OwnerLink, Section, StatusIcon } from "../scope-primitives";

/** The question, model time domain and the core intervention plan owned by this revision. */
export function QuestionDetails({
  context,
  model,
}: {
  readonly context: ScopeContext;
  readonly model: DynamicalModelSpec;
}) {
  const name = (id: ConstructId) => {
    const construct = context.entities.constructs.find((item) => item.id === id);
    return construct ? (
      <OwnerLink onClick={() => context.select({ kind: "construct", id })}>
        {humanize(construct.name)}
      </OwnerLink>
    ) : (
      <span className="font-mono">{id}</span>
    );
  };
  const query = model.core_query;
  return (
    <Section title="Question" wide>
      <p className="m-0">{model.question}</p>
      <Hint>Outcome: {name(model.outcome)}</Hint>
      <Hint>
        Model: from {model.time.window.start} for {model.time.window.horizon}, step{" "}
        {model.time.step}
      </Hint>
      <div className="border-t pt-2">
        <p className="m-0 font-medium">Core query</p>
        <Hint>
          From {query.window.start} for {query.window.horizon}
        </Hint>
        {query.interventions.map((event) => (
          <p key={`${event.target}-${event.after ?? "start"}`} className="m-0">
            {event.after ? `After ${event.after}` : "At the start"}: set {name(event.target)} to{" "}
            {event.value}
          </p>
        ))}
      </div>
    </Section>
  );
}

/** Every check of the question against the model and record, passing or not. */
export function QuestionChecks({ context }: { readonly context: ScopeContext }) {
  const checks = context.modelSnapshot.question_checks;
  if (!checks) return null;
  return (
    <Section title="Question checks" wide>
      {checks.findings.map((finding) => {
        const subject = finding.subject;
        const label = "outcome" in subject ? "Outcome" : `Core query · ${humanize(finding.code)}`;
        return (
          <p
            key={`${finding.code}:${JSON.stringify(subject)}`}
            className="m-0 flex items-start gap-1.5"
          >
            <StatusIcon status={finding.kind === "evaluated" ? finding.outcome : "not_evaluated"} />
            <span>
              <span className="font-medium">{label}</span>:{" "}
              {finding.kind === "evaluated" ? finding.evidence : finding.detail}
            </span>
          </p>
        );
      })}
    </Section>
  );
}
