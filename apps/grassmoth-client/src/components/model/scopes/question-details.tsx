import type { ConstructId, QuestionSpec } from "@grassmoth/api-types";
import { presentEntries } from "@/lib/model-accessors";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { humanize } from "@/lib/model-asset/selection";
import { Hint, OwnerLink, Section, StatusIcon } from "../scope-primitives";

/** The study question: the user's words, its outcome and each query's dated contrast. */
export function QuestionDetails({
  context,
  question: output,
}: {
  context: ScopeContext;
  question: QuestionSpec;
}) {
  const question = output;
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
  const queries = presentEntries(question.queries);
  return (
    <Section title="Question" wide>
      <p className="m-0">{question.text}</p>
      <Hint>Outcome: {name(question.outcome)}</Hint>
      {queries.map(([label, query]) => (
        <div key={label} className="border-t pt-2">
          <p className="m-0 font-medium">{label}</p>
          <Hint>
            From {query.start} for {query.horizon}
          </Hint>
          {query.interventions.map((event) => (
            <p key={`${event.target}-${event.after ?? "start"}`} className="m-0">
              {event.after ? `After ${event.after}` : "At the start"}: set {name(event.target)} to{" "}
              {event.value}
            </p>
          ))}
        </div>
      ))}
    </Section>
  );
}

/** Every check of the question against the model and record, passing or not. */
export function QuestionChecks({ context }: { context: ScopeContext }) {
  const checks = context.modelSnapshot.question_checks;
  if (!checks) return null;
  return (
    <Section title="Question checks" wide>
      {checks.findings.map((finding) => {
        const subject = finding.subject;
        const label =
          "outcome" in subject ? "Outcome" : `${subject.query} · ${humanize(finding.code)}`;
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
