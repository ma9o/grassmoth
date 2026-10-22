import { presentEntries } from "@/lib/model-accessors";
import type { ScopeContext } from "@/lib/model-asset/scope";
import type { ActionSuccess } from "@grassmoth/api-types";
import { callModelOutput } from "@/lib/model-asset/compose-call-view";
import { humanize, type EntitySelection } from "@/lib/model-asset/selection";
import { Hint, OwnerLink, Section, StatusIcon } from "./scope-primitives";

/** Current findings for the saved call remain visible when read from cache. */
export function ActionFindings({ context, call }: { context: ScopeContext; call: ActionSuccess }) {
  const { modelSnapshot, entities, select } = context;
  const checks = callModelOutput(call)?.checks;
  const identification = call.action === "edit_model" ? call.body.identification : null;
  const validation = call.action === "fit" ? call.body.checks : null;
  const data = validation?.data ?? (call.action === "prepare_data" ? call.body.profile : null);
  const question = call.action === "fit" ? call.body.checks.question : checks?.question;
  const findings: Array<{
    label: string;
    reason: string | null;
    status: "passed" | "failed" | "warning" | "error" | "not_evaluated";
    owner?: EntitySelection;
  }> = [];
  for (const finding of [...(checks?.specification ?? []), ...(validation?.preflight ?? [])]) {
    if (finding.kind !== "evaluated" || finding.outcome !== "passed")
      findings.push({
        label: humanize(finding.code),
        reason: finding.kind === "evaluated" ? finding.evidence : finding.detail,
        status: finding.kind === "evaluated" ? finding.outcome : "not_evaluated",
      });
  }
  for (const finding of [
    ...(data?.findings ?? []),
    ...presentEntries(data?.indicators ?? {}).flatMap(([, audit]) => audit.findings),
  ]) {
    if (finding.kind === "evaluated" && finding.outcome === "passed") continue;
    const id = typeof finding.subject === "string" ? null : finding.subject.id;
    const indicator = entities.indicators.find((item) => item.observation.id === id);
    const variable = modelSnapshot.metadata?.variables.find((item) => item.id === id);
    findings.push({
      label: humanize(indicator?.observation.name ?? variable?.name ?? "Dataset"),
      reason: finding.kind === "evaluated" ? finding.evidence : finding.detail,
      status: finding.kind === "evaluated" ? finding.outcome : "not_evaluated",
      ...(indicator ? { owner: { kind: "indicator" as const, id: indicator.observation.id } } : {}),
    });
  }
  for (const construct of entities.constructs) {
    const finding = identification?.treatments[construct.id];
    if (finding?.status === "not_identified")
      findings.push({
        label: humanize(construct.name),
        reason: [
          "Not identified.",
          finding.confounders.length
            ? `Confounded by ${entities.constructs
                .filter((entity) => finding.confounders.includes(entity.id))
                .map((entity) => humanize(entity.name))
                .join(", ")}.`
            : "",
          finding.notes,
        ]
          .filter(Boolean)
          .join(" "),
        status: "failed",
        owner: { kind: "construct", id: construct.id },
      });
  }
  for (const finding of question?.findings ?? []) {
    if (finding.kind === "evaluated" && finding.outcome === "passed") continue;
    const subject = finding.subject;
    const construct =
      "outcome" in subject ? subject.outcome.id : "target" in subject ? subject.target.id : null;
    const entity = entities.constructs.find((item) => item.id === construct);
    findings.push({
      label:
        "outcome" in subject
          ? "Question outcome"
          : `${subject.query} · ${humanize(finding.code)}${entity ? ` · ${humanize(entity.name)}` : ""}`,
      reason: finding.kind === "evaluated" ? finding.evidence : finding.detail,
      status: finding.kind === "evaluated" ? finding.outcome : "not_evaluated",
      ...(entity ? { owner: { kind: "construct" as const, id: entity.id } } : {}),
    });
  }
  if (
    call.action === "prepare_data" &&
    call.body.extraction.workers.some((worker) => worker.status === "failed")
  )
    findings.push({
      label: "Extraction incomplete",
      reason: "Some extraction workers failed. The prepared data is incomplete.",
      status: "warning",
    });
  return (
    <>
      {identification && Object.keys(identification.treatments).length > 0 && (
        <Section title="Identification">
          <Hint>
            {
              presentEntries(identification.treatments)
                .map(([, finding]) => finding)
                .filter((finding) => finding.status === "identified").length
            }{" "}
            identified;{" "}
            {
              presentEntries(identification.treatments)
                .map(([, finding]) => finding)
                .filter((finding) => finding.status === "not_identified").length
            }{" "}
            not identified.
          </Hint>
        </Section>
      )}
      {findings.length > 0 && (
        <Section title="Problems">
          {findings.map(({ owner, ...finding }, index) => (
            <div key={index} className="flex items-start gap-2">
              <StatusIcon status={finding.status} />
              <p className="min-w-0">
                {owner ? (
                  <OwnerLink onClick={() => select(owner)}>{finding.label}</OwnerLink>
                ) : (
                  finding.label
                )}
                {finding.reason !== null && <>: {finding.reason}</>}
              </p>
            </div>
          ))}
        </Section>
      )}
    </>
  );
}
