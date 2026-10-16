import type { RecordDependency, TimelineRevision } from "@nof1-causal-lab/api-types";

/** Input roles supplying the saved outputs composed into each action's view. */
const viewArguments = {
  edit_question: [],
  edit_model: ["parent"],
  prepare_data: ["dynamical_model_spec"],
  fit: ["dynamical_model_spec", "data", "fit"],
  simulate: ["dynamical_model_spec"],
  model_diff: ["after"],
  data_diff: ["right"],
} satisfies Record<TimelineRevision["record"]["attempt"]["action"], string[]>;

export function viewDependencies(
  entry: TimelineRevision,
  dependencies: readonly RecordDependency[],
): readonly RecordDependency[] {
  const roles: readonly string[] = viewArguments[entry.record.attempt.action];
  return dependencies.filter(
    (item) => item.seq === entry.record.seq && roles.includes(item.argument),
  );
}

/** Resolve a recorded commit or artifact to its first successful producing call. */
export function producingCall(
  attempts: readonly TimelineRevision[],
  identity: string | undefined,
): TimelineRevision | undefined {
  return attempts.find((entry) => {
    const { outcome } = entry.record.attempt;
    return (
      entry.call_id !== null &&
      outcome.status === "applied" &&
      (entry.commit_id === identity ||
        outcome.effects.produced.some((artifact) => artifact.revision === identity))
    );
  });
}

/** Follow parent references using timeline metadata, without loading intermediate model bodies. */
export function questionCall(
  entry: TimelineRevision,
  attempts: readonly TimelineRevision[],
  dependencies: readonly RecordDependency[],
): TimelineRevision {
  if (entry.record.attempt.action === "edit_question") return entry;
  const request = entry.record.attempt.request;
  const argument =
    request?.action === "fit" && request.input.target.kind === "continue"
      ? "fit"
      : entry.record.attempt.action === "edit_model"
        ? "parent"
        : "dynamical_model_spec";
  const dependency = dependencies.find(
    (item) => item.seq === entry.record.seq && item.argument === argument,
  );
  const parent = attempts.find((item) => item.record.seq === dependency?.source_seq);
  if (!parent) throw new Error(`Missing ${argument} dependency for call ${entry.record.seq}`);
  return questionCall(parent, attempts, dependencies);
}

/** Only declared input dependencies belong to a call's view; journal neighbors do not. */
export function callDependencies(
  selected: TimelineRevision,
  attempts: readonly TimelineRevision[],
  dependencies: readonly RecordDependency[],
): readonly TimelineRevision[] {
  const bySeq = new Map(attempts.map((entry) => [entry.record.seq, entry]));
  const visited = new Set<number>();
  const calls: TimelineRevision[] = [];
  function visit(entry: TimelineRevision) {
    const seq = entry.record.seq;
    if (visited.has(seq)) return;
    visited.add(seq);
    if (entry.record.attempt.action === "edit_model") {
      visit(questionCall(entry, attempts, dependencies));
    } else {
      for (const dependency of viewDependencies(entry, dependencies)) {
        const parent = bySeq.get(dependency.source_seq);
        if (parent === undefined)
          throw new Error(`Missing recorded dependency ${dependency.source_seq} for call ${seq}`);
        visit(parent);
      }
    }
    calls.push(entry);
  }
  visit(selected);
  return calls;
}

/** A node addresses its produced model, or the model selected by its recorded inputs. */
export function modelReference(
  entry: TimelineRevision | undefined,
  attempts: readonly TimelineRevision[],
): string | undefined {
  if (!entry) return undefined;
  const { outcome, request } = entry.record.attempt;
  if (outcome.status === "applied") {
    const own = outcome.effects.produced.find(
      (artifact) => artifact.artifact_id === "model" || artifact.artifact_id === "question",
    );
    if (own) return own.revision;
  }
  if (!request) return undefined;
  switch (request.action) {
    case "edit_question":
      return undefined;
    case "edit_model":
      return request.input.parent_ref;
    case "fit":
      return request.input.target.kind === "new"
        ? request.input.target.dynamical_model_spec_ref
        : modelReference(producingCall(attempts, request.input.target.fit_ref), attempts);
    case "prepare_data":
    case "simulate":
      return request.input.dynamical_model_spec_ref;
    case "model_diff":
      return request.input.after_ref;
    case "data_diff":
      return modelReference(
        producingCall(attempts, request.input.right_ref[0]?.revision),
        attempts,
      );
  }
}
