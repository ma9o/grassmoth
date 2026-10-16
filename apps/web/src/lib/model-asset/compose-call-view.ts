import type {
  ActionSuccess,
  EditModelOutput,
  FitOutput,
  ModelSnapshot,
  PrepareDataOutput,
  RecordDependency,
  TimelineRevision,
} from "@nof1-causal-lab/api-types";
import { questionCall } from "./call-dependencies";

export function callModelOutput(result: ActionSuccess | undefined): EditModelOutput | null {
  return result?.action === "edit_model" ? result.body : null;
}

interface ViewParts {
  artifacts: ModelSnapshot["state"]["current"];
  question: ModelSnapshot["question"];
  modelOutput: EditModelOutput | null;
  dynamicalModelSpec: ModelSnapshot["dynamical_model_spec"];
  fitOutput: FitOutput | null;
  prepared: PrepareDataOutput | null;
  data: ModelSnapshot["state"]["data"];
  fit: ModelSnapshot["fit"];
  simulation: ModelSnapshot["simulation"];
}

const empty: ViewParts = {
  artifacts: {},
  question: null,
  modelOutput: null,
  dynamicalModelSpec: null,
  fitOutput: null,
  prepared: null,
  data: null,
  fit: null,
  simulation: null,
};

/** Select already computed outputs by their recorded roles; never reconstruct science here. */
export function composeCallView(
  workspaceId: string,
  selected: TimelineRevision,
  attempts: readonly TimelineRevision[],
  dependencies: readonly RecordDependency[],
  results: ReadonlyMap<number, ActionSuccess>,
): ModelSnapshot {
  const bySeq = new Map(attempts.map((entry) => [entry.record.seq, entry]));
  const views = new Map<number, ViewParts>();

  function inputCall(entry: TimelineRevision, argument: string): TimelineRevision {
    const dependency = dependencies.find(
      (item) => item.seq === entry.record.seq && item.argument === argument,
    );
    if (!dependency) throw new Error(`Missing ${argument} dependency for call ${entry.record.seq}`);
    const parent = bySeq.get(dependency.source_seq);
    if (!parent) throw new Error(`Missing ${argument} dependency for call ${entry.record.seq}`);
    return parent;
  }

  function input(entry: TimelineRevision, argument: string): ViewParts {
    return parts(inputCall(entry, argument));
  }

  function parts(entry: TimelineRevision): ViewParts {
    const seq = entry.record.seq;
    const cached = views.get(seq);
    if (cached) return cached;
    const result = results.get(seq);
    if (!result) throw new Error(`Missing saved result for call ${seq}`);
    const request = entry.record.attempt.request;
    const outcome = entry.record.attempt.outcome;
    const produced: ModelSnapshot["state"]["current"] = Object.fromEntries(
      outcome.status === "applied"
        ? outcome.effects.produced.map((artifact) => [artifact.artifact_id, artifact])
        : [],
    );
    let view: ViewParts;
    switch (result.action) {
      case "edit_question":
        view = { ...empty, question: result.body.question, artifacts: produced };
        break;
      case "edit_model": {
        const question = parts(questionCall(entry, attempts, dependencies));
        view = {
          ...empty,
          question: question.question,
          artifacts: { ...question.artifacts, ...produced },
          modelOutput: result.body,
          dynamicalModelSpec: result.body.dynamical_model_spec,
        };
        break;
      }
      case "prepare_data": {
        const modelView = input(entry, "dynamical_model_spec");
        const panel = produced.panel;
        view = {
          ...modelView,
          artifacts: {
            ...Object.fromEntries(
              Object.entries(modelView.artifacts).filter(
                ([key]) => key === "model" || key === "question",
              ),
            ),
            ...produced,
          },
          prepared: result.body,
          fitOutput: null,
          data: panel ? { revision: panel.revision, replicate_index: 0 } : null,
          simulation: null,
        };
        break;
      }
      case "fit": {
        if (request?.action !== "fit") throw new Error("Saved fit is missing its input references");
        if (request.input.target.kind === "continue") {
          const parent = input(entry, "fit");
          view = {...parent, artifacts: {...parent.artifacts, ...produced},
            dynamicalModelSpec: result.body.dynamical_model_spec ?? parent.dynamicalModelSpec,
            fitOutput: result.body, fit: result.body.inference.core};
          break;
        }
        const dataCall = inputCall(entry, "data");
        const data = parts(dataCall);
        const modelView = input(entry, "dynamical_model_spec");
        view = {
          ...empty,
          question: modelView.question,
          artifacts: {
            ...Object.fromEntries(
              Object.entries(modelView.artifacts).filter(
                ([key]) => key === "model" || key === "question",
              ),
            ),
            ...(dataCall.record.attempt.action === "prepare_data"
              ? Object.fromEntries(
                  Object.entries(data.artifacts).filter(
                    ([key]) => key === "panel" || key === "raw_data",
                  ),
                )
              : {}),
            ...produced,
          },
          modelOutput: modelView.modelOutput,
          dynamicalModelSpec: result.body.dynamical_model_spec ?? modelView.dynamicalModelSpec,
          fitOutput: result.body,
          prepared: dataCall.record.attempt.action === "prepare_data" ? data.prepared : null,
          data: request.input.target.data_ref,
          fit: result.body.inference.core,
        };
        break;
      }
      case "simulate": {
        if (request?.action !== "simulate")
          throw new Error("Saved simulation is missing its input references");
        const modelView = input(entry, "dynamical_model_spec");
        view = {
          ...modelView,
          artifacts: {
            ...Object.fromEntries(
              Object.entries(modelView.artifacts).filter(
                ([key]) => key === "model" || key === "question",
              ),
            ),
          },
          prepared: null,
          data: null,
          simulation: result.body.report,
        };
        break;
      }
      case "model_diff":
        view = input(entry, "after");
        break;
      case "data_diff":
        view = input(entry, "right");
        break;
    }
    views.set(seq, view);
    return view;
  }

  const view = parts(selected);
  const modelOutput = view.modelOutput;
  const prepared = view.prepared;
  return {
    workspace_id: workspaceId,
    commit_id: selected.commit_id,
    selected_seq: selected.record.seq,
    question: view.question,
    state: {
      current: view.artifacts,
      data: view.data,
    },
    dynamical_model_spec: view.dynamicalModelSpec,
    metadata: prepared?.metadata ?? null,
    profile: prepared?.profile ?? null,
    identification: modelOutput?.identification ?? null,
    fit_checks: view.fitOutput?.checks ?? null,
    specification: modelOutput?.checks.specification ?? null,
    question_checks: modelOutput
      ? {
          findings: [
            ...modelOutput.checks.question.findings,
            ...(view.fitOutput?.checks.question.findings ?? []),
          ],
        }
      : null,
    fit: view.fit,
    simulation: view.simulation,
  };
}
