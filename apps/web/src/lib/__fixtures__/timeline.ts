/** Slim action records owned by the journal and timeline contract tests. */
import type { Applied, TimelineRecord, TimelineRevision } from "@nof1-causal-lab/api-types";

export const applied: Applied<null> = {
  status: "applied",
  result: null,
  effects: { produced: [], retracted: [], reports: {} },
};

export function revision(seq: number, attempt: TimelineRecord["attempt"]): TimelineRevision {
  return {
    call_id: `call:${seq.toString(16).padStart(64, "0")}`,
    commit_id: seq.toString(16).padEnd(40, "0"),
    parent_ids: [(seq - 1).toString(16).padEnd(40, "0")],
    record: { seq, ts: "2026-01-01T00:00:00Z", messages: [], trace_ids: [], attempt },
  };
}
export const simulation = revision(5, {
  action: "simulate",
  request: {
    action: "simulate",
    input: {
      dynamical_model_spec_ref: "3".repeat(40),
      simulation: { start: "2026-01-01", horizon: "1d", interventions: [] },
      seed: 0,
      num_replicates: 100,
    },
    reasoning: null,
  },
  outcome: applied,
});
export const failedFit = revision(4, {
  action: "fit",
  request: null,
  outcome: { status: "raised", error_type: "FitError", error_message: "Fit failed.", details: [] },
});
export const comparison = revision(6, {
  action: "data_diff",
  request: {
    action: "data_diff",
    input: {
      left_ref: [{ revision: "2".repeat(40), replicate_index: null }],
      right_ref: [{ revision: "5".repeat(40), replicate_index: 0 }],
    },
    reasoning: null,
  },
  outcome: applied,
});
