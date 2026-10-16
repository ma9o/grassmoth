import type { EditModelOutput, FitOutput, ModelSnapshot } from "@nof1-causal-lab/api-types";
import { fixtureValue } from "./fixture-value";

/** Retained model projections shared by isolated action-response fixtures. */
export function modelResult(snapshot: ModelSnapshot): EditModelOutput {
  return {
    dynamical_model_spec: fixtureValue(snapshot.dynamical_model_spec),
    checks: {
      specification: snapshot.specification ?? [],
      question: fixtureValue(snapshot.question_checks),
    },
    identification: fixtureValue(snapshot.identification),
    pruning: { constructs: [], edges: [], parameters: [], distributions: [] },
  };
}

/** Fit fixtures expose only the values owned by fitting. */
export function fitResult(snapshot: ModelSnapshot): FitOutput {
  return {
    dynamical_model_spec: fixtureValue(snapshot.dynamical_model_spec),
    checks: fixtureValue(snapshot.fit_checks),
    inference: {
      evidence: {
        checkpoint: null,
        latent_free_mask: null,
        chain_extra_fields: {},
        observation_log_probs: null,
        observed_rows: null,
        exact_observation_rows: null,
        phase_extra_fields: {},
        warmup_complete_log_posterior_history: null,
        all_complete_log_posterior_history: null,
        initial_latent_delta: null,
        final_latent_delta: null,
      },
      core: fixtureValue(snapshot.fit),
      detail: { trace_data: [], rank_histograms: [], pareto_k: [], loo_pit: [] },
    },
  };
}
