/* eslint-disable */
/**
 * AUTO-GENERATED — DO NOT EDIT
 *
 * Generated from Python contract owners via:
 *   cd apps/grassmoth-server && uv run python -m scripts.codegen.export_api
 *   cd packages/api-types && bun run scripts/generate.ts
 *
 * Metadata exporter: apps/grassmoth-server/scripts/codegen/export_api.py
 */

import type { ArtifactId } from "./models";
export const ARTIFACT_IDS = ["question","raw_data","model","panel"] as const satisfies readonly ArtifactId[];

const _OBS_HYPERS_BY_DIST = {
  "student_t": [
    "obs_df"
  ],
  "gamma": [
    "obs_shape"
  ],
  "negative_binomial": [
    "obs_r"
  ],
  "beta": [
    "obs_concentration"
  ],
  "ordered_logistic": [
    "obs_ordered_base",
    "obs_ordered_gaps"
  ],
  "categorical": [
    "obs_cat_intercepts",
    "obs_cat_slopes"
  ]
} as const;

export type ObservationHyperparameter =
  typeof _OBS_HYPERS_BY_DIST[keyof typeof _OBS_HYPERS_BY_DIST][number];

export const OBSERVATION_HYPERPARAMETERS_BY_DISTRIBUTION: Partial<
  Record<string, readonly ObservationHyperparameter[]>
> = _OBS_HYPERS_BY_DIST;
