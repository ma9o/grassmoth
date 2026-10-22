"use client";

import type { IndicatorId, ModelSnapshot } from "@grassmoth/api-types";
import {
  historyView,
  pathsView,
  drawsView,
  type PathSeriesView,
  type SimulationPathsView,
} from "@/lib/model-asset/result-values";
import { presentEntries } from "@/lib/model-accessors";
import { useActionResult, useViewedSimulationResult } from "./use-model-snapshot";

export function useObservationHistory(modelSnapshot: ModelSnapshot, id: IndicatorId) {
  const query = useActionResult(
    modelSnapshot.workspace_id,
    modelSnapshot.state.data?.revision ?? modelSnapshot.commit_id,
  );
  const result = query.data;
  const history =
    result?.action === "simulate"
      ? modelSnapshot.state.data && result.body.data[modelSnapshot.state.data.replicate_index]
      : result?.action === "prepare_data"
        ? result.body.data
        : null;
  const selected = history?.[id];
  const owner =
    result?.action === "prepare_data"
      ? result.body.metadata
      : result?.action === "simulate"
        ? {
            ...result.body.report.evidence.observation_layout,
            time_origin: result.body.report.evidence.time_origin,
          }
        : null;
  const variable = owner?.variables.find((variable) => variable.id === id);
  return {
    ...query,
    data:
      selected && variable && owner
        ? {
            ...historyView(selected),
            label: variable.name,
            levels: variable.ordinal_levels ?? variable.categorical_levels,
            time_origin: owner.time_origin,
          }
        : null,
  };
}

/** Which saved draws a chart shows: a contiguous page, or every draw. */
export interface DrawSelection {
  readonly start: number;
  readonly count: number;
}

export const ALL_DRAWS: DrawSelection = { start: 0, count: Number.POSITIVE_INFINITY };

/**
 * The server returns every draw of the viewed model's simulation; a selection pages them.
 * Another model revision's simulation is never fetched.
 */
export function useSimulationPaths(
  modelSnapshot: ModelSnapshot,
  selection: DrawSelection = ALL_DRAWS,
) {
  const simulation = modelSnapshot.simulation;
  const query = useViewedSimulationResult(modelSnapshot, simulation !== null);
  const paths =
    query.data?.action === "simulate" && modelSnapshot.dynamical_model_spec
      ? pathsView(query.data.body, modelSnapshot.dynamical_model_spec)
      : null;
  const { start, count } = selection;
  const page = (series: PathSeriesView): PathSeriesView => ({
    ...series,
    action: series.action.slice(start, start + count),
    reference: series.reference.slice(start, start + count),
  });
  const data: SimulationPathsView | null =
    paths && simulation
      ? {
          ...paths,
          start,
          count: Math.min(count, paths.total_draws - start),
          states: Object.fromEntries(
            presentEntries(paths.states).map(([id, series]) => [id, page(series)]),
          ),
          indicators: Object.fromEntries(
            presentEntries(paths.indicators).map(([id, series]) => [id, page(series)]),
          ),
          effect: paths.effect ? page(paths.effect) : null,
        }
      : null;
  return { ...query, data };
}

export function useParameterDraws(modelSnapshot: ModelSnapshot) {
  const query = useActionResult(
    modelSnapshot.workspace_id,
    modelSnapshot.state.current.model?.revision,
    modelSnapshot.fit != null,
  );
  return { ...query, data: query.data?.action === "fit" ? drawsView(query.data.body) : null };
}
