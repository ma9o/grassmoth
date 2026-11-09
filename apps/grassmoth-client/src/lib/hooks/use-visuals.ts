"use client";

import type { ActionSuccess, IndicatorId, ModelSnapshot } from "@grassmoth/api-types";
import { useMemo } from "react";
import { presentEntries } from "@/lib/model-accessors";
import {
  drawsView,
  historyView,
  type PathSeriesView,
  pathsView,
  type SimulationPathsView,
} from "@/lib/model-asset/result-values";
import { useActionResult, useViewedSimulationResult } from "./use-model-snapshot";

function savedObservations(modelSnapshot: ModelSnapshot, result: ActionSuccess | undefined) {
  if (result?.action === "simulate") {
    return {
      history:
        modelSnapshot.state.data && result.body.data[modelSnapshot.state.data.replicate_index],
      owner: {
        ...result.body.report.evidence.observation_layout,
        time_origin: result.body.report.evidence.time_origin,
      },
    };
  }
  if (result?.action === "prepare_data") {
    return { history: result.body.data, owner: result.body.metadata };
  }
  return { history: null, owner: null };
}

export function useObservationHistory(modelSnapshot: ModelSnapshot, id: IndicatorId) {
  const query = useActionResult(
    modelSnapshot.workspace_id,
    modelSnapshot.state.data?.revision ?? modelSnapshot.commit_id,
  );
  const result = query.data;
  const { history, owner } = savedObservations(modelSnapshot, result);
  const selected = history?.[id];
  const variable = owner?.variables.find((observation) => observation.id === id);
  return {
    ...query,
    data:
      selected && variable && owner
        ? {
            ...historyView(selected),
            label: variable.name,
            levels: "levels" in variable.measurement ? variable.measurement.levels : null,
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
  const data = useMemo((): SimulationPathsView | null => {
    if (!paths || !simulation) return null;
    if (start === 0 && count >= paths.total_draws) return paths;
    const page = (series: PathSeriesView): PathSeriesView => ({
      ...series,
      action: series.action.slice(start, start + count),
      reference: series.reference.slice(start, start + count),
    });
    return {
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
    };
  }, [paths, simulation, start, count]);
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
