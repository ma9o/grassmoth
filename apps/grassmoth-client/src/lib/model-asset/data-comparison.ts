import type {
  ActionSuccess,
  DataComparisonReport,
  GitOid,
  PrepareDataOutput,
} from "@grassmoth/api-types";
import { historyView, type HistoryView } from "./result-values";

export type DataSourceResult = Extract<ActionSuccess, { action: "prepare_data" | "simulate" }>;

interface DataSeriesView {
  readonly variable: PrepareDataOutput["metadata"]["variables"][number];
  readonly time_origin: string;
  readonly history: HistoryView;
}

export type DataVariableView = DataComparisonReport["variables"][number] & {
  readonly left: readonly (DataSeriesView | null)[];
  readonly right: readonly (DataSeriesView | null)[];
};

export type DataComparisonView = Omit<DataComparisonReport, "variables"> & {
  readonly variables: readonly DataVariableView[];
};

/** Attach original histories for display; all comparisons and statistics stay in the saved report. */
export function dataComparisonView(
  report: DataComparisonReport,
  sources: ReadonlyMap<GitOid, DataSourceResult>,
): DataComparisonView {
  const side = (name: "left" | "right", id: DataVariableView["indicator_id"]) =>
    report[name].map((source): DataSeriesView | null => {
      const result = coordinate(sources.get(source.revision));
      const data =
        result.action === "prepare_data"
          ? result.body.data
          : coordinate(result.body.data[source.replicate_index]);
      const recorded = data[id];
      if (recorded === undefined) return null;
      const variables =
        result.action === "prepare_data"
          ? result.body.metadata.variables
          : result.body.report.evidence.observation_layout.variables;
      const history = historyView(recorded);
      const timeOrigin =
        result.action === "prepare_data"
          ? result.body.metadata.time_origin
          : result.body.report.evidence.time_origin;
      return {
        variable: coordinate(variables.find((variable) => variable.id === id)),
        time_origin: timeOrigin,
        history,
      };
    });
  return {
    ...report,
    variables: report.variables.map((variable) => ({
      ...variable,
      left: side("left", variable.indicator_id),
      right: side("right", variable.indicator_id),
    })),
  };
}

function coordinate<T>(value: T | undefined): T {
  if (value === undefined) throw new Error("Saved comparison source has a missing coordinate");
  return value;
}
