import type {
  DataComparisonReport,
  ObservationData,
  ObservationHistory,
  PreparedDataMetadata,
} from "@grassmoth/api-types";
import { dump } from "npyjs";
import { expect, it } from "vitest";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { scenarioResult } from "@/components/__fixtures__/study-scenario";
import { dataComparisonChart } from "@/components/charts/series-adapters";
import { outcome } from "@/lib/__fixtures__/model";
import { dataComparisonView, type DataSourceResult } from "@/lib/model-asset/data-comparison";

const simulationReport = scenarioResult("simulated").body.report;

const observation = {
  ...fixtureValue(outcome.indicators[0]).observation,
  observation_window: "1d",
};
const panel = "a".repeat(40);
const simulation = "b".repeat(40);
const emissions = { npy: new Uint8Array(dump([0, 1, 2, 2, 3, 4], [2, 3, 1], { dtype: "f8" })) };
const mask = { npy: new Uint8Array(dump([1, 1, 1, 1, 0, 0], [2, 3, 1], { dtype: "b1" })) };
const origin = "2026-01-01T00:00:00Z";
const metadata: PreparedDataMetadata = {
  preparation: {
    default_window: "1d",
    variables: [
      {
        observation,
        extraction: { kind: "semantic", how_to_measure: "Read Y", source_columns: [] },
      },
    ],
    context: "",
  },
  time_origin: origin,
  variables: [observation],
};
const history: ObservationHistory = {
  times: [5, 6, 7],
  values: [1, 4, null],
  support_start: [4, 5, 6],
  support_end: [5, 6, 7],
  empirical: [],
};
const envelope = {
  status: "success" as const,
  call_id: `call:${"a".repeat(64)}` as const,
  messages: [],
};
const replicate = (index: number): ObservationData => ({
  [observation.id]: {
    ...history,
    times: [3, 4, 5],
    support_start: [2, 3, 4],
    support_end: [3, 4, 5],
    values: {
      array: emissions,
      indices: [index, null, 0],
      start: 0,
      stop: null,
      mask: {
        array: mask,
        indices: [index, null, 0],
        start: 0,
        stop: null,
        mask: null,
      },
    },
  },
});
const sources = new Map<string, DataSourceResult>([
  [
    panel,
    {
      ...envelope,
      commit_id: panel,
      action: "prepare_data",
      body: {
        data: { [observation.id]: history },
        metadata,
        extraction: { workers: [], extraction_reused: null },
        profile: { indicators: {}, findings: [] },
      },
    },
  ],
  [
    simulation,
    {
      ...envelope,
      commit_id: simulation,
      action: "simulate",
      body: {
        report: {
          ...simulationReport,
          evidence: {
            ...simulationReport.evidence,
            time_origin: "2026-01-03T00:00:00Z",
            observation_layout: {
              ...simulationReport.evidence.observation_layout,
              variables: [observation],
            },
          },
        },
        data: [replicate(0), replicate(1)],
      },
    },
  ],
]);
const report: DataComparisonReport = {
  left: [{ revision: panel, replicate_index: 0 }],
  right: [1, 0].map((replicate_index) => ({ revision: simulation, replicate_index })),
  variables: [
    {
      indicator_id: observation.id,
      changes: [],
      findings: [],
      statistics: [{ statistic: "mean", left: [2.5], right: [3, 1] }],
      predictive: {
        reference_side: "left",
        evaluation: {
          kind: "not_evaluated",
          code: "data_comparison",
          subject: { kind: "indicator", id: observation.id },
          reason: "MISSING_REPLICATE_VALUES",
          detail: "Replicas contain missing values at observed anchors",
        },
      },
    },
  ],
};

it("joins selected source histories in report order and retains computed evidence unchanged", () => {
  const view = dataComparisonView(report, sources);
  const variable = fixtureValue(view.variables[0]);
  expect(view.left).toBe(report.left);
  expect(view.right).toBe(report.right);
  expect(variable.statistics).toBe(report.variables[0]?.statistics);
  const retained = fixtureValue(report.variables[0]);
  if (variable.predictive === null || retained.predictive === null)
    throw new Error("Expected the recorded predictive comparison");
  expect(variable.predictive).toBe(retained.predictive);
  expect(variable.left[0]?.history.values).toEqual([1, 4, null]);
  expect(variable.right.map((series) => series?.history.values)).toEqual([
    [2, null, null],
    [0, 1, 2],
  ]);
  expect(variable.right[0]?.time_origin).toBe("2026-01-03T00:00:00Z");
  expect(variable.right[0]?.history).toEqual({
    times: [3, 4, 5],
    support_start: [2, 3, 4],
    support_end: [3, 4, 5],
    values: [2, null, null],
    empirical: [],
  });
  const chart = dataComparisonChart(variable);
  expect(chart.layers.find((layer) => layer.key === "right")?.points).toBe(true);
  expect(chart.observed?.values).toEqual([1, 4, null]);
  const missing = dataComparisonView(
    {
      ...report,
      variables: [{ ...fixtureValue(report.variables[0]), indicator_id: "indicator:absent" }],
    },
    sources,
  );
  expect(missing.variables[0]?.right).toEqual([null, null]);
});

it("plots exact point changes against canonical history coordinates without changing their values", () => {
  const point = { anchor_time: "2026-01-07T00:00:00Z", support_start: null, support_end: null };
  const selected: DataComparisonReport = {
    ...report,
    right: [{ revision: simulation, replicate_index: 0 }],
    variables: [
      {
        indicator_id: observation.id,
        statistics: [],
        findings: [],
        changes: [
          { kind: "revised", before: { ...point, value: 4 }, after: { ...point, value: 1 } },
        ],
        predictive: null,
      },
    ],
  };
  const view = dataComparisonView(selected, sources);
  const chart = dataComparisonChart(fixtureValue(view.variables[0]));
  expect(chart.times).toEqual([0, 1, 2]);
  expect(chart.timeOrigin).toBe("2026-01-06T00:00:00.000Z");
  expect(chart.layers.find((layer) => layer.key === "right")?.rows[0]?.values).toEqual([0, 1, 2]);
  expect(chart.layers.find((layer) => layer.key === "revised-left")?.rows[0]?.values).toEqual([
    null,
    4,
    null,
  ]);
  expect(chart.layers.find((layer) => layer.key === "revised-right")?.rows[0]?.values).toEqual([
    null,
    1,
    null,
  ]);
});
