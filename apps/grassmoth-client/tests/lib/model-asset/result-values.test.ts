import type { ArrayVector, FitOutput } from "@grassmoth/api-types";
import { dump } from "npyjs";
import { expect, it } from "vitest";
import { fitResult } from "@/components/__fixtures__/action-results";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { fittedSnapshot } from "@/lib/__fixtures__/snapshot";
import { drawsView, inferenceView, scalarValues } from "@/lib/model-asset/result-values";

function array(shape: number[], values: number[], dtype: "f8" | "b1" = "f8") {
  return { npy: new Uint8Array(dump(values, shape, { dtype })) };
}

it("selects original joint posterior atoms and native chain evidence without reducing them", () => {
  const base = fitResult(fittedSnapshot);
  const identity = fixtureValue(base.inference.core.inference_metadata.distribution);
  const model = fixtureValue(base.dynamical_model_spec);
  const atoms = array([4, 4], [-5, 1, 90, 91, 5, -1, 92, 93, -5, 0, 94, 95, 5, 1, 96, 97]);
  const divergences = array([2, 2], [0, 1, 0, 0], "b1");
  const delta = array([2, 2], [1, 2, 3, 4]);
  const marginals = base.inference.core.posterior_marginals.map((marginal, column) => ({
    ...marginal,
    empirical:
      column === 0
        ? [
            { value: -5, probability: 0.5 },
            { value: 5, probability: 1 },
          ]
        : [
            { value: -1, probability: 0.25 },
            { value: 0, probability: 0.5 },
            { value: 1, probability: 1 },
          ],
  }));
  const saved: FitOutput = {
    ...base,
    dynamical_model_spec: {
      ...model,
      parameters: Object.fromEntries(
        marginals.map(({ subject }) => [
          subject.parameter_id,
          {
            ...fixtureValue(model.parameters[subject.parameter_id]),
            distribution: identity,
            transform: { kind: "identity" },
          },
        ]),
      ),
      distributions: {
        [identity]: {
          distribution: "MixtureSameFamily",
          params: {
            mixing_distribution: {
              distribution: "CategoricalProbs",
              params: { probs: [0.25, 0.25, 0.25, 0.25] },
            },
            component_distribution: {
              distribution: "Delta",
              params: {
                v: atoms,
                event_dim: 1,
              },
            },
          },
        },
      },
      law_layouts: {
        [identity]: {
          parameters: marginals.map(({ subject }) => [subject.parameter_id, [subject.element_id]]),
          constructs: ["construct:00000000000000000003"],
          time_points: [0, 1],
          time_origin: "relative",
          labels: Object.fromEntries(
            marginals.map(({ subject, parameter }) => [subject.element_id, parameter]),
          ),
        },
      },
    },
    inference: {
      ...base.inference,
      evidence: {
        ...base.inference.evidence,
        chain_extra_fields: { diverging: divergences },
        initial_latent_delta: delta,
      },
      core: {
        ...base.inference.core,
        inference_metadata: {
          ...base.inference.core.inference_metadata,
          num_samples_total: 4,
          num_chains: 2,
        },
        posterior_marginals: marginals,
      },
      detail: {
        ...base.inference.detail,
        trace_data: [
          {
            subject: fixtureValue(marginals[0]).subject,
            chains: [0, 2].map((start) => ({
              array: atoms,
              indices: [null, 0],
              start,
              stop: start + 2,
              mask: null,
            })),
          },
        ],
      },
    },
  };
  const columns = drawsView(saved);
  expect(columns.map(({ values }) => values)).toEqual([
    [-5, 5, -5, 5],
    [1, -1, 0, 1],
  ]);
  expect(columns[0]?.empirical).toBe(marginals[0]?.empirical);
  const report = inferenceView(saved);
  expect(report.detail.trace_data[0]?.chains).toEqual([
    [-5, 5],
    [-5, 5],
  ]);
  expect(report.detail.divergent).toEqual([false, true, false, false]);
  expect(report.detail.initial_latent_delta).toEqual([
    [1, 2],
    [3, 4],
  ]);
  expect(saved.inference.detail.trace_data[0]?.chains[0]).toHaveProperty("array", atoms);
});

it("selects a saved replicate without exposing masked or non-finite observations", () => {
  const observations = array(
    [2, 4, 2],
    [90, 91, 92, 93, 94, 95, 96, 97, 0, 1, 2, 5, 3, 7, 4, Number.POSITIVE_INFINITY],
  );
  const mask = array([2, 4, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], "b1");
  const selection: ArrayVector = {
    array: observations,
    indices: [1, null, 1],
    start: 1,
    stop: 4,
    mask: { array: mask, indices: [1, null, 1], start: 1, stop: 4, mask: null },
  };
  const displayed = scalarValues(selection);
  expect(displayed).toEqual([5, null, null]);
  expect(selection).toHaveProperty("array", observations);
});
