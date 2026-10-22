import type { SimulateOutput } from "@grassmoth/api-types";
import { dump } from "npyjs";
import { expect, it } from "vitest";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { scenarioResult } from "@/components/__fixtures__/study-scenario";
import { outcome } from "@/lib/__fixtures__/model";
import { fittedSnapshot } from "@/lib/__fixtures__/snapshot";
import { pathsView } from "@/lib/model-asset/result-values";

const simulation = scenarioResult("simulated").body;

function array(shape: number[], values: number[], dtype: "f8" | "b1" = "f8") {
  return { npy: new Uint8Array(dump(values, shape, { dtype })) };
}

it.each([
  "single",
  "paired",
] as const)("projects %s evidence with saved summaries and observation masks", (kind) => {
  const observations = array([2, 4, 1], [91, 93, 95, 97, 1, 5, 7, Number.POSITIVE_INFINITY]);
  const mask = array([2, 4, 1], [1, 1, 1, 1, 1, 1, 0, 1], "b1");
  const latents = array([2, 4, 1], [-5, -4, -5, -5, 5, 4, 5, 5]);
  const reference = array([2, 4, 1], [-6, -5, -6, -6, 2, 1, 2, 2]);
  const differences = array([2, 4], [1, 1, 1, 1, 3, 3, 3, 3]);
  const identity = "indicator:recorded";
  const action = { latent_paths: latents, observations };
  const saved: SimulateOutput = {
    data: [{}],
    report: {
      ...simulation.report,
      evidence: {
        ...simulation.report.evidence,
        draws: 2,
        times: [0, 0.25, 8, 10],
        state_ids: [outcome.id],
        arms:
          kind === "single"
            ? { kind, action }
            : {
                kind,
                action,
                reference: { latent_paths: reference, observations },
                causal: {
                  outcome: outcome.id,
                  labels: { [outcome.id]: outcome.name },
                  differences,
                  frame: [1, 3],
                  summary: { mean: 2, median: 2, lower_95: 1, upper_95: 3, prob_positive: 1 },
                  reference_mean: 0,
                  manifest_effects: {},
                  warnings: [],
                },
              },
        observation_layout: {
          ...simulation.report.evidence.observation_layout,
          mask,
          variables: [
            {
              ...fixtureValue(simulation.report.evidence.observation_layout.variables[0]),
              id: identity,
              name: "Recorded observations",
            },
          ],
        },
      },
      summary: {
        state_frames: { [outcome.id]: [-8, 5] },
        indicator_frames: { [identity]: [5, 9] },
        action_category_probabilities: {},
        reference_category_probabilities: {},
      },
    },
  };
  const model = fixtureValue(fittedSnapshot.dynamical_model_spec);
  const displayed = pathsView(saved, model);
  expect(pathsView(saved, model)).toBe(displayed);
  expect(displayed.times).toBe(saved.report.evidence.times);
  const series = fixtureValue(displayed.indicators[identity]);
  expect(series.action[1]).toEqual({ draw: 1, values: [1, 5, null, null] });
  expect(series.frame).toEqual([5, 9]);
  expect(displayed.states[outcome.id]?.action[1]?.values).toEqual([5, 4, 5, 5]);
  expect(series.reference).toHaveLength(kind === "paired" ? 2 : 0);
  if (kind === "paired") {
    expect(displayed.states[outcome.id]?.reference[1]?.values).toEqual([2, 1, 2, 2]);
    expect(displayed.effect?.action[1]?.values).toEqual([3, 3, 3, 3]);
    expect(displayed.effect?.frame).toEqual([1, 3]);
  } else expect(displayed.effect).toBeNull();
  expect(displayed.action_category_probabilities).toBe(
    saved.report.summary.action_category_probabilities,
  );
  const renamed = pathsView(saved, {
    ...model,
    constructs: {
      ...model.constructs,
      [outcome.id]: { ...fixtureValue(model.constructs[outcome.id]), name: "Renamed outcome" },
    },
  });
  expect(renamed.states[outcome.id]?.label).toBe("Renamed outcome");
  expect(displayed.states[outcome.id]?.label).toBe(outcome.name);
});
