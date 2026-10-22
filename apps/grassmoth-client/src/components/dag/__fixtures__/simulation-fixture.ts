import { decodeFixture, fixtureValue } from "@/components/__fixtures__/fixture-value";
import { causalEffect } from "@/lib/simulation-report";
import rawReports from "./simulation-reports.json";

const reports = decodeFixture(rawReports);
const selected = fixtureValue(
  reports.filter((report) => causalEffect(report) !== undefined).at(-1),
);
const arms = selected.evidence.arms;
if (arms.kind !== "paired" || "kind" in arms.causal)
  throw new Error("The illustrative fixture needs paired causal effects");
export const demoSimulationResult = {
  ...selected,
  evidence: { ...selected.evidence, arms: { ...arms, causal: arms.causal } },
};
