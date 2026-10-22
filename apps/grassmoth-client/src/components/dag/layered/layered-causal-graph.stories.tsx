import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useMemo, useState } from "react";
import { scenarioWorkspace } from "@/components/__fixtures__/generated/study";
import { scenarioRevision, type ScenarioName } from "@/components/__fixtures__/study-scenario";
import { ScenarioProviders, startScenarioWorker } from "@/components/__fixtures__/study-story";
import { useActionResult, useModelSnapshot } from "@/lib/hooks/use-model-snapshot";
import { indexModel } from "@/lib/model-asset/entities";
import { pathsView } from "@/lib/model-asset/result-values";
import type { EntitySelection } from "@/lib/model-asset/selection";
import { LayeredCausalGraph } from "./layered-causal-graph";

/** Story selections use the production HTTP reader and saved-result composition. */
function ScenarioGraph({ call, simulate = false }: { call: ScenarioName; simulate?: boolean }) {
  const [selection, setSelection] = useState<EntitySelection | null>(null);
  const snapshot = useModelSnapshot(scenarioWorkspace, scenarioRevision(call).commit_id);
  const result = useActionResult(
    scenarioWorkspace,
    scenarioRevision("simulated").commit_id,
    simulate,
  );
  const entities = useMemo(() => indexModel(snapshot.data?.dynamical_model_spec), [snapshot.data]);
  if (snapshot.error) return <p role="alert">{snapshot.error.message}</p>;
  if (!snapshot.data) return <p role="status">Loading synthetic graph…</p>;
  const simulated = result.data?.action === "simulate" ? result.data.body : undefined;
  return (
    <LayeredCausalGraph
      modelSnapshot={snapshot.data}
      entities={entities}
      selection={selection}
      onSelect={setSelection}
      simulation={simulated?.report ?? null}
      simulationPaths={
        simulated && snapshot.data.dynamical_model_spec
          ? pathsView(simulated, snapshot.data.dynamical_model_spec)
          : null
      }
    />
  );
}

const meta = {
  title: "DAG/Layered Causal Graph",
  component: ScenarioGraph,
  tags: ["svg-materialize"],
  parameters: {
    layout: "fullscreen",
    docs: {
      description: {
        component:
          "Progressive layers from server-validated synthetic API examples, loaded through the production reader. No study or scientific execution is needed.",
      },
    },
    svgMaterializer: { selector: 'svg[aria-label="Layered causal graph"]', auto: true },
  },
  decorators: [
    (Story) => (
      <ScenarioProviders>
        <Story />
      </ScenarioProviders>
    ),
  ],
  beforeEach: startScenarioWorker,
} satisfies Meta<typeof ScenarioGraph>;
export default meta;
type Story = StoryObj<typeof meta>;

export const Structure: Story = { name: "1 · Structure", args: { call: "structure" } };
export const Measurement: Story = { name: "2 · + Measurement", args: { call: "measurement" } };
export const Design: Story = { name: "3 · + Design", args: { call: "prepared" } };
export const Specification: Story = { name: "4 · + Specification", args: { call: "authored" } };
export const Fit: Story = { name: "5 · + Fit", args: { call: "fitted" } };
export const Simulation: Story = {
  name: "6 · + Simulation",
  args: { call: "simulated", simulate: true },
};
