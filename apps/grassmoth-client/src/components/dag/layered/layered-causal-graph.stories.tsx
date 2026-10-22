import { demoModelSnapshot, demoSnapshotAt } from "@/components/__fixtures__/demo-artifacts";
import { demoSimulationResult } from "../__fixtures__/simulation-fixture";
import type { EntitySelection } from "@/lib/model-asset/selection";
import { TooltipProvider } from "@/components/ui/tooltip";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { indexModel } from "@/lib/model-asset/entities";
import { useMemo, useState } from "react";
import { LayeredCausalGraph, type LayeredCausalGraphProps } from "./layered-causal-graph";

/** Stories own the selection the way the asset view does. */
function SelectableLayeredCausalGraph(
  props: Omit<LayeredCausalGraphProps, "selection" | "onSelect" | "entities">,
) {
  const [selection, setSelection] = useState<EntitySelection | null>(null);
  const entities = useMemo(
    () => indexModel(props.modelSnapshot.dynamical_model_spec),
    [props.modelSnapshot],
  );
  return (
    <LayeredCausalGraph
      entities={entities}
      {...props}
      selection={selection}
      onSelect={setSelection}
    />
  );
}

const structureModel = demoSnapshotAt(3);
const measurementModel = demoSnapshotAt(4);
const designModel = demoSnapshotAt(4);
const specificationModel = demoSnapshotAt(7);
const fitModel = demoSnapshotAt(8);
const simulationResult = demoSimulationResult;

const meta = {
  title: "DAG/Layered Causal Graph",
  component: SelectableLayeredCausalGraph,
  tags: ["svg-materialize"],
  parameters: {
    layout: "fullscreen",
    docs: {
      description: {
        component:
          "Six cumulative artifact layers over the complete authored DAG. Saved identification findings decorate constructs without changing their structural membership. Every story uses the canonical HEALTHDEMO fixture.",
      },
    },
    svgMaterializer: {
      selector: 'svg[aria-label="Layered causal graph"]',
      auto: true,
    },
  },
  decorators: [
    (Story) => (
      <TooltipProvider>
        <Story />
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof SelectableLayeredCausalGraph>;

export default meta;

type Story = StoryObj<typeof meta>;

export const Structure: Story = {
  name: "1 · Structure",
  args: { modelSnapshot: structureModel },
};

export const Measurement: Story = {
  name: "2 · + Measurement",
  args: { modelSnapshot: measurementModel },
};

export const Design: Story = {
  name: "3 · + Design",
  args: { modelSnapshot: designModel },
};

export const Specification: Story = {
  name: "4 · + Specification",
  args: { modelSnapshot: specificationModel },
};

export const Fit: Story = {
  name: "5 · + Fit",
  args: { modelSnapshot: fitModel },
};

export const Simulation: Story = {
  name: "6 · + Simulation",
  args: { modelSnapshot: demoModelSnapshot, simulation: simulationResult },
};
