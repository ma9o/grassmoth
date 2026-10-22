import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { modelConstructs } from "@/lib/model-accessors";
import { demoModelSnapshot } from "@/components/__fixtures__/demo-artifacts";
const indicators = modelConstructs(fixtureValue(demoModelSnapshot.dynamical_model_spec)).flatMap(
  (construct) => construct.indicators.map((indicator) => indicator.observation),
);
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { predictiveChecks } from "@/components/__fixtures__/inference-data";
import { PPCWarningsTable } from "./ppc-warnings-table";

const ppc = predictiveChecks;

const meta = {
  args: { indicators },
  title: "V1/Pipeline/Outputs/Posterior/PPCWarningsTable",
  component: PPCWarningsTable,
  decorators: [withContainer()],
} satisfies Meta<typeof PPCWarningsTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    warnings: ppc.findings,
    testStats: ppc.test_stats,
    overlays: ppc.overlays,
  },
};

export const WarningsOnly: Story = {
  args: {
    warnings: ppc.findings,
    testStats: [],
    overlays: [],
  },
};
