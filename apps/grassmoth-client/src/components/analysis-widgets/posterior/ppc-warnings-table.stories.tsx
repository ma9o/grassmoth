import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { scenarioResult } from "@/components/__fixtures__/study-scenario";
import { withContainer } from "@/components/story-decorators";
import { PPCWarningsTable } from "./ppc-warnings-table";

const indicators = [...scenarioResult("prepared").body.metadata.variables];
const comparison = fixtureValue(scenarioResult("dataCompared").body.report.variables[0]);
if (!("predictive" in comparison) || "kind" in comparison.predictive.evaluation)
  throw new Error("The synthetic comparison must include predictive check examples");
const ppc = comparison.predictive.evaluation;

const meta = {
  args: { indicators },
  title: "V2/Model/Findings/PPCWarningsTable",
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
