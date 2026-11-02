import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { expect, userEvent, waitFor, within } from "storybook/test";
import { scenarioTimeline, scenarioWorkspace } from "@/components/__fixtures__/generated/study";
import { ScenarioProviders, startScenarioWorker } from "@/components/__fixtures__/study-story";
import { useStudyJournal } from "@/lib/hooks/use-study-journal";
import { CausalModelAsset } from "./causal-model-asset";

function WorkbenchStory() {
  const journal = useStudyJournal(scenarioWorkspace);
  if (journal.error) return <p role="alert">{journal.error.message}</p>;
  if (!journal.data) return <p role="status">Loading the workbench…</p>;
  return (
    <>
      <div
        role="note"
        className="flex h-6 items-center justify-end border-b px-5 text-[10px] text-muted-foreground"
      >
        Synthetic interface examples
      </div>
      <div className="md:h-[calc(100dvh-24px)] [&>div]:md:h-full [&>div]:md:min-h-0">
        <CausalModelAsset workspaceId={scenarioWorkspace} journal={journal.data} />
      </div>
    </>
  );
}

const meta = {
  title: "Model/Workbench",
  component: WorkbenchStory,
  parameters: {
    layout: "fullscreen",
    docs: {
      description: {
        component:
          "A synthetic study covering the seven public actions, progressive authoring, numerical paths, comparisons, failures and a running call. Python validates the examples with the server contracts and serializes its normal HTTP payloads. MSW serves those payloads; the production API client, hooks and view composition build the workbench. These are rendering examples, not computed scientific results.",
      },
    },
  },
  decorators: [
    (Story) => (
      <ScenarioProviders>
        <Story />
      </ScenarioProviders>
    ),
  ],
  beforeEach: startScenarioWorker,
} satisfies Meta<typeof WorkbenchStory>;
export default meta;
type Story = StoryObj<typeof meta>;

/** Select a recorded node by its timeline label, as a viewer does. */
async function selectTick(canvasElement: HTMLElement, label: string) {
  const canvas = within(canvasElement);
  await userEvent.click(await canvas.findByRole("button", { name: new RegExp(label) }));
  const record = within(await canvas.findByRole("complementary", { name: "Action record" }));
  await expect(await record.findByText(label)).toBeVisible();
  return record;
}

/** The workbench as it opens: at the latest node, with a fit still running. */
export const Complete: Story = {
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    await expect(await canvas.findByRole("log", { name: "fit · running" })).toBeVisible();
    await expect(canvasElement.querySelectorAll("path[data-argument]")).toHaveLength(
      scenarioTimeline.dependencies.length,
    );
  },
};

const GRAPH = 'svg[aria-label="Layered causal graph"]';

/**
 * The workbench at the recorded node whose timeline label is the story's name, so a story
 * cannot show a different node than it names. The graph pane is the workbench's own, at the
 * size the workbench gives it; the SVG materializer saves the graph once the node has drawn.
 * Each story states its svg-materialize tag literally, so the story index can read it.
 */
const atNode = (
  drawn: (canvasElement: HTMLElement) => Promise<unknown> = async () => {},
): Story => ({
  parameters: { svgMaterializer: { selector: GRAPH, auto: true } },
  play: async ({ canvasElement, name }) => {
    await selectTick(canvasElement, name);
    await waitFor(() =>
      expect(canvasElement.querySelector("main")).toHaveAttribute("aria-busy", "false"),
    );
    await within(canvasElement).findByRole("img", { name: "Layered causal graph" });
    await drawn(canvasElement);
  },
});

export const EditModel2: Story = {
  ...atNode(),
  name: "edit_model · 0000002",
  tags: ["svg-materialize"],
};
export const EditModel3: Story = {
  ...atNode(),
  name: "edit_model · 0000003",
  tags: ["svg-materialize"],
};
export const EditModel4: Story = {
  ...atNode(),
  name: "edit_model · 0000004",
  tags: ["svg-materialize"],
};
export const PrepareData: Story = {
  ...atNode(),
  name: "prepare_data · 0000005",
  tags: ["svg-materialize"],
};
export const Fit: Story = { ...atNode(), name: "fit · 0000008", tags: ["svg-materialize"] };
export const Simulate: Story = {
  ...atNode((canvasElement) =>
    within(canvasElement).findAllByRole("img", { name: /individual simulation draws/ }),
  ),
  name: "simulate · 0000009",
  tags: ["svg-materialize"],
};
export const ModelDiff: Story = {
  ...atNode((canvasElement) =>
    waitFor(() => expect(canvasElement.querySelector("[data-comparison-mark]")).not.toBeNull()),
  ),
  name: "model_diff · 000000c",
  tags: ["svg-materialize"],
};
export const DataDiff: Story = {
  ...atNode((canvasElement) =>
    within(canvasElement).findAllByRole("button", { name: /comparison$/ }),
  ),
  name: "data_diff · 000000d",
  tags: ["svg-materialize"],
};
