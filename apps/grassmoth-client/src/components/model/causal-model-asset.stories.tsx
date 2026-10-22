import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { expect, userEvent, within } from "storybook/test";
import {
  scenarioQuestion,
  scenarioTimeline,
  scenarioWorkspace,
} from "@/components/__fixtures__/generated/study";
import { scenarioRevision, type ScenarioName } from "@/components/__fixtures__/study-scenario";
import { ScenarioProviders, startScenarioWorker } from "@/components/__fixtures__/study-story";
import { useStudyJournal } from "@/lib/hooks/use-study-journal";
import { timelineTickLabel } from "@/lib/model-asset/timeline-presentation";
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
        <CausalModelAsset
          workspaceId={scenarioWorkspace}
          question={scenarioQuestion}
          journal={journal.data}
        />
      </div>
    </>
  );
}

const meta = {
  title: "V2/Model/Workbench",
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

// Keep one comprehensive workbench story and exercise the production reader throughout.
export const Complete: Story = {
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    const select = async (name: ScenarioName) => {
      const label = timelineTickLabel(scenarioRevision(name));
      await userEvent.click(await canvas.findByRole("button", { name: new RegExp(label) }));
      const record = within(await canvas.findByRole("complementary", { name: "Action record" }));
      await expect(await record.findByText(label)).toBeVisible();
      return record;
    };
    await canvas.findByRole("button", {
      name: new RegExp(timelineTickLabel(scenarioRevision("dataCompared"))),
    });
    await expect(canvasElement.querySelectorAll("path[data-argument]")).toHaveLength(
      scenarioTimeline.dependencies.length,
    );
    await select("editFailed");
    const failure = await canvas.findByRole("region", { name: "Edit model failed" });
    await expect(failure).toBeVisible();
    await expect(within(failure).getByText(/SyntheticProposalError:/)).toBeVisible();
    await select("fitRejected");
    await expect(await canvas.findByRole("region", { name: "Fit failed" })).toBeVisible();
    for (const name of [
      "question",
      "structure",
      "measurement",
      "authored",
      "prepared",
      "fitted",
      "simulated",
      "edited",
      "effectUnavailable",
      "modelCompared",
      "dataCompared",
    ] as const) {
      const record = await select(name);
      await expect(await canvas.findByRole("region", { name: "Model details" })).toBeVisible();
      if (name === "authored") {
        await userEvent.click(record.getByText("Agent conversation"));
        await expect(
          await record.findByText("Synthetic authoring conversation for the story trace panel."),
        ).toBeVisible();
      }
      if (name === "prepared") {
        await expect(await canvas.findByRole("region", { name: "Prepared data" })).toBeVisible();
        await expect(await canvas.findByText("input/synthetic.csv")).toBeVisible();
      }
      if (name === "simulated") {
        await expect(await canvas.findByRole("log", { name: "Simulator log" })).toBeVisible();
        await expect(
          (await canvas.findAllByRole("img", { name: /recorded paired effects/ }))[0],
        ).toBeInTheDocument();
      }
      if (name === "effectUnavailable") {
        await expect(
          await canvas.findByText("Synthetic example of an unavailable causal effect."),
        ).toBeVisible();
      }
      if (name === "modelCompared") {
        await expect(await record.findByRole("region", { name: "Model comparison" })).toBeVisible();
      }
      if (name === "dataCompared") {
        await expect(await record.findByRole("region", { name: "Data comparison" })).toBeVisible();
        await expect(
          await canvas.findByText("Synthetic calibration warning for interface coverage."),
        ).toBeVisible();
      }
    }
    await userEvent.click(await canvas.findByRole("button", { name: "Return to latest" }));
    await expect(await canvas.findByRole("log", { name: "fit · running" })).toBeVisible();
  },
};
