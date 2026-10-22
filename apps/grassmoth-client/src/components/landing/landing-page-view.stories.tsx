import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { LandingPageView } from "./landing-page-view";

const meta = {
  title: "Landing/LandingPageView",
  component: LandingPageView,
  args: {
    data: {
      "local-adhd-pilot": "Local ADHD pilot workspace with merged EMA and wearable measurements.",
      "local-sleep-study":
        "Local sleep study workspace with irregular actigraphy and survey exports.",
      "local-medication-trial": null,
    },
    error: null,
    isLoading: false,
  },
} satisfies Meta<typeof LandingPageView>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {};

export const EmptyWorkspaces: Story = {
  args: { data: {} },
};

export const WorkspacesLoading: Story = {
  args: { data: undefined, isLoading: true },
};

export const WorkspacesError: Story = {
  args: { data: undefined, error: "Failed to load workspaces." },
};
