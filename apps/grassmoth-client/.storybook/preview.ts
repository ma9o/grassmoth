import type { Preview } from "@storybook/nextjs-vite";
import "katex/dist/katex.min.css";
import "../src/app/globals.css";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: false, refetchOnWindowFocus: false } },
});

const preview: Preview = {
  decorators: [
    (Story) =>
      React.createElement(QueryClientProvider, { client: queryClient }, React.createElement(Story)),
  ],
  parameters: {
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },
    options: {
      // Keep the old pipeline and new workbench in separate namespaces.
      // Everything unlisted falls back to alphabetical.
      storySort: {
        method: "alphabetical",
        order: ["V1", ["Pipeline", "*"], "V2", ["Model", "*"], "Charts", "UI", "Landing", "*"],
      },
    },
  },
};

export default preview;
