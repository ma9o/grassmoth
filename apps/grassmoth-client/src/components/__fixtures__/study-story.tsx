import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { setupWorker } from "msw/browser";
import { useState, type ReactNode } from "react";
import { TooltipProvider } from "@/components/ui/tooltip";
import { scenarioHandlers } from "./study-scenario";

const worker = setupWorker();

export async function startScenarioWorker() {
  worker.resetHandlers(...scenarioHandlers());
  await worker.start({
    quiet: true,
    onUnhandledRequest: (request, print) => {
      if (new URL(request.url).pathname.startsWith("/api/")) print.error();
    },
  });
  return () => worker.stop();
}

export function ScenarioProviders({ children }: { children: ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: { queries: { retry: false, refetchOnWindowFocus: false } },
      }),
  );
  return (
    <QueryClientProvider client={client}>
      <TooltipProvider>{children}</TooltipProvider>
    </QueryClientProvider>
  );
}
