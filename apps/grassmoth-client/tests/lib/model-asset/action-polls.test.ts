import { createModelClient } from "@grassmoth/api-types";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import { afterAll, beforeAll, expect, it } from "vitest";
import {
  failedPoll,
  modelComparison,
  rejectedPoll,
  runningPoll,
} from "@/lib/__fixtures__/action-results";

const polls = [failedPoll, rejectedPoll, runningPoll, modelComparison];
const server = setupServer(
  http.get("http://unit.test/api/studies/TEST/:action/:callId", ({ params }) => {
    const poll = polls.find(
      (entry) => entry.action === params.action && entry.call_id === params.callId,
    );
    if (!poll) throw new Error("Unknown unit call");
    return HttpResponse.json(poll);
  }),
);
beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterAll(() => server.close());

it.each(polls)("reads $action / $status through the production HTTP client", async (poll) => {
  const client = createModelClient({ baseUrl: "http://unit.test" });
  const response = await client.GET("/api/studies/{workspace_id}/{action}/{call_id}", {
    params: { path: { workspace_id: "TEST", action: poll.action, call_id: poll.call_id } },
  });
  expect(response.data).toEqual(poll);
  if (poll.status === "failed") {
    expect(response.data?.body).toBeNull();
    expect(response.data?.messages.some((message) => message.kind === "failure")).toBe(true);
  } else if (poll.status === "running") {
    expect(response.data?.body).toBeNull();
    expect(response.data?.commit_id).toBeNull();
  }
});
