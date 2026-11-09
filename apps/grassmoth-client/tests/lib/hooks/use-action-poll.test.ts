import { QueryClient } from "@tanstack/react-query";
import { expect, it, vi } from "vitest";
import { readActionPoll } from "@/lib/api/endpoints";
import { actionPollQuery } from "@/lib/hooks/use-action-poll";
import { modelComparison, runningPoll } from "@/lib/__fixtures__/action-results";
import { executionMessages } from "@/lib/__fixtures__/execution-log";

vi.mock("@/lib/api/endpoints", () => ({ readActionPoll: vi.fn() }));

it("shares one response across readers and retains messages when polling completes", async () => {
  const client = new QueryClient();
  const live = {
    ...runningPoll,
    action: modelComparison.action,
    call_id: modelComparison.call_id,
    messages: executionMessages,
  };
  const completed = { ...modelComparison, messages: executionMessages };
  vi.mocked(readActionPoll).mockResolvedValueOnce(live).mockResolvedValueOnce(completed);
  const options = actionPollQuery("STUDY", live.action, live.call_id);
  const responses = await Promise.all([client.fetchQuery(options), client.fetchQuery(options)]);
  expect(responses).toEqual([live, live]);
  expect(readActionPoll).toHaveBeenCalledTimes(1);
  expect(await client.fetchQuery(options)).toEqual(completed);
  expect(await client.fetchQuery(options)).toEqual(completed);
  expect(readActionPoll).toHaveBeenCalledTimes(2);
  expect(client.getQueryCache().getAll()).toHaveLength(1);
  client.clear();
});
