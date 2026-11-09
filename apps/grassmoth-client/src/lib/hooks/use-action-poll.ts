"use client";

import type {
  ActionId,
  ActionPoll,
  ActionSuccess,
  CallId,
  TimelineRevision,
} from "@grassmoth/api-types";
import { queryOptions, skipToken, useQuery } from "@tanstack/react-query";
import { readActionPoll } from "@/lib/api/endpoints";

/** Results, execution messages and conversations share one response and one cache entry. */
export function actionPollQuery(
  workspaceId: string,
  action: ActionId | undefined,
  callId: CallId | null,
) {
  return queryOptions({
    queryKey: ["action-result", workspaceId, callId],
    queryFn:
      callId === null || action === undefined
        ? skipToken
        : ({ signal }) => readActionPoll(workspaceId, action, callId, signal),
    refetchInterval: (query) => (query.state.data?.status === "running" ? 500 : false),
    // A running snapshot can have been cached by an observer that has since unmounted.
    staleTime: (query) => (query.state.data?.status === "running" ? 0 : Infinity),
    retry: false,
  });
}

export function recordedCallQuery(workspaceId: string, call: TimelineRevision | undefined) {
  return actionPollQuery(workspaceId, call?.record.attempt.action, call?.call_id ?? null);
}

export type UseActionPoll = (
  action: ActionId,
  callId: CallId | null,
) => {
  data: ActionPoll | undefined;
  error: Error | null;
};

export function useActionPoll(workspaceId: string, action: ActionId, callId: CallId | null) {
  return useQuery(actionPollQuery(workspaceId, action, callId));
}

/** Scientific view consumers wait for the same response to finish successfully. */
export function successfulResult(poll: ActionPoll): ActionSuccess | undefined {
  if (poll.status === "failed") throw new Error("The saved successful call is unavailable");
  return poll.status === "success" ? poll : undefined;
}
