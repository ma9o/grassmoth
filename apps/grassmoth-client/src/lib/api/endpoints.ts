import type { ActionId, ActionPoll, CallId } from "@grassmoth/api-types";
import { apiClient } from "./client";

/** GET reads the recorded call; the viewer never submits scientific work. */
export async function readActionPoll(
  workspaceId: string,
  action: ActionId,
  callId: CallId,
  signal?: AbortSignal,
): Promise<ActionPoll> {
  const response = await apiClient.GET("/api/studies/{workspace_id}/{action}/{call_id}", {
    params: {
      path: {
        workspace_id: workspaceId,
        action,
        call_id: callId,
      },
    },
    ...(signal === undefined ? {} : { signal }),
  });
  if (response.data === undefined)
    throw new Error(
      `Cannot read ${action} (${response.response.status}): ${JSON.stringify(response.error)}`,
    );
  return response.data;
}
