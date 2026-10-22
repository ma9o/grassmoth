import { apiClient } from "@/lib/api/client";
import type { WorkspaceList } from "@/lib/server/workspaces";

export function getWorkspacesQueryKey() {
  return ["workspaces"] as const;
}

export async function getWorkspaces(): Promise<WorkspaceList> {
  const { data, response } = await apiClient.GET("/api/workspaces", {
    cache: "no-store",
  });
  if (data === undefined) throw new Error(`Workspace API error ${response.status}`);
  return data;
}
