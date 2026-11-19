import { apiClient } from "@/lib/api/client";

export function getWorkspacesQueryKey() {
  return ["workspaces"] as const;
}

export async function getWorkspaces() {
  const { data, response } = await apiClient.GET("/api/workspaces", {
    cache: "no-store",
  });
  if (data === undefined) throw new Error(`Workspace API error ${response.status}`);
  return data;
}
