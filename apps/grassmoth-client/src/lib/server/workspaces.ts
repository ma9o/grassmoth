import { createModelClient } from "@grassmoth/api-types";
import { getToolServerUrl } from "@/lib/runtime-urls";
export type WorkspaceList = Readonly<Record<string, string | null>>;
const client = createModelClient({ baseUrl: getToolServerUrl() });
export async function listWorkspaces(headers?: Headers): Promise<WorkspaceList> {
  const { data, error, response } = await client.GET("/api/workspaces", { cache: "no-store" });
  if (data === undefined)
    throw new Error(`Workspace facade error ${response.status}: ${JSON.stringify(error)}`);
  const enabled = response.headers.get("X-Actions-Enabled");
  if (enabled !== "true" && enabled !== "false")
    throw new Error("Workspace response has no action capability");
  headers?.set("X-Actions-Enabled", enabled);
  return data;
}
