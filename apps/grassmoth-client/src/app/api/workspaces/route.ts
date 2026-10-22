import { NextResponse } from "next/server";
import { listWorkspaces } from "@/lib/server/workspaces";

export const dynamic = "force-dynamic";

export async function GET() {
  const headers = new Headers();
  const data = await listWorkspaces(headers);
  return NextResponse.json(data, { headers });
}
