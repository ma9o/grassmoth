import { getToolServerUrl } from "@/lib/runtime-urls";

/** Preserve content-named action arguments and complete results end to end. */
export async function proxyStudyRequest(request: Request) {
  const url = new URL(request.url);
  const response = await fetch(`${getToolServerUrl()}${url.pathname}${url.search}`, {
    method: request.method,
    headers: {
      Accept: "application/msgpack, application/json",
      "Content-Type": "application/json",
    },
    ...(request.method === "GET" ? {} : { body: await request.text() }),
    cache: "no-store",
    signal: request.signal,
  });
  const headers = new Headers({ "Cache-Control": "no-store" });
  const contentType = response.headers.get("Content-Type");
  if (contentType !== null) headers.set("Content-Type", contentType);
  return new Response(response.body, { status: response.status, headers });
}
