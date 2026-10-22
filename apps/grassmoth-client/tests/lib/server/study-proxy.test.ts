import { afterEach, expect, it, vi } from "vitest";
import { proxyStudyRequest } from "@/lib/server/study-proxy";

vi.mock("@/lib/runtime-urls", () => ({ getToolServerUrl: () => "http://tool-server" }));
afterEach(() => vi.unstubAllGlobals());

it("preserves named inputs, payload and backend rejections for calls", async () => {
  const fetcher = vi
    .fn()
    .mockResolvedValue(Response.json({ detail: "Input unavailable" }, { status: 422 }));
  vi.stubGlobal("fetch", fetcher);
  const body = JSON.stringify({
    action: "edit_model",
    input: { parent_ref: "a".repeat(40), dynamical_model_spec: {} },
  });
  const path = "/api/studies/STUDY/edit_model";
  const response = await proxyStudyRequest(
    new Request(`http://web${path}`, {
      method: "POST",
      body,
    }),
  );
  expect(fetcher).toHaveBeenCalledWith(
    `http://tool-server${path}`,
    expect.objectContaining({
      method: "POST",
      body,
    }),
  );
  expect(response.status).toBe(422);
  expect(await response.json()).toEqual({ detail: "Input unavailable" });
});

it("streams MessagePack bytes unchanged and preserves their content type", async () => {
  const payload = new Uint8Array([0x81, 0xa3, 0x6e, 0x70, 0x79, 0xc4, 0x02, 0x00, 0xff]);
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      new Response(payload, {
        headers: { "Content-Type": "application/msgpack" },
      }),
    ),
  );
  const response = await proxyStudyRequest(
    new Request("http://web/api/studies/STUDY/fit/call:test"),
  );
  expect(response.headers.get("Content-Type")).toBe("application/msgpack");
  expect(new Uint8Array(await response.arrayBuffer())).toEqual(payload);
});
