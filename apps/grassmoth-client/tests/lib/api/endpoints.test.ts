import { Buffer } from "node:buffer";
import { ExtData, encode } from "@msgpack/msgpack";
import {
  createModelClient,
  decodeModelMessage,
  readNumericalArray,
  type TimelineRevision,
} from "@grassmoth/api-types";
import { dump } from "npyjs";
import { beforeEach, expect, it, vi } from "vitest";
import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { readActionPoll } from "@/lib/api/endpoints";

const { fetcher } = vi.hoisted(() => ({
  fetcher: vi.fn<(request: Request) => Promise<Response>>(),
}));
vi.mock("@/lib/api/client", () => ({
  apiClient: createModelClient({ baseUrl: "http://viewer", fetch: fetcher }),
}));
beforeEach(() => fetcher.mockReset());

function messagePack(value: unknown) {
  return new Response(encode(value), { headers: { "Content-Type": "application/msgpack" } });
}

const comparison: TimelineRevision = {
  call_id: `call:${"3".repeat(64)}`,
  commit_id: "3".repeat(40),
  parent_ids: [],
  record: {
    seq: 1,
    ts: "2026-10-06T12:00:00Z",
    attempt: {
      action: "model_diff",
      request: {
        action: "model_diff",
        input: { before_ref: "1".repeat(40), after_ref: "2".repeat(40) },
        reasoning: "Compare the revised assumptions before fitting.",
      },
      outcome: {
        status: "applied",
        result: null,
        effects: { produced: [], retracted: [], reports: { "model-diff": "4".repeat(40) } },
      },
    },
  },
};

it("reads a recorded comparison by call ID without submitting inputs", async () => {
  const result = {
    call_id: comparison.call_id,
    action: "model_diff",
    status: "success",
    commit_id: comparison.commit_id,
    body: {},
    messages: [],
  };
  fetcher.mockResolvedValue(messagePack(result));
  expect(await readActionPoll("user-1", "model_diff", fixtureValue(comparison.call_id))).toEqual(
    result,
  );
  const [call] = fixtureValue(fetcher.mock.calls.at(0));
  expect(decodeURIComponent(call.url)).toBe(
    `http://viewer/api/studies/user-1/model_diff/${comparison.call_id}`,
  );
  expect(call.method).toBe("GET");
  expect(await call.text()).toBe("");
});

it("reports HTTP failures without retrying execution", async () => {
  fetcher.mockResolvedValue(Response.json({ detail: "Unknown call" }, { status: 404 }));
  await expect(
    readActionPoll("user-1", "model_diff", fixtureValue(comparison.call_id)),
  ).rejects.toThrow("Cannot read model_diff (404)");
  expect(fetcher).toHaveBeenCalledTimes(1);
});

it("reads retained traces from a failed call's messages", async () => {
  const trace = {
    messages: [{ role: "assistant", content: "Inspecting the observations." }],
    model: "test",
    total_time_seconds: 1,
    usage: { input_tokens: 3, output_tokens: 5, reasoning_tokens: null },
  };
  fetcher.mockResolvedValue(
    messagePack({
      call_id: comparison.call_id,
      action: "model_diff",
      status: "failed",
      commit_id: comparison.commit_id,
      body: null,
      messages: [{ kind: "trace", timestamp: comparison.record.ts, trace_id: "worker", trace }],
    }),
  );
  const result = await readActionPoll("user-1", "model_diff", fixtureValue(comparison.call_id));
  expect(result.messages).toEqual([
    { kind: "trace", timestamp: comparison.record.ts, trace_id: "worker", trace },
  ]);
  expect(fixtureValue(fetcher.mock.calls[0])[0].method).toBe("GET");
});

it("decodes binary NPY buffers once with shape, non-finite values and exact 64-bit integers", async () => {
  const floats = {
    npy: new Uint8Array(dump(new Float64Array([1, NaN, Infinity, -Infinity]), [2, 2])),
  };
  const integers = {
    npy: new Uint8Array(dump(new BigUint64Array([BigInt("18446744073709551615")]), [1])),
  };
  const body = {
    changes: {
      distributions: {
        "distribution:floats": { distribution: "Normal", params: { loc: floats, scale: floats } },
        "distribution:integers": { distribution: "Delta", params: { v: integers } },
      },
    },
  };
  fetcher.mockResolvedValue(
    messagePack({
      call_id: comparison.call_id,
      action: "model_diff",
      status: "success",
      commit_id: comparison.commit_id,
      body: {
        changes: {
          distributions: {
            ...body.changes.distributions,
            "distribution:floats": {
              distribution: "Normal",
              params: { loc: floats, scale: { npy: new ExtData(42, new Uint8Array(8)) } },
            },
          },
        },
      },
      messages: [],
    }),
  );
  const saved = await readActionPoll("user-1", "model_diff", fixtureValue(comparison.call_id));
  expect(saved.body).toEqual(body);
  if (saved.status !== "success" || saved.action !== "model_diff")
    throw new Error("Expected model comparison");
  const laws = fixtureValue(saved.body.changes.distributions);
  const law = fixtureValue(laws["distribution:floats"]);
  const integerLaw = fixtureValue(laws["distribution:integers"]);
  function buffer(value: unknown) {
    if (
      value === null ||
      typeof value !== "object" ||
      !("npy" in value) ||
      !(value.npy instanceof Uint8Array)
    )
      throw new Error("Expected owned numerical value");
    return { npy: value.npy };
  }
  const array = buffer(law.params?.loc);
  const shared = buffer(law.params?.scale);
  expect(shared.npy).toBe(array.npy);
  expect(array.npy).toBeInstanceOf(Uint8Array);
  const decoded = readNumericalArray(array);
  expect(decoded.shape).toEqual([2, 2]);
  expect(Array.from(decoded.values)).toEqual([1, NaN, Infinity, -Infinity]);
  expect(readNumericalArray(array)).toBe(decoded);
  expect(readNumericalArray(shared)).toBe(decoded);
  const nodeBuffer = Buffer.concat([Buffer.from("padding"), Buffer.from(floats.npy)]);
  expect(Array.from(readNumericalArray({ npy: nodeBuffer.subarray(7) }).values)).toEqual([
    1,
    NaN,
    Infinity,
    -Infinity,
  ]);
  expect(Array.from(readNumericalArray(buffer(integerLaw.params?.v)).values)).toEqual([
    BigInt("18446744073709551615"),
  ]);
});

it("restores shared buffers in wire order when JavaScript reorders numeric object keys", () => {
  const first = new Uint8Array([11, 12]);
  const second = new Uint8Array([21, 22]);
  const reference = new ExtData(42, new Uint8Array(8));
  // A Python map can emit these keys in insertion order; a JavaScript object
  // enumerates them as 1, 2, 10 after MessagePack decoding.
  const payload = Buffer.concat([
    Buffer.from([0x83]),
    encode("10"),
    encode(first),
    encode("2"),
    encode(second),
    encode("1"),
    encode(reference),
  ]);
  expect(decodeModelMessage(Uint8Array.from(payload))).toEqual({ 1: first, 2: second, 10: first });
  const forwardReference = Buffer.concat([Buffer.from([0x92]), encode(reference), encode(first)]);
  expect(() => decodeModelMessage(forwardReference)).toThrow("earlier value");
});

it("restores chunked NPY buffers as one shared logical value among ordinary buffers", () => {
  const npy = new Uint8Array(dump(new Float32Array([1, 2, 3, 4]), [2, 2]));
  const length = new Uint8Array(8);
  new DataView(length.buffer).setUint32(4, npy.byteLength);
  const parts = Array.from({ length: Math.ceil(npy.byteLength / 17) }, (_, index) =>
    npy.subarray(index * 17, (index + 1) * 17),
  );
  const reference = (index: number) => {
    const data = new Uint8Array(8);
    new DataView(data.buffer).setUint32(4, index);
    return new ExtData(42, data);
  };
  const decoded = decodeModelMessage(
    encode([
      new Uint8Array([9]),
      [new ExtData(43, length), ...parts],
      reference(1),
      new Uint8Array([8]),
      reference(2),
    ]),
  );
  if (!Array.isArray(decoded)) throw new Error("Expected decoded array");
  const buffer: unknown = decoded[1];
  if (!(buffer instanceof Uint8Array)) throw new Error("Expected restored numerical buffer");
  expect(buffer).toEqual(npy);
  expect(decoded[2]).toBe(buffer);
  expect(decoded[4]).toBe(decoded[3]);
  expect(Array.from(readNumericalArray({ npy: buffer }).values)).toEqual([1, 2, 3, 4]);
  expect(() =>
    decodeModelMessage(encode([[new ExtData(43, length), new Uint8Array([1])]])),
  ).toThrow("byte length");
});

it("keeps JSON requests, timeline responses and HTTP errors alongside binary action responses", async () => {
  const client = createModelClient({ baseUrl: "http://viewer", fetch: fetcher });
  fetcher.mockResolvedValueOnce(messagePack({ status: "running", body: null, messages: [] }));
  const request = {
    action: "edit_question" as const,
    input: { question: { text: "Why?", outcome: "construct:outcome" as const } },
  };
  const running = await client.POST("/api/studies/{workspace_id}/edit_question", {
    params: { path: { workspace_id: "user-1" } },
    body: request,
  });
  expect(running.data?.status).toBe("running");
  const sent = fixtureValue(fetcher.mock.calls[0])[0];
  expect(sent.headers.get("Content-Type")).toBe("application/json");
  expect(await sent.json()).toEqual(request);
  const timeline = { attempts: [], dependencies: [], running: [] };
  fetcher.mockResolvedValueOnce(Response.json(timeline));
  expect(
    (
      await client.GET("/api/studies/{workspace_id}/timeline", {
        params: { path: { workspace_id: "user-1" } },
      })
    ).data,
  ).toEqual(timeline);
  fetcher.mockResolvedValueOnce(Response.json({ detail: "Unavailable" }, { status: 422 }));
  const error = await client.GET("/api/studies/{workspace_id}/timeline", {
    params: { path: { workspace_id: "user-1" } },
  });
  expect(error.response.status).toBe(422);
  expect(error.error).toEqual({ detail: "Unavailable" });
});
