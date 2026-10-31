import { decode, ExtData } from "@msgpack/msgpack";

/** Restore logical binary values, including chunked buffers, from the Python codec. */
export function decodeModelMessage(payload: Uint8Array): unknown {
  const decoded = decode(payload);
  const buffers: { readonly value: Uint8Array; readonly wireOffset: number }[] = [];
  const chunked = new WeakMap<object, Uint8Array>();
  function collect(value: unknown): void {
    if (value instanceof Uint8Array) {
      buffers.push({ value, wireOffset: value.byteOffset });
    } else if (Array.isArray(value)) {
      const marker: unknown = value[0];
      if (marker instanceof ExtData && marker.type === 43) {
        if (
          !(marker.data instanceof Uint8Array) ||
          marker.data.byteLength !== 8 ||
          value.length < 2
        )
          throw new Error("Invalid numerical buffer chunks");
        const header = new DataView(marker.data.buffer, marker.data.byteOffset, 8);
        const length = header.getUint32(0) * 2 ** 32 + header.getUint32(4);
        const parts: Uint8Array[] = [];
        let size = 0;
        for (const part of value.slice(1)) {
          if (!(part instanceof Uint8Array) || part.byteLength === 0)
            throw new Error("Invalid numerical buffer chunks");
          parts.push(part);
          size += part.byteLength;
        }
        if (!Number.isSafeInteger(length) || size !== length)
          throw new Error("Numerical buffer chunks do not match their byte length");
        const buffer = new Uint8Array(length);
        let offset = 0;
        for (const part of parts) {
          buffer.set(part, offset);
          offset += part.byteLength;
        }
        chunked.set(value, buffer);
        buffers.push({ value: buffer, wireOffset: marker.data.byteOffset });
      } else {
        for (const item of value) collect(item);
      }
    } else if (value !== null && typeof value === "object" && !(value instanceof ExtData)) {
      for (const item of Object.values(value)) collect(item);
    }
  }
  collect(decoded);
  // The decoder returns binary views into the payload. Their offsets preserve
  // wire order even when JavaScript enumerates integer object keys differently.
  buffers.sort((left, right) => left.wireOffset - right.wireOffset);

  function restore(value: unknown): unknown {
    if (value instanceof Uint8Array) return value;
    if (value instanceof ExtData) {
      if (value.type !== 42 || !(value.data instanceof Uint8Array) || value.data.byteLength !== 8)
        throw new Error("Unknown numerical MessagePack extension");
      const view = new DataView(value.data.buffer, value.data.byteOffset, value.data.byteLength);
      const index = view.getUint32(0) * 2 ** 32 + view.getUint32(4);
      const buffer = buffers[index];
      if (
        !Number.isSafeInteger(index) ||
        buffer === undefined ||
        buffer.wireOffset >= value.data.byteOffset
      )
        throw new Error("Numerical buffer reference must select an earlier value");
      return buffer.value;
    }
    if (Array.isArray(value)) return chunked.get(value) ?? value.map(restore);
    if (value !== null && typeof value === "object")
      return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, restore(item)]));
    return value;
  }
  return restore(decoded);
}
