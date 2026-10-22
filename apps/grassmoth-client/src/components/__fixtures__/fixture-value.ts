/** A required part of an authored test/story fixture fails at its loading boundary. */
export function fixtureValue<T>(value: T | null | undefined): T {
  if (value == null) throw new Error("Fixture is missing a required value");
  return value;
}

/** JSON fixture files spell binary buffers as byte lists; the API uses Uint8Array. */
type EncodedFixture<T> = T extends Uint8Array
  ? readonly number[] | { readonly buffer: number }
  : T extends object
    ? { readonly [K in keyof T]: EncodedFixture<T[K]> }
    : T;
declare const fixtureContract: unique symbol;
interface FixtureContract<T> {
  readonly [fixtureContract]?: T;
}
export type BinaryFixture<T> = EncodedFixture<T> & FixtureContract<T>;

export function decodeFixture<T>(value: FixtureContract<T>): T;
export function decodeFixture(value: unknown): unknown {
  const buffers: Uint8Array[] = [];
  function restore(item: unknown): unknown {
    if (Array.isArray(item)) return item.map(restore);
    if (item !== null && typeof item === "object") {
      if ("npy" in item) {
        if (Array.isArray(item.npy)) {
          const npy = new Uint8Array(item.npy);
          buffers.push(npy);
          return { npy };
        }
        if (
          item.npy !== null &&
          typeof item.npy === "object" &&
          "buffer" in item.npy &&
          typeof item.npy.buffer === "number"
        )
          return { npy: fixtureValue(buffers[item.npy.buffer]) };
        throw new Error("Invalid fixture buffer");
      }
      return Object.fromEntries(Object.entries(item).map(([key, child]) => [key, restore(child)]));
    }
    return item;
  }
  return restore(value);
}
