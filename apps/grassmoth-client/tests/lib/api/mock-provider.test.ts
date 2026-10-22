import { afterEach, describe, expect, it } from "vitest";
import { getMockFixture, isMockMode } from "@/lib/api/mock-provider";

function unsetEnv(key: string) {
  Reflect.deleteProperty(process.env, key);
}

describe("isMockMode", () => {
  const original = process.env.NEXT_PUBLIC_MOCK_DATA;

  afterEach(() => {
    if (original !== undefined) {
      process.env.NEXT_PUBLIC_MOCK_DATA = original;
    } else {
      unsetEnv("NEXT_PUBLIC_MOCK_DATA");
    }
  });

  it.each([
    [undefined, false],
    ["", false],
    ["false", false],
    ["true", true],
    ["healthdemo", true],
  ])("interprets NEXT_PUBLIC_MOCK_DATA=%s as mock mode %s", (value, expected) => {
    if (value === undefined) {
      unsetEnv("NEXT_PUBLIC_MOCK_DATA");
    } else {
      process.env.NEXT_PUBLIC_MOCK_DATA = value;
    }

    expect(isMockMode()).toBe(expected);
  });
});

describe("getMockFixture", () => {
  const original = process.env.NEXT_PUBLIC_MOCK_DATA;

  afterEach(() => {
    if (original !== undefined) {
      process.env.NEXT_PUBLIC_MOCK_DATA = original;
    } else {
      unsetEnv("NEXT_PUBLIC_MOCK_DATA");
    }
  });

  it.each([
    [undefined, "DEFAULT"],
    ["", "DEFAULT"],
    ["true", "DEFAULT"],
    ["healthdemo", "HEALTHDEMO"],
  ])("maps NEXT_PUBLIC_MOCK_DATA=%s to fixture %s", (value, expected) => {
    if (value === undefined) {
      unsetEnv("NEXT_PUBLIC_MOCK_DATA");
    } else {
      process.env.NEXT_PUBLIC_MOCK_DATA = value;
    }

    expect(getMockFixture()).toBe(expected);
  });
});
