import { describe, expect, it } from "vitest";
import { formatNumber } from "@/lib/utils/format";

describe("formatNumber", () => {
  it("formats a positive number with default decimals", () => {
    expect(formatNumber(Math.PI)).toBe("3.142");
  });

  it("formats with custom decimals", () => {
    expect(formatNumber(Math.PI, 1)).toBe("3.1");
  });

  it("returns NaN for NaN", () => {
    expect(formatNumber(Number.NaN)).toBe("NaN");
  });

  it("returns +Inf for positive infinity", () => {
    expect(formatNumber(Number.POSITIVE_INFINITY)).toBe("+Inf");
  });

  it("returns -Inf for negative infinity", () => {
    expect(formatNumber(Number.NEGATIVE_INFINITY)).toBe("-Inf");
  });

  it("formats zero", () => {
    expect(formatNumber(0)).toBe("0.000");
  });

  it("formats negative numbers", () => {
    expect(formatNumber(-1.5, 2)).toBe("-1.50");
  });
});
