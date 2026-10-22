import { describe, expect, it } from "vitest";
import { distributionArgumentText, distributionText } from "@/lib/utils/distribution-format";

describe("native distribution display", () => {
  it("renders array values without exposing wire dtype tags or flattening dimensions", () => {
    expect(
      distributionArgumentText({
        array: [
          [1, 2],
          [3, 4],
        ],
        dtype: "float32",
      }),
    ).toBe("[[1, 2], [3, 4]]");
  });

  it("retains nested laws and their constructor arguments", () => {
    expect(
      distributionText({
        distribution: "Independent",
        params: {
          base_dist: {
            distribution: "Normal",
            params: {
              loc: { array: [0, 1], dtype: "float32" },
              scale: 2,
              validate_args: false,
            },
          },
          reinterpreted_batch_ndims: 1,
          validate_args: true,
        },
      }),
    ).toBe("Independent(base_dist=Normal(loc=[0, 1], scale=2), reinterpreted_batch_ndims=1)");
  });
});
