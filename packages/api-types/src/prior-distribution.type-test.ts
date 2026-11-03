import type {
  DistributionId,
  DynamicalModelSpec,
  NumPyroDistribution,
  ParameterId,
  ParameterSpec,
  ReasoningSpec,
} from "./generated/models";

type Expect<T extends true> = T;
type Extends<A, B> = A extends B ? true : false;

// @ts-expect-error Parameter rationale uses the shared reasoning field.
export type NoParameterProvenance = ParameterSpec["prior_reasoning"];
export type ParameterOwnsIdentity = Expect<Extends<ParameterSpec["id"], ParameterId>>;
export type ParameterCarriesNativeLaw = Expect<
  Extends<NonNullable<ParameterSpec["distribution"]>, NumPyroDistribution | DistributionId>
>;
export type ModelOwnsParameters = Expect<
  Extends<NonNullable<DynamicalModelSpec["parameters"][ParameterId]>, Omit<ParameterSpec, "id">>
>;

export type ParameterRequiresReasoning = Expect<Extends<ParameterSpec["reasoning"], ReasoningSpec>>;
export type ReferenceUrlsAreStrings = Expect<
  Extends<ParameterSpec["reasoning"]["reference_urls"], readonly string[]>
>;
