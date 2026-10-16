import type { MethodResponse } from "openapi-fetch";
import type openapi from "../schemas/openapi.json";
import type { createModelClient } from "./client";
import type { paths } from "./generated/model-api";
import type {
  ActionId,
  ActionPoll,
  Assessment,
  Attempt,
  Change,
  ConstructId,
  ConstructSpec,
  DistributionId,
  DynamicalModelSpec,
  EdgeId,
  EditModelRequest,
  FitOutput,
  GitOid,
  IndicatorId,
  IndicatorSpec,
  InferenceReportCore,
  ModelDiffOutput,
  ModelSnapshot,
  NumPyroDistribution,
  ObservationData,
  ObservationHistory,
  ObservationSpec,
  ParameterId,
  ParameterSpec,
  Rejected,
  TimelineResponse,
} from "./generated/models";

type Expect<T extends true> = T;
type Extends<A, B> = A extends B ? true : false;
type Equal<A, B> = [A, B] extends [B, A] ? true : false;

export type GenericAttemptCorrelatesRequest = Expect<
  Equal<
    Attempt<"edit_model", EditModelRequest<GitOid>, null>["request"],
    EditModelRequest<GitOid> | null
  >
>;
export type GenericAssessmentRetainsSubject = Expect<
  Equal<Assessment<string, number>["subject"], string>
>;
export type GenericChangeRetainsPayload = Expect<
  Equal<Extract<Change<DynamicalModelSpec>, { kind: "added" }>["after"], DynamicalModelSpec>
>;
export type GenericObservationRetainsWindow = Expect<
  Equal<ObservationSpec<"1d">["observation_window"], "1d">
>;
// @ts-expect-error Rejections use the common finding subject vocabulary.
export type RejectionHasNoUnrelatedReason = Rejected<number>;

export type ModelComparisonUsesTheEditDocument = Expect<
  Equal<ModelDiffOutput["changes"], EditModelRequest<GitOid>["input"]["dynamical_model_spec"]>
>;
export type EmptyModelDiffIsValid = Expect<
  Extends<Record<string, never>, ModelDiffOutput["changes"]>
>;
export type ModelDiffKeepsDeletionEntries = Expect<
  Extends<null, NonNullable<ModelDiffOutput["changes"]["constructs"]>[ConstructId]>
>;

export type CanonicalDefinition = Expect<
  Equal<NonNullable<ModelSnapshot["dynamical_model_spec"]>, DynamicalModelSpec>
>;
export type CanonicalInferenceCore = Expect<
  Equal<NonNullable<ModelSnapshot["fit"]>, InferenceReportCore>
>;
export type CanonicalParameter = Expect<
  Equal<NonNullable<DynamicalModelSpec["parameters"][ParameterId]>, Omit<ParameterSpec, "id">>
>;
export type CanonicalConstruct = Expect<
  Equal<NonNullable<DynamicalModelSpec["edges"][EdgeId]>["cause"], ConstructId>
>;
export type OwnedIndicator = Expect<Equal<ConstructSpec["indicators"][number], IndicatorSpec>>;
// @ts-expect-error Indicator ownership is declared by containment.
export type NoIndependentIndicatorOwner = IndicatorSpec["construct_id"];

type FitCall = paths["/api/studies/{workspace_id}/fit"]["post"];
type FetchedCall = MethodResponse<
  ReturnType<typeof createModelClient>,
  "post",
  "/api/studies/{workspace_id}/fit"
>;
export type GeneratedCallRetainsOwnedFitResult = Expect<
  Equal<Extract<FetchedCall, { status: "success"; action: "fit" }>["body"], FitOutput>
>;
export type GeneratedCallRetainsCanonicalOutcome = Expect<Equal<FetchedCall, ActionPoll>>;

export type SparseLawLookup = Expect<
  Equal<DynamicalModelSpec["distributions"][DistributionId], NumPyroDistribution | undefined>
>;
export type SparseObservationLookup = Expect<
  Equal<ObservationData[IndicatorId], ObservationHistory | undefined>
>;
export type DistinctScientificIds = Expect<Equal<Extends<IndicatorId, ConstructId>, false>>;
export type RequiredNullableResponse = Expect<
  Equal<
    Record<string, never> extends Pick<ModelSnapshot, "dynamical_model_spec"> ? true : false,
    false
  >
>;

// Coverage follows the exported schema; adding an endpoint cannot silently leave it out of the client.
export type EveryExportedPath = Expect<Equal<keyof paths, keyof typeof openapi.paths>>;
type HttpMethod = "get" | "post" | "put" | "patch" | "delete" | "head" | "options" | "trace";
type ExportedOperations = {
  [Path in keyof typeof openapi.paths]: `${Extract<keyof (typeof openapi.paths)[Path], HttpMethod>} ${Path}`;
}[keyof typeof openapi.paths];
type GeneratedOperations = {
  [Path in keyof paths]: {
    [Method in Extract<keyof paths[Path], HttpMethod>]: paths[Path][Method] extends undefined
      ? never
      : `${Method} ${Path}`;
  }[Extract<keyof paths[Path], HttpMethod>];
}[keyof paths];
export type EveryExportedOperation = Expect<Equal<GeneratedOperations, ExportedOperations>>;
type CallRoute = `/api/studies/{workspace_id}/${ActionId}`;
type CallInput = paths[CallRoute]["post"]["requestBody"]["content"]["application/json"];
export type CanonicalCallAlternatives = Expect<Equal<NonNullable<CallInput["action"]>, ActionId>>;
export type CanonicalTimeline = Expect<
  Equal<
    MethodResponse<
      ReturnType<typeof createModelClient>,
      "get",
      "/api/studies/{workspace_id}/timeline"
    >,
    TimelineResponse
  >
>;
export type CanonicalWorkspaces = Expect<
  Equal<
    MethodResponse<ReturnType<typeof createModelClient>, "get", "/api/workspaces">,
    Readonly<Record<string, string | null>>
  >
>;
type Upload = paths["/api/upload"]["post"]["requestBody"]["content"]["multipart/form-data"];
export type NativeUploadFile = Expect<Equal<Upload["file"], Blob>>;
type FitInput = FitCall["requestBody"]["content"]["application/json"];
export type RequestDefaultsMayBeOmitted = Expect<
  Extends<
    {
      action: "fit";
      input: {
        target: {
          kind: "new";
          dynamical_model_spec_ref: string;
          data_ref: { revision: string; replicate_index: number };
        };
      };
    },
    FitInput
  >
>;
type EditInput =
  paths["/api/studies/{workspace_id}/edit_model"]["post"]["requestBody"]["content"]["application/json"];
export type PartialModelEditsKeepNestedFieldsOptional = Expect<
  Extends<
    {
      action: "edit_model";
      input: {
        parent_ref: string;
        dynamical_model_spec: {
          edges: {
            "edge:remove": null;
            "edge:revise": {
              mechanisms: {
                "mechanism:weight": { expression: { left: { left: { value: 4 } } } };
              };
            };
          };
        };
      };
    },
    EditInput
  >
>;
export type DefaultsAreAbsent = Expect<
  Equal<
    Extends<
      {
        action: "fit";
        input: {
          dynamical_model_spec_ref: string;
          data_ref: string;
          replicate_index: number;
          settings: undefined;
        };
      },
      FitInput
    >,
    false
  >
>;

declare const definition: DynamicalModelSpec;
declare const edge: NonNullable<DynamicalModelSpec["edges"][EdgeId]>;
// @ts-expect-error Published entity maps are read-only.
definition.parameters["parameter:new"] = {};
// @ts-expect-error Read-only guarantees extend to nested values.
edge.description = "changed";
// @ts-expect-error Published maps are read-only.
definition.distributions["distribution:changed"] = {};
// @ts-expect-error A map retains its scientific key kind.
export type WrongMapKey = DynamicalModelSpec["distributions"][IndicatorId];
