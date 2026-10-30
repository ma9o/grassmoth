/** AUTO-GENERATED from Python's OpenAPI graph. Run bun run codegen. */
import type * as Domain from "./models";
export interface paths {
    readonly "/api/studies/{workspace_id}/{action}/{call_id}": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Poll Action
         * @description Read an existing call by ID. Never resolve inputs, start a workflow, execute, or retry. The envelope and accumulated messages match POST, including cached failures.
         */
        readonly get: operations["poll_action"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/edit_question": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly get?: never;
        readonly put?: never;
        /**
         * Edit Question
         * @description Set the study question from input.question. POST returns a call_id; GET polls it. Identical resolved calls reuse both successes and failures.
         */
        readonly post: operations["edit_question"];
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/edit_model": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly get?: never;
        readonly put?: never;
        /**
         * Edit Model
         * @description Merge input.dynamical_model_spec into input.parent_ref: start empty for a question revision, or retain omitted fields from a model revision and its pinned question. Null entity entries delete their IDs. Prune constructs outside the outcome ancestry with warnings, then validate and evaluate data-independent model checks. The body contains the produced model and its findings; messages retain all execution logging. GET polls the returned call_id.
         */
        readonly post: operations["edit_model"];
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/prepare_data": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly get?: never;
        readonly put?: never;
        /**
         * Prepare Data
         * @description Prepare observations from uploaded tables using the selected model's definitions. input.source is a folder under data/{workspace_id}/, including subfolders. CSV and Parquet tables must supply a date or datetime timestamp column and are concatenated in captured file order. Source bytes and revision selectors are pinned before computing call identity. GET polls call_id.
         */
        readonly post: operations["prepare_data"];
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/fit": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly get?: never;
        readonly put?: never;
        /**
         * Fit
         * @description Fit target.new model and data, or continue target.fit_ref at its last retained state. Counts request additional draws; zero warmup freezes tuning and algorithm changes require warmup. Zero warmup and zero samples compile both sweep programs without execution. Zero-draw fits return checks and inference without a conditioned model. repeat distinguishes timing attempts. GET polls call_id.
         */
        readonly post: operations["fit"];
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/simulate": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly get?: never;
        readonly put?: never;
        /**
         * Simulate
         * @description Simulate input.dynamical_model_spec_ref using input.simulation. The model owns deterministic inputs and retained trajectory coordinates; authored initial states apply at simulation.start. body.data contains an array of observation histories of the same type returned by prepare_data. The successful body requires data and report; report owns single or paired numerical evidence, calculated summaries and causal findings; messages retain traces. GET polls call_id.
         */
        readonly post: operations["simulate"];
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/data_diff": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly get?: never;
        readonly put?: never;
        /**
         * Data Diff
         * @description Compare input.left_ref and input.right_ref, each a nonempty list of {revision, replicate_index} references. Omit the index to compare all recorded histories. Data latest selects prepared user data; simulations require exact Git hashes. Retain the complete comparison in Git. GET polls call_id. Identical resolved calls reuse successes and failures; the comparison is the body.
         */
        readonly post: operations["data_diff"];
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/model_diff": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly get?: never;
        readonly put?: never;
        /**
         * Model Diff
         * @description Compare input.before_ref and input.after_ref as saved DynamicalModelSpec documents. Return changes using the same partial DynamicalModelSpec contract as edit_model: omissions are unchanged and null map entries delete identities. Retain the patch in Git. GET polls call_id. Identical resolved calls reuse successes and failures; the comparison is the body.
         */
        readonly post: operations["model_diff"];
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/studies/{workspace_id}/timeline": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * Get Timeline
         * @description Slim call log: call IDs, retained arguments, outcome summaries and dependencies. The viewer reads complete results and messages with GET using each call_id.
         */
        readonly get: operations["get_timeline_api_studies__workspace_id__timeline_get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/workspaces": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        /**
         * List Workspaces
         * @description Available workspaces and their immutable study questions.
         *
         *     X-Actions-Enabled reports whether the facade accepts new calls.
         */
        readonly get: operations["list_workspaces_api_workspaces_get"];
        readonly put?: never;
        readonly post?: never;
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
    readonly "/api/upload": {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly get?: never;
        readonly put?: never;
        /**
         * Upload File
         * @description Stage one named raw input file in the workspace's input folder. prepare_data reads that folder when input.source is "input" and captures each file's SHA-256.
         */
        readonly post: operations["upload_file_api_upload_post"];
        readonly delete?: never;
        readonly options?: never;
        readonly head?: never;
        readonly patch?: never;
        readonly trace?: never;
    };
}
export type webhooks = Record<string, never>;
export interface components {
    schemas: {
        /** @description A closed action attempt pairs its request with only that action's successful result or failure outcome. */
        readonly ActionAttempt: Domain.Attempt<"edit_question", Domain.EditQuestionRequest, Domain.GitOid> | Domain.Attempt<"edit_model", Domain.EditModelRequest<Domain.GitOid>, Domain.GitOid> | Domain.Attempt<"prepare_data", Domain.PrepareDataRequest<Domain.GitOid, Domain.FileSourceRef>, Domain.GitOid> | Domain.Attempt<"fit", Domain.FitRequest<Domain.GitOid>, Domain.GitOid> | Domain.Attempt<"simulate", Domain.SimulateRequest<Domain.GitOid>, Domain.GitOid> | Domain.Attempt<"data_diff", Domain.DataDiffRequest<Domain.GitOid>, Domain.GitOid> | Domain.Attempt<"model_diff", Domain.ModelDiffRequest<Domain.GitOid>, Domain.GitOid>;
        /**
         * ActionEffects
         * @description What an executed action did to the store: the workflow installs this.
         */
        readonly ActionEffects: {
            /** Produced */
            readonly produced: readonly components["schemas"]["ArtifactRecord"][];
            /** Retracted */
            readonly retracted: readonly components["schemas"]["RetractedArtifact"][];
            /** Reports */
            readonly reports: Readonly<Partial<Record<components["schemas"]["ActionReportName"], components["schemas"]["GitOid-Output"]>>>;
        };
        readonly ActionId: components["schemas"]["ScientificActionId"] | ("data_diff" | "model_diff");
        /**
         * ActionMessage
         * @description A label emitted by an attempt; measurements belong in its scientific result.
         */
        readonly ActionMessage: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "log";
            /**
             * Timestamp
             * Format: date-time
             */
            readonly timestamp: string;
            /**
             * Severity
             * @enum {string}
             */
            readonly severity: "debug" | "info" | "warning" | "error";
            /** Code */
            readonly code: string;
            readonly subject: components["schemas"]["FindingSubject"];
            /** Detail */
            readonly detail: string;
        };
        /** @description A call returns its identity, publication status, scientific body, and accumulated execution messages. */
        readonly ActionPoll: components["schemas"]["RunningPoll"] | components["schemas"]["FailedPoll"] | components["schemas"]["ActionSuccess"];
        /** @enum {string} */
        readonly ActionReportName: "checks" | "identification" | "validation" | "data-profile" | "inference" | "simulation" | "data-diff" | "model-diff";
        /** @description A successful call pairs its action with that action's scientific result body. */
        readonly ActionSuccess: Domain.SuccessfulPoll<"edit_question", Domain.EditQuestionOutput> | Domain.SuccessfulPoll<"edit_model", Domain.EditModelOutput> | Domain.SuccessfulPoll<"prepare_data", Domain.PrepareDataOutput> | Domain.SuccessfulPoll<"fit", Domain.FitOutput> | Domain.SuccessfulPoll<"simulate", Domain.SimulateOutput> | Domain.SuccessfulPoll<"data_diff", Domain.DataDiffOutput> | Domain.SuccessfulPoll<"model_diff", Domain.ModelDiffOutput>;
        /** @enum {string} */
        readonly AdaptationScheme: "simple" | "dual_averaging";
        /**
         * AdaptedLatentStepSpec
         * @description Warmup adapts the per-time scale towards the assigned movement target.
         */
        readonly "AdaptedLatentStepSpec-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "adapted";
            /**
             * Target
             * @description Fraction of proposals at each time step that should change its state. In a new fit every time step's scale starts at 0.01, adapts towards this target during warmup and is then frozen. Corenflos & Särkkä (2025, section 4) target 0.5 per time step, below Finke & Thiery's 1 - (N + 1)^(-1/3), because a higher target can be unreachable at some time steps and leaves the sampler stuck there. Corenflos & Finke (2024, section 5.2) use 0.75 and note that the optimum depends on N and on the state dimension.
             * @default 0.5
             */
            readonly target?: number;
        };
        /**
         * AdaptedLatentStepSpec
         * @description Warmup adapts the per-time scale towards the assigned movement target.
         */
        readonly "AdaptedLatentStepSpec-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "adapted";
            /**
             * Target
             * @description Fraction of proposals at each time step that should change its state. In a new fit every time step's scale starts at 0.01, adapts towards this target during warmup and is then frozen. Corenflos & Särkkä (2025, section 4) target 0.5 per time step, below Finke & Thiery's 1 - (N + 1)^(-1/3), because a higher target can be unreachable at some time steps and leaves the sampler stuck there. Corenflos & Finke (2024, section 5.2) use 0.75 and note that the optimum depends on N and on the state dimension.
             * @default 0.5
             */
            readonly target: number;
        };
        /**
         * Added[DataPoint]
         * @description The right payload exists at an identity absent on the left.
         */
        readonly Added_DataPoint_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "added";
            readonly after: components["schemas"]["DataPoint"];
        };
        /**
         * Applied[GitOid]
         * @description A successful action result and the artifact changes it proposes to publish.
         */
        readonly Applied_GitOid_: {
            /**
             * @description Discriminator identifying an applied outcome. (enum property replaced by openapi-typescript)
             * @enum {string}
             */
            readonly status: "applied";
            /** @description Complete saved result reference, or the transient evidence awaiting completion. */
            readonly result: components["schemas"]["GitOid-Output"];
            /** @description Produced and retracted artifacts and retained report references. */
            readonly effects: components["schemas"]["ActionEffects"];
        };
        /**
         * Applied[NoneType]
         * @description A successful action result and the artifact changes it proposes to publish.
         */
        readonly Applied_NoneType_: {
            /**
             * @description Discriminator identifying an applied outcome. (enum property replaced by openapi-typescript)
             * @enum {string}
             */
            readonly status: "applied";
            /**
             * Result
             * @description Complete saved result reference, or the transient evidence awaiting completion.
             */
            readonly result: null;
            /** @description Produced and retracted artifacts and retained report references. */
            readonly effects: components["schemas"]["ActionEffects"];
        };
        /**
         * ArrayVector
         * @description A vector view of an owned numerical value; the wire codec shares its buffer.
         */
        readonly ArrayVector: {
            readonly array: components["schemas"]["NumericalArray"];
            /** Indices */
            readonly indices: readonly (number | null)[];
            /** @default null */
            readonly mask: components["schemas"]["ArrayVector"] | null;
            /**
             * Start
             * @default 0
             */
            readonly start: number;
            /**
             * Stop
             * @default null
             */
            readonly stop: number | null;
        };
        /**
         * ArtifactFiles
         * @description Execution staging or an uploaded input owns its physical files.
         */
        readonly ArtifactFiles: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "files";
        };
        /**
         * @description An artifact identity selects one node in the study's artifact graph.
         * @enum {string}
         */
        readonly ArtifactId: "question" | "raw_data" | "model" | "panel";
        /**
         * ArtifactRecord
         * @description An immutable artifact revision with its producer and exact input dependencies.
         *
         *     ``derived_from`` pins the versions used to compute the payload. Initial
         *     models pin their question; edits also pin their model parent. The producing
         *     activity stamps ``created_at`` outside deterministic workflow execution.
         */
        readonly ArtifactRecord: {
            readonly artifact_id: components["schemas"]["ArtifactId"];
            readonly revision: components["schemas"]["GitOid-Output"];
            /** Derived From */
            readonly derived_from: Readonly<Partial<Record<components["schemas"]["ArtifactId"], components["schemas"]["GitOid-Output"]>>>;
            /**
             * Produced By
             * @default null
             */
            readonly produced_by: string | null;
            /**
             * Created At
             * @default
             */
            readonly created_at: string;
            readonly source: components["schemas"]["ArtifactSource"];
        };
        /**
         * ArtifactResult
         * @description A published scientific artifact selects a field of its owning action result.
         */
        readonly ArtifactResult: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "result";
            readonly result: components["schemas"]["GitOid-Output"];
        };
        /** @description An artifact owns staged input files or references its complete action result. */
        readonly ArtifactSource: components["schemas"]["ArtifactFiles"] | components["schemas"]["ArtifactResult"];
        readonly Assessment_ConvergenceAssessmentSubject_NumericCriterionEvidence_: Domain.Evaluated<Domain.ConvergenceAssessmentSubject, Domain.NumericCriterionEvidence> | Domain.NotEvaluated<Domain.ConvergenceAssessmentSubject>;
        readonly Assessment_IndicatorCheckSubject_NumericCriterionEvidence_: Domain.Evaluated<Domain.IndicatorCheckSubject, Domain.NumericCriterionEvidence> | Domain.NotEvaluated<Domain.IndicatorCheckSubject>;
        /**
         * AttemptRecord
         * @description Stored inside the Git object, with no self-referential publication ID.
         */
        readonly AttemptRecord: {
            /**
             * Seq
             * @description Attempt's position in the study journal.
             */
            readonly seq: number;
            /**
             * Attempt Id
             * @description Execution identity used for progress tracking, when retained.
             * @default null
             */
            readonly attempt_id: string | null;
            /**
             * Ts
             * @description Recorded attempt timestamp.
             */
            readonly ts: string;
            /**
             * Messages
             * @description Structured action messages emitted during execution.
             * @default []
             */
            readonly messages: readonly components["schemas"]["ActionMessage"][];
            /**
             * Trace Ids
             * @description References identifying the retained execution traces.
             * @default []
             */
            readonly trace_ids: readonly string[];
            readonly attempt: components["schemas"]["ActionAttempt"];
        };
        /**
         * Attempt[ActionId, Union[ScientificActionRequest, DataDiffRequest[GitOid], ModelDiffRequest[GitOid]], NoneType]
         * @description One action's request and successful result share the same attempt owner.
         */
        readonly Attempt_ActionId_Union_ScientificActionRequest__DataDiffRequest_GitOid___ModelDiffRequest_GitOid___NoneType_: {
            readonly action: components["schemas"]["ActionId"];
            /**
             * Request
             * @description Parsed arguments, or null for a historical attempt whose arguments were not retained
             */
            readonly request: components["schemas"]["ScientificActionRequest"] | Domain.DataDiffRequest<Domain.GitOid> | Domain.ModelDiffRequest<Domain.GitOid> | null;
            /** Outcome */
            readonly outcome: Domain.Applied<null> | components["schemas"]["Rejected"] | components["schemas"]["Raised"];
        };
        /**
         * Attempt[Literal['data_diff'], DataDiffRequest[GitOid], GitOid]
         * @description One action's request and successful result share the same attempt owner.
         */
        readonly Attempt_Literal__data_diff___DataDiffRequest_GitOid__GitOid_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "data_diff";
            /** @description Parsed arguments, or null for a historical attempt whose arguments were not retained */
            readonly request: Domain.DataDiffRequest<Domain.GitOid> | null;
            /** Outcome */
            readonly outcome: Domain.Applied<Domain.GitOid> | components["schemas"]["Rejected"] | components["schemas"]["Raised"];
        };
        /**
         * Attempt[Literal['edit_model'], EditModelRequest[GitOid], GitOid]
         * @description One action's request and successful result share the same attempt owner.
         */
        readonly Attempt_Literal__edit_model___EditModelRequest_GitOid__GitOid_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "edit_model";
            /** @description Parsed arguments, or null for a historical attempt whose arguments were not retained */
            readonly request: Domain.EditModelRequest<Domain.GitOid> | null;
            /** Outcome */
            readonly outcome: Domain.Applied<Domain.GitOid> | components["schemas"]["Rejected"] | components["schemas"]["Raised"];
        };
        /**
         * Attempt[Literal['edit_question'], EditQuestionRequest, GitOid]
         * @description One action's request and successful result share the same attempt owner.
         */
        readonly Attempt_Literal__edit_question___EditQuestionRequest_GitOid_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "edit_question";
            /** @description Parsed arguments, or null for a historical attempt whose arguments were not retained */
            readonly request: components["schemas"]["EditQuestionRequest-Output"] | null;
            /** Outcome */
            readonly outcome: Domain.Applied<Domain.GitOid> | components["schemas"]["Rejected"] | components["schemas"]["Raised"];
        };
        /**
         * Attempt[Literal['fit'], FitRequest[GitOid], GitOid]
         * @description One action's request and successful result share the same attempt owner.
         */
        readonly Attempt_Literal__fit___FitRequest_GitOid__GitOid_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "fit";
            /** @description Parsed arguments, or null for a historical attempt whose arguments were not retained */
            readonly request: Domain.FitRequest<Domain.GitOid> | null;
            /** Outcome */
            readonly outcome: Domain.Applied<Domain.GitOid> | components["schemas"]["Rejected"] | components["schemas"]["Raised"];
        };
        /**
         * Attempt[Literal['model_diff'], ModelDiffRequest[GitOid], GitOid]
         * @description One action's request and successful result share the same attempt owner.
         */
        readonly Attempt_Literal__model_diff___ModelDiffRequest_GitOid__GitOid_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "model_diff";
            /** @description Parsed arguments, or null for a historical attempt whose arguments were not retained */
            readonly request: Domain.ModelDiffRequest<Domain.GitOid> | null;
            /** Outcome */
            readonly outcome: Domain.Applied<Domain.GitOid> | components["schemas"]["Rejected"] | components["schemas"]["Raised"];
        };
        /**
         * Attempt[Literal['prepare_data'], PrepareDataRequest[GitOid, FileSourceRef], GitOid]
         * @description One action's request and successful result share the same attempt owner.
         */
        readonly Attempt_Literal__prepare_data___PrepareDataRequest_GitOid__FileSourceRef__GitOid_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "prepare_data";
            /** @description Parsed arguments, or null for a historical attempt whose arguments were not retained */
            readonly request: Domain.PrepareDataRequest<Domain.GitOid, Domain.FileSourceRef> | null;
            /** Outcome */
            readonly outcome: Domain.Applied<Domain.GitOid> | components["schemas"]["Rejected"] | components["schemas"]["Raised"];
        };
        /**
         * Attempt[Literal['simulate'], SimulateRequest[GitOid], GitOid]
         * @description One action's request and successful result share the same attempt owner.
         */
        readonly Attempt_Literal__simulate___SimulateRequest_GitOid__GitOid_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "simulate";
            /** @description Parsed arguments, or null for a historical attempt whose arguments were not retained */
            readonly request: Domain.SimulateRequest<Domain.GitOid> | null;
            /** Outcome */
            readonly outcome: Domain.Applied<Domain.GitOid> | components["schemas"]["Rejected"] | components["schemas"]["Raised"];
        };
        /**
         * AuthoredLawProvenance
         * @description The current laws have authored ancestry without retained fitting.
         */
        readonly AuthoredLawProvenance: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "authored";
            /**
             * Interpretation
             * @description Prior-predictive interpretation of draws made entirely from authored parameter laws.
             * @constant
             */
            readonly interpretation: "prior_predictive";
        };
        /**
         * BernoulliLogitsLawSpec[Expression]
         * @description The BernoulliLogits conditional law.
         */
        readonly "BernoulliLogitsLawSpec_Expression_-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "BernoulliLogits";
            readonly logits: components["schemas"]["Expression-Output"];
        };
        /**
         * BernoulliProbsLawSpec[Expression]
         * @description The BernoulliProbs conditional law.
         */
        readonly "BernoulliProbsLawSpec_Expression_-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "BernoulliProbs";
            readonly probs: components["schemas"]["Expression-Output"];
        };
        /**
         * BetaLawSpec[Expression]
         * @description The Beta conditional law.
         */
        readonly "BetaLawSpec_Expression_-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "Beta";
            readonly concentration1: components["schemas"]["Expression-Output"];
            readonly concentration0: components["schemas"]["Expression-Output"];
        };
        /**
         * BinaryExpression
         * @description A supported scalar operation composing two expressions.
         */
        readonly "BinaryExpression-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "binary";
            readonly operator: components["schemas"]["BinaryOperator"];
            readonly left: components["schemas"]["Expression-Output"];
            readonly right: components["schemas"]["Expression-Output"];
        };
        /**
         * @description A binary operator combines two scalar expression operands.
         * @enum {string}
         */
        readonly BinaryOperator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
        /** Body_upload_file_api_upload_post */
        readonly Body_upload_file_api_upload_post: {
            /** File */
            readonly file: Blob;
            /** Workspaceid */
            readonly workspaceId: string;
        };
        /**
         * CallExpression
         * @description A supported mathematical function, including explicit discrete contrasts.
         */
        readonly "CallExpression-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "call";
            readonly function: components["schemas"]["ExpressionFunction"];
            /** Arguments */
            readonly arguments: readonly components["schemas"]["Expression-Output"][];
        };
        readonly "CallId-Input": `call:${string}`;
        readonly "CallId-Output": `call:${string}`;
        /**
         * CategoricalLawSpec[Expression]
         * @description The Categorical conditional law.
         */
        readonly "CategoricalLawSpec_Expression_-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "Categorical";
            readonly logits: components["schemas"]["Expression-Output"];
        };
        /**
         * CategoryProbabilitySummary
         * @description Predictive probabilities for each declared level; unobserved anchors are null.
         */
        readonly CategoryProbabilitySummary: {
            /** Probabilities */
            readonly probabilities: {
                readonly [key: string]: readonly (number | null)[];
            };
            /** N Draws */
            readonly n_draws: readonly number[];
        };
        /**
         * CausalEdgeSpec
         * @description A specification of a directed causal relationship between two constructs.
         */
        readonly "CausalEdgeSpec-Output": {
            /** @description Persistent identity. Preserve when revising the same edge. */
            readonly id: components["schemas"]["EdgeId-Output"];
            /**
             * Mechanisms
             * @default []
             */
            readonly mechanisms: readonly components["schemas"]["DriftMechanismSpec-Output"][];
            /** @description Cause construct; shared endpoints have one identity. */
            readonly cause: components["schemas"]["ConstructSpec-Output"] | components["schemas"]["ConstructRef-Output"];
            /** @description Effect construct; shared endpoints have one identity. */
            readonly effect: components["schemas"]["ConstructSpec-Output"] | components["schemas"]["ConstructRef-Output"];
            /**
             * Description
             * @description Theoretical justification for this causal link
             */
            readonly description: string;
            /**
             * Sources
             * @description Literature sources supporting this causal link
             */
            readonly sources: readonly components["schemas"]["LiteratureSource-Output"][];
        };
        /**
         * CausalEffectResult
         * @description Causal effects and realized trajectories under the enclosing report's design.
         */
        readonly CausalEffectResult: {
            readonly outcome: components["schemas"]["ConstructId-Output"];
            /** Labels */
            readonly labels: Readonly<Partial<Record<components["schemas"]["ConstructId-Output"], string>>>;
            /** @description Paired outcome contrasts, [draw, time]. */
            readonly differences: components["schemas"]["NumericalArray"];
            /** Frame */
            readonly frame: readonly [
                number,
                number
            ];
            readonly summary: components["schemas"]["EffectSummary"];
            /**
             * Reference Mean
             * @description Mean reference outcome at the final time.
             */
            readonly reference_mean: number;
            /**
             * Manifest Effects
             * @description Mean final-time indicator contrasts where every paired draw is finite.
             */
            readonly manifest_effects: Readonly<Partial<Record<components["schemas"]["IndicatorId-Output"], number>>>;
            /** Warnings */
            readonly warnings: readonly string[];
        };
        /**
         * CentredParameterizationRecord
         * @description The original frozen affine parameterization, reconstructed from the pinned target and seed.
         */
        readonly CentredParameterizationRecord: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "centred";
        };
        /**
         * ChainDiagnostics
         * @description Compact retained-chain measurements; plot series compose the report detail.
         */
        readonly ChainDiagnostics: {
            /** Num Chains */
            readonly num_chains: number;
            /** Num Samples Per Chain */
            readonly num_samples_per_chain: number;
            /** Per Parameter */
            readonly per_parameter: readonly components["schemas"]["ParameterDiagnostics"][];
            /**
             * Per Latent
             * @default null
             */
            readonly per_latent: readonly components["schemas"]["LatentDiagnostics"][] | null;
            /**
             * Num Divergences
             * @default null
             */
            readonly num_divergences: number | null;
            /**
             * Divergence Rate
             * @default null
             */
            readonly divergence_rate: number | null;
            /**
             * Tree Depth Mean
             * @default null
             */
            readonly tree_depth_mean: number | null;
            /**
             * Tree Depth Max
             * @default null
             */
            readonly tree_depth_max: number | null;
            /**
             * Accept Prob Mean
             * @default null
             */
            readonly accept_prob_mean: number | null;
            /**
             * Latent Accept Prob Mean
             * @default null
             */
            readonly latent_accept_prob_mean: number | null;
            /**
             * Parameter Accept Prob Mean
             * @default null
             */
            readonly parameter_accept_prob_mean: number | null;
            /** @default null */
            readonly energy: components["schemas"]["EnergyDiagnostics"] | null;
        };
        readonly Change_DataPoint_: Domain.Added<Domain.DataPoint> | Domain.Removed<Domain.DataPoint> | Domain.Revised<Domain.DataPoint>;
        /** @enum {string} */
        readonly CheckOutcome: "passed" | "failed";
        readonly CheckStatus: components["schemas"]["CheckOutcome"] | "not_evaluated";
        /**
         * CoefficientExpression
         * @description A scientifically typed coefficient operand, literal or parameter reference.
         */
        readonly "CoefficientExpression-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "coefficient";
            readonly role: components["schemas"]["CoefficientRole"];
            /**
             * Value
             * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
             * @default null
             */
            readonly value: number | components["schemas"]["ParameterId-Output"] | null;
            /**
             * Construct Ids
             * @description Additional constructs participating in this coefficient use.
             * @default []
             */
            readonly construct_ids: readonly components["schemas"]["ConstructId-Output"][];
        };
        /**
         * @description A coefficient role identifies an expression operand’s scientific quantity and support.
         * @enum {string}
         */
        readonly CoefficientRole: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
        /** @enum {string} */
        readonly ComparisonSide: "left" | "right";
        /**
         * CompletedExtractionWorker
         * @description Counts of extracted observations and windows retained from a completed worker.
         */
        readonly CompletedExtractionWorker: {
            /** Worker Id */
            readonly worker_id: number;
            /** N Extractions */
            readonly n_extractions: number;
            /** N Windows */
            readonly n_windows: number;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly status: "completed";
        };
        /**
         * ComputedExtractionSpec
         * @description Compute a deterministic support-window measurement from source columns.
         */
        readonly "ComputedExtractionSpec-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "computed";
            /**
             * How To Measure
             * @description Description of the deterministic measurement.
             */
            readonly how_to_measure: string;
            /** Source Columns */
            readonly source_columns: readonly [
                string,
                ...string[]
            ];
            /**
             * @description Optional deterministic support-window expression over the declared source columns. It must return one scalar per window with the observation's declared summary operator. Omitted uses a direct single-column aggregation.
             * @default null
             */
            readonly computed_rule?: components["schemas"]["WindowExpression"] | null;
            /**
             * Fill Null
             * @description Optional Polars null filling after aggregation on the sorted time grid within the selected data span. Use forward, backward, min, max, mean, zero, one, or a numeric constant. Fills every null, including explicit unknown readings. Omitted leaves nulls unknown. Forward carries the last value and leaves leading nulls unknown.
             * @default null
             */
            readonly fill_null?: ("forward" | "backward" | "min" | "max" | "mean" | "zero" | "one") | number | null;
            /**
             * Fill Null Limit
             * @description Maximum consecutive nulls filled by forward/backward; omitted is unlimited. Only valid when fill_null is forward or backward.
             * @default null
             */
            readonly fill_null_limit?: number | null;
        };
        /**
         * ComputedExtractionSpec
         * @description Compute a deterministic support-window measurement from source columns.
         */
        readonly "ComputedExtractionSpec-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "computed";
            /**
             * How To Measure
             * @description Description of the deterministic measurement.
             */
            readonly how_to_measure: string;
            /** Source Columns */
            readonly source_columns: readonly [
                string,
                ...string[]
            ];
            /**
             * @description Optional deterministic support-window expression over the declared source columns. It must return one scalar per window with the observation's declared summary operator. Omitted uses a direct single-column aggregation.
             * @default null
             */
            readonly computed_rule: components["schemas"]["WindowExpression"] | null;
            /**
             * Fill Null
             * @description Optional Polars null filling after aggregation on the sorted time grid within the selected data span. Use forward, backward, min, max, mean, zero, one, or a numeric constant. Fills every null, including explicit unknown readings. Omitted leaves nulls unknown. Forward carries the last value and leaves leading nulls unknown.
             * @default null
             */
            readonly fill_null: ("forward" | "backward" | "min" | "max" | "mean" | "zero" | "one") | number | null;
            /**
             * Fill Null Limit
             * @description Maximum consecutive nulls filled by forward/backward; omitted is unlimited. Only valid when fill_null is forward or backward.
             * @default null
             */
            readonly fill_null_limit: number | null;
        };
        /** @description A persistent construct identity survives changes to its display name. */
        readonly "ConstructId-Input": `construct:${string}`;
        /** @description A persistent construct identity survives changes to its display name. */
        readonly "ConstructId-Output": `construct:${string}`;
        /**
         * ConstructRef
         * @description A construct identity independent of its current display name or model revision.
         */
        readonly "ConstructRef-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "construct";
            readonly id: components["schemas"]["ConstructId-Output"];
        };
        /**
         * ConstructSpec
         * @description A specification of a theoretical entity in the scientific causal model.
         */
        readonly "ConstructSpec-Output": {
            /** @description Persistent identity. Preserve when revising or renaming. */
            readonly id: components["schemas"]["ConstructId-Output"];
            /**
             * Name
             * @description Construct name (e.g., 'stress', 'sleep_quality')
             */
            readonly name: string;
            /**
             * Description
             * @description What this theoretical construct represents
             */
            readonly description: string;
            /**
             * Indicators
             * @default []
             */
            readonly indicators: readonly components["schemas"]["IndicatorSpec-Output"][];
            /**
             * Dynamics
             * @default []
             */
            readonly dynamics: readonly components["schemas"]["DynamicsMechanismSpec-Output"][];
            /**
             * Coefficients
             * @default []
             */
            readonly coefficients: readonly components["schemas"]["CoefficientExpression-Output"][];
            /**
             * Innovation Family
             * @default gaussian
             * @enum {string}
             */
            readonly innovation_family: "gaussian" | "student_t";
            /**
             * @description Trajectory law in DynamicalModelSpec.distributions, with coordinates in its law_layouts entry. Exogenous trajectories use Delta and hold each value until the next point, including after the last point.
             * @default null
             */
            readonly distribution: components["schemas"]["DistributionId-Output"] | null;
            /** @description 'endogenous' means modeled, with or without parents; 'exogenous' means supplied by a deterministic trajectory law, without endogenous dynamics or noise. */
            readonly role: components["schemas"]["Role"];
            /** @description 'time_varying' (changes over time) or 'time_invariant' (fixed) */
            readonly temporal_status: components["schemas"]["TemporalStatus"];
        };
        /**
         * ContinueFitTarget[GitOid]
         * @description Continue the original target at the last retained draw of a prior fit.
         */
        readonly ContinueFitTarget_GitOid_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "continue";
            readonly fit_ref: components["schemas"]["GitOid-Output"];
        };
        /**
         * ContinueFitTarget[RevisionSelector]
         * @description Continue the original target at the last retained draw of a prior fit.
         */
        readonly ContinueFitTarget_RevisionSelector_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "continue";
            readonly fit_ref: components["schemas"]["RevisionSelector"];
        };
        /**
         * ContinuedInitializationDiagnostics
         * @description Starting values come from a previous fit; no initializer was executed.
         */
        readonly ContinuedInitializationDiagnostics: {
            /**
             * Source
             * @default continued
             * @constant
             */
            readonly source: "continued";
            /** Reused Tuning */
            readonly reused_tuning: boolean;
        };
        readonly ConvergenceAssessmentSubject: components["schemas"]["ConvergenceSubject"] | components["schemas"]["LatentConvergenceSubject"] | ("recorded_parameter_chains" | "recorded_latent_chains");
        /**
         * ConvergenceCheckpointReport
         * @description The passing native production check that ended this execution's retained phase.
         */
        readonly ConvergenceCheckpointReport: {
            /**
             * Draws
             * @description New retained draws per chain at the declared checkpoint.
             */
            readonly draws: number;
            readonly report: components["schemas"]["ConvergenceReport"];
        };
        /**
         * ConvergenceReport
         * @description Recorded-chain criteria cover parameters and independent latent coordinates.
         */
        readonly ConvergenceReport: {
            /** Findings */
            readonly findings: readonly Domain.Assessment<Domain.ConvergenceAssessmentSubject, Domain.NumericCriterionEvidence>[];
            /**
             * Warnings
             * @default []
             */
            readonly warnings: readonly string[];
            /**
             * Checked
             * @description Number of distinct parameter and construct references with convergence findings.
             */
            readonly checked: number;
            /** @description Convergence verdict with failed checks taking precedence over unevaluated checks. */
            readonly status: components["schemas"]["CheckStatus"];
            /**
             * Messages
             * @description Explanations for convergence checks that failed or could not be evaluated.
             */
            readonly messages: readonly string[];
        };
        /**
         * ConvergenceStoppingReport
         * @description Actual stopping point and its passing evidence for an opted-in execution.
         */
        readonly ConvergenceStoppingReport: {
            /**
             * Retained Draws
             * @description Actual newly retained draws per chain in this segment.
             */
            readonly retained_draws: number;
            /** @description Null when no eligible checkpoint passed, including zero-draw execution. */
            readonly checkpoint: components["schemas"]["ConvergenceCheckpointReport"] | null;
        };
        /**
         * ConvergenceStoppingSpec
         * @description Declared retained-draw checkpoints for the native production convergence gate.
         */
        readonly ConvergenceStoppingSpec: {
            /**
             * Checkpoint Draws
             * @description Check after each multiple of this many newly retained draws per chain.
             */
            readonly checkpoint_draws: number;
            /**
             * Min Draws
             * @description Minimum newly retained draws per chain before a passing checkpoint may stop this execution. The requested sample count remains the ceiling.
             */
            readonly min_draws: number;
        };
        /**
         * ConvergenceSubject
         * @description A convergence criterion on one stable scientific scalar.
         */
        readonly ConvergenceSubject: {
            readonly parameter: components["schemas"]["ParameterRef"];
            /** Label */
            readonly label: string;
        };
        /**
         * DataComparisonReport
         * @description Statistical comparison of two nonempty selections of immutable saved histories.
         *
         *     Compare data preparation revisions, observations with prior or posterior
         *     simulations, or simulations across model specifications, priors, fits and
         *     intervention scenarios.
         *
         *     Left/right name comparison sides, not temporal revisions or an editing language.
         *     Resolved references retain every selected replicate in request order; a history
         *     cannot occur twice within one side. Variables are ordered by indicator identity.
         *     Exogenous indicators supplied by a simulation's model are excluded.
         *
         *     One-versus-one comparisons expose point changes. One-versus-many comparisons
         *     evaluate predictive checks against the singleton reference. Many-versus-many
         *     comparisons retain per-history statistics without pairing draws or evaluating
         *     reference-based checks. IndicatorComparison owns the exact change semantics.
         *     Source definitions and complete observations belong to prepare_data/simulate
         *     results; this report owns only selection identity and newly computed evidence.
         */
        readonly DataComparisonReport: {
            /** Left */
            readonly left: readonly Domain.DataRef<Domain.GitOid, number>[];
            /** Right */
            readonly right: readonly Domain.DataRef<Domain.GitOid, number>[];
            /** Variables */
            readonly variables: readonly components["schemas"]["IndicatorComparison"][];
        };
        /**
         * DataDiffInput[GitOid]
         * @description Compare saved histories statistically; left/right do not imply before/after revisions.
         *
         *     DataComparisonReport owns selection and comparison semantics. The action reads
         *     retained observations without generating new draws or changing scientific state.
         */
        readonly DataDiffInput_GitOid_: {
            /** @description References on the left side. Each reference selects one replicate by index, or all its retained histories when the index is omitted. */
            readonly left_ref: Domain.DataSelection<Domain.GitOid>;
            /** @description References on the right side, using the same selection rule. */
            readonly right_ref: Domain.DataSelection<Domain.GitOid>;
        };
        /**
         * DataDiffInput[RevisionSelector]
         * @description Compare saved histories statistically; left/right do not imply before/after revisions.
         *
         *     DataComparisonReport owns selection and comparison semantics. The action reads
         *     retained observations without generating new draws or changing scientific state.
         */
        readonly DataDiffInput_RevisionSelector_: {
            /** @description References on the left side. Each reference selects one replicate by index, or all its retained histories when the index is omitted. */
            readonly left_ref: components["schemas"]["DataSelection_RevisionSelector_"];
            /** @description References on the right side, using the same selection rule. */
            readonly right_ref: components["schemas"]["DataSelection_RevisionSelector_"];
        };
        /**
         * DataDiffOutput
         * @description The computed comparison report; original histories remain in their producing results.
         */
        readonly DataDiffOutput: {
            /** @description Exact history selections, point changes, per-history statistics, compatibility findings and predictive checks. */
            readonly report: components["schemas"]["DataComparisonReport"];
        };
        /**
         * DataDiffRequest[GitOid]
         * @description Compare the saved observation histories; return ``DataDiffOutput``.
         */
        readonly DataDiffRequest_GitOid_: {
            /**
             * Action
             * @description Scientific action that owns this request or result.
             * @default data_diff
             * @constant
             */
            readonly action: "data_diff";
            /** @description Typed arguments of the scientific action. */
            readonly input: Domain.DataDiffInput<Domain.GitOid>;
            /**
             * Reasoning
             * @description Why the caller is taking this action and what goal it serves. Retained with the original call and shown at the top of its action log; excluded from call identity.
             * @default null
             */
            readonly reasoning: string | null;
        };
        /**
         * DataDiffRequest[RevisionSelector]
         * @description Compare the saved observation histories; return ``DataDiffOutput``.
         */
        readonly DataDiffRequest_RevisionSelector_: {
            /**
             * @description Scientific action that owns this request or result. (enum property replaced by openapi-typescript)
             * @enum {string}
             */
            readonly action: "data_diff";
            /** @description Typed arguments of the scientific action. */
            readonly input: components["schemas"]["DataDiffInput_RevisionSelector_"];
            /**
             * Reasoning
             * @description Why the caller is taking this action and what goal it serves. Retained with the original call and shown at the top of its action log; excluded from call identity.
             * @default null
             */
            readonly reasoning?: string | null;
        };
        readonly DataFinding: Domain.Evaluated<Domain.IndicatorRef | "dataset", string> | Domain.NotEvaluated<Domain.IndicatorRef | "dataset">;
        /**
         * DataPoint
         * @description One recorded anchor, support and value; a null value is a present but missing observation.
         *
         *     Row absence, represented by Added or Removed, is distinct from a present point's null value.
         */
        readonly DataPoint: {
            /**
             * Anchor Time
             * Format: date-time
             */
            readonly anchor_time: string;
            /** Support Start */
            readonly support_start: string | null;
            /** Support End */
            readonly support_end: string | null;
            /** Value */
            readonly value: number | null;
        };
        /**
         * DataPreparationResult
         * @description Preparation artifacts and the measurements actually retained by extraction.
         */
        readonly DataPreparationResult: {
            /**
             * Workers
             * @default []
             */
            readonly workers: readonly components["schemas"]["ExtractionWorkerResult"][];
            /**
             * Extraction Reused
             * @default null
             */
            readonly extraction_reused: number | null;
        };
        /**
         * DataPreparationSpec
         * @description The model-owned observation definitions resolved for extraction.
         */
        readonly DataPreparationSpec: {
            /** Default Window */
            readonly default_window: string;
            /** Variables */
            readonly variables: readonly [
                components["schemas"]["DataVariableSpec"],
                ...components["schemas"]["DataVariableSpec"][]
            ];
            /**
             * Context
             * @description Optional context for interpreting the source data.
             * @default
             */
            readonly context: string;
        };
        /**
         * DataProfileReport
         * @description Model-independent empirical measurements and data-quality findings.
         */
        readonly DataProfileReport: {
            /** Indicators */
            readonly indicators: Readonly<Partial<Record<components["schemas"]["IndicatorId-Output"], components["schemas"]["IndicatorAudit"]>>>;
            /** Findings */
            readonly findings: readonly components["schemas"]["DataFinding"][];
        };
        /**
         * DataRef[GitOid, Annotated[Union[int, NoneType], FieldInfo(annotation=NoneType, required=False, default=None)]]
         * @description A prepared-data or simulation gitref with a selection of recorded histories.
         *
         *     An integer selects one history by its zero-based index: zero for prepared user
         *     data, or less than the saved replicate count for a simulation. None selects
         *     every retained history when the consuming action permits it.
         */
        readonly DataRef_GitOid_Annotated_Union_int__NoneType___FieldInfo_annotation_NoneType__required_False__default_None___: {
            readonly revision: components["schemas"]["GitOid-Output"];
            /**
             * Replicate Index
             * @default null
             */
            readonly replicate_index: number | null;
        };
        /**
         * DataRef[GitOid, int]
         * @description A prepared-data or simulation gitref with a selection of recorded histories.
         *
         *     An integer selects one history by its zero-based index: zero for prepared user
         *     data, or less than the saved replicate count for a simulation. None selects
         *     every retained history when the consuming action permits it.
         */
        readonly DataRef_GitOid_int_: {
            readonly revision: components["schemas"]["GitOid-Output"];
            /** Replicate Index */
            readonly replicate_index: number;
        };
        /**
         * DataRef[RevisionSelector, Annotated[Union[int, NoneType], FieldInfo(annotation=NoneType, required=False, default=None)]]
         * @description A prepared-data or simulation gitref with a selection of recorded histories.
         *
         *     An integer selects one history by its zero-based index: zero for prepared user
         *     data, or less than the saved replicate count for a simulation. None selects
         *     every retained history when the consuming action permits it.
         */
        readonly DataRef_RevisionSelector_Annotated_Union_int__NoneType___FieldInfo_annotation_NoneType__required_False__default_None___: {
            readonly revision: components["schemas"]["RevisionSelector"];
            /**
             * Replicate Index
             * @default null
             */
            readonly replicate_index?: number | null;
        };
        /**
         * DataRef[RevisionSelector, int]
         * @description A prepared-data or simulation gitref with a selection of recorded histories.
         *
         *     An integer selects one history by its zero-based index: zero for prepared user
         *     data, or less than the saved replicate count for a simulation. None selects
         *     every retained history when the consuming action permits it.
         */
        readonly DataRef_RevisionSelector_int_: {
            readonly revision: components["schemas"]["RevisionSelector"];
            /** Replicate Index */
            readonly replicate_index: number;
        };
        /** @description One or more saved observation-history references. */
        readonly DataSelection_GitOid_: readonly Domain.DataRef<Domain.GitOid, number | null>[];
        /** @description One or more saved observation-history references. */
        readonly DataSelection_RevisionSelector_: readonly components["schemas"]["DataRef_RevisionSelector_Annotated_Union_int__NoneType___FieldInfo_annotation_NoneType__required_False__default_None___"][];
        readonly DataStatisticComparison: components["schemas"]["ScalarStatisticComparison"] | components["schemas"]["ProportionComparison"];
        /**
         * DataVariableSpec
         * @description Compose an observed variable with its data-owned extraction instructions.
         */
        readonly DataVariableSpec: {
            readonly observation: Domain.ObservationSpec<Domain.ObservationWindow | null>;
            readonly extraction: components["schemas"]["ExtractionSpec-Output"];
        };
        /**
         * DeltaLawSpec[Expression]
         * @description The Delta conditional law.
         */
        readonly "DeltaLawSpec_Expression_-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "Delta";
            readonly v: components["schemas"]["Expression-Output"];
        };
        /**
         * DensityCurve
         * @description Aligned density ordinates; the owning field distinguishes PDF samples from histogram heights.
         */
        readonly DensityCurve: {
            /**
             * X
             * @default []
             */
            readonly x: readonly number[];
            /**
             * Density
             * @default []
             */
            readonly density: readonly number[];
        };
        /** @enum {string} */
        readonly DiagnosticPhase: "warmup" | "post_warmup";
        readonly "DistributionId-Input": `distribution:${string}`;
        readonly "DistributionId-Output": `distribution:${string}`;
        /**
         * DriftMechanismSpec
         * @description An additive drift contribution on a construct or directed edge.
         */
        readonly "DriftMechanismSpec-Output": {
            readonly id: components["schemas"]["MechanismId-Output"];
            readonly expression: components["schemas"]["Expression-Output"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "drift";
        };
        /**
         * DsmcNodeWeightDiagnostics
         * @description One tree node before resampling; level zero is a leaf, stop is exclusive.
         *
         *     Entirely padded nodes are omitted. A partial node's range contains only real
         *     time indices, while its candidate count includes the algorithm's padding.
         */
        readonly DsmcNodeWeightDiagnostics: {
            /** Level */
            readonly level: number;
            /** Start */
            readonly start: number;
            /** Stop */
            readonly stop: number;
            readonly weights: components["schemas"]["ParticleWeightDiagnostics"];
        };
        /**
         * DsmcSpec
         * @description Conditional dSMC auxiliary law, particle proposal, and assigned scale policy.
         *
         *     Each sweep refreshes an auxiliary point near the current path, draws candidates
         *     around it at every time step, and selects a new path through the conditional dSMC
         *     tree (Corenflos, Chopin & Särkkä 2022, Algorithm 4), whose depth grows with log T.
         *     The auxiliary law is Particle-RWM or Particle-aMALA (Corenflos & Finke 2024).
         */
        readonly "DsmcSpec-Input": {
            /**
             * @description Auxiliary law for conditional dSMC. particle_rwm (kappa 0) centres the auxiliary point on the current state. particle_amala (Algorithm 3) drifts it along the gradient of the local prior and observation terms; particle_amala_plus (Algorithm 5) adds the next transition. With gradients the scale need only shrink like D^(-1/3) as the state dimension D grows, against D^(-1) without (Corenflos & Finke 2024, figure 1). There aMALA+ had up to three times Particle-RWM's ESS and twice aMALA's, for twice aMALA's gradients (section 5.4). On a GPU, Corenflos & Särkkä (2025) found gradient and plain proposals equally efficient per second (section 4.1), with gradients more robust to sticky dynamics and very informative observations (section 4.4). (enum property replaced by openapi-typescript)
             * @enum {string}
             */
            readonly method: "particle_amala" | "particle_amala_plus" | "particle_rwm";
            /**
             * Num Particles
             * @description Particles per sweep, including the current path. More particles give each time step more candidates, but each dSMC merge scores N x N candidate pairs, so memory and work per merge grow as N^2 (Corenflos, Chopin & Särkkä 2022, section 4).
             * @default 64
             */
            readonly num_particles?: number;
            /** @description Per-time-step scale: adapted towards a movement target, or fixed. */
            readonly latent_step?: components["schemas"]["LatentStepSpec-Input"];
            /**
             * @description State coordinates proposed per sweep, drawn uniformly from the free ones; the others keep their current paths at every time step. 1 proposes one coordinate's whole path, all proposes the full state. Importance weights degenerate as the proposed dimension grows, which blocking counters (Corenflos & Särkkä 2025, section 3.1): smaller blocks keep larger scales at the same movement rate, but each coordinate is proposed only every D/k sweeps, and coupled coordinates move slowly when proposed apart.
             * @default 1
             */
            readonly latent_block_coords?: components["schemas"]["LatentBlockSize"];
            /**
             * Pilot Mixture
             * @description Mix two fixed Gaussians into every candidate draw: 85% of candidates come from the local auxiliary proposal, 10% from a Gaussian at the initializer's mean path with a quarter of its variance, and 5% from one with four times its variance. Weights use the full mixture density, so the target is unchanged (a defensive mixture, Hesterberg 1995). It adds candidates away from the current path, as Corenflos, Chopin & Särkkä (2022, section 4) suggest for separable proposals. The auxiliary factor still scores each candidate against the refreshed point, so at a small scale distant pilot draws are rarely kept. The moments come from the num_chains starting paths.
             * @default true
             */
            readonly pilot_mixture?: boolean;
        };
        /**
         * DsmcSpec
         * @description Conditional dSMC auxiliary law, particle proposal, and assigned scale policy.
         *
         *     Each sweep refreshes an auxiliary point near the current path, draws candidates
         *     around it at every time step, and selects a new path through the conditional dSMC
         *     tree (Corenflos, Chopin & Särkkä 2022, Algorithm 4), whose depth grows with log T.
         *     The auxiliary law is Particle-RWM or Particle-aMALA (Corenflos & Finke 2024).
         */
        readonly "DsmcSpec-Output": {
            /**
             * @description Auxiliary law for conditional dSMC. particle_rwm (kappa 0) centres the auxiliary point on the current state. particle_amala (Algorithm 3) drifts it along the gradient of the local prior and observation terms; particle_amala_plus (Algorithm 5) adds the next transition. With gradients the scale need only shrink like D^(-1/3) as the state dimension D grows, against D^(-1) without (Corenflos & Finke 2024, figure 1). There aMALA+ had up to three times Particle-RWM's ESS and twice aMALA's, for twice aMALA's gradients (section 5.4). On a GPU, Corenflos & Särkkä (2025) found gradient and plain proposals equally efficient per second (section 4.1), with gradients more robust to sticky dynamics and very informative observations (section 4.4). (enum property replaced by openapi-typescript)
             * @enum {string}
             */
            readonly method: "particle_amala" | "particle_amala_plus" | "particle_rwm";
            /**
             * Num Particles
             * @description Particles per sweep, including the current path. More particles give each time step more candidates, but each dSMC merge scores N x N candidate pairs, so memory and work per merge grow as N^2 (Corenflos, Chopin & Särkkä 2022, section 4).
             * @default 64
             */
            readonly num_particles: number;
            /** @description Per-time-step scale: adapted towards a movement target, or fixed. */
            readonly latent_step: components["schemas"]["LatentStepSpec-Output"];
            /**
             * @description State coordinates proposed per sweep, drawn uniformly from the free ones; the others keep their current paths at every time step. 1 proposes one coordinate's whole path, all proposes the full state. Importance weights degenerate as the proposed dimension grows, which blocking counters (Corenflos & Särkkä 2025, section 3.1): smaller blocks keep larger scales at the same movement rate, but each coordinate is proposed only every D/k sweeps, and coupled coordinates move slowly when proposed apart.
             * @default 1
             */
            readonly latent_block_coords: components["schemas"]["LatentBlockSize"];
            /**
             * Pilot Mixture
             * @description Mix two fixed Gaussians into every candidate draw: 85% of candidates come from the local auxiliary proposal, 10% from a Gaussian at the initializer's mean path with a quarter of its variance, and 5% from one with four times its variance. Weights use the full mixture density, so the target is unchanged (a defensive mixture, Hesterberg 1995). It adds candidates away from the current path, as Corenflos, Chopin & Särkkä (2022, section 4) suggest for separable proposals. The auxiliary factor still scores each candidate against the refreshed point, so at a small scale distant pilot draws are rarely kept. The moments come from the num_chains starting paths.
             * @default true
             */
            readonly pilot_mixture: boolean;
        };
        /**
         * DsmcWeightDiagnostics
         * @description dSMC tree and terminal weight diagnostics.
         *
         *     Tree nodes are ordered by increasing level, then start time. Terminal weights
         *     include the exact full-path correction and precede the modified Gibbs move;
         *     their reference probability is not that move's rejection probability.
         */
        readonly DsmcWeightDiagnostics: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "dsmc";
            readonly phase: components["schemas"]["DiagnosticPhase"];
            /** Tree */
            readonly tree: readonly components["schemas"]["DsmcNodeWeightDiagnostics"][];
            readonly terminal: components["schemas"]["ParticleWeightDiagnostics"];
        };
        /**
         * DynamicalModelSpec
         * @description One identity-addressed scientific document, usable for creation and partial edits.
         *
         *     Omission carries no update. Null entity entries are deletion instructions. The
         *     editing boundary materializes and checks the complete document before publication.
         *     Scientific consumers resolve its owned entities at that boundary.
         */
        readonly "DynamicalModelSpec-Input": {
            readonly constructs?: Readonly<Partial<Record<components["schemas"]["ConstructId-Input"], {
                /** @description Construct name (e.g., 'stress', 'sleep_quality') */
                readonly name?: string;
                /** @description What this theoretical construct represents */
                readonly description?: string;
                readonly indicators?: Readonly<Partial<Record<components["schemas"]["IndicatorId-Input"], {
                    readonly observation?: {
                        /** @description Indicator name (e.g., 'hrv', 'self_reported_stress') */
                        readonly name?: string;
                        readonly measurement?: {
                            /** @enum {string} */
                            readonly dtype?: "continuous" | "binary" | "count";
                        } | {
                            /** @enum {string} */
                            readonly dtype?: "ordinal" | "categorical";
                            /** @description At least two distinct level labels, coded 0..K-1 in this order. Ordinal levels run from lowest to highest (e.g., ['low', 'medium', 'high']); categorical levels list every category (e.g., ['home', 'work', 'other']). */
                            readonly levels?: readonly [
                                string,
                                string,
                                ...string[]
                            ];
                        };
                        /**
                         * @description How measurements within a support window are reduced to one observation.
                         * @enum {string}
                         */
                        readonly aggregation?: "first" | "last" | "sum" | "count" | "mean" | "std";
                        /** @description Optional support window: positive fixed units s, m, h, d or w (for example '2w'), or whole calendar months/years parsed by pytimeparse2 (for example '3mo', '2 years', '1y6mo'). Calendar windows align to whole-month boundaries anchored to January 1970 and retain their actual lengths, including leap days. Resolved by the preparation window or the generative model clock. */
                        readonly observation_window?: string | null;
                    };
                    readonly likelihood?: {
                        readonly law?: {
                            /** @constant */
                            readonly distribution?: "Delta";
                            readonly v?: {
                                /** @constant */
                                readonly kind?: "literal";
                                readonly value?: number;
                            } | {
                                /** @constant */
                                readonly kind?: "state";
                                readonly construct_id?: `construct:${string}`;
                            } | {
                                /** @constant */
                                readonly kind?: "coefficient";
                                /** @enum {string} */
                                readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
                                readonly value?: number | `parameter:${string}` | null;
                                /** @description Additional constructs participating in this coefficient use. */
                                readonly construct_ids?: readonly `construct:${string}`[];
                            } | {
                                /** @constant */
                                readonly kind?: "binary";
                                /** @enum {string} */
                                readonly operator?: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                                readonly right?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                            } | {
                                /** @constant */
                                readonly kind?: "call";
                                /** @enum {string} */
                                readonly function?: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments?: readonly components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"][];
                            };
                        } | {
                            /** @constant */
                            readonly distribution?: "Normal";
                            readonly loc?: {
                                /** @constant */
                                readonly kind?: "literal";
                                readonly value?: number;
                            } | {
                                /** @constant */
                                readonly kind?: "state";
                                readonly construct_id?: `construct:${string}`;
                            } | {
                                /** @constant */
                                readonly kind?: "coefficient";
                                /** @enum {string} */
                                readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
                                readonly value?: number | `parameter:${string}` | null;
                                /** @description Additional constructs participating in this coefficient use. */
                                readonly construct_ids?: readonly `construct:${string}`[];
                            } | {
                                /** @constant */
                                readonly kind?: "binary";
                                /** @enum {string} */
                                readonly operator?: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                                readonly right?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                            } | {
                                /** @constant */
                                readonly kind?: "call";
                                /** @enum {string} */
                                readonly function?: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments?: readonly components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"][];
                            };
                            readonly scale?: {
                                /** @constant */
                                readonly kind?: "literal";
                                readonly value?: number;
                            } | {
                                /** @constant */
                                readonly kind?: "state";
                                readonly construct_id?: `construct:${string}`;
                            } | {
                                /** @constant */
                                readonly kind?: "coefficient";
                                /** @enum {string} */
                                readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
                                readonly value?: number | `parameter:${string}` | null;
                                /** @description Additional constructs participating in this coefficient use. */
                                readonly construct_ids?: readonly `construct:${string}`[];
                            } | {
                                /** @constant */
                                readonly kind?: "binary";
                                /** @enum {string} */
                                readonly operator?: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                                readonly right?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                            } | {
                                /** @constant */
                                readonly kind?: "call";
                                /** @enum {string} */
                                readonly function?: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments?: readonly components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"][];
                            };
                        } | {
                            /** @constant */
                            readonly distribution?: "StudentT";
                            readonly df?: {
                                /** @constant */
                                readonly kind?: "literal";
                                readonly value?: number;
                            } | {
                                /** @constant */
                                readonly kind?: "state";
                                readonly construct_id?: `construct:${string}`;
                            } | {
                                /** @constant */
                                readonly kind?: "coefficient";
                                /** @enum {string} */
                                readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
                                readonly value?: number | `parameter:${string}` | null;
                                /** @description Additional constructs participating in this coefficient use. */
                                readonly construct_ids?: readonly `construct:${string}`[];
                            } | {
                                /** @constant */
                                readonly kind?: "binary";
                                /** @enum {string} */
                                readonly operator?: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                                readonly right?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                            } | {
                                /** @constant */
                                readonly kind?: "call";
                                /** @enum {string} */
                                readonly function?: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments?: readonly components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"][];
                            };
                            readonly loc?: {
                                /** @constant */
                                readonly kind?: "literal";
                                readonly value?: number;
                            } | {
                                /** @constant */
                                readonly kind?: "state";
                                readonly construct_id?: `construct:${string}`;
                            } | {
                                /** @constant */
                                readonly kind?: "coefficient";
                                /** @enum {string} */
                                readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
                                readonly value?: number | `parameter:${string}` | null;
                                /** @description Additional constructs participating in this coefficient use. */
                                readonly construct_ids?: readonly `construct:${string}`[];
                            } | {
                                /** @constant */
                                readonly kind?: "binary";
                                /** @enum {string} */
                                readonly operator?: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                                readonly right?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                            } | {
                                /** @constant */
                                readonly kind?: "call";
                                /** @enum {string} */
                                readonly function?: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments?: readonly components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"][];
                            };
                            readonly scale?: {
                                /** @constant */
                                readonly kind?: "literal";
                                readonly value?: number;
                            } | {
                                /** @constant */
                                readonly kind?: "state";
                                readonly construct_id?: `construct:${string}`;
                            } | {
                                /** @constant */
                                readonly kind?: "coefficient";
                                /** @enum {string} */
                                readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
                                readonly value?: number | `parameter:${string}` | null;
                                /** @description Additional constructs participating in this coefficient use. */
                                readonly construct_ids?: readonly `construct:${string}`[];
                            } | {
                                /** @constant */
                                readonly kind?: "binary";
                                /** @enum {string} */
                                readonly operator?: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                                readonly right?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                            } | {
                                /** @constant */
                                readonly kind?: "call";
                                /** @enum {string} */
                                readonly function?: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments?: readonly components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"][];
                            };
                        } | {
                            /** @constant */
                            readonly distribution?: "Poisson";
                            readonly rate?: {
                                /** @constant */
                                readonly kind?: "literal";
                                readonly value?: number;
                            } | {
                                /** @constant */
                                readonly kind?: "state";
                                readonly construct_id?: `construct:${string}`;
                            } | {
                                /** @constant */
                                readonly kind?: "coefficient";
                                /** @enum {string} */
                                readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
                                readonly value?: number | `parameter:${string}` | null;
                                /** @description Additional constructs participating in this coefficient use. */
                                readonly construct_ids?: readonly `construct:${string}`[];
                            } | {
                                /** @constant */
                                readonly kind?: "binary";
                                /** @enum {string} */
                                readonly operator?: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                                readonly right?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                            } | {
                                /** @constant */
                                readonly kind?: "call";
                                /** @enum {string} */
                                readonly function?: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments?: readonly components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"][];
                            };
                        } | {
                            /** @constant */
                            readonly distribution?: "Gamma";
                            readonly concentration?: {
                                /** @constant */
                                readonly kind?: "literal";
                                readonly value?: number;
                            } | {
                                /** @constant */
                                readonly kind?: "state";
                                readonly construct_id?: `construct:${string}`;
                            } | {
                                /** @constant */
                                readonly kind?: "coefficient";
                                /** @enum {string} */
                                readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
                                readonly value?: number | `parameter:${string}` | null;
                                /** @description Additional constructs participating in this coefficient use. */
                                readonly construct_ids?: readonly `construct:${string}`[];
                            } | {
                                /** @constant */
                                readonly kind?: "binary";
                                /** @enum {string} */
                                readonly operator?: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                                readonly right?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                            } | {
                                /** @constant */
                                readonly kind?: "call";
                                /** @enum {string} */
                                readonly function?: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments?: readonly components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"][];
                            };
                            readonly rate?: {
                                /** @constant */
                                readonly kind?: "literal";
                                readonly value?: number;
                            } | {
                                /** @constant */
                                readonly kind?: "state";
                                readonly construct_id?: `construct:${string}`;
                            } | {
                                /** @constant */
                                readonly kind?: "coefficient";
                                /** @enum {string} */
                                readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
                                readonly value?: number | `parameter:${string}` | null;
                                /** @description Additional constructs participating in this coefficient use. */
                                readonly construct_ids?: readonly `construct:${string}`[];
                            } | {
                                /** @constant */
                                readonly kind?: "binary";
                                /** @enum {string} */
                                readonly operator?: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                                readonly right?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                            } | {
                                /** @constant */
                                readonly kind?: "call";
                                /** @enum {string} */
                                readonly function?: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments?: readonly components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"][];
                            };
                        } | {
                            /** @constant */
                            readonly distribution?: "BernoulliLogits";
                            readonly logits?: {
                                /** @constant */
                                readonly kind?: "literal";
                                readonly value?: number;
                            } | {
                                /** @constant */
                                readonly kind?: "state";
                                readonly construct_id?: `construct:${string}`;
                            } | {
                                /** @constant */
                                readonly kind?: "coefficient";
                                /** @enum {string} */
                                readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
                                readonly value?: number | `parameter:${string}` | null;
                                /** @description Additional constructs participating in this coefficient use. */
                                readonly construct_ids?: readonly `construct:${string}`[];
                            } | {
                                /** @constant */
                                readonly kind?: "binary";
                                /** @enum {string} */
                                readonly operator?: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                                readonly right?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                            } | {
                                /** @constant */
                                readonly kind?: "call";
                                /** @enum {string} */
                                readonly function?: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments?: readonly components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"][];
                            };
                        } | {
                            /** @constant */
                            readonly distribution?: "BernoulliProbs";
                            readonly probs?: {
                                /** @constant */
                                readonly kind?: "literal";
                                readonly value?: number;
                            } | {
                                /** @constant */
                                readonly kind?: "state";
                                readonly construct_id?: `construct:${string}`;
                            } | {
                                /** @constant */
                                readonly kind?: "coefficient";
                                /** @enum {string} */
                                readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
                                readonly value?: number | `parameter:${string}` | null;
                                /** @description Additional constructs participating in this coefficient use. */
                                readonly construct_ids?: readonly `construct:${string}`[];
                            } | {
                                /** @constant */
                                readonly kind?: "binary";
                                /** @enum {string} */
                                readonly operator?: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                                readonly right?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                            } | {
                                /** @constant */
                                readonly kind?: "call";
                                /** @enum {string} */
                                readonly function?: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments?: readonly components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"][];
                            };
                        } | {
                            /** @constant */
                            readonly distribution?: "NegativeBinomial2";
                            readonly mean?: {
                                /** @constant */
                                readonly kind?: "literal";
                                readonly value?: number;
                            } | {
                                /** @constant */
                                readonly kind?: "state";
                                readonly construct_id?: `construct:${string}`;
                            } | {
                                /** @constant */
                                readonly kind?: "coefficient";
                                /** @enum {string} */
                                readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
                                readonly value?: number | `parameter:${string}` | null;
                                /** @description Additional constructs participating in this coefficient use. */
                                readonly construct_ids?: readonly `construct:${string}`[];
                            } | {
                                /** @constant */
                                readonly kind?: "binary";
                                /** @enum {string} */
                                readonly operator?: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                                readonly right?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                            } | {
                                /** @constant */
                                readonly kind?: "call";
                                /** @enum {string} */
                                readonly function?: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments?: readonly components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"][];
                            };
                            readonly concentration?: {
                                /** @constant */
                                readonly kind?: "literal";
                                readonly value?: number;
                            } | {
                                /** @constant */
                                readonly kind?: "state";
                                readonly construct_id?: `construct:${string}`;
                            } | {
                                /** @constant */
                                readonly kind?: "coefficient";
                                /** @enum {string} */
                                readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
                                readonly value?: number | `parameter:${string}` | null;
                                /** @description Additional constructs participating in this coefficient use. */
                                readonly construct_ids?: readonly `construct:${string}`[];
                            } | {
                                /** @constant */
                                readonly kind?: "binary";
                                /** @enum {string} */
                                readonly operator?: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                                readonly right?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                            } | {
                                /** @constant */
                                readonly kind?: "call";
                                /** @enum {string} */
                                readonly function?: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments?: readonly components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"][];
                            };
                        } | {
                            /** @constant */
                            readonly distribution?: "Beta";
                            readonly concentration1?: {
                                /** @constant */
                                readonly kind?: "literal";
                                readonly value?: number;
                            } | {
                                /** @constant */
                                readonly kind?: "state";
                                readonly construct_id?: `construct:${string}`;
                            } | {
                                /** @constant */
                                readonly kind?: "coefficient";
                                /** @enum {string} */
                                readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
                                readonly value?: number | `parameter:${string}` | null;
                                /** @description Additional constructs participating in this coefficient use. */
                                readonly construct_ids?: readonly `construct:${string}`[];
                            } | {
                                /** @constant */
                                readonly kind?: "binary";
                                /** @enum {string} */
                                readonly operator?: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                                readonly right?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                            } | {
                                /** @constant */
                                readonly kind?: "call";
                                /** @enum {string} */
                                readonly function?: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments?: readonly components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"][];
                            };
                            readonly concentration0?: {
                                /** @constant */
                                readonly kind?: "literal";
                                readonly value?: number;
                            } | {
                                /** @constant */
                                readonly kind?: "state";
                                readonly construct_id?: `construct:${string}`;
                            } | {
                                /** @constant */
                                readonly kind?: "coefficient";
                                /** @enum {string} */
                                readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
                                readonly value?: number | `parameter:${string}` | null;
                                /** @description Additional constructs participating in this coefficient use. */
                                readonly construct_ids?: readonly `construct:${string}`[];
                            } | {
                                /** @constant */
                                readonly kind?: "binary";
                                /** @enum {string} */
                                readonly operator?: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                                readonly right?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                            } | {
                                /** @constant */
                                readonly kind?: "call";
                                /** @enum {string} */
                                readonly function?: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments?: readonly components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"][];
                            };
                        } | {
                            /** @constant */
                            readonly distribution?: "OrderedLogistic";
                            readonly predictor?: {
                                /** @constant */
                                readonly kind?: "literal";
                                readonly value?: number;
                            } | {
                                /** @constant */
                                readonly kind?: "state";
                                readonly construct_id?: `construct:${string}`;
                            } | {
                                /** @constant */
                                readonly kind?: "coefficient";
                                /** @enum {string} */
                                readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
                                readonly value?: number | `parameter:${string}` | null;
                                /** @description Additional constructs participating in this coefficient use. */
                                readonly construct_ids?: readonly `construct:${string}`[];
                            } | {
                                /** @constant */
                                readonly kind?: "binary";
                                /** @enum {string} */
                                readonly operator?: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                                readonly right?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                            } | {
                                /** @constant */
                                readonly kind?: "call";
                                /** @enum {string} */
                                readonly function?: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments?: readonly components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"][];
                            };
                            readonly cutpoints?: {
                                /** @constant */
                                readonly kind?: "literal";
                                readonly value?: number;
                            } | {
                                /** @constant */
                                readonly kind?: "state";
                                readonly construct_id?: `construct:${string}`;
                            } | {
                                /** @constant */
                                readonly kind?: "coefficient";
                                /** @enum {string} */
                                readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
                                readonly value?: number | `parameter:${string}` | null;
                                /** @description Additional constructs participating in this coefficient use. */
                                readonly construct_ids?: readonly `construct:${string}`[];
                            } | {
                                /** @constant */
                                readonly kind?: "binary";
                                /** @enum {string} */
                                readonly operator?: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                                readonly right?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                            } | {
                                /** @constant */
                                readonly kind?: "call";
                                /** @enum {string} */
                                readonly function?: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments?: readonly components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"][];
                            };
                        } | {
                            /** @constant */
                            readonly distribution?: "Categorical";
                            readonly logits?: {
                                /** @constant */
                                readonly kind?: "literal";
                                readonly value?: number;
                            } | {
                                /** @constant */
                                readonly kind?: "state";
                                readonly construct_id?: `construct:${string}`;
                            } | {
                                /** @constant */
                                readonly kind?: "coefficient";
                                /** @enum {string} */
                                readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
                                readonly value?: number | `parameter:${string}` | null;
                                /** @description Additional constructs participating in this coefficient use. */
                                readonly construct_ids?: readonly `construct:${string}`[];
                            } | {
                                /** @constant */
                                readonly kind?: "binary";
                                /** @enum {string} */
                                readonly operator?: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                                readonly right?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                            } | {
                                /** @constant */
                                readonly kind?: "call";
                                /** @enum {string} */
                                readonly function?: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments?: readonly components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"][];
                            };
                        };
                        /** @description Whether observations are mean-centered and scaled before fitting. */
                        readonly standardized?: boolean;
                        /** @description Why this conditional law was chosen for the indicator */
                        readonly reasoning?: string;
                        readonly sources?: readonly {
                            /** @description Title of the source (paper, meta-analysis, textbook, etc.) */
                            readonly title?: string;
                            /** @description URL of the source if available */
                            readonly url?: string | null;
                            /** @description Relevant excerpt or paraphrase from the source */
                            readonly snippet?: string;
                        }[];
                    } | null;
                    /**
                     * @description Whether the measurement increases or decreases with its owning construct.
                     * @enum {string}
                     */
                    readonly construct_polarity?: "positive" | "negative";
                } | null>>>;
                readonly dynamics?: Readonly<Partial<Record<components["schemas"]["MechanismId-Input"], ({
                    readonly expression?: {
                        /** @constant */
                        readonly kind?: "literal";
                        readonly value?: number;
                    } | {
                        /** @constant */
                        readonly kind?: "state";
                        readonly construct_id?: `construct:${string}`;
                    } | {
                        /** @constant */
                        readonly kind?: "coefficient";
                        /** @enum {string} */
                        readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                        /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
                        readonly value?: number | `parameter:${string}` | null;
                        /** @description Additional constructs participating in this coefficient use. */
                        readonly construct_ids?: readonly `construct:${string}`[];
                    } | {
                        /** @constant */
                        readonly kind?: "binary";
                        /** @enum {string} */
                        readonly operator?: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                        readonly left?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                        readonly right?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                    } | {
                        /** @constant */
                        readonly kind?: "call";
                        /** @enum {string} */
                        readonly function?: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                        readonly arguments?: readonly components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"][];
                    };
                    /** @constant */
                    readonly kind?: "drift";
                } | {
                    readonly expression?: {
                        /** @constant */
                        readonly kind?: "literal";
                        readonly value?: number;
                    } | {
                        /** @constant */
                        readonly kind?: "state";
                        readonly construct_id?: `construct:${string}`;
                    } | {
                        /** @constant */
                        readonly kind?: "coefficient";
                        /** @enum {string} */
                        readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                        /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
                        readonly value?: number | `parameter:${string}` | null;
                        /** @description Additional constructs participating in this coefficient use. */
                        readonly construct_ids?: readonly `construct:${string}`[];
                    } | {
                        /** @constant */
                        readonly kind?: "binary";
                        /** @enum {string} */
                        readonly operator?: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                        readonly left?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                        readonly right?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                    } | {
                        /** @constant */
                        readonly kind?: "call";
                        /** @enum {string} */
                        readonly function?: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                        readonly arguments?: readonly components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"][];
                    };
                    /** @constant */
                    readonly kind?: "potential";
                }) | null>>>;
                readonly coefficients?: readonly {
                    /** @constant */
                    readonly kind?: "coefficient";
                    /** @enum {string} */
                    readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                    /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
                    readonly value?: number | `parameter:${string}` | null;
                    /** @description Additional constructs participating in this coefficient use. */
                    readonly construct_ids?: readonly `construct:${string}`[];
                }[];
                /** @enum {string} */
                readonly innovation_family?: "gaussian" | "student_t";
                /** @description Trajectory law in DynamicalModelSpec.distributions, with coordinates in its law_layouts entry. Exogenous trajectories use Delta and hold each value until the next point, including after the last point. */
                readonly distribution?: `distribution:${string}` | null;
                /**
                 * @description Whether a construct is modeled as endogenous or supplied as an exogenous input.
                 * @enum {string}
                 */
                readonly role?: "endogenous" | "exogenous";
                /**
                 * @description Temporal status states whether a construct varies within the individual over time.
                 * @enum {string}
                 */
                readonly temporal_status?: "time_varying" | "time_invariant";
            } | null>>>;
            readonly edges?: Readonly<Partial<Record<components["schemas"]["EdgeId-Input"], {
                readonly mechanisms?: Readonly<Partial<Record<components["schemas"]["MechanismId-Input"], {
                    readonly expression?: {
                        /** @constant */
                        readonly kind?: "literal";
                        readonly value?: number;
                    } | {
                        /** @constant */
                        readonly kind?: "state";
                        readonly construct_id?: `construct:${string}`;
                    } | {
                        /** @constant */
                        readonly kind?: "coefficient";
                        /** @enum {string} */
                        readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                        /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
                        readonly value?: number | `parameter:${string}` | null;
                        /** @description Additional constructs participating in this coefficient use. */
                        readonly construct_ids?: readonly `construct:${string}`[];
                    } | {
                        /** @constant */
                        readonly kind?: "binary";
                        /** @enum {string} */
                        readonly operator?: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                        readonly left?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                        readonly right?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
                    } | {
                        /** @constant */
                        readonly kind?: "call";
                        /** @enum {string} */
                        readonly function?: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                        readonly arguments?: readonly components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"][];
                    };
                    /** @constant */
                    readonly kind?: "drift";
                } | null>>>;
                /** @description Persistent identity. Preserve when revising or renaming. */
                readonly cause?: components["schemas"]["ConstructId-Input"];
                /** @description Persistent identity. Preserve when revising or renaming. */
                readonly effect?: components["schemas"]["ConstructId-Input"];
                /** @description Theoretical justification for this causal link */
                readonly description?: string;
                /** @description Literature sources supporting this causal link */
                readonly sources?: readonly {
                    /** @description Title of the source (paper, meta-analysis, textbook, etc.) */
                    readonly title?: string;
                    /** @description URL of the source if available */
                    readonly url?: string | null;
                    /** @description Relevant excerpt or paraphrase from the source */
                    readonly snippet?: string;
                }[];
            } | null>>>;
            readonly parameters?: Readonly<Partial<Record<components["schemas"]["ParameterId-Input"], {
                /** @description Authored parameter label; relationships use its persistent ID */
                readonly name?: string;
                /** @description Human-readable description of what this parameter represents */
                readonly description?: string;
                readonly transform?: {
                    /** @constant */
                    readonly kind?: "identity";
                } | {
                    /** @constant */
                    readonly kind?: "dt_persistence_to_ct_decay";
                    readonly interval_days?: number | "model_clock";
                } | {
                    /** @constant */
                    readonly kind?: "dt_effect_to_ct_rate";
                    readonly interval_days?: number | "model_clock";
                } | {
                    /** @constant */
                    readonly kind?: "initial_state_correlation";
                };
                /** @description Membership in a native law in DynamicalModelSpec.distributions; may be joint. None means the law has not been assigned yet. */
                readonly distribution?: `distribution:${string}` | null;
                /** @description Why the authored prior law fits this quantity, and where its values come from. */
                readonly reasoning?: string | null;
                /** @description Evidence behind the authored prior law. */
                readonly sources?: readonly {
                    /** @description Title of the source (paper, meta-analysis, textbook, etc.) */
                    readonly title?: string;
                    /** @description URL of the source if available */
                    readonly url?: string | null;
                    /** @description Relevant excerpt or paraphrase from the source */
                    readonly snippet?: string;
                }[];
            } | null>>>;
            /** @description All explicit probability laws. Members are the parameters and constructs referring to each ID. Event coordinates are parameters by ID and element ID, then constructs by ID and time point. A scalar law belongs to one parameter and applies independently to its elements. */
            readonly distributions?: Readonly<Partial<Record<components["schemas"]["DistributionId-Input"], {
                readonly distribution?: string;
                readonly params?: {
                    readonly [key: string]: Domain.NumPyroValue;
                };
            } | null>>>;
            /** @description Scientific coordinates and production labels of each joint law, beside its native atoms. */
            readonly law_layouts?: Readonly<Partial<Record<components["schemas"]["DistributionId-Input"], {
                readonly parameters?: readonly (readonly [
                    components["schemas"]["ParameterId-Input"],
                    readonly components["schemas"]["ParameterElementId-Input"][]
                ])[];
                readonly constructs?: readonly `construct:${string}`[];
                readonly time_points?: readonly number[];
                /** @description Calendar instant of model day zero, or relative coordinates bound at execution. Fitting retains its calendar origin with the law. */
                readonly time_origin?: string | "relative";
                readonly labels?: Readonly<Partial<Record<components["schemas"]["ParameterElementId-Input"], string | null>>>;
            } | null>>>;
            readonly measurement_clock?: string | null;
        };
        /**
         * DynamicalModelSpec
         * @description One identity-addressed scientific document, usable for creation and partial edits.
         *
         *     Omission carries no update. Null entity entries are deletion instructions. The
         *     editing boundary materializes and checks the complete document before publication.
         *     Scientific consumers resolve its owned entities at that boundary.
         */
        readonly "DynamicalModelSpec-Output": {
            readonly constructs: Readonly<Partial<Record<components["schemas"]["ConstructId-Output"], {
                /** @description Construct name (e.g., 'stress', 'sleep_quality') */
                readonly name: string;
                /** @description What this theoretical construct represents */
                readonly description: string;
                readonly indicators: Readonly<Partial<Record<components["schemas"]["IndicatorId-Output"], {
                    readonly observation: {
                        /** @description Indicator name (e.g., 'hrv', 'self_reported_stress') */
                        readonly name: string;
                        readonly measurement: {
                            /** @enum {string} */
                            readonly dtype: "continuous" | "binary" | "count";
                        } | {
                            /** @enum {string} */
                            readonly dtype: "ordinal" | "categorical";
                            /** @description At least two distinct level labels, coded 0..K-1 in this order. Ordinal levels run from lowest to highest (e.g., ['low', 'medium', 'high']); categorical levels list every category (e.g., ['home', 'work', 'other']). */
                            readonly levels: readonly [
                                string,
                                string,
                                ...string[]
                            ];
                        };
                        /**
                         * @description How measurements within a support window are reduced to one observation.
                         * @enum {string}
                         */
                        readonly aggregation: "first" | "last" | "sum" | "count" | "mean" | "std";
                        /**
                         * @description Optional support window: positive fixed units s, m, h, d or w (for example '2w'), or whole calendar months/years parsed by pytimeparse2 (for example '3mo', '2 years', '1y6mo'). Calendar windows align to whole-month boundaries anchored to January 1970 and retain their actual lengths, including leap days. Resolved by the preparation window or the generative model clock.
                         * @default null
                         */
                        readonly observation_window: string | null;
                    };
                    /** @default null */
                    readonly likelihood: {
                        readonly law: {
                            /**
                             * @default Delta
                             * @constant
                             */
                            readonly distribution: "Delta";
                            readonly v: {
                                /**
                                 * @default literal
                                 * @constant
                                 */
                                readonly kind: "literal";
                                readonly value: number;
                            } | {
                                /**
                                 * @default state
                                 * @constant
                                 */
                                readonly kind: "state";
                                readonly construct_id: `construct:${string}`;
                            } | {
                                /**
                                 * @default coefficient
                                 * @constant
                                 */
                                readonly kind: "coefficient";
                                /** @enum {string} */
                                readonly role: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /**
                                 * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
                                 * @default null
                                 */
                                readonly value: number | `parameter:${string}` | null;
                                /**
                                 * @description Additional constructs participating in this coefficient use.
                                 * @default []
                                 */
                                readonly construct_ids: readonly `construct:${string}`[];
                            } | {
                                /**
                                 * @default binary
                                 * @constant
                                 */
                                readonly kind: "binary";
                                /** @enum {string} */
                                readonly operator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left: components["schemas"]["Expression-Output"];
                                readonly right: components["schemas"]["Expression-Output"];
                            } | {
                                /**
                                 * @default call
                                 * @constant
                                 */
                                readonly kind: "call";
                                /** @enum {string} */
                                readonly function: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments: readonly components["schemas"]["Expression-Output"][];
                            };
                        } | {
                            /**
                             * @default Normal
                             * @constant
                             */
                            readonly distribution: "Normal";
                            readonly loc: {
                                /**
                                 * @default literal
                                 * @constant
                                 */
                                readonly kind: "literal";
                                readonly value: number;
                            } | {
                                /**
                                 * @default state
                                 * @constant
                                 */
                                readonly kind: "state";
                                readonly construct_id: `construct:${string}`;
                            } | {
                                /**
                                 * @default coefficient
                                 * @constant
                                 */
                                readonly kind: "coefficient";
                                /** @enum {string} */
                                readonly role: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /**
                                 * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
                                 * @default null
                                 */
                                readonly value: number | `parameter:${string}` | null;
                                /**
                                 * @description Additional constructs participating in this coefficient use.
                                 * @default []
                                 */
                                readonly construct_ids: readonly `construct:${string}`[];
                            } | {
                                /**
                                 * @default binary
                                 * @constant
                                 */
                                readonly kind: "binary";
                                /** @enum {string} */
                                readonly operator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left: components["schemas"]["Expression-Output"];
                                readonly right: components["schemas"]["Expression-Output"];
                            } | {
                                /**
                                 * @default call
                                 * @constant
                                 */
                                readonly kind: "call";
                                /** @enum {string} */
                                readonly function: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments: readonly components["schemas"]["Expression-Output"][];
                            };
                            readonly scale: {
                                /**
                                 * @default literal
                                 * @constant
                                 */
                                readonly kind: "literal";
                                readonly value: number;
                            } | {
                                /**
                                 * @default state
                                 * @constant
                                 */
                                readonly kind: "state";
                                readonly construct_id: `construct:${string}`;
                            } | {
                                /**
                                 * @default coefficient
                                 * @constant
                                 */
                                readonly kind: "coefficient";
                                /** @enum {string} */
                                readonly role: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /**
                                 * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
                                 * @default null
                                 */
                                readonly value: number | `parameter:${string}` | null;
                                /**
                                 * @description Additional constructs participating in this coefficient use.
                                 * @default []
                                 */
                                readonly construct_ids: readonly `construct:${string}`[];
                            } | {
                                /**
                                 * @default binary
                                 * @constant
                                 */
                                readonly kind: "binary";
                                /** @enum {string} */
                                readonly operator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left: components["schemas"]["Expression-Output"];
                                readonly right: components["schemas"]["Expression-Output"];
                            } | {
                                /**
                                 * @default call
                                 * @constant
                                 */
                                readonly kind: "call";
                                /** @enum {string} */
                                readonly function: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments: readonly components["schemas"]["Expression-Output"][];
                            };
                        } | {
                            /**
                             * @default StudentT
                             * @constant
                             */
                            readonly distribution: "StudentT";
                            readonly df: {
                                /**
                                 * @default literal
                                 * @constant
                                 */
                                readonly kind: "literal";
                                readonly value: number;
                            } | {
                                /**
                                 * @default state
                                 * @constant
                                 */
                                readonly kind: "state";
                                readonly construct_id: `construct:${string}`;
                            } | {
                                /**
                                 * @default coefficient
                                 * @constant
                                 */
                                readonly kind: "coefficient";
                                /** @enum {string} */
                                readonly role: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /**
                                 * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
                                 * @default null
                                 */
                                readonly value: number | `parameter:${string}` | null;
                                /**
                                 * @description Additional constructs participating in this coefficient use.
                                 * @default []
                                 */
                                readonly construct_ids: readonly `construct:${string}`[];
                            } | {
                                /**
                                 * @default binary
                                 * @constant
                                 */
                                readonly kind: "binary";
                                /** @enum {string} */
                                readonly operator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left: components["schemas"]["Expression-Output"];
                                readonly right: components["schemas"]["Expression-Output"];
                            } | {
                                /**
                                 * @default call
                                 * @constant
                                 */
                                readonly kind: "call";
                                /** @enum {string} */
                                readonly function: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments: readonly components["schemas"]["Expression-Output"][];
                            };
                            readonly loc: {
                                /**
                                 * @default literal
                                 * @constant
                                 */
                                readonly kind: "literal";
                                readonly value: number;
                            } | {
                                /**
                                 * @default state
                                 * @constant
                                 */
                                readonly kind: "state";
                                readonly construct_id: `construct:${string}`;
                            } | {
                                /**
                                 * @default coefficient
                                 * @constant
                                 */
                                readonly kind: "coefficient";
                                /** @enum {string} */
                                readonly role: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /**
                                 * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
                                 * @default null
                                 */
                                readonly value: number | `parameter:${string}` | null;
                                /**
                                 * @description Additional constructs participating in this coefficient use.
                                 * @default []
                                 */
                                readonly construct_ids: readonly `construct:${string}`[];
                            } | {
                                /**
                                 * @default binary
                                 * @constant
                                 */
                                readonly kind: "binary";
                                /** @enum {string} */
                                readonly operator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left: components["schemas"]["Expression-Output"];
                                readonly right: components["schemas"]["Expression-Output"];
                            } | {
                                /**
                                 * @default call
                                 * @constant
                                 */
                                readonly kind: "call";
                                /** @enum {string} */
                                readonly function: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments: readonly components["schemas"]["Expression-Output"][];
                            };
                            readonly scale: {
                                /**
                                 * @default literal
                                 * @constant
                                 */
                                readonly kind: "literal";
                                readonly value: number;
                            } | {
                                /**
                                 * @default state
                                 * @constant
                                 */
                                readonly kind: "state";
                                readonly construct_id: `construct:${string}`;
                            } | {
                                /**
                                 * @default coefficient
                                 * @constant
                                 */
                                readonly kind: "coefficient";
                                /** @enum {string} */
                                readonly role: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /**
                                 * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
                                 * @default null
                                 */
                                readonly value: number | `parameter:${string}` | null;
                                /**
                                 * @description Additional constructs participating in this coefficient use.
                                 * @default []
                                 */
                                readonly construct_ids: readonly `construct:${string}`[];
                            } | {
                                /**
                                 * @default binary
                                 * @constant
                                 */
                                readonly kind: "binary";
                                /** @enum {string} */
                                readonly operator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left: components["schemas"]["Expression-Output"];
                                readonly right: components["schemas"]["Expression-Output"];
                            } | {
                                /**
                                 * @default call
                                 * @constant
                                 */
                                readonly kind: "call";
                                /** @enum {string} */
                                readonly function: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments: readonly components["schemas"]["Expression-Output"][];
                            };
                        } | {
                            /**
                             * @default Poisson
                             * @constant
                             */
                            readonly distribution: "Poisson";
                            readonly rate: {
                                /**
                                 * @default literal
                                 * @constant
                                 */
                                readonly kind: "literal";
                                readonly value: number;
                            } | {
                                /**
                                 * @default state
                                 * @constant
                                 */
                                readonly kind: "state";
                                readonly construct_id: `construct:${string}`;
                            } | {
                                /**
                                 * @default coefficient
                                 * @constant
                                 */
                                readonly kind: "coefficient";
                                /** @enum {string} */
                                readonly role: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /**
                                 * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
                                 * @default null
                                 */
                                readonly value: number | `parameter:${string}` | null;
                                /**
                                 * @description Additional constructs participating in this coefficient use.
                                 * @default []
                                 */
                                readonly construct_ids: readonly `construct:${string}`[];
                            } | {
                                /**
                                 * @default binary
                                 * @constant
                                 */
                                readonly kind: "binary";
                                /** @enum {string} */
                                readonly operator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left: components["schemas"]["Expression-Output"];
                                readonly right: components["schemas"]["Expression-Output"];
                            } | {
                                /**
                                 * @default call
                                 * @constant
                                 */
                                readonly kind: "call";
                                /** @enum {string} */
                                readonly function: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments: readonly components["schemas"]["Expression-Output"][];
                            };
                        } | {
                            /**
                             * @default Gamma
                             * @constant
                             */
                            readonly distribution: "Gamma";
                            readonly concentration: {
                                /**
                                 * @default literal
                                 * @constant
                                 */
                                readonly kind: "literal";
                                readonly value: number;
                            } | {
                                /**
                                 * @default state
                                 * @constant
                                 */
                                readonly kind: "state";
                                readonly construct_id: `construct:${string}`;
                            } | {
                                /**
                                 * @default coefficient
                                 * @constant
                                 */
                                readonly kind: "coefficient";
                                /** @enum {string} */
                                readonly role: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /**
                                 * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
                                 * @default null
                                 */
                                readonly value: number | `parameter:${string}` | null;
                                /**
                                 * @description Additional constructs participating in this coefficient use.
                                 * @default []
                                 */
                                readonly construct_ids: readonly `construct:${string}`[];
                            } | {
                                /**
                                 * @default binary
                                 * @constant
                                 */
                                readonly kind: "binary";
                                /** @enum {string} */
                                readonly operator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left: components["schemas"]["Expression-Output"];
                                readonly right: components["schemas"]["Expression-Output"];
                            } | {
                                /**
                                 * @default call
                                 * @constant
                                 */
                                readonly kind: "call";
                                /** @enum {string} */
                                readonly function: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments: readonly components["schemas"]["Expression-Output"][];
                            };
                            readonly rate: {
                                /**
                                 * @default literal
                                 * @constant
                                 */
                                readonly kind: "literal";
                                readonly value: number;
                            } | {
                                /**
                                 * @default state
                                 * @constant
                                 */
                                readonly kind: "state";
                                readonly construct_id: `construct:${string}`;
                            } | {
                                /**
                                 * @default coefficient
                                 * @constant
                                 */
                                readonly kind: "coefficient";
                                /** @enum {string} */
                                readonly role: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /**
                                 * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
                                 * @default null
                                 */
                                readonly value: number | `parameter:${string}` | null;
                                /**
                                 * @description Additional constructs participating in this coefficient use.
                                 * @default []
                                 */
                                readonly construct_ids: readonly `construct:${string}`[];
                            } | {
                                /**
                                 * @default binary
                                 * @constant
                                 */
                                readonly kind: "binary";
                                /** @enum {string} */
                                readonly operator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left: components["schemas"]["Expression-Output"];
                                readonly right: components["schemas"]["Expression-Output"];
                            } | {
                                /**
                                 * @default call
                                 * @constant
                                 */
                                readonly kind: "call";
                                /** @enum {string} */
                                readonly function: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments: readonly components["schemas"]["Expression-Output"][];
                            };
                        } | {
                            /**
                             * @default BernoulliLogits
                             * @constant
                             */
                            readonly distribution: "BernoulliLogits";
                            readonly logits: {
                                /**
                                 * @default literal
                                 * @constant
                                 */
                                readonly kind: "literal";
                                readonly value: number;
                            } | {
                                /**
                                 * @default state
                                 * @constant
                                 */
                                readonly kind: "state";
                                readonly construct_id: `construct:${string}`;
                            } | {
                                /**
                                 * @default coefficient
                                 * @constant
                                 */
                                readonly kind: "coefficient";
                                /** @enum {string} */
                                readonly role: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /**
                                 * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
                                 * @default null
                                 */
                                readonly value: number | `parameter:${string}` | null;
                                /**
                                 * @description Additional constructs participating in this coefficient use.
                                 * @default []
                                 */
                                readonly construct_ids: readonly `construct:${string}`[];
                            } | {
                                /**
                                 * @default binary
                                 * @constant
                                 */
                                readonly kind: "binary";
                                /** @enum {string} */
                                readonly operator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left: components["schemas"]["Expression-Output"];
                                readonly right: components["schemas"]["Expression-Output"];
                            } | {
                                /**
                                 * @default call
                                 * @constant
                                 */
                                readonly kind: "call";
                                /** @enum {string} */
                                readonly function: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments: readonly components["schemas"]["Expression-Output"][];
                            };
                        } | {
                            /**
                             * @default BernoulliProbs
                             * @constant
                             */
                            readonly distribution: "BernoulliProbs";
                            readonly probs: {
                                /**
                                 * @default literal
                                 * @constant
                                 */
                                readonly kind: "literal";
                                readonly value: number;
                            } | {
                                /**
                                 * @default state
                                 * @constant
                                 */
                                readonly kind: "state";
                                readonly construct_id: `construct:${string}`;
                            } | {
                                /**
                                 * @default coefficient
                                 * @constant
                                 */
                                readonly kind: "coefficient";
                                /** @enum {string} */
                                readonly role: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /**
                                 * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
                                 * @default null
                                 */
                                readonly value: number | `parameter:${string}` | null;
                                /**
                                 * @description Additional constructs participating in this coefficient use.
                                 * @default []
                                 */
                                readonly construct_ids: readonly `construct:${string}`[];
                            } | {
                                /**
                                 * @default binary
                                 * @constant
                                 */
                                readonly kind: "binary";
                                /** @enum {string} */
                                readonly operator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left: components["schemas"]["Expression-Output"];
                                readonly right: components["schemas"]["Expression-Output"];
                            } | {
                                /**
                                 * @default call
                                 * @constant
                                 */
                                readonly kind: "call";
                                /** @enum {string} */
                                readonly function: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments: readonly components["schemas"]["Expression-Output"][];
                            };
                        } | {
                            /**
                             * @default NegativeBinomial2
                             * @constant
                             */
                            readonly distribution: "NegativeBinomial2";
                            readonly mean: {
                                /**
                                 * @default literal
                                 * @constant
                                 */
                                readonly kind: "literal";
                                readonly value: number;
                            } | {
                                /**
                                 * @default state
                                 * @constant
                                 */
                                readonly kind: "state";
                                readonly construct_id: `construct:${string}`;
                            } | {
                                /**
                                 * @default coefficient
                                 * @constant
                                 */
                                readonly kind: "coefficient";
                                /** @enum {string} */
                                readonly role: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /**
                                 * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
                                 * @default null
                                 */
                                readonly value: number | `parameter:${string}` | null;
                                /**
                                 * @description Additional constructs participating in this coefficient use.
                                 * @default []
                                 */
                                readonly construct_ids: readonly `construct:${string}`[];
                            } | {
                                /**
                                 * @default binary
                                 * @constant
                                 */
                                readonly kind: "binary";
                                /** @enum {string} */
                                readonly operator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left: components["schemas"]["Expression-Output"];
                                readonly right: components["schemas"]["Expression-Output"];
                            } | {
                                /**
                                 * @default call
                                 * @constant
                                 */
                                readonly kind: "call";
                                /** @enum {string} */
                                readonly function: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments: readonly components["schemas"]["Expression-Output"][];
                            };
                            readonly concentration: {
                                /**
                                 * @default literal
                                 * @constant
                                 */
                                readonly kind: "literal";
                                readonly value: number;
                            } | {
                                /**
                                 * @default state
                                 * @constant
                                 */
                                readonly kind: "state";
                                readonly construct_id: `construct:${string}`;
                            } | {
                                /**
                                 * @default coefficient
                                 * @constant
                                 */
                                readonly kind: "coefficient";
                                /** @enum {string} */
                                readonly role: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /**
                                 * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
                                 * @default null
                                 */
                                readonly value: number | `parameter:${string}` | null;
                                /**
                                 * @description Additional constructs participating in this coefficient use.
                                 * @default []
                                 */
                                readonly construct_ids: readonly `construct:${string}`[];
                            } | {
                                /**
                                 * @default binary
                                 * @constant
                                 */
                                readonly kind: "binary";
                                /** @enum {string} */
                                readonly operator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left: components["schemas"]["Expression-Output"];
                                readonly right: components["schemas"]["Expression-Output"];
                            } | {
                                /**
                                 * @default call
                                 * @constant
                                 */
                                readonly kind: "call";
                                /** @enum {string} */
                                readonly function: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments: readonly components["schemas"]["Expression-Output"][];
                            };
                        } | {
                            /**
                             * @default Beta
                             * @constant
                             */
                            readonly distribution: "Beta";
                            readonly concentration1: {
                                /**
                                 * @default literal
                                 * @constant
                                 */
                                readonly kind: "literal";
                                readonly value: number;
                            } | {
                                /**
                                 * @default state
                                 * @constant
                                 */
                                readonly kind: "state";
                                readonly construct_id: `construct:${string}`;
                            } | {
                                /**
                                 * @default coefficient
                                 * @constant
                                 */
                                readonly kind: "coefficient";
                                /** @enum {string} */
                                readonly role: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /**
                                 * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
                                 * @default null
                                 */
                                readonly value: number | `parameter:${string}` | null;
                                /**
                                 * @description Additional constructs participating in this coefficient use.
                                 * @default []
                                 */
                                readonly construct_ids: readonly `construct:${string}`[];
                            } | {
                                /**
                                 * @default binary
                                 * @constant
                                 */
                                readonly kind: "binary";
                                /** @enum {string} */
                                readonly operator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left: components["schemas"]["Expression-Output"];
                                readonly right: components["schemas"]["Expression-Output"];
                            } | {
                                /**
                                 * @default call
                                 * @constant
                                 */
                                readonly kind: "call";
                                /** @enum {string} */
                                readonly function: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments: readonly components["schemas"]["Expression-Output"][];
                            };
                            readonly concentration0: {
                                /**
                                 * @default literal
                                 * @constant
                                 */
                                readonly kind: "literal";
                                readonly value: number;
                            } | {
                                /**
                                 * @default state
                                 * @constant
                                 */
                                readonly kind: "state";
                                readonly construct_id: `construct:${string}`;
                            } | {
                                /**
                                 * @default coefficient
                                 * @constant
                                 */
                                readonly kind: "coefficient";
                                /** @enum {string} */
                                readonly role: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /**
                                 * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
                                 * @default null
                                 */
                                readonly value: number | `parameter:${string}` | null;
                                /**
                                 * @description Additional constructs participating in this coefficient use.
                                 * @default []
                                 */
                                readonly construct_ids: readonly `construct:${string}`[];
                            } | {
                                /**
                                 * @default binary
                                 * @constant
                                 */
                                readonly kind: "binary";
                                /** @enum {string} */
                                readonly operator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left: components["schemas"]["Expression-Output"];
                                readonly right: components["schemas"]["Expression-Output"];
                            } | {
                                /**
                                 * @default call
                                 * @constant
                                 */
                                readonly kind: "call";
                                /** @enum {string} */
                                readonly function: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments: readonly components["schemas"]["Expression-Output"][];
                            };
                        } | {
                            /**
                             * @default OrderedLogistic
                             * @constant
                             */
                            readonly distribution: "OrderedLogistic";
                            readonly predictor: {
                                /**
                                 * @default literal
                                 * @constant
                                 */
                                readonly kind: "literal";
                                readonly value: number;
                            } | {
                                /**
                                 * @default state
                                 * @constant
                                 */
                                readonly kind: "state";
                                readonly construct_id: `construct:${string}`;
                            } | {
                                /**
                                 * @default coefficient
                                 * @constant
                                 */
                                readonly kind: "coefficient";
                                /** @enum {string} */
                                readonly role: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /**
                                 * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
                                 * @default null
                                 */
                                readonly value: number | `parameter:${string}` | null;
                                /**
                                 * @description Additional constructs participating in this coefficient use.
                                 * @default []
                                 */
                                readonly construct_ids: readonly `construct:${string}`[];
                            } | {
                                /**
                                 * @default binary
                                 * @constant
                                 */
                                readonly kind: "binary";
                                /** @enum {string} */
                                readonly operator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left: components["schemas"]["Expression-Output"];
                                readonly right: components["schemas"]["Expression-Output"];
                            } | {
                                /**
                                 * @default call
                                 * @constant
                                 */
                                readonly kind: "call";
                                /** @enum {string} */
                                readonly function: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments: readonly components["schemas"]["Expression-Output"][];
                            };
                            readonly cutpoints: {
                                /**
                                 * @default literal
                                 * @constant
                                 */
                                readonly kind: "literal";
                                readonly value: number;
                            } | {
                                /**
                                 * @default state
                                 * @constant
                                 */
                                readonly kind: "state";
                                readonly construct_id: `construct:${string}`;
                            } | {
                                /**
                                 * @default coefficient
                                 * @constant
                                 */
                                readonly kind: "coefficient";
                                /** @enum {string} */
                                readonly role: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /**
                                 * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
                                 * @default null
                                 */
                                readonly value: number | `parameter:${string}` | null;
                                /**
                                 * @description Additional constructs participating in this coefficient use.
                                 * @default []
                                 */
                                readonly construct_ids: readonly `construct:${string}`[];
                            } | {
                                /**
                                 * @default binary
                                 * @constant
                                 */
                                readonly kind: "binary";
                                /** @enum {string} */
                                readonly operator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left: components["schemas"]["Expression-Output"];
                                readonly right: components["schemas"]["Expression-Output"];
                            } | {
                                /**
                                 * @default call
                                 * @constant
                                 */
                                readonly kind: "call";
                                /** @enum {string} */
                                readonly function: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments: readonly components["schemas"]["Expression-Output"][];
                            };
                        } | {
                            /**
                             * @default Categorical
                             * @constant
                             */
                            readonly distribution: "Categorical";
                            readonly logits: {
                                /**
                                 * @default literal
                                 * @constant
                                 */
                                readonly kind: "literal";
                                readonly value: number;
                            } | {
                                /**
                                 * @default state
                                 * @constant
                                 */
                                readonly kind: "state";
                                readonly construct_id: `construct:${string}`;
                            } | {
                                /**
                                 * @default coefficient
                                 * @constant
                                 */
                                readonly kind: "coefficient";
                                /** @enum {string} */
                                readonly role: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                                /**
                                 * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
                                 * @default null
                                 */
                                readonly value: number | `parameter:${string}` | null;
                                /**
                                 * @description Additional constructs participating in this coefficient use.
                                 * @default []
                                 */
                                readonly construct_ids: readonly `construct:${string}`[];
                            } | {
                                /**
                                 * @default binary
                                 * @constant
                                 */
                                readonly kind: "binary";
                                /** @enum {string} */
                                readonly operator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                                readonly left: components["schemas"]["Expression-Output"];
                                readonly right: components["schemas"]["Expression-Output"];
                            } | {
                                /**
                                 * @default call
                                 * @constant
                                 */
                                readonly kind: "call";
                                /** @enum {string} */
                                readonly function: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                                readonly arguments: readonly components["schemas"]["Expression-Output"][];
                            };
                        };
                        /**
                         * @description Whether observations are mean-centered and scaled before fitting.
                         * @default false
                         */
                        readonly standardized: boolean;
                        /** @description Why this conditional law was chosen for the indicator */
                        readonly reasoning: string;
                        /** @default [] */
                        readonly sources: readonly {
                            /** @description Title of the source (paper, meta-analysis, textbook, etc.) */
                            readonly title: string;
                            /**
                             * @description URL of the source if available
                             * @default null
                             */
                            readonly url: string | null;
                            /** @description Relevant excerpt or paraphrase from the source */
                            readonly snippet: string;
                        }[];
                    } | null;
                    /**
                     * @description Whether the measurement increases or decreases with its owning construct.
                     * @enum {string}
                     */
                    readonly construct_polarity: "positive" | "negative";
                }>>>;
                readonly dynamics: Readonly<Partial<Record<components["schemas"]["MechanismId-Output"], {
                    readonly expression: {
                        /**
                         * @default literal
                         * @constant
                         */
                        readonly kind: "literal";
                        readonly value: number;
                    } | {
                        /**
                         * @default state
                         * @constant
                         */
                        readonly kind: "state";
                        readonly construct_id: `construct:${string}`;
                    } | {
                        /**
                         * @default coefficient
                         * @constant
                         */
                        readonly kind: "coefficient";
                        /** @enum {string} */
                        readonly role: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                        /**
                         * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
                         * @default null
                         */
                        readonly value: number | `parameter:${string}` | null;
                        /**
                         * @description Additional constructs participating in this coefficient use.
                         * @default []
                         */
                        readonly construct_ids: readonly `construct:${string}`[];
                    } | {
                        /**
                         * @default binary
                         * @constant
                         */
                        readonly kind: "binary";
                        /** @enum {string} */
                        readonly operator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                        readonly left: components["schemas"]["Expression-Output"];
                        readonly right: components["schemas"]["Expression-Output"];
                    } | {
                        /**
                         * @default call
                         * @constant
                         */
                        readonly kind: "call";
                        /** @enum {string} */
                        readonly function: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                        readonly arguments: readonly components["schemas"]["Expression-Output"][];
                    };
                    /**
                     * @default drift
                     * @constant
                     */
                    readonly kind: "drift";
                } | {
                    readonly expression: {
                        /**
                         * @default literal
                         * @constant
                         */
                        readonly kind: "literal";
                        readonly value: number;
                    } | {
                        /**
                         * @default state
                         * @constant
                         */
                        readonly kind: "state";
                        readonly construct_id: `construct:${string}`;
                    } | {
                        /**
                         * @default coefficient
                         * @constant
                         */
                        readonly kind: "coefficient";
                        /** @enum {string} */
                        readonly role: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                        /**
                         * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
                         * @default null
                         */
                        readonly value: number | `parameter:${string}` | null;
                        /**
                         * @description Additional constructs participating in this coefficient use.
                         * @default []
                         */
                        readonly construct_ids: readonly `construct:${string}`[];
                    } | {
                        /**
                         * @default binary
                         * @constant
                         */
                        readonly kind: "binary";
                        /** @enum {string} */
                        readonly operator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                        readonly left: components["schemas"]["Expression-Output"];
                        readonly right: components["schemas"]["Expression-Output"];
                    } | {
                        /**
                         * @default call
                         * @constant
                         */
                        readonly kind: "call";
                        /** @enum {string} */
                        readonly function: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                        readonly arguments: readonly components["schemas"]["Expression-Output"][];
                    };
                    /**
                     * @default potential
                     * @constant
                     */
                    readonly kind: "potential";
                }>>>;
                /** @default [] */
                readonly coefficients: readonly {
                    /**
                     * @default coefficient
                     * @constant
                     */
                    readonly kind: "coefficient";
                    /** @enum {string} */
                    readonly role: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                    /**
                     * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
                     * @default null
                     */
                    readonly value: number | `parameter:${string}` | null;
                    /**
                     * @description Additional constructs participating in this coefficient use.
                     * @default []
                     */
                    readonly construct_ids: readonly `construct:${string}`[];
                }[];
                /**
                 * @default gaussian
                 * @enum {string}
                 */
                readonly innovation_family: "gaussian" | "student_t";
                /**
                 * @description Trajectory law in DynamicalModelSpec.distributions, with coordinates in its law_layouts entry. Exogenous trajectories use Delta and hold each value until the next point, including after the last point.
                 * @default null
                 */
                readonly distribution: `distribution:${string}` | null;
                /**
                 * @description Whether a construct is modeled as endogenous or supplied as an exogenous input.
                 * @enum {string}
                 */
                readonly role: "endogenous" | "exogenous";
                /**
                 * @description Temporal status states whether a construct varies within the individual over time.
                 * @enum {string}
                 */
                readonly temporal_status: "time_varying" | "time_invariant";
            }>>>;
            readonly edges: Readonly<Partial<Record<components["schemas"]["EdgeId-Output"], {
                readonly mechanisms: Readonly<Partial<Record<components["schemas"]["MechanismId-Output"], {
                    readonly expression: {
                        /**
                         * @default literal
                         * @constant
                         */
                        readonly kind: "literal";
                        readonly value: number;
                    } | {
                        /**
                         * @default state
                         * @constant
                         */
                        readonly kind: "state";
                        readonly construct_id: `construct:${string}`;
                    } | {
                        /**
                         * @default coefficient
                         * @constant
                         */
                        readonly kind: "coefficient";
                        /** @enum {string} */
                        readonly role: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
                        /**
                         * @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned.
                         * @default null
                         */
                        readonly value: number | `parameter:${string}` | null;
                        /**
                         * @description Additional constructs participating in this coefficient use.
                         * @default []
                         */
                        readonly construct_ids: readonly `construct:${string}`[];
                    } | {
                        /**
                         * @default binary
                         * @constant
                         */
                        readonly kind: "binary";
                        /** @enum {string} */
                        readonly operator: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
                        readonly left: components["schemas"]["Expression-Output"];
                        readonly right: components["schemas"]["Expression-Output"];
                    } | {
                        /**
                         * @default call
                         * @constant
                         */
                        readonly kind: "call";
                        /** @enum {string} */
                        readonly function: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
                        readonly arguments: readonly components["schemas"]["Expression-Output"][];
                    };
                    /**
                     * @default drift
                     * @constant
                     */
                    readonly kind: "drift";
                }>>>;
                /** @description Persistent identity. Preserve when revising or renaming. */
                readonly cause: components["schemas"]["ConstructId-Output"];
                /** @description Persistent identity. Preserve when revising or renaming. */
                readonly effect: components["schemas"]["ConstructId-Output"];
                /** @description Theoretical justification for this causal link */
                readonly description: string;
                /** @description Literature sources supporting this causal link */
                readonly sources: readonly {
                    /** @description Title of the source (paper, meta-analysis, textbook, etc.) */
                    readonly title: string;
                    /**
                     * @description URL of the source if available
                     * @default null
                     */
                    readonly url: string | null;
                    /** @description Relevant excerpt or paraphrase from the source */
                    readonly snippet: string;
                }[];
            }>>>;
            readonly parameters: Readonly<Partial<Record<components["schemas"]["ParameterId-Output"], {
                /** @description Authored parameter label; relationships use its persistent ID */
                readonly name: string;
                /** @description Human-readable description of what this parameter represents */
                readonly description: string;
                readonly transform: {
                    /**
                     * @default identity
                     * @constant
                     */
                    readonly kind: "identity";
                } | {
                    /**
                     * @default dt_persistence_to_ct_decay
                     * @constant
                     */
                    readonly kind: "dt_persistence_to_ct_decay";
                    readonly interval_days: number | "model_clock";
                } | {
                    /**
                     * @default dt_effect_to_ct_rate
                     * @constant
                     */
                    readonly kind: "dt_effect_to_ct_rate";
                    readonly interval_days: number | "model_clock";
                } | {
                    /**
                     * @default initial_state_correlation
                     * @constant
                     */
                    readonly kind: "initial_state_correlation";
                };
                /**
                 * @description Membership in a native law in DynamicalModelSpec.distributions; may be joint. None means the law has not been assigned yet.
                 * @default null
                 */
                readonly distribution: `distribution:${string}` | null;
                /**
                 * @description Why the authored prior law fits this quantity, and where its values come from.
                 * @default null
                 */
                readonly reasoning: string | null;
                /**
                 * @description Evidence behind the authored prior law.
                 * @default []
                 */
                readonly sources: readonly {
                    /** @description Title of the source (paper, meta-analysis, textbook, etc.) */
                    readonly title: string;
                    /**
                     * @description URL of the source if available
                     * @default null
                     */
                    readonly url: string | null;
                    /** @description Relevant excerpt or paraphrase from the source */
                    readonly snippet: string;
                }[];
            }>>>;
            /** @description All explicit probability laws. Members are the parameters and constructs referring to each ID. Event coordinates are parameters by ID and element ID, then constructs by ID and time point. A scalar law belongs to one parameter and applies independently to its elements. */
            readonly distributions: Readonly<Partial<Record<components["schemas"]["DistributionId-Output"], Domain.NumPyroDistribution>>>;
            /** @description Scientific coordinates and production labels of each joint law, beside its native atoms. */
            readonly law_layouts: Readonly<Partial<Record<components["schemas"]["DistributionId-Output"], components["schemas"]["JointLawLayout-Output"]>>>;
            /** @default null */
            readonly measurement_clock: string | null;
        };
        /** @description A supported scalar operation composing two expressions. */
        readonly "DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__BinaryExpression-Input__1_": {
            /** @constant */
            readonly kind?: "binary";
            /** @enum {string} */
            readonly operator?: "add" | "subtract" | "multiply" | "divide" | "power" | "maximum";
            readonly left?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
            readonly right?: components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"];
        };
        readonly "DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_": {
            /** @constant */
            readonly kind?: "literal";
            readonly value?: number;
        } | {
            /** @constant */
            readonly kind?: "state";
            readonly construct_id?: `construct:${string}`;
        } | {
            /** @constant */
            readonly kind?: "coefficient";
            /** @enum {string} */
            readonly role?: "center" | "decay" | "quartic" | "intercept" | "weight" | "emax" | "ec50" | "exponent" | "loading" | "observation_intercept" | "observation_scale" | "degrees_of_freedom" | "shape" | "dispersion" | "concentration" | "cutpoint_base" | "cutpoint_gaps" | "category_intercepts" | "category_slopes" | "diffusion_scale" | "diffusion_loading" | "process_degrees_of_freedom" | "initial_mean" | "initial_scale" | "initial_correlation";
            /** @description Fixed coefficients are finite literals; uncertain coefficients reference a persistent parameter ID. Null leaves the operand unassigned. */
            readonly value?: number | `parameter:${string}` | null;
            /** @description Additional constructs participating in this coefficient use. */
            readonly construct_ids?: readonly `construct:${string}`[];
        } | components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__BinaryExpression-Input__1_"] | {
            /** @constant */
            readonly kind?: "call";
            /** @enum {string} */
            readonly function?: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
            readonly arguments?: readonly components["schemas"]["DynamicalModelSpecDocument___components_schemas_grassmoth__artifacts__model__expressions__Expression-Input__1_"][];
        };
        /** @description A dynamics mechanism declares one contribution to continuous-time drift. */
        readonly "DynamicsMechanismSpec-Output": components["schemas"]["DriftMechanismSpec-Output"] | components["schemas"]["PotentialMechanismSpec-Output"];
        /** @description A persistent edge identity identifies one authored causal relationship. */
        readonly "EdgeId-Input": `edge:${string}`;
        /** @description A persistent edge identity identifies one authored causal relationship. */
        readonly "EdgeId-Output": `edge:${string}`;
        /**
         * EdgeRef
         * @description A causal-edge identity independent of edits to its scientific definition.
         */
        readonly EdgeRef: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "edge";
            readonly id: components["schemas"]["EdgeId-Output"];
        };
        /**
         * EditModelInput[GitOid]
         * @description Create or revise a model from a question or model parent.
         */
        readonly EditModelInput_GitOid_: {
            /** @description Question or model revision to start from. A model parent supplies its pinned question. 'latest' selects the current model, otherwise the current question. An exact Git hash can select either parent without being the current head. */
            readonly parent_ref: components["schemas"]["GitOid-Input"];
            /** @description Scientific definitions merged by identity into a model parent, or into an empty model for a question parent. Omitted fields are retained; null entity entries delete their identities. The action retains the merged spec and its findings. */
            readonly dynamical_model_spec: components["schemas"]["DynamicalModelSpec-Input"];
        };
        /**
         * EditModelInput[RevisionSelector]
         * @description Create or revise a model from a question or model parent.
         */
        readonly EditModelInput_RevisionSelector_: {
            /** @description Question or model revision to start from. A model parent supplies its pinned question. 'latest' selects the current model, otherwise the current question. An exact Git hash can select either parent without being the current head. */
            readonly parent_ref: components["schemas"]["RevisionSelector"];
            /** @description Scientific definitions merged by identity into a model parent, or into an empty model for a question parent. Omitted fields are retained; null entity entries delete their identities. The action retains the merged spec and its findings. */
            readonly dynamical_model_spec: components["schemas"]["DynamicalModelSpec-Input"];
        };
        /**
         * EditModelOutput
         * @description The produced model and its recorded scientific findings.
         */
        readonly EditModelOutput: {
            /** @description Saved scientific definition. */
            readonly dynamical_model_spec: components["schemas"]["DynamicalModelSpec-Output"];
            /** @description Combined findings retained by the action that produced the model. */
            readonly checks: components["schemas"]["ModelCheckReport"];
            /** @description Causal identification findings. */
            readonly identification: components["schemas"]["IdentificationReport"];
            /** @description Entities removed because they do not serve the question. */
            readonly pruning: components["schemas"]["ModelEditResult"];
        };
        /**
         * EditModelRequest[GitOid]
         * @description Create or revise a model from its selected parent; return ``EditModelOutput``.
         */
        readonly EditModelRequest_GitOid_: {
            /**
             * @description Scientific action that owns this request or result. (enum property replaced by openapi-typescript)
             * @enum {string}
             */
            readonly action: "edit_model";
            /** @description Typed arguments of the scientific action. */
            readonly input: Domain.EditModelInput<Domain.GitOid>;
            /**
             * Reasoning
             * @description Why the caller is taking this action and what goal it serves. Retained with the original call and shown at the top of its action log; excluded from call identity.
             * @default null
             */
            readonly reasoning: string | null;
        };
        /**
         * EditModelRequest[RevisionSelector]
         * @description Create or revise a model from its selected parent; return ``EditModelOutput``.
         */
        readonly EditModelRequest_RevisionSelector_: {
            /**
             * @description Scientific action that owns this request or result. (enum property replaced by openapi-typescript)
             * @enum {string}
             */
            readonly action: "edit_model";
            /** @description Typed arguments of the scientific action. */
            readonly input: components["schemas"]["EditModelInput_RevisionSelector_"];
            /**
             * Reasoning
             * @description Why the caller is taking this action and what goal it serves. Retained with the original call and shown at the top of its action log; excluded from call identity.
             * @default null
             */
            readonly reasoning?: string | null;
        };
        /**
         * EditQuestionInput
         * @description The study question, its required outcome, and any named intervention queries.
         */
        readonly "EditQuestionInput-Input": {
            /** @description The replacement study question, including the required outcome and any intervention queries to assess against candidate models. */
            readonly question: components["schemas"]["QuestionSpec-Input"];
        };
        /**
         * EditQuestionInput
         * @description The study question, its required outcome, and any named intervention queries.
         */
        readonly "EditQuestionInput-Output": {
            /** @description The replacement study question, including the required outcome and any intervention queries to assess against candidate models. */
            readonly question: components["schemas"]["QuestionSpec-Output"];
        };
        /**
         * EditQuestionOutput
         * @description The question saved by this action.
         */
        readonly EditQuestionOutput: {
            /** @description The saved question, including its outcome and intervention queries. */
            readonly question: components["schemas"]["QuestionSpec-Output"];
        };
        /**
         * EditQuestionRequest
         * @description Save the supplied question; the successful body is ``EditQuestionOutput``.
         */
        readonly "EditQuestionRequest-Input": {
            /**
             * @description Scientific action that owns this request or result. (enum property replaced by openapi-typescript)
             * @enum {string}
             */
            readonly action: "edit_question";
            /** @description Typed arguments of the scientific action. */
            readonly input: components["schemas"]["EditQuestionInput-Input"];
            /**
             * Reasoning
             * @description Why the caller is taking this action and what goal it serves. Retained with the original call and shown at the top of its action log; excluded from call identity.
             * @default null
             */
            readonly reasoning?: string | null;
        };
        /**
         * EditQuestionRequest
         * @description Save the supplied question; the successful body is ``EditQuestionOutput``.
         */
        readonly "EditQuestionRequest-Output": {
            /**
             * @description Scientific action that owns this request or result. (enum property replaced by openapi-typescript)
             * @enum {string}
             */
            readonly action: "edit_question";
            /** @description Typed arguments of the scientific action. */
            readonly input: components["schemas"]["EditQuestionInput-Output"];
            /**
             * Reasoning
             * @description Why the caller is taking this action and what goal it serves. Retained with the original call and shown at the top of its action log; excluded from call identity.
             * @default null
             */
            readonly reasoning: string | null;
        };
        /**
         * EffectSummary
         * @description An effect summary reports posterior location, uncertainty, and sign probability.
         */
        readonly EffectSummary: {
            /** Mean */
            readonly mean: number;
            /** Median */
            readonly median: number;
            /** Lower 95 */
            readonly lower_95: number;
            /** Upper 95 */
            readonly upper_95: number;
            /** Prob Positive */
            readonly prob_positive: number;
        };
        /**
         * EmpiricalPoint
         * @description A step of an empirical cumulative distribution.
         */
        readonly EmpiricalPoint: {
            /**
             * Value
             * @description Distinct finite observed or sampled value.
             */
            readonly value: number;
            /**
             * Probability
             * @description Fraction of finite values less than or equal to `value`.
             */
            readonly probability: number;
        };
        /**
         * EnergyDiagnostics
         * @description Energy distributions and the producer's chain-specific BFMI values.
         */
        readonly EnergyDiagnostics: {
            readonly energy_hist: components["schemas"]["DensityCurve"];
            readonly energy_transition_hist: components["schemas"]["DensityCurve"];
            /** Bfmi */
            readonly bfmi: readonly number[];
        };
        /** @description An entity reference identifies a construct, edge, indicator, or mechanism by its persistent identity. */
        readonly EntityRef: components["schemas"]["ConstructRef-Output"] | components["schemas"]["EdgeRef"] | components["schemas"]["IndicatorRef"] | components["schemas"]["MechanismRef"];
        /**
         * Evaluated[ConvergenceAssessmentSubject, NumericCriterionEvidence]
         * @description One producer's measured outcome and the evidence supporting it.
         */
        readonly Evaluated_ConvergenceAssessmentSubject_NumericCriterionEvidence_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "evaluated";
            /** Code */
            readonly code: string;
            readonly subject: components["schemas"]["ConvergenceAssessmentSubject"];
            readonly outcome: components["schemas"]["CheckOutcome"];
            readonly evidence: components["schemas"]["NumericCriterionEvidence"];
        };
        /**
         * Evaluated[IndicatorCheckSubject, NumericCriterionEvidence]
         * @description One producer's measured outcome and the evidence supporting it.
         */
        readonly Evaluated_IndicatorCheckSubject_NumericCriterionEvidence_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "evaluated";
            /** Code */
            readonly code: string;
            readonly subject: components["schemas"]["IndicatorCheckSubject"];
            readonly outcome: components["schemas"]["CheckOutcome"];
            readonly evidence: components["schemas"]["NumericCriterionEvidence"];
        };
        /**
         * Evaluated[PredictiveSubject, tuple[NumericCriterionEvidence, ...]]
         * @description One producer's measured outcome and the evidence supporting it.
         */
        readonly Evaluated_PredictiveSubject_tuple_NumericCriterionEvidence__________: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "evaluated";
            /** Code */
            readonly code: string;
            readonly subject: components["schemas"]["PredictiveSubject"];
            readonly outcome: components["schemas"]["CheckOutcome"];
            /** Evidence */
            readonly evidence: readonly components["schemas"]["NumericCriterionEvidence"][];
        };
        /**
         * Evaluated[QuestionSubject, str]
         * @description One producer's measured outcome and the evidence supporting it.
         */
        readonly Evaluated_QuestionSubject_str_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "evaluated";
            /** Code */
            readonly code: string;
            readonly subject: components["schemas"]["QuestionSubject"];
            readonly outcome: components["schemas"]["CheckOutcome"];
            /** Evidence */
            readonly evidence: string;
        };
        /**
         * Evaluated[Union[IndicatorRef, Literal['dataset']], str]
         * @description One producer's measured outcome and the evidence supporting it.
         */
        readonly Evaluated_Union_IndicatorRef__Literal__dataset____str_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "evaluated";
            /** Code */
            readonly code: string;
            /** Subject */
            readonly subject: components["schemas"]["IndicatorRef"] | "dataset";
            readonly outcome: components["schemas"]["CheckOutcome"];
            /** Evidence */
            readonly evidence: string;
        };
        /**
         * Evaluated[str, str]
         * @description One producer's measured outcome and the evidence supporting it.
         */
        readonly Evaluated_str_str_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "evaluated";
            /** Code */
            readonly code: string;
            /** Subject */
            readonly subject: string;
            readonly outcome: components["schemas"]["CheckOutcome"];
            /** Evidence */
            readonly evidence: string;
        };
        /** @description An execution message retains a lifecycle entry, structured progress, an LLM trace, or failure details. */
        readonly ExecutionMessage: components["schemas"]["ActionMessage"] | components["schemas"]["ProgressMessage"] | components["schemas"]["TraceLogMessage"] | components["schemas"]["FailureMessage"];
        /** @description A scalar expression composes supported arithmetic with scientific state and coefficient references. */
        readonly "Expression-Output": components["schemas"]["LiteralExpression-Output"] | components["schemas"]["StateExpression-Output"] | components["schemas"]["CoefficientExpression-Output"] | components["schemas"]["BinaryExpression-Output"] | components["schemas"]["CallExpression-Output"];
        /**
         * @description An expression function transforms scalar operands or constructs structured observation arguments.
         * @enum {string}
         */
        readonly ExpressionFunction: "exp" | "sigmoid" | "normal_cdf" | "ordered_cutpoints" | "category_logits";
        /**
         * ExtractionPlanEvent
         * @description The extraction fan-out plan.
         */
        readonly ExtractionPlanEvent: {
            /**
             * Attempt Id
             * Format: uuid
             */
            readonly attempt_id: string;
            /**
             * Cursor
             * @default
             */
            readonly cursor: string;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly event: "grassmoth.extraction.plan";
            /** Total Workers */
            readonly total_workers: number;
            /**
             * Max Concurrent Workers
             * @default null
             */
            readonly max_concurrent_workers: number | null;
        };
        /**
         * ExtractionSnapshotEvent
         * @description Aggregate extraction worker counts.
         */
        readonly ExtractionSnapshotEvent: {
            /**
             * Attempt Id
             * Format: uuid
             */
            readonly attempt_id: string;
            /**
             * Cursor
             * @default
             */
            readonly cursor: string;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly event: "grassmoth.extraction.snapshot";
            /** Total Workers */
            readonly total_workers: number;
            /** Pending Workers */
            readonly pending_workers: number;
            /** Running Workers */
            readonly running_workers: number;
            /** Completed Workers */
            readonly completed_workers: number;
            /** Failed Workers */
            readonly failed_workers: number;
        };
        readonly "ExtractionSpec-Input": components["schemas"]["ComputedExtractionSpec-Input"] | components["schemas"]["SemanticExtractionSpec-Input"] | components["schemas"]["RecordedExtractionSpec-Input"];
        readonly "ExtractionSpec-Output": components["schemas"]["ComputedExtractionSpec-Output"] | components["schemas"]["SemanticExtractionSpec-Output"] | components["schemas"]["RecordedExtractionSpec-Output"];
        /**
         * ExtractionWorkerEvent
         * @description One extraction worker's state; a worker reports its LLM calls when it finishes.
         */
        readonly ExtractionWorkerEvent: {
            /**
             * Attempt Id
             * Format: uuid
             */
            readonly attempt_id: string;
            /**
             * Cursor
             * @default
             */
            readonly cursor: string;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly event: "grassmoth.extraction.worker";
            /** Worker Id */
            readonly worker_id: number;
            /**
             * State
             * @enum {string}
             */
            readonly state: "pending" | "running" | "completed" | "failed";
            /** N Windows */
            readonly n_windows: number;
            /**
             * N Extractions
             * @default null
             */
            readonly n_extractions: number | null;
            /**
             * N Llm Calls
             * @default null
             */
            readonly n_llm_calls: number | null;
            /**
             * Error
             * @default null
             */
            readonly error: string | null;
        };
        readonly ExtractionWorkerResult: components["schemas"]["CompletedExtractionWorker"] | components["schemas"]["FailedExtractionChunk"];
        /**
         * FailedExtractionChunk
         * @description A failed extraction chunk with its error, retained counts, and execution details.
         */
        readonly FailedExtractionChunk: {
            /** Worker Id */
            readonly worker_id: number;
            /** N Extractions */
            readonly n_extractions: number;
            /** N Windows */
            readonly n_windows: number;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly status: "failed";
            /** Error */
            readonly error: string;
            /**
             * N Llm Calls
             * @default 0
             */
            readonly n_llm_calls: number | null;
            /**
             * Reused
             * @default false
             */
            readonly reused: boolean | null;
        };
        /**
         * FailedPoll
         * @description A terminal failure with its full details in the accumulated messages.
         */
        readonly FailedPoll: {
            readonly call_id: components["schemas"]["CallId-Output"];
            readonly action: components["schemas"]["ActionId"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly status: "failed";
            readonly commit_id: components["schemas"]["GitOid-Output"] | null;
            /** Messages */
            readonly messages: readonly components["schemas"]["ExecutionMessage"][];
            /**
             * Body
             * @description The shared envelope has no scientific result after failure.
             */
            readonly body: null;
        };
        /**
         * FailureMessage
         * @description The complete terminal failure carried by the public execution log.
         */
        readonly FailureMessage: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "failure";
            /**
             * Timestamp
             * Format: date-time
             */
            readonly timestamp: string;
            /** Failure */
            readonly failure: components["schemas"]["Rejected"] | components["schemas"]["Raised"];
        };
        /**
         * FileSourceRef
         * @description Captured source files with workspace-relative paths and their content hashes.
         */
        readonly FileSourceRef: {
            /** Files */
            readonly files: readonly [
                string,
                ...string[]
            ];
            /**
             * Hashes
             * @description Call-time SHA-256 of every captured source file, retained with the resolved call.
             */
            readonly hashes: {
                readonly [key: string]: string;
            };
            /**
             * Start
             * @description Inclusive UTC source-coverage date.
             * @default null
             */
            readonly start: string | null;
            /**
             * End
             * @description Exclusive UTC source-coverage date.
             * @default null
             */
            readonly end: string | null;
        };
        readonly FindingSubject: string | components["schemas"]["EntityRef"] | components["schemas"]["PredictiveSubject"] | components["schemas"]["IndicatorCheckSubject"] | components["schemas"]["ConvergenceSubject"] | components["schemas"]["LatentConvergenceSubject"] | components["schemas"]["QuestionSubject"];
        /**
         * FitCheckReport
         * @description Compatibility and question findings owned by one completed fit.
         */
        readonly FitCheckReport: {
            readonly data: components["schemas"]["DataProfileReport"];
            /** Preflight */
            readonly preflight: readonly components["schemas"]["SpecificationAssessment"][];
            readonly question: components["schemas"]["QuestionCheckReport"];
        };
        /**
         * FitComponentTiming
         * @description One independently compiled native operation; timing includes synchronization.
         */
        readonly FitComponentTiming: {
            /**
             * Component
             * @enum {string}
             */
            readonly component: "full_sweep" | "parameter_move" | "path_smoother";
            readonly timing: components["schemas"]["TimingSpread"];
        };
        /**
         * FitInput[GitOid]
         * @description Start or continue a fit; sample counts always mean additional retained draws.
         */
        readonly FitInput_GitOid_: {
            /** @description A model and saved history to fit, or a prior fit to continue. */
            readonly target: Domain.FitTarget<Domain.GitOid>;
            /** @description Sampler overrides for this call. */
            readonly settings: components["schemas"]["FitSettingsSpec-Output"];
            /**
             * Repeat
             * @description Timing repeat identity; identical repeats reuse their saved result.
             * @default 0
             */
            readonly repeat: number;
        };
        /**
         * FitInput[RevisionSelector]
         * @description Start or continue a fit; sample counts always mean additional retained draws.
         */
        readonly FitInput_RevisionSelector_: {
            /** @description A model and saved history to fit, or a prior fit to continue. */
            readonly target: components["schemas"]["FitTarget_RevisionSelector_"];
            /** @description Sampler overrides for this call. */
            readonly settings?: components["schemas"]["FitSettingsSpec-Input"];
            /**
             * Repeat
             * @description Timing repeat identity; identical repeats reuse their saved result.
             * @default 0
             */
            readonly repeat?: number;
        };
        /**
         * FitOutput
         * @description A conditioned model, its completed checks, and self-contained inference evidence.
         */
        readonly FitOutput: {
            /** @description The combined conditioned model, or null when this fit and its parents have no retained draws. Zero-draw continuations reuse their parent's model without producing a revision. */
            readonly dynamical_model_spec: components["schemas"]["DynamicalModelSpec-Output"] | null;
            /** @description Completed compatibility and question findings for the selected history. */
            readonly checks: components["schemas"]["FitCheckReport"];
            /** @description Run provenance, native sampler evidence, diagnostics and parameter summaries. */
            readonly inference: components["schemas"]["InferenceReport"];
            /**
             * @description Timings for this execution only when settings.profile is true.
             * @default null
             */
            readonly profile: components["schemas"]["FitProfileReport"] | null;
        };
        /**
         * FitProfileReport
         * @description Wall-clock microbenchmarks on a copied state, never used by inference.
         *
         *     Every operation is primed once and measured ten times at fixed state, tuning
         *     and random keys. Full sweep uses the production executable with count one;
         *     components are compiled separately. The path timing includes recurrence
         *     replay at the fixed selected context. No device-trace
         *     kernel times are used. Separately compiled parts need not sum to a full sweep.
         */
        readonly FitProfileReport: {
            /**
             * Phase
             * @enum {string}
             */
            readonly phase: "after_warmup" | "after_first_sweep" | "compile_only";
            /** Repetitions */
            readonly repetitions: number;
            /**
             * Compile Seconds
             * @description Union of production compile-request intervals, including tracing, lowering and compilation or cache load. Excludes profiling-only programs and artifact writes.
             * @default 0
             */
            readonly compile_seconds: number;
            /**
             * Compilation
             * @description Production compile requests, in request order.
             * @default []
             */
            readonly compilation: readonly components["schemas"]["ProgramCompileReport"][];
            /**
             * Profiling Compilation
             * @description Separately compiled timing probes, excluded from compile_seconds.
             * @default []
             */
            readonly profiling_compilation: readonly components["schemas"]["ProgramCompileReport"][];
            /**
             * Device
             * @default []
             */
            readonly device: readonly string[];
            /** Chains */
            readonly chains: number;
            /** Time Steps */
            readonly time_steps: number;
            /** @default null */
            readonly particle_dimensions: components["schemas"]["ParticleProfileDimensions"] | null;
            /**
             * Components
             * @default []
             */
            readonly components: readonly components["schemas"]["FitComponentTiming"][];
            /**
             * @description Sum of parameter and path timings per repetition, summarized alongside full_sweep. Other sweep bookkeeping is not part of this sum.
             * @default null
             */
            readonly split_total: components["schemas"]["TimingSpread"] | null;
        };
        /** @enum {string} */
        readonly FitReliability: "not_fitted" | "converged" | "unconverged" | "unknown";
        /**
         * FitRequest[GitOid]
         * @description Fit the selected model to the selected history; return ``FitOutput``.
         */
        readonly FitRequest_GitOid_: {
            /**
             * @description Scientific action that owns this request or result. (enum property replaced by openapi-typescript)
             * @enum {string}
             */
            readonly action: "fit";
            /** @description Typed arguments of the scientific action. */
            readonly input: Domain.FitInput<Domain.GitOid>;
            /**
             * Reasoning
             * @description Why the caller is taking this action and what goal it serves. Retained with the original call and shown at the top of its action log; excluded from call identity.
             * @default null
             */
            readonly reasoning: string | null;
        };
        /**
         * FitRequest[RevisionSelector]
         * @description Fit the selected model to the selected history; return ``FitOutput``.
         */
        readonly FitRequest_RevisionSelector_: {
            /**
             * @description Scientific action that owns this request or result. (enum property replaced by openapi-typescript)
             * @enum {string}
             */
            readonly action: "fit";
            /** @description Typed arguments of the scientific action. */
            readonly input: components["schemas"]["FitInput_RevisionSelector_"];
            /**
             * Reasoning
             * @description Why the caller is taking this action and what goal it serves. Retained with the original call and shown at the top of its action log; excluded from call identity.
             * @default null
             */
            readonly reasoning?: string | null;
        };
        /**
         * FitRuntime
         * @description Actual numerical source, environment, compiler settings and devices.
         */
        readonly FitRuntime: {
            /** Source Digest */
            readonly source_digest: string;
            /** Versions */
            readonly versions: {
                readonly [key: string]: string;
            };
            /** Compiler */
            readonly compiler: {
                readonly [key: string]: string;
            };
            /** Devices */
            readonly devices: readonly string[];
            /**
             * Time Limit Seconds
             * @description Enforced Modal phase limit; null for local execution.
             * @default null
             */
            readonly time_limit_seconds: number | null;
        };
        /**
         * FitSegment
         * @description One execution's retained draw interval within every combined chain.
         */
        readonly FitSegment: {
            /** Start */
            readonly start: number;
            /** Stop */
            readonly stop: number;
            readonly settings: components["schemas"]["SamplerSpec"];
            readonly runtime: components["schemas"]["FitRuntime"];
            /** Phases */
            readonly phases: readonly components["schemas"]["InferencePhase"][];
            /** @description Production diagnostics for this execution's retained draws alone. */
            readonly diagnostics: components["schemas"]["ChainDiagnostics"] | null;
            /** @description Production convergence assessment for this segment, before combining chains. */
            readonly convergence: components["schemas"]["ConvergenceReport"];
            /** @default null */
            readonly parameterization: components["schemas"]["JointParameterizationRecord"] | null;
            /** @default null */
            readonly convergence_stopping: components["schemas"]["ConvergenceStoppingReport"] | null;
        };
        /**
         * FitSettingsSpec
         * @description Optional numerical controls applied to the configured joint sampler.
         *
         *     Each field means what the SamplerSpec field of the same name means. Null keeps the
         *     server's configured value, or the continued fit's, except convergence stopping,
         *     which is opt-in for each execution. Profiling also applies only to this execution.
         */
        readonly "FitSettingsSpec-Input": {
            /**
             * @description Opt in for this execution only; absent/null keeps the fixed schedule, including continuations.
             * @default null
             */
            readonly stop_when_converged?: components["schemas"]["ConvergenceStoppingSpec"] | null;
            /**
             * Profile
             * @description Profile this execution only, including continuations. Time native compiled parts on frozen copies after warmup, or after the first sweep without warmup. Report named production and profiling-only compile requests with cache status; save named compiled analysis beside the cached fit after timing completes.
             * @default false
             */
            readonly profile?: boolean;
            /** @default null */
            readonly hardware?: components["schemas"]["InferenceHardware"] | null;
            /**
             * Time Limit Seconds
             * @default null
             */
            readonly time_limit_seconds?: number | null;
            /** @default null */
            readonly algorithm?: components["schemas"]["InferenceAlgorithmSpec-Input"] | null;
            /**
             * Num Samples Per Chain
             * @default null
             */
            readonly num_samples_per_chain?: number | null;
            /**
             * Num Warmup
             * @default null
             */
            readonly num_warmup?: number | null;
            /**
             * Num Chains
             * @default null
             */
            readonly num_chains?: number | null;
            /**
             * Seed
             * @default null
             */
            readonly seed?: number | null;
        };
        /**
         * FitSettingsSpec
         * @description Optional numerical controls applied to the configured joint sampler.
         *
         *     Each field means what the SamplerSpec field of the same name means. Null keeps the
         *     server's configured value, or the continued fit's, except convergence stopping,
         *     which is opt-in for each execution. Profiling also applies only to this execution.
         */
        readonly "FitSettingsSpec-Output": {
            /**
             * @description Opt in for this execution only; absent/null keeps the fixed schedule, including continuations.
             * @default null
             */
            readonly stop_when_converged: components["schemas"]["ConvergenceStoppingSpec"] | null;
            /**
             * Profile
             * @description Profile this execution only, including continuations. Time native compiled parts on frozen copies after warmup, or after the first sweep without warmup. Report named production and profiling-only compile requests with cache status; save named compiled analysis beside the cached fit after timing completes.
             * @default false
             */
            readonly profile: boolean;
            /** @default null */
            readonly hardware: components["schemas"]["InferenceHardware"] | null;
            /**
             * Time Limit Seconds
             * @default null
             */
            readonly time_limit_seconds: number | null;
            /** @default null */
            readonly algorithm: components["schemas"]["InferenceAlgorithmSpec-Output"] | null;
            /**
             * Num Samples Per Chain
             * @default null
             */
            readonly num_samples_per_chain: number | null;
            /**
             * Num Warmup
             * @default null
             */
            readonly num_warmup: number | null;
            /**
             * Num Chains
             * @default null
             */
            readonly num_chains: number | null;
            /**
             * Seed
             * @default null
             */
            readonly seed: number | null;
        };
        readonly FitTarget_GitOid_: Domain.NewFitTarget<Domain.GitOid> | Domain.ContinueFitTarget<Domain.GitOid>;
        readonly FitTarget_RevisionSelector_: components["schemas"]["NewFitTarget_RevisionSelector_"] | components["schemas"]["ContinueFitTarget_RevisionSelector_"];
        /**
         * FittedLawProvenance
         * @description All current laws retain one committed fit's model and observation panel.
         */
        readonly FittedLawProvenance: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "fitted";
            readonly fitted_data: Domain.DataRef<Domain.GitOid, number>;
            readonly fitted_model_revision: components["schemas"]["GitOid-Output"];
            /**
             * Interpretation
             * @enum {string}
             */
            readonly interpretation: "in_sample_posterior_predictive" | "posterior_predictive";
        };
        /**
         * FixedLatentStepSpec
         * @description An assigned scale which remains fixed during warmup and sampling.
         */
        readonly "FixedLatentStepSpec-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "fixed";
            /**
             * Delta
             * @description Variance of the candidates around each proposed state, at every time step: half comes from refreshing the auxiliary point and half from drawing around it (Corenflos & Finke 2024, Algorithm 3). It is in the state's own units and shared by all its coordinates. Fix it to compare proposals at a matched scale, or when adaptation settles on a scale at which the path stops moving.
             * @default 0.01
             */
            readonly delta?: number;
        };
        /**
         * FixedLatentStepSpec
         * @description An assigned scale which remains fixed during warmup and sampling.
         */
        readonly "FixedLatentStepSpec-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "fixed";
            /**
             * Delta
             * @description Variance of the candidates around each proposed state, at every time step: half comes from refreshing the auxiliary point and half from drawing around it (Corenflos & Finke 2024, Algorithm 3). It is in the state's own units and shared by all its coordinates. Fix it to compare proposals at a matched scale, or when adaptation settles on a scale at which the path stops moving.
             * @default 0.01
             */
            readonly delta: number;
        };
        /**
         * FixedTrajectorySpec
         * @description A fixed number of leapfrog steps, with the step size jittered by up to 20%.
         */
        readonly "FixedTrajectorySpec-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "fixed";
            /**
             * Steps
             * @description Leapfrog steps per transition. Each transition scales the adapted step size by a uniform factor in [0.8, 1.2], so trajectories cannot stay periodic (Neal 2011, section 5.3.2). Every transition costs the same number of gradients, so chains advanced together never wait for the longest NUTS tree, which limits NUTS on accelerators (Hoffman, Radul & Sountsov 2021).
             * @default 8
             */
            readonly steps?: number;
        };
        /**
         * FixedTrajectorySpec
         * @description A fixed number of leapfrog steps, with the step size jittered by up to 20%.
         */
        readonly "FixedTrajectorySpec-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "fixed";
            /**
             * Steps
             * @description Leapfrog steps per transition. Each transition scales the adapted step size by a uniform factor in [0.8, 1.2], so trajectories cannot stay periodic (Neal 2011, section 5.3.2). Every transition costs the same number of gradients, so chains advanced together never wait for the longest NUTS tree, which limits NUTS on accelerators (Hoffman, Radul & Sountsov 2021).
             * @default 8
             */
            readonly steps: number;
        };
        /**
         * FrequencyRefresh
         * @description Frequency-localized movement for one construct's complete retained trajectory.
         */
        readonly FrequencyRefresh: {
            readonly subject: components["schemas"]["ConstructRef-Output"];
            /** Label */
            readonly label: string;
            /** Bands */
            readonly bands: readonly components["schemas"]["FrequencyRefreshBand"][];
        };
        /**
         * FrequencyRefreshBand
         * @description Median squared adjacent-draw jump divided by twice pooled mode variance.
         */
        readonly FrequencyRefreshBand: {
            /**
             * Start Mode
             * @description Inclusive orthonormal DCT-II mode index.
             */
            readonly start_mode: number;
            /**
             * Stop Mode
             * @description Exclusive orthonormal DCT-II mode index.
             */
            readonly stop_mode: number;
            /**
             * Median Refresh
             * @description Null without two retained draws per chain; zero for constant modes.
             */
            readonly median_refresh: number | null;
        };
        readonly FrozenSamplerTuning: components["schemas"]["JointTuning"] | components["schemas"]["ParticleTuning"] | components["schemas"]["TransferredTuning"];
        /**
         * GammaLawSpec[Expression]
         * @description The Gamma conditional law.
         */
        readonly "GammaLawSpec_Expression_-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "Gamma";
            readonly concentration: components["schemas"]["Expression-Output"];
            readonly rate: components["schemas"]["Expression-Output"];
        };
        /**
         * GaussianInitializationDiagnostics
         * @description Unadapted Gaussian starting points, not posterior or convergence evidence.
         */
        readonly GaussianInitializationDiagnostics: {
            /**
             * Source
             * @default gaussian
             * @constant
             */
            readonly source: "gaussian";
            /** Joint Dimension */
            readonly joint_dimension: number;
            /** Scale */
            readonly scale: number;
        };
        /**
         * GaussianInitializationSpec
         * @description Gaussian starts around a data-conditioned MPGibbs path center.
         *
         *     A damped iterated extended Kalman smoother conditions the path center on
         *     point readings at the initial parameter center, preserving the native
         *     Gaussian jitter and parameter starts. Interval readings are omitted from
         *     this seed. Exact replay restores recurrences, static values and constraints;
         *     subsequent particle transitions and diagnostics use the exact nonlinear target.
         */
        readonly "GaussianInitializationSpec-Input": {
            /**
             * Scale
             * @description Standard deviation of the starts around the native starting values, in the centred chart. The initial parameter preconditioner is scale times the identity, and MPGibbs re-estimates it in warmup windows.
             * @default 0.1
             */
            readonly scale?: number;
        };
        /**
         * GaussianInitializationSpec
         * @description Gaussian starts around a data-conditioned MPGibbs path center.
         *
         *     A damped iterated extended Kalman smoother conditions the path center on
         *     point readings at the initial parameter center, preserving the native
         *     Gaussian jitter and parameter starts. Interval readings are omitted from
         *     this seed. Exact replay restores recurrences, static values and constraints;
         *     subsequent particle transitions and diagnostics use the exact nonlinear target.
         */
        readonly "GaussianInitializationSpec-Output": {
            /**
             * Scale
             * @description Standard deviation of the starts around the native starting values, in the centred chart. The initial parameter preconditioner is scale times the identity, and MPGibbs re-estimates it in warmup windows.
             * @default 0.1
             */
            readonly scale: number;
        };
        /**
         * GaussianReferenceTemperingSpec
         * @description Replica exchange; each ladder retains only its exact beta-one chain.
         *
         *     The ladder spans beta one to zero. The reference is fixed at initialization,
         *     its endpoint is independently refreshed every sweep, and even/odd neighboring
         *     swaps alternate. Rejection-based ladder adaptation runs during the first half
         *     of warmup, leaving the rest to settle before retaining posterior draws.
         *
         *     It is for posteriors whose chains stay in separate modes: states that move freely
         *     under the reference reach the exact chain through swaps (Syed et al. 2022). Its
         *     limits are in docs/assumptions.md, under Model class.
         */
        readonly "GaussianReferenceTemperingSpec-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "gaussian_reference";
            /**
             * Num Temperatures
             * @description Replicas per chain, initially evenly spaced from beta one (the exact posterior) to beta zero (the Gaussian reference). Each replica runs a full sweep, so cost grows linearly. With alternating even/odd swaps the round-trip rate tends to 1 / (2 + 2Λ) instead of collapsing as replicas are added, and about 2Λ replicas are optimal, where Λ is the posterior's communication barrier (Syed et al. 2022, sections 4.2 and 5.3).
             * @default 48
             */
            readonly num_temperatures?: number;
            /**
             * Reference Scale
             * @description Multiplies the reference's standard deviations for parameters and path innovations. The reference is built at initialization from the starting draws and the transition Jacobians along their mean path. Above 1 it covers mass the starting draws missed, but overlaps the posterior less, which raises the barrier and so the replicas needed.
             * @default 1
             */
            readonly reference_scale?: number;
            /**
             * Adaptation Interval
             * @description Sweeps between ladder updates in the first half of warmup. Each update respaces the temperatures to equalize the swap rejection rates estimated since the last update (Syed et al. 2022, sections 5.1-5.2). An interval above half the warmup leaves the evenly spaced ladder unadapted.
             * @default 500
             */
            readonly adaptation_interval?: number;
        };
        /**
         * GaussianReferenceTemperingSpec
         * @description Replica exchange; each ladder retains only its exact beta-one chain.
         *
         *     The ladder spans beta one to zero. The reference is fixed at initialization,
         *     its endpoint is independently refreshed every sweep, and even/odd neighboring
         *     swaps alternate. Rejection-based ladder adaptation runs during the first half
         *     of warmup, leaving the rest to settle before retaining posterior draws.
         *
         *     It is for posteriors whose chains stay in separate modes: states that move freely
         *     under the reference reach the exact chain through swaps (Syed et al. 2022). Its
         *     limits are in docs/assumptions.md, under Model class.
         */
        readonly "GaussianReferenceTemperingSpec-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "gaussian_reference";
            /**
             * Num Temperatures
             * @description Replicas per chain, initially evenly spaced from beta one (the exact posterior) to beta zero (the Gaussian reference). Each replica runs a full sweep, so cost grows linearly. With alternating even/odd swaps the round-trip rate tends to 1 / (2 + 2Λ) instead of collapsing as replicas are added, and about 2Λ replicas are optimal, where Λ is the posterior's communication barrier (Syed et al. 2022, sections 4.2 and 5.3).
             * @default 48
             */
            readonly num_temperatures: number;
            /**
             * Reference Scale
             * @description Multiplies the reference's standard deviations for parameters and path innovations. The reference is built at initialization from the starting draws and the transition Jacobians along their mean path. Above 1 it covers mass the starting draws missed, but overlaps the posterior less, which raises the barrier and so the replicas needed.
             * @default 1
             */
            readonly reference_scale: number;
            /**
             * Adaptation Interval
             * @description Sweeps between ladder updates in the first half of warmup. Each update respaces the temperatures to equalize the swap rejection rates estimated since the last update (Syed et al. 2022, sections 5.1-5.2). An interval above half the warmup leaves the evenly spaced ladder unadapted.
             * @default 500
             */
            readonly adaptation_interval: number;
        };
        /** @description A native Git object identity for an immutable tree or commit. */
        readonly "GitOid-Input": string;
        /** @description A native Git object identity for an immutable tree or commit. */
        readonly "GitOid-Output": string;
        /**
         * GuidedPathSpec
         * @description Sequential conditional SMC with exact Gaussian transition-informed proposals.
         *
         *     Corenflos & Finke (2024). Parameter labels remain marginalized, and exact nonlinear
         *     transitions and densities pay every weight, so each method's approximation costs
         *     mixing, never the target.
         */
        readonly "GuidedPathSpec-Input": {
            /**
             * @description particle_agrad keeps the auxiliary variable in the weights (section 4.1); strong future information or multiple modes can reduce its mixing. twisted_particle_agrad adds a Gaussian look-ahead built at the auxiliary path: a parallel backward information scan guides the sequential filter, and proposals combine exact nonlinear transition means with the approximate twist, whose normalized densities and twist ratios are paid in full (section 4.4); poor local fits and multiple modes cost efficiency. particle_mgrad integrates the path auxiliary out of the particle weights with the untwisted proposal (section 4.2); it supports point readings without deterministic recurrences, and separated path components and informative future readings can slow its mixing. (enum property replaced by openapi-typescript)
             * @enum {string}
             */
            readonly method: "particle_agrad" | "particle_mgrad" | "twisted_particle_agrad";
            /**
             * Num Particles
             * @description Particles per sweep, including the current path. Each extends a parent chosen at the previous time, so work grows linearly in particles and times, and the filter's depth grows linearly with the number of times (Corenflos & Finke 2024, section 4.7).
             * @default 64
             */
            readonly num_particles?: number;
            /** @description Per-time-step scale: adapted towards a movement target, or fixed. Small scales keep proposals near the current path; large ones let the dynamics dominate, approaching conditional SMC with the transition as proposal (Corenflos & Finke 2024, section 4.6). */
            readonly latent_step?: components["schemas"]["LatentStepSpec-Input"];
            /**
             * Kappa
             * @description 1 shifts the auxiliary point along the gradient of the observation densities, mixed over parameter labels; 0 omits it (Corenflos & Finke 2024, Algorithm 6). The shift is a Langevin step on the observation term: for a Gaussian reading it overshoots once the scale exceeds four times that reading's variance.
             * @default 1
             * @enum {integer}
             */
            readonly kappa?: 0 | 1;
            /**
             * Scan Unroll
             * @description Time steps per iteration of the forward filter and backward-selection loops. Unrolling changes how XLA compiles these sequential loops, not the algorithm or its target. Each step runs small kernels, so on a GPU per-iteration overhead is a large share of a sweep; unrolling amortizes it at the cost of a larger program and longer compilation. A CPU has little such overhead to remove. 1 disables unrolling.
             * @default 8
             */
            readonly scan_unroll?: number;
            /**
             * Prefix Mixture
             * @description Opt in to per-parameter-label local Gaussian proposals, mixed with the unfloored prefix label probabilities. Every particle pays the full mixture density; the shared auxiliary law and look-ahead remain unchanged. Requires twisted_particle_agrad with kappa=1 and point readings without replay. Early label collapse, poor shared guidance and separated modes can reduce mixing.
             * @default false
             */
            readonly prefix_mixture?: boolean;
        };
        /**
         * GuidedPathSpec
         * @description Sequential conditional SMC with exact Gaussian transition-informed proposals.
         *
         *     Corenflos & Finke (2024). Parameter labels remain marginalized, and exact nonlinear
         *     transitions and densities pay every weight, so each method's approximation costs
         *     mixing, never the target.
         */
        readonly "GuidedPathSpec-Output": {
            /**
             * @description particle_agrad keeps the auxiliary variable in the weights (section 4.1); strong future information or multiple modes can reduce its mixing. twisted_particle_agrad adds a Gaussian look-ahead built at the auxiliary path: a parallel backward information scan guides the sequential filter, and proposals combine exact nonlinear transition means with the approximate twist, whose normalized densities and twist ratios are paid in full (section 4.4); poor local fits and multiple modes cost efficiency. particle_mgrad integrates the path auxiliary out of the particle weights with the untwisted proposal (section 4.2); it supports point readings without deterministic recurrences, and separated path components and informative future readings can slow its mixing. (enum property replaced by openapi-typescript)
             * @enum {string}
             */
            readonly method: "particle_agrad" | "particle_mgrad" | "twisted_particle_agrad";
            /**
             * Num Particles
             * @description Particles per sweep, including the current path. Each extends a parent chosen at the previous time, so work grows linearly in particles and times, and the filter's depth grows linearly with the number of times (Corenflos & Finke 2024, section 4.7).
             * @default 64
             */
            readonly num_particles: number;
            /** @description Per-time-step scale: adapted towards a movement target, or fixed. Small scales keep proposals near the current path; large ones let the dynamics dominate, approaching conditional SMC with the transition as proposal (Corenflos & Finke 2024, section 4.6). */
            readonly latent_step: components["schemas"]["LatentStepSpec-Output"];
            /**
             * Kappa
             * @description 1 shifts the auxiliary point along the gradient of the observation densities, mixed over parameter labels; 0 omits it (Corenflos & Finke 2024, Algorithm 6). The shift is a Langevin step on the observation term: for a Gaussian reading it overshoots once the scale exceeds four times that reading's variance.
             * @default 1
             * @enum {integer}
             */
            readonly kappa: 0 | 1;
            /**
             * Scan Unroll
             * @description Time steps per iteration of the forward filter and backward-selection loops. Unrolling changes how XLA compiles these sequential loops, not the algorithm or its target. Each step runs small kernels, so on a GPU per-iteration overhead is a large share of a sweep; unrolling amortizes it at the cost of a larger program and longer compilation. A CPU has little such overhead to remove. 1 disables unrolling.
             * @default 8
             */
            readonly scan_unroll: number;
            /**
             * Prefix Mixture
             * @description Opt in to per-parameter-label local Gaussian proposals, mixed with the unfloored prefix label probabilities. Every particle pays the full mixture density; the shared auxiliary law and look-ahead remain unchanged. Requires twisted_particle_agrad with kappa=1 and point readings without replay. Early label collapse, poor shared guidance and separated modes can reduce mixing.
             * @default false
             */
            readonly prefix_mixture: boolean;
        };
        /** HTTPValidationError */
        readonly HTTPValidationError: {
            /** Detail */
            readonly detail?: readonly components["schemas"]["ValidationError"][];
        };
        /**
         * HistogramBin
         * @description A histogram bin gives its interval, center, and number of posterior draws.
         */
        readonly HistogramBin: {
            /** Bin Center */
            readonly bin_center: number;
            /** Bin Start */
            readonly bin_start: number;
            /** Bin End */
            readonly bin_end: number;
            /** Count */
            readonly count: number;
        };
        /**
         * IdentificationReport
         * @description Positive and negative causal identification findings for the study question's outcome.
         */
        readonly IdentificationReport: {
            readonly outcome: components["schemas"]["ConstructId-Output"] | null;
            /**
             * Treatments
             * @description One tagged identification result per treatment, including its supporting evidence
             */
            readonly treatments: Readonly<Partial<Record<components["schemas"]["ConstructId-Output"], components["schemas"]["IdentifiedTreatmentStatus"] | components["schemas"]["NonIdentifiableTreatmentStatus"]>>>;
        };
        /**
         * IdentifiedTreatmentStatus
         * @description Details on how a treatment effect is identified.
         */
        readonly IdentifiedTreatmentStatus: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly status: "identified";
            /**
             * Estimand
             * @description Nonparametric estimand returned by do-calculus
             */
            readonly estimand: string;
            /**
             * Marginalized Confounders
             * @description Unobserved confounders the estimand integrates out
             */
            readonly marginalized_confounders: readonly components["schemas"]["ConstructId-Output"][];
            /**
             * Instruments
             * @description Instrument constructs appearing in the nonparametric identification argument
             */
            readonly instruments: readonly components["schemas"]["ConstructId-Output"][];
        };
        /**
         * IdentityTransformSpec
         * @description Keep the authored probability law on the scientific quantity's native scale.
         */
        readonly "IdentityTransformSpec-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "identity";
        };
        /**
         * IndicatorAudit
         * @description Empirical measurements and validation findings for one observation indicator.
         */
        readonly IndicatorAudit: {
            /** @default null */
            readonly profile: components["schemas"]["IndicatorEmpiricalProfile"] | null;
            /** Findings */
            readonly findings: readonly components["schemas"]["DataFinding"][];
        };
        /**
         * IndicatorCheckSubject
         * @description The indicator and criterion remain present when evaluation is unavailable.
         */
        readonly IndicatorCheckSubject: {
            readonly target: components["schemas"]["IndicatorRef"];
        };
        /**
         * IndicatorComparison
         * @description Computed evidence for one persistent indicator identity, without copying its histories.
         *
         *     Point changes are directional, from left to right, and only computed for one
         *     history on each side. Match by calendar instant:
         *     a new anchor is Added, a lost anchor Removed, and an exact value or support change
         *     at a retained anchor Revised. Omit unchanged points and order changes by anchor.
         *     Renames and measurement-definition changes are not point revisions; definition
         *     mismatches are findings. No tolerance, interpolation, or imputation is
         *     used. Reversing sides exchanges additions/removals and before/after payloads.
         *
         *     An empty changes tuple can mean identical points or an inapplicable point diff
         *     (replicated selections); it never asserts that whole
         *     datasets are equal. Missing variables and differing schedules remain explicit
         *     findings. Predictive evaluation owns its own applicability, so extra replicate
         *     anchors can produce a schedule finding without preventing checks at observed anchors.
         */
        readonly IndicatorComparison: {
            readonly indicator_id: components["schemas"]["IndicatorId-Output"];
            /** Changes */
            readonly changes: readonly Domain.Change<Domain.DataPoint>[];
            /** Statistics */
            readonly statistics: readonly components["schemas"]["DataStatisticComparison"][];
            /** Findings */
            readonly findings: readonly components["schemas"]["DataFinding"][];
            /** @description Reference history against the other side's replicates, when exactly one side is a single history of a compatible numeric indicator; otherwise null. An alignment failure is reported as a finding. */
            readonly predictive: components["schemas"]["PredictiveComparison"] | null;
        };
        /**
         * IndicatorEmpiricalProfile
         * @description Count, recorded range, quartiles and mean of one indicator's usable observations.
         */
        readonly IndicatorEmpiricalProfile: {
            /** N Obs */
            readonly n_obs: number;
            /** Min */
            readonly min: number | null;
            /** Q25 */
            readonly q25: number | null;
            /** Q50 */
            readonly q50: number | null;
            /** Q75 */
            readonly q75: number | null;
            /** Max */
            readonly max: number | null;
            /** Mean */
            readonly mean: number | null;
        };
        /** @description A persistent indicator identity survives changes to its measurement label. */
        readonly "IndicatorId-Input": `indicator:${string}`;
        /** @description A persistent indicator identity survives changes to its measurement label. */
        readonly "IndicatorId-Output": `indicator:${string}`;
        /**
         * IndicatorPolarity
         * @description Whether the measurement increases or decreases with its owning construct.
         * @enum {string}
         */
        readonly IndicatorPolarity: "positive" | "negative";
        /**
         * IndicatorRef
         * @description An observation identity independent of its current display name or model revision.
         */
        readonly IndicatorRef: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "indicator";
            readonly id: components["schemas"]["IndicatorId-Output"];
        };
        /**
         * IndicatorSpec
         * @description Bind an observed-variable ID to a construct and an emission likelihood.
         *
         *     Extraction instructions belong to DataPreparationSpec. The shared observation
         *     schema also permits generative models before any observations have been collected.
         */
        readonly "IndicatorSpec-Output": {
            readonly observation: Domain.ObservationSpec<Domain.ObservationWindow | null>;
            /** @default null */
            readonly likelihood: components["schemas"]["LikelihoodSpec-Output"] | null;
            /** @description Whether higher values move with (positive) or against (negative) the construct. */
            readonly construct_polarity: components["schemas"]["IndicatorPolarity"];
        };
        readonly "InferenceAlgorithmSpec-Input": components["schemas"]["NutsDASpec-Input"] | components["schemas"]["MarginalParticleGibbsSpec-Input"];
        readonly "InferenceAlgorithmSpec-Output": components["schemas"]["NutsDASpec-Output"] | components["schemas"]["MarginalParticleGibbsSpec-Output"];
        /**
         * InferenceEvidence
         * @description Native telemetry buffers; posterior atoms and coordinates belong to the model.
         */
        readonly InferenceEvidence: {
            /** @default null */
            readonly checkpoint: components["schemas"]["SamplerCheckpoint"] | null;
            /** @default null */
            readonly latent_free_mask: components["schemas"]["NumericalArray"] | null;
            /** Chain Extra Fields */
            readonly chain_extra_fields: {
                readonly [key: string]: components["schemas"]["NumericalArray"];
            };
            /** @default null */
            readonly observation_log_probs: components["schemas"]["NumericalArray"] | null;
            /** @default null */
            readonly observed_rows: components["schemas"]["NumericalArray"] | null;
            /** @default null */
            readonly exact_observation_rows: components["schemas"]["NumericalArray"] | null;
            /** Phase Extra Fields */
            readonly phase_extra_fields: {
                readonly [key: string]: {
                    readonly [key: string]: components["schemas"]["NumericalArray"];
                };
            };
            /** @default null */
            readonly warmup_complete_log_posterior_history: components["schemas"]["NumericalArray"] | null;
            /** @default null */
            readonly all_complete_log_posterior_history: components["schemas"]["NumericalArray"] | null;
            /** @default null */
            readonly initial_latent_delta: components["schemas"]["NumericalArray"] | null;
            /** @default null */
            readonly final_latent_delta: components["schemas"]["NumericalArray"] | null;
        };
        /**
         * InferenceHardware
         * @description Run here, or request one of Modal's named GPU types.
         * @enum {string}
         */
        readonly InferenceHardware: "local" | "T4" | "L4" | "A10" | "L40S" | "A100" | "A100-40GB" | "A100-80GB" | "RTX-PRO-6000" | "H100" | "H100!" | "H200" | "B200";
        /**
         * InferenceMetadata
         * @description The production run's law, chain layout and sampler measurements.
         */
        readonly InferenceMetadata: {
            readonly distribution: components["schemas"]["DistributionId-Output"] | null;
            /** Num Samples Total */
            readonly num_samples_total: number;
            /**
             * Segments
             * @default []
             */
            readonly segments: readonly components["schemas"]["FitSegment"][];
            /** Num Chains */
            readonly num_chains: number;
            /** Duration Seconds */
            readonly duration_seconds: number;
            readonly engine: components["schemas"]["MCMCEvidence"];
            /** Sampler Diagnostics */
            readonly sampler_diagnostics: components["schemas"]["ParticleSamplerDiagnostics"] | components["schemas"]["NutsDADiagnostics"] | null;
        };
        /**
         * InferencePhase
         * @description Measured interval relative to this execution's start; intervals may overlap.
         */
        readonly InferencePhase: {
            /** Sampler */
            readonly sampler: string;
            /** Stage */
            readonly stage: string;
            /** Start Seconds */
            readonly start_seconds: number;
            /** End Seconds */
            readonly end_seconds: number;
            readonly outcome: components["schemas"]["PhaseOutcome"];
        };
        /**
         * InferenceReport
         * @description One fit's provenance, run metadata, native evidence and computed findings.
         */
        readonly InferenceReport: {
            readonly evidence: components["schemas"]["InferenceEvidence"];
            readonly core: components["schemas"]["InferenceReportCore"];
            readonly detail: components["schemas"]["InferenceReportDetail"];
        };
        /**
         * InferenceReportCore
         * @description Compact scientific report shared by snapshots and the full report.
         */
        readonly InferenceReportCore: {
            readonly inference_metadata: components["schemas"]["InferenceMetadata"];
            readonly inference_diagnostics: components["schemas"]["ChainDiagnostics"] | null;
            readonly convergence: components["schemas"]["ConvergenceReport"];
            /** @default null */
            readonly loo_diagnostics: components["schemas"]["LOODiagnostics"] | null;
            /** Posterior Marginals */
            readonly posterior_marginals: readonly components["schemas"]["PosteriorMarginal"][];
            /**
             * Prior Densities
             * @description Input laws evaluated on the quantity scale of the posterior summaries.
             */
            readonly prior_densities: Readonly<Partial<Record<components["schemas"]["ParameterId-Output"], components["schemas"]["DensityCurve"]>>>;
            /**
             * Frequency Refresh
             * @default []
             */
            readonly frequency_refresh: readonly components["schemas"]["FrequencyRefresh"][];
            /**
             * Regional Movement
             * @default []
             */
            readonly regional_movement: readonly components["schemas"]["RegionalMovement"][];
        };
        /**
         * InferenceReportDetail
         * @description Retained plot series served in full by the action result.
         */
        readonly InferenceReportDetail: {
            /**
             * Trace Data
             * @default []
             */
            readonly trace_data: readonly components["schemas"]["TraceSeries"][];
            /**
             * Rank Histograms
             * @default []
             */
            readonly rank_histograms: readonly components["schemas"]["RankHistogram"][];
            /**
             * Pareto K
             * @default []
             */
            readonly pareto_k: readonly components["schemas"]["ParetoKPoint"][];
            /**
             * Loo Pit
             * @default []
             */
            readonly loo_pit: readonly components["schemas"]["LOOPITPoint"][];
        };
        /**
         * InitialCorrelationTransformSpec
         * @description Constrain an initial-state correlation to its scientific support [-1, 1].
         */
        readonly "InitialCorrelationTransformSpec-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "initial_state_correlation";
        };
        readonly InitializationDiagnostics: components["schemas"]["GaussianInitializationDiagnostics"] | components["schemas"]["ContinuedInitializationDiagnostics"];
        /**
         * IntervalEffectTransformSpec
         * @description Divide an interval effect by its explicit duration in days.
         */
        readonly "IntervalEffectTransformSpec-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "dt_effect_to_ct_rate";
            /** Interval Days */
            readonly interval_days: number | "model_clock";
        };
        /**
         * InterventionSpec
         * @description Set a state some time after the design's start, then let its dynamics resume.
         */
        readonly "InterventionSpec-Input": {
            readonly target: components["schemas"]["ConstructId-Input"];
            /**
             * After
             * @description Offset from the design's start; omitted means at the start.
             * @default null
             */
            readonly after?: string | null;
            /** Value */
            readonly value: number;
        };
        /**
         * InterventionSpec
         * @description Set a state some time after the design's start, then let its dynamics resume.
         */
        readonly "InterventionSpec-Output": {
            readonly target: components["schemas"]["ConstructId-Output"];
            /**
             * After
             * @description Offset from the design's start; omitted means at the start.
             * @default null
             */
            readonly after: string | null;
            /** Value */
            readonly value: number;
        };
        /**
         * JointLawLayout
         * @description An explicit law membership ordered independently of native tensor positions.
         *
         *     Parameters sort by identity, then by scientific element identity. Trajectories
         *     follow in construct-identity order, retaining the supplied time-point order.
         *     Element identities include their categorical or covariance basis.
         */
        readonly "JointLawLayout-Output": {
            /** Parameters */
            readonly parameters: readonly (readonly [
                components["schemas"]["ParameterId-Output"],
                readonly components["schemas"]["ParameterElementId-Output"][]
            ])[];
            /** Constructs */
            readonly constructs: readonly components["schemas"]["ConstructId-Output"][];
            /** Time Points */
            readonly time_points: readonly number[];
            /**
             * Time Origin
             * @description Calendar instant of model day zero, or relative coordinates bound at execution. Fitting retains its calendar origin with the law.
             * @default relative
             */
            readonly time_origin: string | "relative";
            /** Labels */
            readonly labels: Readonly<Partial<Record<components["schemas"]["ParameterElementId-Output"], string>>>;
        };
        /** @enum {string} */
        readonly JointParameterization: "centred" | "non_centred" | "mixed";
        readonly JointParameterizationRecord: components["schemas"]["CentredParameterizationRecord"] | components["schemas"]["NonCentredParameterizationRecord"] | components["schemas"]["MixedParameterizationRecord"];
        readonly "JointTrajectorySpec-Input": components["schemas"]["NutsTrajectorySpec-Input"] | components["schemas"]["FixedTrajectorySpec-Input"];
        readonly "JointTrajectorySpec-Output": components["schemas"]["NutsTrajectorySpec-Output"] | components["schemas"]["FixedTrajectorySpec-Output"];
        /**
         * JointTuning
         * @description Joint coordinates are reconstructed from the original model and trace seed.
         */
        readonly JointTuning: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "joint";
            readonly parameterization: components["schemas"]["JointParameterizationRecord"];
        };
        /** @description A JSON scalar transports a string, number, boolean, or null. */
        readonly JsonScalar: boolean | number | string | null;
        /**
         * LLMTrace
         * @description An LLM trace records a conversation, its model, elapsed time, and token usage.
         */
        readonly LLMTrace: {
            /** Messages */
            readonly messages: readonly components["schemas"]["TraceMessage"][];
            /**
             * Model
             * @default
             */
            readonly model: string;
            /**
             * Total Time Seconds
             * @default 0
             */
            readonly total_time_seconds: number;
            readonly usage: components["schemas"]["TraceUsage"];
        };
        /**
         * LOODiagnostics
         * @description Exact-emission leave-one-measurement-row-out interpolation diagnostics.
         */
        readonly LOODiagnostics: {
            /** Elpd Loo */
            readonly elpd_loo: number;
            /** P Loo */
            readonly p_loo: number;
            /** Se */
            readonly se: number;
            /** N Data Points */
            readonly n_data_points: number;
            /**
             * N Bad K
             * @default null
             */
            readonly n_bad_k: number | null;
            /**
             * N Warn K
             * @default null
             */
            readonly n_warn_k: number | null;
            /**
             * Pareto Warning Limit
             * @default 0.5
             */
            readonly pareto_warning_limit: number;
            /**
             * Pareto Failure Limit
             * @default 0.7
             */
            readonly pareto_failure_limit: number;
        };
        /**
         * LOOPITPoint
         * @description A retained PIT value and its empirical cumulative probability.
         */
        readonly LOOPITPoint: {
            /** Pit */
            readonly pit: number;
            /** Ecdf */
            readonly ecdf: number;
        };
        /** @enum {string} */
        readonly LabeledDtype: "ordinal" | "categorical";
        /**
         * LabeledMeasurement
         * @description Readings recorded as labels, coded 0..K-1 in level order.
         */
        readonly LabeledMeasurement: {
            /**
             * @description 'ordinal' or 'categorical' (enum property replaced by openapi-typescript)
             * @enum {string}
             */
            readonly dtype: "categorical" | "ordinal";
            /**
             * Levels
             * @description At least two distinct level labels, coded 0..K-1 in this order. Ordinal levels run from lowest to highest (e.g., ['low', 'medium', 'high']); categorical levels list every category (e.g., ['home', 'work', 'other']).
             */
            readonly levels: readonly [
                string,
                string,
                ...string[]
            ];
        };
        /** @enum {unknown} */
        readonly LatentBlockSize: 1 | 2 | "all";
        /**
         * LatentConvergenceSubject
         * @description A convergence criterion over a construct's independent latent coordinates.
         */
        readonly LatentConvergenceSubject: {
            readonly state: components["schemas"]["ConstructRef-Output"];
            /** Label */
            readonly label: string;
        };
        /**
         * LatentDiagnostics
         * @description Worst chain estimators across a construct's independent, unobserved time coordinates.
         */
        readonly LatentDiagnostics: {
            readonly subject: components["schemas"]["ConstructRef-Output"];
            /** Label */
            readonly label: string;
            /** Coordinates */
            readonly coordinates: number;
            /** R Hat */
            readonly r_hat: number | null;
            /** Ess Bulk */
            readonly ess_bulk: number | null;
            /** Ess Tail */
            readonly ess_tail: number | null;
            /** Mcse Mean */
            readonly mcse_mean: number | null;
        };
        readonly "LatentStepSpec-Input": components["schemas"]["AdaptedLatentStepSpec-Input"] | components["schemas"]["FixedLatentStepSpec-Input"];
        readonly "LatentStepSpec-Output": components["schemas"]["AdaptedLatentStepSpec-Output"] | components["schemas"]["FixedLatentStepSpec-Output"];
        /**
         * LikelihoodSpec
         * @description An indicator's conditional probability law and its scientific justification.
         */
        readonly "LikelihoodSpec-Output": {
            readonly law: components["schemas"]["ObservationLawSpec-Output"];
            /**
             * Standardized
             * @description Whether observations are mean-centered and scaled before fitting.
             * @default false
             */
            readonly standardized: boolean;
            /**
             * Reasoning
             * @description Why this conditional law was chosen for the indicator
             */
            readonly reasoning: string;
            /**
             * Sources
             * @default []
             */
            readonly sources: readonly components["schemas"]["LiteratureSource-Output"][];
        };
        /**
         * LiteralExpression
         * @description A finite scalar constant in a model equation.
         */
        readonly "LiteralExpression-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "literal";
            /** Value */
            readonly value: number;
        };
        /**
         * LiteratureSource
         * @description A literature source records cited evidence supporting a scientific modeling decision.
         */
        readonly "LiteratureSource-Output": {
            /**
             * Title
             * @description Title of the source (paper, meta-analysis, textbook, etc.)
             */
            readonly title: string;
            /**
             * Url
             * @description URL of the source if available
             * @default null
             */
            readonly url: string | null;
            /**
             * Snippet
             * @description Relevant excerpt or paraphrase from the source
             */
            readonly snippet: string;
        };
        /**
         * MCMCEvidence
         * @description The algorithm that sampled the exact joint state and parameter target.
         */
        readonly MCMCEvidence: {
            /**
             * Method
             * @default mpgibbs
             * @enum {string}
             */
            readonly method: "nuts_da" | "mpgibbs";
        };
        /**
         * MarginalParticleGibbsSpec
         * @description Marginalized parameter ensemble and exact conditional trajectory updates.
         *
         *     m-PGibbs (Corenflos 2025), the default sampler, proposes parameter particles around
         *     a shared auxiliary point, runs conditional SMC against the posterior mixture over
         *     them and selects a parameter together with the new path. Parameter moves are thus
         *     marginalized over paths, as in PMMH, instead of conditioned on one. The proposal
         *     covariance starts from the initializer and is re-estimated at the end of each
         *     warmup window (section 3.4).
         */
        readonly "MarginalParticleGibbsSpec-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly method: "mpgibbs";
            /**
             * N Parameter Particles
             * @description Parameter particles M per sweep: the current value and M - 1 proposals around one auxiliary point. Work grows as O(TNM); Corenflos (2025, section 5) recommends M = 2, given the weak performance of larger ensembles (Pozza & Zanella 2024).
             * @default 2
             */
            readonly n_parameter_particles?: number;
            /** @description untempered runs the exact posterior alone; gaussian_reference runs Gaussian-reference parallel tempering. */
            readonly tempering?: components["schemas"]["TemperingSpec-Input"];
            /** @description Exact path update. */
            readonly path_update?: components["schemas"]["PathUpdateSpec-Input"];
            /**
             * Compiler Reparameterization
             * @description Compute time-indexed path translations from Gauss-Newton mean cross curvature, including deterministic recurrences and interval readings. Build at mean initial parameters and their IEKS path; rebuild at cold-chain mean parameters and their IEKS path at warmup window ends. Freeze before retained draws and preserve the chart in native continuation state. Every density and reported path retains the exact physical model. Local slopes, strong nonlinearity and multiple modes can limit mixing.
             * @default false
             */
            readonly compiler_reparameterization?: boolean;
            /**
             * Scale Chart
             * @description Rescale each stochastic state's deviations from a center path by the change in its own diffusion scale, in the sampler's actual parameter coordinates: structural non-centring of the diffusion. The center is the mean warmup path in the translated chart, learned in the existing warmup windows and frozen before retained draws. Every label pays the exact log-Jacobian, and all densities and reported paths retain the exact physical model. Requires a guided path update and point readings without deterministic recurrences. Data-pinned deviations, a poor center or a state-dependent diffusion cost efficiency, never the target.
             * @default false
             */
            readonly scale_chart?: boolean;
            /**
             * @description random_walk splits a preconditioned Gaussian step into two symmetric halves through the auxiliary point. pseudo_langevin drifts the first half along the log-posterior gradient at a path fixed when the kernel is built, and corrects the asymmetry exactly in the selection weights; the current path would break that correction (Corenflos 2025, section 3.1). It is kept as the control for gradients within the same kernel.
             * @default random_walk
             */
            readonly parameter_proposal?: components["schemas"]["ParameterProposal"];
            /**
             * @description Optional traces to record: particle_identity (the selected particle per time step, reference-path hit rate and distinct selected particles), parameter_movement (RMS parameter jump) and particle_weights (dSMC weight ESS, entropy and spread per tree node). frequency_refresh summarizes retained path movement in DCT bands; regional_movement summarizes movement and frozen step sizes over existing question windows and reading coverage, with the ten worst times. They cost memory and output size, and no convergence check reads them; they explain a failing sampler.
             * @default []
             */
            readonly diagnostic_metrics?: components["schemas"]["ParticleDiagnosticMetrics"];
            /**
             * Param Target Accept
             * @description Target fraction of sweeps in which the parameters move to a proposed particle. Selection is a Barker-type step (Corenflos 2025, section 3.3), whose optimal rate for random-walk proposals is about 0.27 in one dimension and 0.158 in high dimension (Agrawal et al. 2023).
             * @default 0.2
             */
            readonly param_target_accept?: number;
            /**
             * @description How warmup adapts the parameter step: simple uses a constant gain and keeps the last step; dual_averaging keeps the averaged step at the end of warmup (Hoffman & Gelman 2014, section 3.2).
             * @default simple
             */
            readonly adaptation_scheme?: components["schemas"]["AdaptationScheme"];
            /** @description Independent Gaussian starts, which also set the initial proposal scale. */
            readonly initialization?: components["schemas"]["GaussianInitializationSpec-Input"];
        };
        /**
         * MarginalParticleGibbsSpec
         * @description Marginalized parameter ensemble and exact conditional trajectory updates.
         *
         *     m-PGibbs (Corenflos 2025), the default sampler, proposes parameter particles around
         *     a shared auxiliary point, runs conditional SMC against the posterior mixture over
         *     them and selects a parameter together with the new path. Parameter moves are thus
         *     marginalized over paths, as in PMMH, instead of conditioned on one. The proposal
         *     covariance starts from the initializer and is re-estimated at the end of each
         *     warmup window (section 3.4).
         */
        readonly "MarginalParticleGibbsSpec-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly method: "mpgibbs";
            /**
             * N Parameter Particles
             * @description Parameter particles M per sweep: the current value and M - 1 proposals around one auxiliary point. Work grows as O(TNM); Corenflos (2025, section 5) recommends M = 2, given the weak performance of larger ensembles (Pozza & Zanella 2024).
             * @default 2
             */
            readonly n_parameter_particles: number;
            /** @description untempered runs the exact posterior alone; gaussian_reference runs Gaussian-reference parallel tempering. */
            readonly tempering: components["schemas"]["TemperingSpec-Output"];
            /** @description Exact path update. */
            readonly path_update: components["schemas"]["PathUpdateSpec-Output"];
            /**
             * Compiler Reparameterization
             * @description Compute time-indexed path translations from Gauss-Newton mean cross curvature, including deterministic recurrences and interval readings. Build at mean initial parameters and their IEKS path; rebuild at cold-chain mean parameters and their IEKS path at warmup window ends. Freeze before retained draws and preserve the chart in native continuation state. Every density and reported path retains the exact physical model. Local slopes, strong nonlinearity and multiple modes can limit mixing.
             * @default false
             */
            readonly compiler_reparameterization: boolean;
            /**
             * Scale Chart
             * @description Rescale each stochastic state's deviations from a center path by the change in its own diffusion scale, in the sampler's actual parameter coordinates: structural non-centring of the diffusion. The center is the mean warmup path in the translated chart, learned in the existing warmup windows and frozen before retained draws. Every label pays the exact log-Jacobian, and all densities and reported paths retain the exact physical model. Requires a guided path update and point readings without deterministic recurrences. Data-pinned deviations, a poor center or a state-dependent diffusion cost efficiency, never the target.
             * @default false
             */
            readonly scale_chart: boolean;
            /**
             * @description random_walk splits a preconditioned Gaussian step into two symmetric halves through the auxiliary point. pseudo_langevin drifts the first half along the log-posterior gradient at a path fixed when the kernel is built, and corrects the asymmetry exactly in the selection weights; the current path would break that correction (Corenflos 2025, section 3.1). It is kept as the control for gradients within the same kernel.
             * @default random_walk
             */
            readonly parameter_proposal: components["schemas"]["ParameterProposal"];
            /**
             * @description Optional traces to record: particle_identity (the selected particle per time step, reference-path hit rate and distinct selected particles), parameter_movement (RMS parameter jump) and particle_weights (dSMC weight ESS, entropy and spread per tree node). frequency_refresh summarizes retained path movement in DCT bands; regional_movement summarizes movement and frozen step sizes over existing question windows and reading coverage, with the ten worst times. They cost memory and output size, and no convergence check reads them; they explain a failing sampler.
             * @default []
             */
            readonly diagnostic_metrics: components["schemas"]["ParticleDiagnosticMetrics"];
            /**
             * Param Target Accept
             * @description Target fraction of sweeps in which the parameters move to a proposed particle. Selection is a Barker-type step (Corenflos 2025, section 3.3), whose optimal rate for random-walk proposals is about 0.27 in one dimension and 0.158 in high dimension (Agrawal et al. 2023).
             * @default 0.2
             */
            readonly param_target_accept: number;
            /**
             * @description How warmup adapts the parameter step: simple uses a constant gain and keeps the last step; dual_averaging keeps the averaged step at the end of warmup (Hoffman & Gelman 2014, section 3.2).
             * @default simple
             */
            readonly adaptation_scheme: components["schemas"]["AdaptationScheme"];
            /** @description Independent Gaussian starts, which also set the initial proposal scale. */
            readonly initialization: components["schemas"]["GaussianInitializationSpec-Output"];
        };
        readonly Measurement: components["schemas"]["NumericMeasurement"] | components["schemas"]["LabeledMeasurement"];
        /** @description A persistent mechanism identity distinguishes additive terms through reordering and revision. */
        readonly "MechanismId-Input": `mechanism:${string}`;
        /** @description A persistent mechanism identity distinguishes additive terms through reordering and revision. */
        readonly "MechanismId-Output": `mechanism:${string}`;
        /**
         * MechanismRef
         * @description A particular additive term, independently of its position or coefficient values.
         */
        readonly MechanismRef: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "mechanism";
            readonly id: components["schemas"]["MechanismId-Output"];
        };
        /**
         * MixedLawProvenance
         * @description Some laws retain a committed fit and others have different ancestry.
         */
        readonly MixedLawProvenance: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "mixed";
            readonly fitted_data: Domain.DataRef<Domain.GitOid, number>;
            readonly fitted_model_revision: components["schemas"]["GitOid-Output"];
            /**
             * Interpretation
             * @description Mixed interpretation of draws whose parameter laws combine different provenance.
             * @constant
             */
            readonly interpretation: "mixed";
        };
        /**
         * MixedParameterizationRecord
         * @description Centred free coordinates, in the full checkpoint path's time/state axis order.
         */
        readonly MixedParameterizationRecord: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "mixed";
            /** Centred Mask */
            readonly centred_mask: readonly (readonly boolean[])[];
            /**
             * Mask Identity
             * @description Content identity of the actual mask, including its time and state dimensions.
             */
            readonly mask_identity: string;
            /**
             * Equals Non Centred
             * @description Whether the available-reading policy selects no centred free coordinate.
             */
            readonly equals_non_centred: boolean;
        };
        /**
         * ModelCheckReport
         * @description Findings evaluated by the action against its pinned scientific inputs.
         */
        readonly ModelCheckReport: {
            /** Specification */
            readonly specification: readonly components["schemas"]["SpecificationAssessment"][];
            readonly question: components["schemas"]["QuestionCheckReport"];
        };
        /**
         * ModelDiffInput[GitOid]
         * @description Base and target specs for a directional patch, selected by question or model revisions.
         *
         *     The selections need not be chronologically ordered or share an editing parent.
         *     Before/after determine patch direction; ModelDiffOutput defines its contract.
         */
        readonly ModelDiffInput_GitOid_: {
            /** @description Question or model revision used as the base; a question denotes an empty spec. */
            readonly before_ref: components["schemas"]["GitOid-Output"];
            /** @description Question or model revision reconstructed by applying the patch to the base. */
            readonly after_ref: components["schemas"]["GitOid-Output"];
        };
        /**
         * ModelDiffInput[RevisionSelector]
         * @description Base and target specs for a directional patch, selected by question or model revisions.
         *
         *     The selections need not be chronologically ordered or share an editing parent.
         *     Before/after determine patch direction; ModelDiffOutput defines its contract.
         */
        readonly ModelDiffInput_RevisionSelector_: {
            /** @description Question or model revision used as the base; a question denotes an empty spec. */
            readonly before_ref: components["schemas"]["RevisionSelector"];
            /** @description Question or model revision reconstructed by applying the patch to the base. */
            readonly after_ref: components["schemas"]["RevisionSelector"];
        };
        /**
         * ModelDiffOutput
         * @description Directional changes between saved specs, expressed in the model's editing language.
         *
         *     DynamicalModelSpec.changes_from owns the document comparison. Applying its patch with
         *     merge_fields to the before spec reconstructs the after spec; equal specs yield
         *     an empty document. Only saved specification fields participate, including their
         *     exact law-buffer references. No execution findings or statistical comparisons
         *     are computed here; those remain owned by their producing actions.
         */
        readonly ModelDiffOutput: {
            /** @description Merge this document into the before spec to obtain the after spec. Omitted fields are unchanged, supplied fields are added or updated, and null map entries delete their identities. A question revision denotes the empty spec. */
            readonly changes: components["schemas"]["DynamicalModelSpec-Input"];
        };
        /**
         * ModelDiffRequest[GitOid]
         * @description Compare two saved specs as an edit document; return ``ModelDiffOutput``.
         */
        readonly ModelDiffRequest_GitOid_: {
            /**
             * Action
             * @description Scientific action that owns this request or result.
             * @default model_diff
             * @constant
             */
            readonly action: "model_diff";
            /** @description Typed arguments of the scientific action. */
            readonly input: Domain.ModelDiffInput<Domain.GitOid>;
            /**
             * Reasoning
             * @description Why the caller is taking this action and what goal it serves. Retained with the original call and shown at the top of its action log; excluded from call identity.
             * @default null
             */
            readonly reasoning: string | null;
        };
        /**
         * ModelDiffRequest[RevisionSelector]
         * @description Compare two saved specs as an edit document; return ``ModelDiffOutput``.
         */
        readonly ModelDiffRequest_RevisionSelector_: {
            /**
             * @description Scientific action that owns this request or result. (enum property replaced by openapi-typescript)
             * @enum {string}
             */
            readonly action: "model_diff";
            /** @description Typed arguments of the scientific action. */
            readonly input: components["schemas"]["ModelDiffInput_RevisionSelector_"];
            /**
             * Reasoning
             * @description Why the caller is taking this action and what goal it serves. Retained with the original call and shown at the top of its action log; excluded from call identity.
             * @default null
             */
            readonly reasoning?: string | null;
        };
        /**
         * ModelEditResult
         * @description Scientific entities pruned while materializing an accepted model edit.
         */
        readonly ModelEditResult: {
            /**
             * Constructs
             * @default []
             */
            readonly constructs: readonly components["schemas"]["ConstructId-Output"][];
            /**
             * Edges
             * @default []
             */
            readonly edges: readonly components["schemas"]["EdgeId-Output"][];
            /**
             * Parameters
             * @default []
             */
            readonly parameters: readonly components["schemas"]["ParameterId-Output"][];
            /**
             * Distributions
             * @default []
             */
            readonly distributions: readonly components["schemas"]["DistributionId-Output"][];
        };
        /**
         * ModelSnapshot
         * @description Scientific values selected from recorded action dependencies.
         */
        readonly ModelSnapshot: {
            /** @default null */
            readonly question: components["schemas"]["QuestionSpec-Output"] | null;
            /** @default null */
            readonly dynamical_model_spec: components["schemas"]["DynamicalModelSpec-Output"] | null;
            /** Workspace Id */
            readonly workspace_id: string;
            readonly commit_id: components["schemas"]["GitOid-Output"];
            /** Selected Seq */
            readonly selected_seq: number;
            readonly state: components["schemas"]["StudyState"];
            /** @default null */
            readonly metadata: components["schemas"]["PreparedDataMetadata"] | null;
            /** @default null */
            readonly profile: components["schemas"]["DataProfileReport"] | null;
            /** @default null */
            readonly identification: components["schemas"]["IdentificationReport"] | null;
            /** @default null */
            readonly fit_checks: components["schemas"]["FitCheckReport"] | null;
            /** @default null */
            readonly fit: components["schemas"]["InferenceReportCore"] | null;
            /**
             * Specification
             * @default null
             */
            readonly specification: readonly components["schemas"]["SpecificationAssessment"][] | null;
            /** @default null */
            readonly question_checks: components["schemas"]["QuestionCheckReport"] | null;
            /** @default null */
            readonly simulation: components["schemas"]["SimulationReport"] | null;
        };
        /**
         * MovementQuantiles
         * @description Minimum, median and maximum across selected time coordinates or native delta values.
         */
        readonly MovementQuantiles: {
            /** Minimum */
            readonly minimum: number;
            /** Median */
            readonly median: number;
            /** Maximum */
            readonly maximum: number;
        };
        /**
         * MovementRegion
         * @description Refresh and final native delta on independent coordinates in one window.
         *
         *     For adapted kappa=1, delta is the frozen scalar multiplier of the coordinate
         *     geometry; these are not the effective per-coordinate step sizes.
         */
        readonly MovementRegion: {
            readonly window: components["schemas"]["MovementWindow"];
            /** Coordinates */
            readonly coordinates: number;
            readonly refresh_quantiles: components["schemas"]["MovementQuantiles"] | null;
            readonly delta_quantiles: components["schemas"]["MovementQuantiles"] | null;
        };
        /**
         * MovementWindow
         * @description An existing question's half-open window expressed in retained model days.
         */
        readonly MovementWindow: {
            /** Name */
            readonly name: string;
            /** Start Day */
            readonly start_day: number;
            /** End Day */
            readonly end_day: number;
        };
        /**
         * NegativeBinomial2LawSpec[Expression]
         * @description The NegativeBinomial2 conditional law.
         */
        readonly "NegativeBinomial2LawSpec_Expression_-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "NegativeBinomial2";
            readonly mean: components["schemas"]["Expression-Output"];
            readonly concentration: components["schemas"]["Expression-Output"];
        };
        /**
         * NewFitTarget[GitOid]
         * @description Condition the selected model law on exactly one saved history.
         */
        readonly NewFitTarget_GitOid_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "new";
            readonly dynamical_model_spec_ref: components["schemas"]["GitOid-Output"];
            readonly data_ref: Domain.DataRef<Domain.GitOid, number>;
        };
        /**
         * NewFitTarget[RevisionSelector]
         * @description Condition the selected model law on exactly one saved history.
         */
        readonly NewFitTarget_RevisionSelector_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "new";
            readonly dynamical_model_spec_ref: components["schemas"]["RevisionSelector"];
            readonly data_ref: components["schemas"]["DataRef_RevisionSelector_int_"];
        };
        /**
         * NonCentredParameterizationRecord
         * @description Exact conditional innovations in every unpinned independent path coordinate.
         */
        readonly NonCentredParameterizationRecord: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "non_centred";
        };
        /**
         * NonIdentifiableTreatmentStatus
         * @description Context on why a treatment effect is not identifiable.
         */
        readonly NonIdentifiableTreatmentStatus: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly status: "not_identified";
            /**
             * Confounders
             * @description Unobserved constructs blocking identification
             */
            readonly confounders: readonly components["schemas"]["ConstructId-Output"][];
            /**
             * Notes
             * @description Optional explanation if confounders cannot be enumerated
             * @default null
             */
            readonly notes: string | null;
        };
        /**
         * NormalLawSpec[Expression]
         * @description The Normal conditional law.
         */
        readonly "NormalLawSpec_Expression_-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "Normal";
            readonly loc: components["schemas"]["Expression-Output"];
            readonly scale: components["schemas"]["Expression-Output"];
        };
        readonly NotEvaluatedReason: components["schemas"]["PredictiveCheckReason"] | ("NONFINITE_EMISSION_MEAN" | "INSUFFICIENT_TIMES" | "NO_RELAXATION_TERM" | "EDGE_CONTRASTS_EXPLICIT" | "NO_OBSERVATION_SUPPORT" | "NO_OBSERVATIONS" | "STATIC_CONSTRUCT" | "INSUFFICIENT_OBSERVATIONS" | "ZERO_RESIDUAL_VARIANCE" | "ZERO_OBSERVED_VARIANCE" | "NONFINITE_PATHS" | "NONFINITE_SIGNAL" | "COMPARISON_INPUTS_MISSING" | "MISSING_REPLICATE_VALUES" | "CAUSAL_EVALUATION_FAILED" | "INSUFFICIENT_CHAIN_SAMPLES" | "NO_RETAINED_CHAINS" | "NO_OUTCOME" | "CONSTRUCT_UNDEFINED" | "NO_PANEL" | "STATE_NOT_RECORDED");
        /**
         * NotEvaluated[ConvergenceAssessmentSubject]
         * @description An explicit absence of evaluation, with a closed producer reason.
         */
        readonly NotEvaluated_ConvergenceAssessmentSubject_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "not_evaluated";
            /** Code */
            readonly code: string;
            readonly subject: components["schemas"]["ConvergenceAssessmentSubject"];
            readonly reason: components["schemas"]["NotEvaluatedReason"];
            /** Detail */
            readonly detail: string;
        };
        /**
         * NotEvaluated[IndicatorCheckSubject]
         * @description An explicit absence of evaluation, with a closed producer reason.
         */
        readonly NotEvaluated_IndicatorCheckSubject_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "not_evaluated";
            /** Code */
            readonly code: string;
            readonly subject: components["schemas"]["IndicatorCheckSubject"];
            readonly reason: components["schemas"]["NotEvaluatedReason"];
            /** Detail */
            readonly detail: string;
        };
        /**
         * NotEvaluated[IndicatorRef]
         * @description An explicit absence of evaluation, with a closed producer reason.
         */
        readonly NotEvaluated_IndicatorRef_: {
            /**
             * Kind
             * @default not_evaluated
             * @constant
             */
            readonly kind: "not_evaluated";
            /** Code */
            readonly code: string;
            readonly subject: components["schemas"]["IndicatorRef"];
            readonly reason: components["schemas"]["NotEvaluatedReason"];
            /** Detail */
            readonly detail: string;
        };
        /**
         * NotEvaluated[Literal['causal_effect']]
         * @description An explicit absence of evaluation, with a closed producer reason.
         */
        readonly NotEvaluated_Literal__causal_effect___: {
            /**
             * Kind
             * @default not_evaluated
             * @constant
             */
            readonly kind: "not_evaluated";
            /** Code */
            readonly code: string;
            /**
             * Subject
             * @constant
             */
            readonly subject: "causal_effect";
            readonly reason: components["schemas"]["NotEvaluatedReason"];
            /** Detail */
            readonly detail: string;
        };
        /**
         * NotEvaluated[PredictiveSubject]
         * @description An explicit absence of evaluation, with a closed producer reason.
         */
        readonly NotEvaluated_PredictiveSubject_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "not_evaluated";
            /** Code */
            readonly code: string;
            readonly subject: components["schemas"]["PredictiveSubject"];
            readonly reason: components["schemas"]["NotEvaluatedReason"];
            /** Detail */
            readonly detail: string;
        };
        /**
         * NotEvaluated[QuestionSubject]
         * @description An explicit absence of evaluation, with a closed producer reason.
         */
        readonly NotEvaluated_QuestionSubject_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "not_evaluated";
            /** Code */
            readonly code: string;
            readonly subject: components["schemas"]["QuestionSubject"];
            readonly reason: components["schemas"]["NotEvaluatedReason"];
            /** Detail */
            readonly detail: string;
        };
        /**
         * NotEvaluated[Union[IndicatorRef, Literal['dataset']]]
         * @description An explicit absence of evaluation, with a closed producer reason.
         */
        readonly NotEvaluated_Union_IndicatorRef__Literal__dataset____: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "not_evaluated";
            /** Code */
            readonly code: string;
            /** Subject */
            readonly subject: components["schemas"]["IndicatorRef"] | "dataset";
            readonly reason: components["schemas"]["NotEvaluatedReason"];
            /** Detail */
            readonly detail: string;
        };
        /**
         * NotEvaluated[str]
         * @description An explicit absence of evaluation, with a closed producer reason.
         */
        readonly NotEvaluated_str_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "not_evaluated";
            /** Code */
            readonly code: string;
            /** Subject */
            readonly subject: string;
            readonly reason: components["schemas"]["NotEvaluatedReason"];
            /** Detail */
            readonly detail: string;
        };
        readonly "NumPyroArray-Input": readonly Domain.NumPyroValue[];
        readonly "NumPyroArray-Output": readonly Domain.NumPyroValue[];
        /** @description A native NumPyro probability distribution serialized by its constructor tree. */
        readonly "NumPyroDistribution-Output": {
            /** Distribution */
            readonly distribution: string;
            /** Params */
            readonly params: {
                readonly [key: string]: Domain.NumPyroValue;
            };
        };
        readonly "NumPyroObject-Input": {
            readonly [key: string]: Domain.NumPyroValue;
        };
        readonly "NumPyroObject-Output": {
            readonly [key: string]: Domain.NumPyroValue;
        };
        readonly "NumPyroValue-Input": Domain.JsonScalar | components["schemas"]["NumericalArray"] | Domain.NumPyroArray | Domain.NumPyroObject;
        readonly "NumPyroValue-Output": Domain.JsonScalar | components["schemas"]["NumericalArray"] | Domain.NumPyroArray | Domain.NumPyroObject;
        /**
         * NumericCriterionEvidence
         * @description A measured scalar and the producer's numerical acceptance region.
         */
        readonly NumericCriterionEvidence: {
            /** Criterion */
            readonly criterion: string;
            /** Value */
            readonly value: number;
            /**
             * Lower
             * @default null
             */
            readonly lower: number | null;
            /**
             * Upper
             * @default null
             */
            readonly upper: number | null;
            /**
             * Lower Inclusive
             * @default true
             */
            readonly lower_inclusive: boolean;
            /**
             * Upper Inclusive
             * @default true
             */
            readonly upper_inclusive: boolean;
            /**
             * Note
             * @default
             */
            readonly note: string;
            /**
             * Display Value
             * @default
             */
            readonly display_value: string;
            /**
             * Band Label
             * @default
             */
            readonly band_label: string;
        };
        /** @enum {string} */
        readonly NumericDtype: "continuous" | "binary" | "count";
        /**
         * NumericMeasurement
         * @description Readings recorded as numbers: real values, 0/1 outcomes or non-negative counts.
         */
        readonly NumericMeasurement: {
            /**
             * @description 'continuous', 'binary' or 'count' (enum property replaced by openapi-typescript)
             * @enum {string}
             */
            readonly dtype: "binary" | "continuous" | "count";
        };
        /**
         * NumericalArray
         * @description An immutable NPY buffer carried as a MessagePack binary value.
         *
         *     The NPY header owns dtype and shape. Buffers use little-endian, row-major
         *     storage; numerical decoding belongs to the boundary that consumes them.
         */
        readonly NumericalArray: {
            /**
             * Npy
             * Format: binary
             * @description Lossless NPY bytes, including dtype and dimensions, in MessagePack. JSON action responses replace the entire NumericalArray with nested lists in that shape (a scalar for a zero-dimensional array); non-finite values are null.
             */
            readonly npy: Uint8Array;
        };
        /**
         * NutsDADiagnostics
         * @description Realized NUTS adaptation and retained-tree saturation by vectorized chain.
         */
        readonly NutsDADiagnostics: {
            readonly settings: components["schemas"]["SamplerSpec"];
            /** Mcmc Phase Seconds */
            readonly mcmc_phase_seconds: number;
            /** Final Step Size */
            readonly final_step_size: readonly number[];
            /** Max Tree Depth Fraction */
            readonly max_tree_depth_fraction: number | null;
        };
        /**
         * NutsDASpec
         * @description Joint HMC with dual averaging and windowed diagonal mass adaptation.
         *
         *     It moves every parameter and free latent coordinate at once on the exact joint
         *     density, with no particle approximation. It also reconstructs deterministic
         *     recurrences that the particle samplers reject, and is a diagnostic when particle
         *     chains fail to converge.
         */
        readonly "NutsDASpec-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly method: "nuts_da";
            /**
             * @description Coordinates the sampler moves in; the posterior is the same in all three, only its geometry changes. centred samples each path directly, standardized by the prior's per-step covariances at the starting parameters. non_centred samples each free coordinate's Euler-Maruyama innovation and rebuilds the path from the current parameters. mixed keeps coordinates with a direct point reading centred and makes the rest non-centred. Centred mixes well where data pin the states, non-centred where data are weak against the dynamics (Papaspiliopoulos, Roberts & Sköld 2007). Frozen per fit; changing it on a continued fit requires warmup.
             * @default centred
             */
            readonly parameterization?: components["schemas"]["JointParameterization"];
            /**
             * Target Accept Prob
             * @description Mean acceptance statistic that dual averaging steers the step size towards during warmup (Hoffman & Gelman 2014, section 3.2). Raising it towards 0.9-0.99 shrinks the step, which removes divergences in strongly curved regions at the cost of longer trajectories. 0.8 is the NumPyro and Stan default.
             * @default 0.8
             */
            readonly target_accept_prob?: number;
            /** @description NUTS chooses each trajectory's length; fixed runs a set number of steps. */
            readonly trajectory?: components["schemas"]["JointTrajectorySpec-Input"];
        };
        /**
         * NutsDASpec
         * @description Joint HMC with dual averaging and windowed diagonal mass adaptation.
         *
         *     It moves every parameter and free latent coordinate at once on the exact joint
         *     density, with no particle approximation. It also reconstructs deterministic
         *     recurrences that the particle samplers reject, and is a diagnostic when particle
         *     chains fail to converge.
         */
        readonly "NutsDASpec-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly method: "nuts_da";
            /**
             * @description Coordinates the sampler moves in; the posterior is the same in all three, only its geometry changes. centred samples each path directly, standardized by the prior's per-step covariances at the starting parameters. non_centred samples each free coordinate's Euler-Maruyama innovation and rebuilds the path from the current parameters. mixed keeps coordinates with a direct point reading centred and makes the rest non-centred. Centred mixes well where data pin the states, non-centred where data are weak against the dynamics (Papaspiliopoulos, Roberts & Sköld 2007). Frozen per fit; changing it on a continued fit requires warmup.
             * @default centred
             */
            readonly parameterization: components["schemas"]["JointParameterization"];
            /**
             * Target Accept Prob
             * @description Mean acceptance statistic that dual averaging steers the step size towards during warmup (Hoffman & Gelman 2014, section 3.2). Raising it towards 0.9-0.99 shrinks the step, which removes divergences in strongly curved regions at the cost of longer trajectories. 0.8 is the NumPyro and Stan default.
             * @default 0.8
             */
            readonly target_accept_prob: number;
            /** @description NUTS chooses each trajectory's length; fixed runs a set number of steps. */
            readonly trajectory: components["schemas"]["JointTrajectorySpec-Output"];
        };
        /**
         * NutsTrajectorySpec
         * @description A dynamically terminated trajectory, bounded by tree depth (Hoffman & Gelman 2014).
         */
        readonly "NutsTrajectorySpec-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "nuts";
            /**
             * Max Tree Depth
             * @description Maximum number of trajectory doublings per transition, so at most 2^max_tree_depth - 1 leapfrog steps. The report gives the fraction of transitions that reach it; frequent hits mean the step is small against the posterior's longest scale, and each extra level can double the cost. 10 is the NumPyro and Stan default.
             * @default 10
             */
            readonly max_tree_depth?: number;
        };
        /**
         * NutsTrajectorySpec
         * @description A dynamically terminated trajectory, bounded by tree depth (Hoffman & Gelman 2014).
         */
        readonly "NutsTrajectorySpec-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "nuts";
            /**
             * Max Tree Depth
             * @description Maximum number of trajectory doublings per transition, so at most 2^max_tree_depth - 1 leapfrog steps. The report gives the fraction of transitions that reach it; frequent hits mean the step is small against the posterior's longest scale, and each extra level can double the cost. 10 is the NumPyro and Stan default.
             * @default 10
             */
            readonly max_tree_depth: number;
        };
        readonly ObservationData: Readonly<Partial<Record<components["schemas"]["IndicatorId-Output"], components["schemas"]["ObservationHistory"]>>>;
        /**
         * ObservationHistory
         * @description All prepared observations, their true anchors and their measurement support.
         */
        readonly ObservationHistory: {
            /** Times */
            readonly times: readonly number[];
            readonly values: components["schemas"]["ScalarValues"];
            readonly support_start: components["schemas"]["ScalarValues"];
            readonly support_end: components["schemas"]["ScalarValues"];
            /** Empirical */
            readonly empirical: readonly components["schemas"]["EmpiricalPoint"][];
        };
        readonly "ObservationLawSpec-Output": Domain.DeltaLawSpec<Domain.Expression> | Domain.NormalLawSpec<Domain.Expression> | Domain.StudentTLawSpec<Domain.Expression> | Domain.PoissonLawSpec<Domain.Expression> | Domain.GammaLawSpec<Domain.Expression> | Domain.BernoulliLogitsLawSpec<Domain.Expression> | Domain.BernoulliProbsLawSpec<Domain.Expression> | Domain.NegativeBinomial2LawSpec<Domain.Expression> | Domain.BetaLawSpec<Domain.Expression> | Domain.OrderedLogisticLawSpec<Domain.Expression> | Domain.CategoricalLawSpec<Domain.Expression>;
        /** ObservationSpec[Annotated[Union[ObservationWindow, NoneType], FieldInfo(annotation=NoneType, required=False, default=None)]] */
        readonly "ObservationSpec_Annotated_Union_ObservationWindow__NoneType___FieldInfo_annotation_NoneType__required_False__default_None___-Output": {
            /** @description Persistent identity. Preserve when revising or renaming. */
            readonly id: components["schemas"]["IndicatorId-Output"];
            /**
             * Name
             * @description Indicator name (e.g., 'hrv', 'self_reported_stress')
             */
            readonly name: string;
            /** @description How readings are recorded: {dtype: 'continuous' | 'binary' | 'count'}, or {dtype: 'ordinal' | 'categorical', levels: [...]} for labels coded 0..K-1. */
            readonly measurement: components["schemas"]["Measurement"];
            /** @description Aggregation function applied when bucketing raw extractions within the indicator support window. Supported operators: first, last, sum, count, mean, std. A computed_rule must produce this same summary. */
            readonly aggregation: components["schemas"]["SummaryOperator"];
            /**
             * @description Optional support window: positive fixed units s, m, h, d or w (for example '2w'), or whole calendar months/years parsed by pytimeparse2 (for example '3mo', '2 years', '1y6mo'). Calendar windows align to whole-month boundaries anchored to January 1970 and retain their actual lengths, including leap days. Resolved by the preparation window or the generative model clock.
             * @default null
             */
            readonly observation_window: components["schemas"]["ObservationWindow"] | null;
        };
        /** ObservationSpec[ObservationWindow] */
        readonly ObservationSpec_ObservationWindow_: {
            /** @description Persistent identity. Preserve when revising or renaming. */
            readonly id: components["schemas"]["IndicatorId-Output"];
            /**
             * Name
             * @description Indicator name (e.g., 'hrv', 'self_reported_stress')
             */
            readonly name: string;
            /** @description How readings are recorded: {dtype: 'continuous' | 'binary' | 'count'}, or {dtype: 'ordinal' | 'categorical', levels: [...]} for labels coded 0..K-1. */
            readonly measurement: components["schemas"]["Measurement"];
            /** @description Aggregation function applied when bucketing raw extractions within the indicator support window. Supported operators: first, last, sum, count, mean, std. A computed_rule must produce this same summary. */
            readonly aggregation: components["schemas"]["SummaryOperator"];
            /** @description Optional support window: positive fixed units s, m, h, d or w (for example '2w'), or whole calendar months/years parsed by pytimeparse2 (for example '3mo', '2 years', '1y6mo'). Calendar windows align to whole-month boundaries anchored to January 1970 and retain their actual lengths, including leap days. Resolved by the preparation window or the generative model clock. */
            readonly observation_window: components["schemas"]["ObservationWindow"];
        };
        readonly ObservationWindow: string;
        /**
         * OrderedLogisticLawSpec[Expression]
         * @description The OrderedLogistic conditional law.
         */
        readonly "OrderedLogisticLawSpec_Expression_-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "OrderedLogistic";
            readonly predictor: components["schemas"]["Expression-Output"];
            readonly cutpoints: components["schemas"]["Expression-Output"];
        };
        /**
         * OutcomeSubject
         * @description The outcome construct selected by the study question.
         */
        readonly OutcomeSubject: {
            readonly outcome: components["schemas"]["ConstructRef-Output"];
        };
        /**
         * PPCOverlay
         * @description A predictive overlay sets one indicator's observed values against simulated ones.
         *
         *     It carries the predictive median and a few individual replicated series, the
         *     spaghetti plot of a visual predictive check.
         */
        readonly PPCOverlay: {
            /** Times */
            readonly times: readonly number[];
            /**
             * Time Origin
             * Format: date-time
             */
            readonly time_origin: string;
            /** Standardized */
            readonly standardized: boolean;
            readonly indicator_id: components["schemas"]["IndicatorId-Output"];
            /** Observed */
            readonly observed: readonly (number | null)[];
            /** Median */
            readonly median: readonly (number | null)[];
            /** Spaghetti Draws */
            readonly spaghetti_draws: readonly (readonly (number | null)[])[];
            /**
             * Frame
             * @description Plot range spanning the widest per-time central 95% of draws and every observed value.
             */
            readonly frame: readonly [
                number,
                number
            ] | null;
        };
        /** @enum {string} */
        readonly PPCStatistic: "mean" | "sd" | "min" | "max";
        /**
         * PPCTestStat
         * @description An observed summary compared with the same statistic across predictive replicates.
         *
         *     Provides the data for Gabry's ppc_stat plots: a histogram of T(y_rep)
         *     with a vertical line at T(y_observed).
         */
        readonly PPCTestStat: {
            readonly indicator_id: components["schemas"]["IndicatorId-Output"];
            readonly stat_name: components["schemas"]["PPCStatistic"];
            /** Observed Value */
            readonly observed_value: number;
            /** Rep Values */
            readonly rep_values: readonly number[];
            /** P Value */
            readonly p_value: number | null;
            /** Histogram */
            readonly histogram: readonly components["schemas"]["HistogramBin"][];
            /**
             * Frame
             * @description Value range charts show: the replicates' central 95%, covering the observed value.
             */
            readonly frame: readonly [
                number,
                number
            ] | null;
        };
        /**
         * PairedArmSimulation
         * @description Intervention and natural-course histories with matched draw indices.
         */
        readonly PairedArmSimulation: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "paired";
            readonly action: components["schemas"]["SimulationArm"];
            readonly reference: components["schemas"]["SimulationArm"];
            /** Causal */
            readonly causal: components["schemas"]["CausalEffectResult"] | Domain.NotEvaluated<"causal_effect">;
        };
        /**
         * ParameterDiagnostics
         * @description Measurements on one scientifically identified retained scalar chain.
         */
        readonly ParameterDiagnostics: {
            /** Parameter */
            readonly parameter: string;
            readonly subject: components["schemas"]["ParameterRef"];
            /** R Hat */
            readonly r_hat: number | null;
            /** Ess Bulk */
            readonly ess_bulk: number | null;
            /** Ess Tail */
            readonly ess_tail: number | null;
            /** Mcse Mean */
            readonly mcse_mean: number | null;
        };
        /** @description A parameter element identity identifies a logical scalar component across model revisions. */
        readonly "ParameterElementId-Input": `element:${string}`;
        /** @description A parameter element identity identifies a logical scalar component across model revisions. */
        readonly "ParameterElementId-Output": `element:${string}`;
        /** @description A scientific parameter identity connects component coefficients to one parameter definition. */
        readonly "ParameterId-Input": `parameter:${string}`;
        /** @description A scientific parameter identity connects component coefficients to one parameter definition. */
        readonly "ParameterId-Output": `parameter:${string}`;
        /** @enum {string} */
        readonly ParameterProposal: "random_walk" | "pseudo_langevin";
        /**
         * ParameterRef
         * @description A scalar finding identifies its scientific parameter and declared logical component.
         */
        readonly ParameterRef: {
            readonly parameter_id: components["schemas"]["ParameterId-Output"];
            readonly element_id: components["schemas"]["ParameterElementId-Output"];
        };
        /**
         * ParameterSpec
         * @description A named uncertain quantity; fixed coefficients are literals in component slots.
         */
        readonly "ParameterSpec-Output": {
            readonly id: components["schemas"]["ParameterId-Output"];
            /**
             * Name
             * @description Authored parameter label; relationships use its persistent ID
             */
            readonly name: string;
            /**
             * Description
             * @description Human-readable description of what this parameter represents
             */
            readonly description: string;
            readonly transform: components["schemas"]["ParameterTransformSpec-Output"];
            /**
             * @description Membership in a native law in DynamicalModelSpec.distributions; may be joint. None means the law has not been assigned yet.
             * @default null
             */
            readonly distribution: components["schemas"]["DistributionId-Output"] | null;
            /**
             * Reasoning
             * @description Why the authored prior law fits this quantity, and where its values come from.
             * @default null
             */
            readonly reasoning: string | null;
            /**
             * Sources
             * @description Evidence behind the authored prior law.
             * @default []
             */
            readonly sources: readonly components["schemas"]["LiteratureSource-Output"][];
        };
        readonly "ParameterTransformSpec-Output": components["schemas"]["IdentityTransformSpec-Output"] | components["schemas"]["PersistenceTransformSpec-Output"] | components["schemas"]["IntervalEffectTransformSpec-Output"] | components["schemas"]["InitialCorrelationTransformSpec-Output"];
        /**
         * ParetoKPoint
         * @description One PSIS influence measurement with its original row and scientific class.
         */
        readonly ParetoKPoint: {
            /** Timestep */
            readonly timestep: number;
            /** K */
            readonly k: number | ("infinity" | "-infinity" | "undefined");
            readonly status: components["schemas"]["CheckStatus"];
        };
        /** @enum {string} */
        readonly ParticleDiagnosticMetric: "particle_identity" | "parameter_movement" | "particle_weights" | "frequency_refresh" | "regional_movement";
        readonly ParticleDiagnosticMetrics: readonly components["schemas"]["ParticleDiagnosticMetric"][];
        /**
         * ParticleProfileDimensions
         * @description Particle-kernel work dimensions, including any temperature replicas.
         */
        readonly ParticleProfileDimensions: {
            /** Particles */
            readonly particles: number;
            /** Labels */
            readonly labels: number;
            /** Replicas */
            readonly replicas: number;
        };
        /**
         * ParticleSamplerDiagnostics
         * @description Typed exact-sampler settings and transition telemetry from the native producer.
         */
        readonly ParticleSamplerDiagnostics: {
            readonly settings: components["schemas"]["SamplerSpec"];
            /** Parameter Kernel */
            readonly parameter_kernel: string;
            /** Mcmc Phase Seconds */
            readonly mcmc_phase_seconds: number;
            /** Drift */
            readonly drift: string;
            /** Diagnostic Metrics */
            readonly diagnostic_metrics: readonly components["schemas"]["ParticleDiagnosticMetric"][];
            /** Param Target Accept */
            readonly param_target_accept: number;
            /** Parameter Preconditioned */
            readonly parameter_preconditioned: boolean;
            readonly diagnostic_summary_phase: components["schemas"]["DiagnosticPhase"];
            /** Parameter Accept Rate */
            readonly parameter_accept_rate: number;
            /** Latent Update Fraction */
            readonly latent_update_fraction: number;
            /** Latent Frozen Fraction */
            readonly latent_frozen_fraction: number;
            readonly latent_block_coords: components["schemas"]["LatentBlockSize"];
            /** Initial Param Step Size */
            readonly initial_param_step_size: readonly number[];
            /** Final Param Step Size */
            readonly final_param_step_size: readonly number[];
            /**
             * Latent Sign Flip Moves
             * @default null
             */
            readonly latent_sign_flip_moves: boolean | null;
            /** Chain Post Warmup Complete Log Posterior Mean */
            readonly chain_post_warmup_complete_log_posterior_mean: readonly number[];
            /**
             * Latent Move Rms Mean
             * @default null
             */
            readonly latent_move_rms_mean: number | null;
            /**
             * Parameter Jump Rms Mean
             * @default null
             */
            readonly parameter_jump_rms_mean: number | null;
            /**
             * Reference Path Hit Rate Mean
             * @default null
             */
            readonly reference_path_hit_rate_mean: number | null;
            /**
             * Selected Particle Unique Count Mean
             * @default null
             */
            readonly selected_particle_unique_count_mean: number | null;
            /**
             * Amala Grad Norm Mean
             * @default null
             */
            readonly amala_grad_norm_mean: number | null;
            /**
             * Amala Grad Norm Max
             * @default null
             */
            readonly amala_grad_norm_max: number | null;
            readonly initialization: components["schemas"]["InitializationDiagnostics"];
            readonly path_weights: components["schemas"]["PathWeightDiagnostics"] | null;
            /** @default null */
            readonly tempering: components["schemas"]["TemperingDiagnostics"] | null;
        };
        /**
         * ParticleTuning
         * @description Frozen proposal geometry; these values are not posterior draws.
         */
        readonly ParticleTuning: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "particle";
            readonly parameter_chol: components["schemas"]["NumericalArray"];
            readonly pilot_mean: components["schemas"]["NumericalArray"];
            readonly pilot_variance: components["schemas"]["NumericalArray"];
            readonly parameter_reference_path: components["schemas"]["NumericalArray"];
            /**
             * Gaussian Reference
             * @default []
             */
            readonly gaussian_reference: readonly components["schemas"]["NumericalArray"][];
            readonly initialization: components["schemas"]["InitializationDiagnostics"];
        };
        /**
         * ParticleWeightDiagnostics
         * @description Candidate count and weight measurements for one resampling distribution.
         */
        readonly ParticleWeightDiagnostics: {
            /** Candidates */
            readonly candidates: number;
            /** Metrics */
            readonly metrics: Readonly<Partial<Record<components["schemas"]["ParticleWeightMetric"], components["schemas"]["ParticleWeightSummary"]>>>;
        };
        /**
         * ParticleWeightMetric
         * @description Measurements of a categorical particle law, before resampling.
         *
         *     ESS counts weighted candidates, not independent MCMC draws. Entropy uses
         *     natural logarithms. Reference probability is the mass of candidate zero,
         *     including pair (0, 0) at merges, not the mass of all duplicate paths.
         *     Log-weight range and variance use finite log weights only; finite_fraction
         *     separately records their share of candidates. Invalid laws remain nonfinite.
         * @enum {string}
         */
        readonly ParticleWeightMetric: "ess" | "ess_fraction" | "entropy" | "max_probability" | "reference_probability" | "finite_log_weight_range" | "finite_log_weight_variance" | "finite_fraction";
        /**
         * ParticleWeightSummary
         * @description Aggregate across chains and sweeps, without dropping nonfinite measurements.
         *
         *     An undefined or infinite aggregate is null; nonfinite_fraction retains the
         *     frequency of nonfinite measurements instead of concealing them in an average.
         */
        readonly ParticleWeightSummary: {
            /** Mean */
            readonly mean: number | null;
            /** Minimum */
            readonly minimum: number | null;
            /** Maximum */
            readonly maximum: number | null;
            /** Nonfinite Fraction */
            readonly nonfinite_fraction: number;
        };
        readonly "PathUpdateSpec-Input": components["schemas"]["DsmcSpec-Input"] | components["schemas"]["GuidedPathSpec-Input"];
        readonly "PathUpdateSpec-Output": components["schemas"]["DsmcSpec-Output"] | components["schemas"]["GuidedPathSpec-Output"];
        readonly PathWeightDiagnostics: components["schemas"]["DsmcWeightDiagnostics"] | components["schemas"]["SequentialWeightDiagnostics"];
        /**
         * PersistenceTransformSpec
         * @description Map persistence p to -log(p) divided by its explicit interval in days.
         */
        readonly "PersistenceTransformSpec-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "dt_persistence_to_ct_decay";
            /** Interval Days */
            readonly interval_days: number | "model_clock";
        };
        /** @enum {string} */
        readonly PhaseOutcome: "complete" | "failed";
        /**
         * PoissonLawSpec[Expression]
         * @description The Poisson conditional law.
         */
        readonly "PoissonLawSpec_Expression_-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "Poisson";
            readonly rate: components["schemas"]["Expression-Output"];
        };
        /**
         * PosteriorMarginal
         * @description One parameter's posterior interval, scale and density plot.
         */
        readonly PosteriorMarginal: {
            /** Parameter */
            readonly parameter: string;
            readonly subject: components["schemas"]["ParameterRef"];
            readonly density_curve: components["schemas"]["DensityCurve"];
            /** Empirical */
            readonly empirical: readonly components["schemas"]["EmpiricalPoint"][];
            /** Mean */
            readonly mean: number;
            /** Lower */
            readonly lower: number;
            /** Upper */
            readonly upper: number;
            /**
             * Interval Kind
             * @enum {string}
             */
            readonly interval_kind: "hdi" | "equal_tail";
            /** Interval Mass */
            readonly interval_mass: number;
            /** Sd */
            readonly sd: number;
        };
        /**
         * PosteriorPredictiveChecks
         * @description Posterior predictive checks report exact-model checks and their supporting plot data.
         */
        readonly PosteriorPredictiveChecks: {
            /** Findings */
            readonly findings: readonly Domain.Assessment<Domain.IndicatorCheckSubject, Domain.NumericCriterionEvidence>[];
            /**
             * N Subsample
             * @default 0
             */
            readonly n_subsample: number;
            /** Overlays */
            readonly overlays: readonly components["schemas"]["PPCOverlay"][];
            /** Test Stats */
            readonly test_stats: readonly components["schemas"]["PPCTestStat"][];
        };
        /**
         * PotentialMechanismSpec
         * @description A construct potential whose negative gradient contributes to its drift.
         */
        readonly "PotentialMechanismSpec-Output": {
            readonly id: components["schemas"]["MechanismId-Output"];
            readonly expression: components["schemas"]["Expression-Output"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "potential";
        };
        readonly PredictiveAssessment: Domain.Evaluated<Domain.PredictiveSubject, readonly (Domain.NumericCriterionEvidence)[]> | Domain.NotEvaluated<Domain.PredictiveSubject>;
        /**
         * @description A predictive check reason explains why a battery could not be evaluated for the selected model and observations.
         * @enum {string}
         */
        readonly PredictiveCheckReason: "MODEL_INCOMPLETE" | "MODEL_NOT_EXECUTABLE" | "NO_COMPATIBLE_PANEL" | "INSUFFICIENT_OBSERVATION_TIMES" | "SIMULATION_UNSUPPORTED" | "ARCHIVED_MEASUREMENT_NOT_RETAINED";
        /**
         * PredictiveComparison
         * @description One reference history compared with the other side's replicated histories.
         *
         *     The singleton side owns the reference role even when evaluation is unavailable.
         *     Swapping sides changes reference_side and preserves the evaluation. Matching
         *     measurement definitions, calendar binding, and support at every observed anchor
         *     are required. Missing reference values are masked; missing replicate values at
         *     observed anchors prevent evaluation. Replicates may have additional anchors.
         *     The source simulation's law provenance determines prior/posterior interpretation;
         *     the side names and number of selected histories do not establish that provenance.
         */
        readonly PredictiveComparison: {
            readonly reference_side: components["schemas"]["ComparisonSide"];
            /** Evaluation */
            readonly evaluation: components["schemas"]["PosteriorPredictiveChecks"] | Domain.NotEvaluated<Domain.IndicatorRef>;
        };
        readonly PredictiveLawProvenance: components["schemas"]["AuthoredLawProvenance"] | components["schemas"]["FittedLawProvenance"] | components["schemas"]["MixedLawProvenance"] | components["schemas"]["UnknownLawProvenance"];
        /**
         * PredictiveSubject
         * @description One named check and its stable target in a construct's scientific context.
         */
        readonly PredictiveSubject: {
            /** @default null */
            readonly construct_id: components["schemas"]["ConstructId-Output"] | null;
            /** Target */
            readonly target: components["schemas"]["EntityRef"] | ("whole_model" | "observations");
        };
        /**
         * PrepareDataInput[GitOid, FileSourceRef]
         * @description Extract observations using the definitions of the selected model.
         */
        readonly PrepareDataInput_GitOid_FileSourceRef_: {
            /** @description Model revision owning the clock and observation definitions. */
            readonly dynamical_model_spec_ref: components["schemas"]["GitOid-Output"];
            /** @description Folder under `data/{workspace_id}/` at the HTTP boundary; the captured file reference in the saved request. */
            readonly source: components["schemas"]["FileSourceRef"];
            /**
             * Extraction
             * @description Computed, recorded-reading or semantic extraction instructions keyed by the model's observation IDs.
             */
            readonly extraction: Readonly<Partial<Record<components["schemas"]["IndicatorId-Output"], components["schemas"]["ExtractionSpec-Output"]>>>;
            /**
             * Context
             * @description Additional background for interpreting the uploaded source tables.
             * @default
             */
            readonly context: string;
        };
        /**
         * PrepareDataInput[RevisionSelector, SourceFolder]
         * @description Extract observations using the definitions of the selected model.
         */
        readonly PrepareDataInput_RevisionSelector_SourceFolder_: {
            /** @description Model revision owning the clock and observation definitions. */
            readonly dynamical_model_spec_ref: components["schemas"]["RevisionSelector"];
            /** @description Folder under `data/{workspace_id}/` at the HTTP boundary; the captured file reference in the saved request. */
            readonly source: components["schemas"]["SourceFolder"];
            /**
             * Extraction
             * @description Computed, recorded-reading or semantic extraction instructions keyed by the model's observation IDs.
             */
            readonly extraction: Readonly<Partial<Record<components["schemas"]["IndicatorId-Input"], components["schemas"]["ExtractionSpec-Input"]>>>;
            /**
             * Context
             * @description Additional background for interpreting the uploaded source tables.
             * @default
             */
            readonly context?: string;
        };
        /**
         * PrepareDataOutput
         * @description Prepared observations, their resolved metadata, and data-quality findings.
         */
        readonly PrepareDataOutput: {
            /** @description Full observation histories keyed by indicator ID, including measurement support intervals and missing values. */
            readonly data: components["schemas"]["ObservationData"];
            /** @description Preparation recipe, resolved observation schema, and the calendar origin used to interpret model time. */
            readonly metadata: components["schemas"]["PreparedDataMetadata"];
            /** @description Empirical statistics and data-quality findings for the prepared panel. */
            readonly profile: components["schemas"]["DataProfileReport"];
            /** @description Retained extraction workers, failed chunks and reuse counts. */
            readonly extraction: components["schemas"]["DataPreparationResult"];
        };
        /**
         * PrepareDataRequest[GitOid, FileSourceRef]
         * @description Prepare the selected source data; return ``PrepareDataOutput``.
         */
        readonly PrepareDataRequest_GitOid_FileSourceRef_: {
            /**
             * @description Scientific action that owns this request or result. (enum property replaced by openapi-typescript)
             * @enum {string}
             */
            readonly action: "prepare_data";
            /** @description Typed arguments of the scientific action. */
            readonly input: Domain.PrepareDataInput<Domain.GitOid, Domain.FileSourceRef>;
            /**
             * Reasoning
             * @description Why the caller is taking this action and what goal it serves. Retained with the original call and shown at the top of its action log; excluded from call identity.
             * @default null
             */
            readonly reasoning: string | null;
        };
        /**
         * PrepareDataRequest[RevisionSelector, SourceFolder]
         * @description Prepare the selected source data; return ``PrepareDataOutput``.
         */
        readonly PrepareDataRequest_RevisionSelector_SourceFolder_: {
            /**
             * @description Scientific action that owns this request or result. (enum property replaced by openapi-typescript)
             * @enum {string}
             */
            readonly action: "prepare_data";
            /** @description Typed arguments of the scientific action. */
            readonly input: components["schemas"]["PrepareDataInput_RevisionSelector_SourceFolder_"];
            /**
             * Reasoning
             * @description Why the caller is taking this action and what goal it serves. Retained with the original call and shown at the top of its action log; excluded from call identity.
             * @default null
             */
            readonly reasoning?: string | null;
        };
        /**
         * PreparedDataMetadata
         * @description An uploaded panel's recipe owns its resolved observation schema.
         */
        readonly PreparedDataMetadata: {
            readonly preparation: components["schemas"]["DataPreparationSpec"];
            /**
             * Time Origin
             * Format: date-time
             * @description Calendar instant of model day zero.
             */
            readonly time_origin: string;
            /**
             * Variables
             * @description Observation definitions resolved against the clock retained in the preparation recipe.
             */
            readonly variables: readonly Domain.ObservationSpec<Domain.ObservationWindow>[];
        };
        /**
         * ProgramCompileReport
         * @description One named request, from tracing until its executable is ready; excludes dumps.
         */
        readonly ProgramCompileReport: {
            /** Name */
            readonly name: string;
            /** Seconds */
            readonly seconds: number;
            /**
             * Persistent Cache
             * @description The server observes JAX's persistent-cache events in the compiling thread. Disabled means no configured persistent cache; not_consulted means no lookup event (for example, in-memory reuse). Unobserved is for a library-supplied compiler without this shell instrumentation.
             * @default unobserved
             * @enum {string}
             */
            readonly persistent_cache: "unobserved" | "disabled" | "not_consulted" | "miss" | "hit";
        };
        /** @description A progress event records one running attempt's step status or extraction telemetry. */
        readonly ProgressEvent: components["schemas"]["StepEvent"] | components["schemas"]["ExtractionPlanEvent"] | components["schemas"]["ExtractionWorkerEvent"] | components["schemas"]["ExtractionSnapshotEvent"];
        /**
         * ProgressMessage
         * @description A structured progress event with its original cursor and measurements.
         */
        readonly ProgressMessage: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "progress";
            /**
             * Timestamp
             * Format: date-time
             */
            readonly timestamp: string;
            readonly progress: components["schemas"]["ProgressEvent"];
        };
        /**
         * ProportionComparison
         * @description The fraction of observed values at one declared level, separately for each history.
         */
        readonly ProportionComparison: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly statistic: "proportion";
            /** Level */
            readonly level: string;
            /** Left */
            readonly left: readonly (number | null)[];
            /** Right */
            readonly right: readonly (number | null)[];
        };
        /** @description A query name labels one contrast of the study question. */
        readonly QueryName: string;
        /**
         * QueryTargetSubject
         * @description An intervention target in one named query.
         */
        readonly QueryTargetSubject: {
            /** Query */
            readonly query: string;
            readonly target: components["schemas"]["ConstructRef-Output"];
        };
        /**
         * QueryWindowSubject
         * @description The observation window of one named query.
         */
        readonly QueryWindowSubject: {
            /** Query */
            readonly query: string;
        };
        /** @description A question assessment records one check of the question against the model or the record. */
        readonly QuestionAssessment: Domain.Evaluated<Domain.QuestionSubject, string> | Domain.NotEvaluated<Domain.QuestionSubject>;
        /**
         * QuestionCheckReport
         * @description The study question checked against the model and, once prepared, the record.
         */
        readonly QuestionCheckReport: {
            /** Findings */
            readonly findings: readonly components["schemas"]["QuestionAssessment"][];
        };
        /**
         * QuestionSpec
         * @description What the study asks: the user's words, the outcome, and named contrasts.
         *
         *     Each query is a contrast of its interventions against the recorded course.
         *     Constructs are named by identity before a model defines them.
         */
        readonly "QuestionSpec-Input": {
            /**
             * Text
             * @description The user's question in their own words.
             */
            readonly text: string;
            /** @description The construct whose course answers the question. */
            readonly outcome: components["schemas"]["ConstructId-Input"];
            /**
             * Queries
             * @description Named contrasts against the recorded course, each with at least one intervention.
             */
            readonly queries?: Readonly<Partial<Record<components["schemas"]["QueryName"], components["schemas"]["SimulationSpec-Input"]>>>;
        };
        /**
         * QuestionSpec
         * @description What the study asks: the user's words, the outcome, and named contrasts.
         *
         *     Each query is a contrast of its interventions against the recorded course.
         *     Constructs are named by identity before a model defines them.
         */
        readonly "QuestionSpec-Output": {
            /**
             * Text
             * @description The user's question in their own words.
             */
            readonly text: string;
            /** @description The construct whose course answers the question. */
            readonly outcome: components["schemas"]["ConstructId-Output"];
            /**
             * Queries
             * @description Named contrasts against the recorded course, each with at least one intervention.
             */
            readonly queries: Readonly<Partial<Record<components["schemas"]["QueryName"], components["schemas"]["SimulationSpec-Output"]>>>;
        };
        /** @description A question check subject names the outcome, one query's window, or one query's intervention target. */
        readonly QuestionSubject: components["schemas"]["OutcomeSubject"] | components["schemas"]["QueryTargetSubject"] | components["schemas"]["QueryWindowSubject"];
        /**
         * Raised
         * @description An unexpected execution failure retained as part of an attempt.
         */
        readonly Raised: {
            /**
             * @description Discriminator identifying an exception outcome. (enum property replaced by openapi-typescript)
             * @enum {string}
             */
            readonly status: "raised";
            /**
             * Error Type
             * @description Exception or external failure type reported by the execution boundary.
             */
            readonly error_type: string;
            /**
             * Error Message
             * @description Human-readable failure explanation.
             */
            readonly error_message: string;
            /**
             * Details
             * @description Opaque external failure details retained for diagnosis.
             * @default []
             */
            readonly details: readonly string[];
        };
        /**
         * RankHistogram
         * @description Pooled-rank bin counts, grouped in original chain order.
         */
        readonly RankHistogram: {
            readonly subject: components["schemas"]["ParameterRef"];
            /** Expected Per Bin */
            readonly expected_per_bin: number;
            /** Chains */
            readonly chains: readonly (readonly number[])[];
        };
        /**
         * ReadingMovement
         * @description Pooled movement at free grid coordinates with or without any prepared reading.
         */
        readonly ReadingMovement: {
            /** Has Readings */
            readonly has_readings: boolean;
            /** Coordinates */
            readonly coordinates: number;
            /**
             * Stretches
             * @description Contiguous stretches on the retained time grid.
             */
            readonly stretches: number;
            readonly refresh_quantiles: components["schemas"]["MovementQuantiles"] | null;
            readonly delta_quantiles: components["schemas"]["MovementQuantiles"] | null;
        };
        /**
         * RecordDependency
         * @description A journal dependency linking a call argument to the attempt that first published its input.
         */
        readonly RecordDependency: {
            /**
             * Seq
             * @description Sequence number of the consuming attempt.
             */
            readonly seq: number;
            /**
             * Source Seq
             * @description Earlier sequence number that published the selected input.
             */
            readonly source_seq: number;
            /**
             * Argument
             * @description Input reference field name with its `_ref` suffix removed.
             */
            readonly argument: string;
        };
        /**
         * RecordedExtractionSpec
         * @description Import already measured readings and their actual support without re-bucketing.
         */
        readonly "RecordedExtractionSpec-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "recorded";
            /**
             * How To Measure
             * @default Import the recorded reading and temporal support unchanged.
             */
            readonly how_to_measure?: string;
            /** Value Column */
            readonly value_column: string;
            /**
             * Anchor Column
             * @default timestamp
             */
            readonly anchor_column?: string;
            /** Support Start Column */
            readonly support_start_column: string;
            /** Support End Column */
            readonly support_end_column: string;
        };
        /**
         * RecordedExtractionSpec
         * @description Import already measured readings and their actual support without re-bucketing.
         */
        readonly "RecordedExtractionSpec-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "recorded";
            /**
             * How To Measure
             * @default Import the recorded reading and temporal support unchanged.
             */
            readonly how_to_measure: string;
            /** Value Column */
            readonly value_column: string;
            /**
             * Anchor Column
             * @default timestamp
             */
            readonly anchor_column: string;
            /** Support Start Column */
            readonly support_start_column: string;
            /** Support End Column */
            readonly support_end_column: string;
        };
        /**
         * RegionalMovement
         * @description Declared windows and the slowest independent time coordinates of one construct.
         */
        readonly RegionalMovement: {
            readonly subject: components["schemas"]["ConstructRef-Output"];
            /** Label */
            readonly label: string;
            /** Regions */
            readonly regions: readonly components["schemas"]["MovementRegion"][];
            /** Reading Coverage */
            readonly reading_coverage: readonly components["schemas"]["ReadingMovement"][];
            /** Worst Times */
            readonly worst_times: readonly components["schemas"]["WorstTimeMovement"][];
        };
        /**
         * Rejected
         * @description An expected refusal to execute or publish a scientific request.
         */
        readonly Rejected: {
            /**
             * @description Discriminator identifying a rejected outcome. (enum property replaced by openapi-typescript)
             * @enum {string}
             */
            readonly status: "rejected";
            /** Code */
            readonly code: string;
            readonly subject: components["schemas"]["FindingSubject"];
            /**
             * Detail
             * @description Explanation of the specific condition that prevented application.
             */
            readonly detail: string;
        };
        /**
         * Removed[DataPoint]
         * @description The left payload exists at an identity absent on the right.
         */
        readonly Removed_DataPoint_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "removed";
            readonly before: components["schemas"]["DataPoint"];
        };
        /**
         * ReplicaSwapCounts
         * @description Accepted and attempted exchanges at one neighboring temperature pair.
         */
        readonly ReplicaSwapCounts: {
            /** Accepted */
            readonly accepted: number;
            /** Attempted */
            readonly attempted: number;
        };
        /**
         * RetractedArtifact
         * @description A current artifact removed by an action, with the finding that caused it.
         */
        readonly RetractedArtifact: {
            readonly artifact_id: components["schemas"]["ArtifactId"];
            /** Reason Ref */
            readonly reason_ref: string;
        };
        /**
         * Revised[DataPoint]
         * @description Both payloads exist at the same identity and their stored values differ.
         */
        readonly Revised_DataPoint_: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "revised";
            readonly before: components["schemas"]["DataPoint"];
            readonly after: components["schemas"]["DataPoint"];
        };
        /** @description An exact Git hash, or 'latest': the current non-stale model/panel or most recent applied simulation, according to the input type. Resolved once before cache lookup and execution; repeat the returned hashes to poll the same call. */
        readonly RevisionSelector: components["schemas"]["GitOid-Input"] | "latest";
        /**
         * Role
         * @description Whether a construct is modeled as endogenous or supplied as an exogenous input.
         * @enum {string}
         */
        readonly Role: "endogenous" | "exogenous";
        /**
         * RunningAction
         * @description Timeline discovery of the active call and its accumulated messages.
         */
        readonly RunningAction: {
            readonly call_id: components["schemas"]["CallId-Output"];
            readonly action: components["schemas"]["ActionId"];
            /** Request */
            readonly request: components["schemas"]["ScientificActionRequest"] | Domain.DataDiffRequest<Domain.GitOid> | Domain.ModelDiffRequest<Domain.GitOid>;
            /** Messages */
            readonly messages: readonly components["schemas"]["ExecutionMessage"][];
        };
        /**
         * RunningPoll
         * @description An accepted call that can be polled by an external action client.
         */
        readonly RunningPoll: {
            readonly call_id: components["schemas"]["CallId-Output"];
            readonly action: components["schemas"]["ActionId"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly status: "running";
            /**
             * Messages
             * @default []
             */
            readonly messages: readonly components["schemas"]["ExecutionMessage"][];
            /**
             * Commit Id
             * @description A running call has no published revision.
             */
            readonly commit_id: null;
            /**
             * Body
             * @description The shared envelope has no scientific result while running.
             */
            readonly body: null;
        };
        /**
         * SamplerCheckpoint
         * @description An exact native state at the last retained draw, with its next random key.
         */
        readonly SamplerCheckpoint: {
            readonly settings: components["schemas"]["SamplerSpec"];
            readonly parameters: components["schemas"]["NumericalArray"];
            readonly paths: components["schemas"]["NumericalArray"];
            readonly random_key: components["schemas"]["NumericalArray"];
            /** Native State */
            readonly native_state: readonly components["schemas"]["NumericalArray"][];
            readonly tuning: components["schemas"]["FrozenSamplerTuning"];
        };
        /**
         * SamplerSpec
         * @description Fully resolved controls for exact joint state and parameter inference.
         */
        readonly SamplerSpec: {
            /**
             * Num Warmup
             * @description Warmup sweeps per chain. Every adaptive step, scale, covariance and ladder is learned here and frozen before draws are retained. Zero keeps the tuning as it is: a continued fit's frozen tuning, or the starting values of a new fit. With zero samples as well, both sweep programs compile and neither runs. Switching the algorithm or joint parameterization on a continued fit requires warmup.
             * @default 4000
             */
            readonly num_warmup: number;
            /**
             * Num Samples Per Chain
             * @description Requested retained draws per chain after warmup, or a ceiling when stop_when_converged is set. A continued fit appends its actual retained draws. Zero returns checks without a conditioned model.
             * @default 1000
             */
            readonly num_samples_per_chain: number;
            /**
             * Num Chains
             * @description Independent chains, which the convergence checks compare: rank-normalized R-hat, and bulk and tail ESS of at least 100 per chain. Vehtari et al. (2021) recommend at least four. Tempering runs one ladder per chain.
             * @default 4
             */
            readonly num_chains: number;
            /**
             * @description Optional production convergence checks at declared retained-draw checkpoints. Warmup is unchanged; the first passing checkpoint at or above min_draws stops sampling. None retains the full requested count. Each continued execution checks only its newly retained draws.
             * @default null
             */
            readonly stop_when_converged: components["schemas"]["ConvergenceStoppingSpec"] | null;
            /**
             * Seed
             * @description Seeds initialization and sampling; another seed is an independent replicate.
             * @default 0
             */
            readonly seed: number;
            /**
             * @description local runs in the server process; any other value requests that Modal GPU type. Modal may run A100 on an 80 GB card at the same price, and H100 on an H200 unless it is written H100!. The fit records the device it ran on.
             * @default L40S
             */
            readonly hardware: components["schemas"]["InferenceHardware"];
            /**
             * Time Limit Seconds
             * @description Modal startup and execution timeout per phase. Reaching it fails the fit, with no retry. Local fits have no enforced timeout.
             * @default 14400
             */
            readonly time_limit_seconds: number;
            /** @description Exact joint sampler: mpgibbs (default) or nuts_da. */
            readonly algorithm: components["schemas"]["InferenceAlgorithmSpec-Output"];
        };
        /**
         * ScalarStatisticComparison
         * @description One statistic per whole selected history, in the report's source order.
         *
         *     Histories are never pooled or paired across sides. Null denotes an undefined
         *     statistic. Discrete codebooks use level proportions instead of numeric moments.
         *     These descriptive values include each history's own observed positions; the
         *     predictive checks separately use the reference history's observed-value mask.
         */
        readonly ScalarStatisticComparison: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly statistic: "max" | "mean" | "min" | "missing_count" | "observed_count" | "sd";
            /** Left */
            readonly left: readonly (number | null)[];
            /** Right */
            readonly right: readonly (number | null)[];
        };
        /** @description Scalar values are owned inline or selected from an owned numerical value. */
        readonly ScalarValues: readonly (number | null)[] | components["schemas"]["ArrayVector"];
        /**
         * @description A scientific action identity selects setting the question, model editing, data preparation, fitting, or simulation.
         * @enum {string}
         */
        readonly ScientificActionId: "edit_question" | "edit_model" | "prepare_data" | "fit" | "simulate";
        readonly ScientificActionRequest: components["schemas"]["EditQuestionRequest-Output"] | Domain.EditModelRequest<Domain.GitOid> | Domain.PrepareDataRequest<Domain.GitOid, Domain.FileSourceRef> | Domain.FitRequest<Domain.GitOid> | Domain.SimulateRequest<Domain.GitOid>;
        /**
         * SemanticExtractionSpec
         * @description Interpret source records using an explicit measurement rubric.
         */
        readonly "SemanticExtractionSpec-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "semantic";
            /**
             * How To Measure
             * @description Scoring rubric and extraction instructions.
             */
            readonly how_to_measure: string;
            /**
             * Source Columns
             * @description Source columns exposed to the extraction worker.
             * @default []
             */
            readonly source_columns?: readonly string[];
        };
        /**
         * SemanticExtractionSpec
         * @description Interpret source records using an explicit measurement rubric.
         */
        readonly "SemanticExtractionSpec-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "semantic";
            /**
             * How To Measure
             * @description Scoring rubric and extraction instructions.
             */
            readonly how_to_measure: string;
            /**
             * Source Columns
             * @description Source columns exposed to the extraction worker.
             * @default []
             */
            readonly source_columns: readonly string[];
        };
        /**
         * SequentialWeightDiagnostics
         * @description Conditional-filter weights in time order, including final path selection.
         */
        readonly SequentialWeightDiagnostics: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "sequential";
            readonly phase: components["schemas"]["DiagnosticPhase"];
            /** Filters */
            readonly filters: readonly components["schemas"]["ParticleWeightDiagnostics"][];
        };
        /**
         * SimulateInput[GitOid]
         * @description Generate histories from a selected model and an explicit simulation design.
         */
        readonly SimulateInput_GitOid_: {
            /** @description Revision supplying the authored or fitted generative law. */
            readonly dynamical_model_spec_ref: components["schemas"]["GitOid-Output"];
            /** @description Requested start, horizon, and interventions. */
            readonly simulation: components["schemas"]["SimulationSpec-Output"];
            /**
             * Seed
             * @description Generation seed, retained with the histories.
             * @default 0
             */
            readonly seed: number;
            /**
             * Num Replicates
             * @description Number of independent histories.
             * @default 100
             */
            readonly num_replicates: number;
        };
        /**
         * SimulateInput[RevisionSelector]
         * @description Generate histories from a selected model and an explicit simulation design.
         */
        readonly SimulateInput_RevisionSelector_: {
            /** @description Revision supplying the authored or fitted generative law. */
            readonly dynamical_model_spec_ref: components["schemas"]["RevisionSelector"];
            /** @description Requested start, horizon, and interventions. */
            readonly simulation: components["schemas"]["SimulationSpec-Input"];
            /**
             * Seed
             * @description Generation seed, retained with the histories.
             * @default 0
             */
            readonly seed?: number;
            /**
             * Num Replicates
             * @description Number of independent histories.
             * @default 100
             */
            readonly num_replicates?: number;
        };
        /**
         * SimulateOutput
         * @description Generated histories and their complete scientific report, including owned numerical evidence.
         */
        readonly SimulateOutput: {
            /**
             * Data
             * @description One observation history per retained replicate, in replicate order.
             */
            readonly data: readonly [
                components["schemas"]["ObservationData"],
                ...components["schemas"]["ObservationData"][]
            ];
            /** @description Generation evidence and evaluated findings. */
            readonly report: components["schemas"]["SimulationReport"];
        };
        /**
         * SimulateRequest[GitOid]
         * @description Generate observation histories and their complete report; return ``SimulateOutput``.
         */
        readonly SimulateRequest_GitOid_: {
            /**
             * @description Scientific action that owns this request or result. (enum property replaced by openapi-typescript)
             * @enum {string}
             */
            readonly action: "simulate";
            /** @description Typed arguments of the scientific action. */
            readonly input: Domain.SimulateInput<Domain.GitOid>;
            /**
             * Reasoning
             * @description Why the caller is taking this action and what goal it serves. Retained with the original call and shown at the top of its action log; excluded from call identity.
             * @default null
             */
            readonly reasoning: string | null;
        };
        /**
         * SimulateRequest[RevisionSelector]
         * @description Generate observation histories and their complete report; return ``SimulateOutput``.
         */
        readonly SimulateRequest_RevisionSelector_: {
            /**
             * @description Scientific action that owns this request or result. (enum property replaced by openapi-typescript)
             * @enum {string}
             */
            readonly action: "simulate";
            /** @description Typed arguments of the scientific action. */
            readonly input: components["schemas"]["SimulateInput_RevisionSelector_"];
            /**
             * Reasoning
             * @description Why the caller is taking this action and what goal it serves. Retained with the original call and shown at the top of its action log; excluded from call identity.
             * @default null
             */
            readonly reasoning?: string | null;
        };
        /**
         * SimulationArm
         * @description One arm's exact state and observation draws.
         */
        readonly SimulationArm: {
            readonly latent_paths: components["schemas"]["NumericalArray"];
            readonly observations: components["schemas"]["NumericalArray"];
        };
        readonly SimulationArms: components["schemas"]["SingleArmSimulation"] | components["schemas"]["PairedArmSimulation"];
        /**
         * SimulationEvidence
         * @description Exact generated histories with their production coordinates.
         */
        readonly SimulationEvidence: {
            /** Assignments */
            readonly assignments: readonly components["schemas"]["StateAssignment"][];
            /**
             * Time Origin
             * Format: date-time
             * @description Calendar instant of model day zero.
             */
            readonly time_origin: string;
            /** Times */
            readonly times: readonly [
                number,
                number,
                ...number[]
            ];
            /** Draws */
            readonly draws: number;
            /** Seed */
            readonly seed: number;
            /** State Ids */
            readonly state_ids: readonly components["schemas"]["ConstructId-Output"][];
            /** Parameter Draws */
            readonly parameter_draws: {
                readonly [key: string]: components["schemas"]["NumericalArray"];
            };
            readonly arms: components["schemas"]["SimulationArms"];
            readonly observation_layout: components["schemas"]["SimulationObservationLayout"];
        };
        /**
         * SimulationObservationLayout
         * @description Saved observation semantics and coordinates; generation truths remain separate.
         */
        readonly SimulationObservationLayout: {
            /** Variables */
            readonly variables: readonly Domain.ObservationSpec<Domain.ObservationWindow>[];
            readonly support_start_times: components["schemas"]["NumericalArray"];
            readonly support_end_times: components["schemas"]["NumericalArray"];
            readonly mask: components["schemas"]["NumericalArray"];
        };
        /**
         * SimulationReport
         * @description A simulation report records forward histories, resolved execution settings, and certified effects when supported.
         */
        readonly SimulationReport: {
            readonly evidence: components["schemas"]["SimulationEvidence"];
            readonly summary: components["schemas"]["SimulationSummary"];
            readonly law: components["schemas"]["PredictiveLawProvenance"];
            /**
             * Findings
             * @default []
             */
            readonly findings: readonly components["schemas"]["PredictiveAssessment"][];
            readonly fit_reliability: components["schemas"]["FitReliability"];
        };
        /**
         * SimulationSpec
         * @description Generate from a calendar day over a horizon, with interventions placed after the start.
         *
         *     Authored initial states apply at the start. A retained trajectory law with a
         *     calendar origin places the start on that law's model-day axis.
         */
        readonly "SimulationSpec-Input": {
            /**
             * Start
             * Format: date
             * @description Calendar day the window starts, at 00:00 UTC.
             */
            readonly start: string;
            /**
             * Horizon
             * @description How long the window lasts, such as 9w or 61d.
             */
            readonly horizon: string;
            /**
             * Interventions
             * @default []
             */
            readonly interventions?: readonly components["schemas"]["InterventionSpec-Input"][];
        };
        /**
         * SimulationSpec
         * @description Generate from a calendar day over a horizon, with interventions placed after the start.
         *
         *     Authored initial states apply at the start. A retained trajectory law with a
         *     calendar origin places the start on that law's model-day axis.
         */
        readonly "SimulationSpec-Output": {
            /**
             * Start
             * Format: date
             * @description Calendar day the window starts, at 00:00 UTC.
             */
            readonly start: string;
            /**
             * Horizon
             * @description How long the window lasts, such as 9w or 61d.
             */
            readonly horizon: string;
            /**
             * Interventions
             * @default []
             */
            readonly interventions: readonly components["schemas"]["InterventionSpec-Output"][];
        };
        /**
         * SimulationSummary
         * @description Full-draw reductions retained once, independently of a viewer's draw selection.
         */
        readonly SimulationSummary: {
            /** State Frames */
            readonly state_frames: Readonly<Partial<Record<components["schemas"]["ConstructId-Output"], readonly [
                number,
                number
            ]>>>;
            /** Indicator Frames */
            readonly indicator_frames: Readonly<Partial<Record<components["schemas"]["IndicatorId-Output"], readonly [
                number,
                number
            ]>>>;
            /** Action Category Probabilities */
            readonly action_category_probabilities: Readonly<Partial<Record<components["schemas"]["IndicatorId-Output"], components["schemas"]["CategoryProbabilitySummary"]>>>;
            /** Reference Category Probabilities */
            readonly reference_category_probabilities: Readonly<Partial<Record<components["schemas"]["IndicatorId-Output"], components["schemas"]["CategoryProbabilitySummary"]>>>;
        };
        /**
         * SingleArmSimulation
         * @description Natural-course generation without an intervention comparison.
         */
        readonly SingleArmSimulation: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "single";
            readonly action: components["schemas"]["SimulationArm"];
        };
        /** @description Folder of ready-to-use CSV or Parquet tables under data/{workspace_id}/, such as input. Every table must have a date or datetime timestamp column. */
        readonly SourceFolder: string;
        readonly SpecificationAssessment: Domain.Evaluated<string, string> | Domain.NotEvaluated<string>;
        /**
         * StateAssignment
         * @description A state set at one model time: a resolved intervention or a deterministic input point.
         */
        readonly StateAssignment: {
            readonly target: components["schemas"]["ConstructId-Output"];
            /**
             * Time
             * @description Absolute time in model days.
             */
            readonly time: number;
            /** Value */
            readonly value: number;
        };
        /**
         * StateExpression
         * @description A construct's state or declared known input, referenced by identity.
         */
        readonly "StateExpression-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "state";
            readonly construct_id: components["schemas"]["ConstructId-Output"];
        };
        /**
         * StepError
         * @description The error type and message of a failed step.
         */
        readonly StepError: {
            /** Type */
            readonly type: string;
            /** Message */
            readonly message: string;
        };
        /**
         * StepEvent
         * @description Measurement extraction changed status.
         */
        readonly StepEvent: {
            /**
             * Attempt Id
             * Format: uuid
             */
            readonly attempt_id: string;
            /**
             * Cursor
             * @default
             */
            readonly cursor: string;
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly event: "grassmoth.step";
            readonly status: components["schemas"]["StepStatus"];
            /** @default null */
            readonly error: components["schemas"]["StepError"] | null;
        };
        /** @enum {string} */
        readonly StepStatus: "running" | "completed" | "failed";
        /**
         * StudentTLawSpec[Expression]
         * @description The StudentT conditional law.
         */
        readonly "StudentTLawSpec_Expression_-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly distribution: "StudentT";
            readonly df: components["schemas"]["Expression-Output"];
            readonly loc: components["schemas"]["Expression-Output"];
            readonly scale: components["schemas"]["Expression-Output"];
        };
        /**
         * StudyRevision
         * @description Git publication wraps its already-owned record, without copying its fields.
         */
        readonly StudyRevision: {
            readonly commit_id: components["schemas"]["GitOid-Output"];
            /** Parent Ids */
            readonly parent_ids: readonly components["schemas"]["GitOid-Output"][];
            readonly record: components["schemas"]["AttemptRecord"];
        };
        /**
         * StudyState
         * @description Study state projects the artifacts selected by a sequence of attempts.
         *
         *     ``current`` maps artifact id → the revision info that is *current* for the
         *     study. Absent key = the artifact does not exist (either never produced,
         *     or produced-when-nonempty semantics withheld it).
         */
        readonly StudyState: {
            /** Current */
            readonly current: Readonly<Partial<Record<components["schemas"]["ArtifactId"], components["schemas"]["ArtifactRecord"]>>>;
            /** @default null */
            readonly data: Domain.DataRef<Domain.GitOid, number> | null;
        };
        /**
         * SuccessfulPoll[Literal['data_diff'], DataDiffOutput]
         * @description A published scientific result and the complete execution log of its call.
         */
        readonly SuccessfulPoll_Literal__data_diff___DataDiffOutput_: {
            readonly call_id: components["schemas"]["CallId-Output"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "data_diff";
            /**
             * Status
             * @default success
             * @constant
             */
            readonly status: "success";
            readonly commit_id: components["schemas"]["GitOid-Output"];
            readonly body: components["schemas"]["DataDiffOutput"];
            /** Messages */
            readonly messages: readonly components["schemas"]["ExecutionMessage"][];
        };
        /**
         * SuccessfulPoll[Literal['edit_model'], EditModelOutput]
         * @description A published scientific result and the complete execution log of its call.
         */
        readonly SuccessfulPoll_Literal__edit_model___EditModelOutput_: {
            readonly call_id: components["schemas"]["CallId-Output"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "edit_model";
            /**
             * Status
             * @default success
             * @constant
             */
            readonly status: "success";
            readonly commit_id: components["schemas"]["GitOid-Output"];
            readonly body: components["schemas"]["EditModelOutput"];
            /** Messages */
            readonly messages: readonly components["schemas"]["ExecutionMessage"][];
        };
        /**
         * SuccessfulPoll[Literal['edit_question'], EditQuestionOutput]
         * @description A published scientific result and the complete execution log of its call.
         */
        readonly SuccessfulPoll_Literal__edit_question___EditQuestionOutput_: {
            readonly call_id: components["schemas"]["CallId-Output"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "edit_question";
            /**
             * Status
             * @default success
             * @constant
             */
            readonly status: "success";
            readonly commit_id: components["schemas"]["GitOid-Output"];
            readonly body: components["schemas"]["EditQuestionOutput"];
            /** Messages */
            readonly messages: readonly components["schemas"]["ExecutionMessage"][];
        };
        /**
         * SuccessfulPoll[Literal['fit'], FitOutput]
         * @description A published scientific result and the complete execution log of its call.
         */
        readonly SuccessfulPoll_Literal__fit___FitOutput_: {
            readonly call_id: components["schemas"]["CallId-Output"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "fit";
            /**
             * Status
             * @default success
             * @constant
             */
            readonly status: "success";
            readonly commit_id: components["schemas"]["GitOid-Output"];
            readonly body: components["schemas"]["FitOutput"];
            /** Messages */
            readonly messages: readonly components["schemas"]["ExecutionMessage"][];
        };
        /**
         * SuccessfulPoll[Literal['model_diff'], ModelDiffOutput]
         * @description A published scientific result and the complete execution log of its call.
         */
        readonly SuccessfulPoll_Literal__model_diff___ModelDiffOutput_: {
            readonly call_id: components["schemas"]["CallId-Output"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "model_diff";
            /**
             * Status
             * @default success
             * @constant
             */
            readonly status: "success";
            readonly commit_id: components["schemas"]["GitOid-Output"];
            readonly body: components["schemas"]["ModelDiffOutput"];
            /** Messages */
            readonly messages: readonly components["schemas"]["ExecutionMessage"][];
        };
        /**
         * SuccessfulPoll[Literal['prepare_data'], PrepareDataOutput]
         * @description A published scientific result and the complete execution log of its call.
         */
        readonly SuccessfulPoll_Literal__prepare_data___PrepareDataOutput_: {
            readonly call_id: components["schemas"]["CallId-Output"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "prepare_data";
            /**
             * Status
             * @default success
             * @constant
             */
            readonly status: "success";
            readonly commit_id: components["schemas"]["GitOid-Output"];
            readonly body: components["schemas"]["PrepareDataOutput"];
            /** Messages */
            readonly messages: readonly components["schemas"]["ExecutionMessage"][];
        };
        /**
         * SuccessfulPoll[Literal['simulate'], SimulateOutput]
         * @description A published scientific result and the complete execution log of its call.
         */
        readonly SuccessfulPoll_Literal__simulate___SimulateOutput_: {
            readonly call_id: components["schemas"]["CallId-Output"];
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly action: "simulate";
            /**
             * Status
             * @default success
             * @constant
             */
            readonly status: "success";
            readonly commit_id: components["schemas"]["GitOid-Output"];
            readonly body: components["schemas"]["SimulateOutput"];
            /** Messages */
            readonly messages: readonly components["schemas"]["ExecutionMessage"][];
        };
        /**
         * SummaryOperator
         * @description How measurements within a support window are reduced to one observation.
         * @enum {string}
         */
        readonly SummaryOperator: "first" | "last" | "sum" | "count" | "mean" | "std";
        /**
         * TemperingDiagnostics
         * @description Frozen ladder and retained swap counts; columns are neighboring temperature edges.
         */
        readonly TemperingDiagnostics: {
            /** Inverse Temperatures */
            readonly inverse_temperatures: readonly number[];
            /** Ladder Frozen At Warmup Step */
            readonly ladder_frozen_at_warmup_step: number;
            /** Swaps */
            readonly swaps: readonly (readonly components["schemas"]["ReplicaSwapCounts"][])[];
        };
        readonly "TemperingSpec-Input": components["schemas"]["UntemperedSpec-Input"] | components["schemas"]["GaussianReferenceTemperingSpec-Input"];
        readonly "TemperingSpec-Output": components["schemas"]["UntemperedSpec-Output"] | components["schemas"]["GaussianReferenceTemperingSpec-Output"];
        /**
         * TemporalStatus
         * @description Temporal status states whether a construct varies within the individual over time.
         * @enum {string}
         */
        readonly TemporalStatus: "time_varying" | "time_invariant";
        /**
         * TimelineRecord
         * @description A replayable call log, excluding every scientific result and check payload.
         */
        readonly TimelineRecord: {
            /** Seq */
            readonly seq: number;
            /** Ts */
            readonly ts: string;
            /** Messages */
            readonly messages: readonly components["schemas"]["ActionMessage"][];
            /** Trace Ids */
            readonly trace_ids: readonly string[];
            readonly attempt: Domain.Attempt<Domain.ActionId, Domain.ScientificActionRequest | Domain.DataDiffRequest<Domain.GitOid> | Domain.ModelDiffRequest<Domain.GitOid>, null>;
        };
        /**
         * TimelineResponse
         * @description Journal attempts, declared input dependencies, and all currently running actions.
         */
        readonly TimelineResponse: {
            /** Attempts */
            readonly attempts: readonly components["schemas"]["TimelineRevision"][];
            /** Dependencies */
            readonly dependencies: readonly components["schemas"]["RecordDependency"][];
            /** Running */
            readonly running: readonly components["schemas"]["RunningAction"][];
        };
        /**
         * TimelineRevision
         * @description A lightweight attempt record linked to its call identity and Git publication.
         */
        readonly TimelineRevision: {
            readonly call_id: components["schemas"]["CallId-Output"] | null;
            readonly commit_id: components["schemas"]["GitOid-Output"];
            /** Parent Ids */
            readonly parent_ids: readonly components["schemas"]["GitOid-Output"][];
            readonly record: components["schemas"]["TimelineRecord"];
        };
        /**
         * TimingSpread
         * @description Median and full observed range of ten synchronized wall-clock repetitions.
         */
        readonly TimingSpread: {
            /** Median Ms */
            readonly median_ms: number;
            /** Min Ms */
            readonly min_ms: number;
            /** Max Ms */
            readonly max_ms: number;
        };
        /**
         * TraceLogMessage
         * @description The accumulated conversation and tool exchanges for one subroutine.
         */
        readonly TraceLogMessage: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "trace";
            /**
             * Timestamp
             * Format: date-time
             */
            readonly timestamp: string;
            /** Trace Id */
            readonly trace_id: string;
            readonly trace: components["schemas"]["LLMTrace"];
        };
        /**
         * TraceMessage
         * @description One retained conversation step, including any reasoning or tool interaction.
         */
        readonly TraceMessage: {
            /** Role */
            readonly role: string;
            /** Content */
            readonly content: string;
            /**
             * Reasoning
             * @default null
             */
            readonly reasoning: string | null;
            /**
             * Tool Calls
             * @default null
             */
            readonly tool_calls: readonly components["schemas"]["TraceToolCall"][] | null;
            /**
             * Tool Call Id
             * @default null
             */
            readonly tool_call_id: string | null;
            /**
             * Tool Name
             * @default null
             */
            readonly tool_name: string | null;
            /**
             * Tool Result
             * @default null
             */
            readonly tool_result: string | null;
            /**
             * Tool Is Error
             * @default false
             */
            readonly tool_is_error: boolean;
        };
        /**
         * TraceSeries
         * @description Every retained draw, grouped in original chain order.
         */
        readonly TraceSeries: {
            readonly subject: components["schemas"]["ParameterRef"];
            /** Chains */
            readonly chains: readonly components["schemas"]["ScalarValues"][];
        };
        /**
         * TraceToolCall
         * @description A function invocation with the call identity used to match its result.
         */
        readonly TraceToolCall: {
            /** Id */
            readonly id: string;
            /** Name */
            readonly name: string;
            /** Arguments */
            readonly arguments: string;
        };
        /**
         * TraceUsage
         * @description Trace usage records the input, output, and reasoning tokens consumed by a conversation.
         */
        readonly TraceUsage: {
            /**
             * Input Tokens
             * @default 0
             */
            readonly input_tokens: number;
            /**
             * Output Tokens
             * @default 0
             */
            readonly output_tokens: number;
            /**
             * Reasoning Tokens
             * @default null
             */
            readonly reasoning_tokens: number | null;
        };
        /**
         * TransferredTuning
         * @description Only scientific values transfer across kernels; warmup must prepare new tuning.
         */
        readonly TransferredTuning: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "transferred";
        };
        /**
         * UnknownLawProvenance
         * @description Imported laws do not establish a conditioning history.
         */
        readonly UnknownLawProvenance: {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "unknown";
            /**
             * Interpretation
             * @description Unknown predictive interpretation when parameter-law provenance is unavailable.
             * @constant
             */
            readonly interpretation: "unknown";
        };
        /**
         * UntemperedSpec
         * @description The exact posterior without temperature replicas.
         */
        readonly "UntemperedSpec-Input": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "untempered";
        };
        /**
         * UntemperedSpec
         * @description The exact posterior without temperature replicas.
         */
        readonly "UntemperedSpec-Output": {
            /**
             * @description discriminator enum property added by openapi-typescript
             * @enum {string}
             */
            readonly kind: "untempered";
        };
        /** ValidationError */
        readonly ValidationError: {
            /** Location */
            readonly loc: readonly (string | number)[];
            /** Message */
            readonly msg: string;
            /** Error Type */
            readonly type: string;
            /** Input */
            readonly input?: unknown;
            /** Context */
            readonly ctx?: Record<string, never>;
        };
        /** @description Deterministic support-window expression that returns one scalar per window. Use Python-like syntax over source_columns with arithmetic, comparisons, if/else, and helper functions such as any(), sum(), mean(), std(), first(), last(), count_true(), count_non_null(), lower(), contains(), and contains_any(). Use None for missing values. */
        readonly WindowExpression: string;
        /**
         * WorstTimeMovement
         * @description An independent time coordinate ordered by its retained-draw refresh ratio.
         */
        readonly WorstTimeMovement: {
            /** Time Index */
            readonly time_index: number;
            /** Time Day */
            readonly time_day: number;
            /** Refresh */
            readonly refresh: number;
        };
    };
    responses: never;
    parameters: never;
    requestBodies: never;
    headers: never;
    pathItems: never;
}
export type $defs = Record<string, never>;
export interface operations {
    readonly poll_action: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
                readonly action: components["schemas"]["ScientificActionId"] | ("data_diff" | "model_diff");
                readonly call_id: components["schemas"]["CallId-Input"];
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/msgpack": components["schemas"]["ActionPoll"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly edit_question: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody: {
            readonly content: {
                readonly "application/json": components["schemas"]["EditQuestionRequest-Input"];
            };
        };
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/msgpack": components["schemas"]["ActionPoll"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly edit_model: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody: {
            readonly content: {
                readonly "application/json": components["schemas"]["EditModelRequest_RevisionSelector_"];
            };
        };
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/msgpack": components["schemas"]["ActionPoll"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly prepare_data: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody: {
            readonly content: {
                readonly "application/json": components["schemas"]["PrepareDataRequest_RevisionSelector_SourceFolder_"];
            };
        };
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/msgpack": components["schemas"]["ActionPoll"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly fit: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody: {
            readonly content: {
                readonly "application/json": components["schemas"]["FitRequest_RevisionSelector_"];
            };
        };
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/msgpack": components["schemas"]["ActionPoll"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly simulate: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody: {
            readonly content: {
                readonly "application/json": components["schemas"]["SimulateRequest_RevisionSelector_"];
            };
        };
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/msgpack": components["schemas"]["ActionPoll"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly data_diff: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody: {
            readonly content: {
                readonly "application/json": components["schemas"]["DataDiffRequest_RevisionSelector_"];
            };
        };
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/msgpack": components["schemas"]["ActionPoll"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly model_diff: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody: {
            readonly content: {
                readonly "application/json": components["schemas"]["ModelDiffRequest_RevisionSelector_"];
            };
        };
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/msgpack": components["schemas"]["ActionPoll"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly get_timeline_api_studies__workspace_id__timeline_get: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path: {
                readonly workspace_id: string;
            };
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["TimelineResponse"];
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
    readonly list_workspaces_api_workspaces_get: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly requestBody?: never;
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": {
                        readonly [key: string]: string | null;
                    };
                };
            };
        };
    };
    readonly upload_file_api_upload_post: {
        readonly parameters: {
            readonly query?: never;
            readonly header?: never;
            readonly path?: never;
            readonly cookie?: never;
        };
        readonly requestBody: {
            readonly content: {
                readonly "multipart/form-data": components["schemas"]["Body_upload_file_api_upload_post"];
            };
        };
        readonly responses: {
            /** @description Successful Response */
            readonly 200: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": string;
                };
            };
            /** @description Validation Error */
            readonly 422: {
                headers: {
                    readonly [name: string]: unknown;
                };
                content: {
                    readonly "application/json": components["schemas"]["HTTPValidationError"];
                };
            };
        };
    };
}
