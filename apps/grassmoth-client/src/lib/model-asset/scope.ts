import type { ActionSuccess, ModelSnapshot, TimelineRevision } from "@grassmoth/api-types";
import type { EntitySelection } from "@/lib/model-asset/selection";
import type { DataComparisonView } from "./data-comparison";
import type { ModelEntities } from "./entities";

/** The viewed version and the selection shared by the graph and details. */
export interface ScopeContext {
  readonly modelSnapshot: ModelSnapshot;
  readonly entities: ModelEntities;
  readonly select: (selection: EntitySelection | null) => void;
  readonly ticks: readonly TimelineRevision[];
  readonly dataDiff: DataComparisonView | null;
  readonly result: ActionSuccess | undefined;
}
