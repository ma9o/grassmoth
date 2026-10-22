import type { ActionSuccess, ModelSnapshot } from "@grassmoth/api-types";
import type { DataComparisonView } from "./data-comparison";
import type { ModelEntities } from "./entities";
import type { EntitySelection } from "@/lib/model-asset/selection";
import type { TimelineRevision } from "@grassmoth/api-types";

/** The viewed version and the selection shared by the graph and details. */
export interface ScopeContext {
  modelSnapshot: ModelSnapshot;
  entities: ModelEntities;
  select: (selection: EntitySelection | null) => void;
  ticks: readonly TimelineRevision[];
  dataDiff: DataComparisonView | null;
  result: ActionSuccess | undefined;
}
