import { decodeFixture } from "./fixture-value";
import type { ModelSnapshot } from "@grassmoth/api-types";
import rawHistory from "../../../../../data/HEALTHDEMO/fixture/model_history.json";
import rawSnapshot from "../../../../../data/HEALTHDEMO/fixture/model_snapshot.json";

/** Validated by the production reader when read fixtures are generated. */
export const demoModelSnapshot = decodeFixture(rawSnapshot);
const history = decodeFixture(rawHistory);

export function demoSnapshotAt(seq: number): ModelSnapshot {
  if (seq === demoModelSnapshot.selected_seq) return demoModelSnapshot;
  const revision = history[seq];
  if (!revision) throw new Error(`No committed HEALTHDEMO revision ${seq}`);
  return revision;
}
