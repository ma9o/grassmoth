/** The selected display node and its direct neighbors, used only for highlighting. */
export function selectedNeighbors<Id extends string>(
  selected: Id | null,
  links: Iterable<readonly [Id, Id]>,
): Set<Id> | null {
  if (selected === null) return null;
  const neighbors = new Set([selected]);
  for (const [cause, effect] of links) {
    if (cause === selected) neighbors.add(effect);
    if (effect === selected) neighbors.add(cause);
  }
  return neighbors;
}
