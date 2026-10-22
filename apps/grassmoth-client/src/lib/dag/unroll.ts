/** Temporal layout helpers. The backend selects evolving states; ghosts are their previous slice. */

/** Suffix marking a node id as the t−1 (previous-timestep) ghost of its base. */
export const GHOST_SUFFIX = "__p";

/** The t−1 ghost id for a present-time node. */
export const ghostId = (base: string): string => `${base}${GHOST_SUFFIX}`;

/** Whether an id refers to a t−1 ghost rather than a present-time node. */
export const isGhost = (id: string): boolean => id.endsWith(GHOST_SUFFIX);
