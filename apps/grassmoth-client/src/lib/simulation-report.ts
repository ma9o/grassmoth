import type { CausalEffectResult, SimulationReport } from "@grassmoth/api-types";

/** Paired arms own their evaluated effect; ordinary simulations have no causal result. */
export function causalEffect(
  report: SimulationReport | null | undefined,
): CausalEffectResult | undefined {
  const arms = report?.evidence.arms;
  return arms?.kind === "paired" && !("kind" in arms.causal) ? arms.causal : undefined;
}
