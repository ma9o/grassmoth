/** Theme tokens for chart marks: colour names a mark's role, arm or chain, never its draw. */
export const CHART_COLORS = {
  reference: "--chart-reference",
  intervened: "--chart-intervened",
  replicate: "--chart-replicate",
  observed: "--chart-observed",
  prior: "--chart-prior",
  posterior: "--chart-posterior",
  flagged: "--chart-flagged",
  warning: "--chart-warning",
  grid: "--chart-grid",
  axis: "--chart-axis",
  added: "--chart-added",
  removed: "--chart-removed",
  revised: "--chart-revised",
} as const;

const CHAIN_COLORS = [
  "--chart-chain-1",
  "--chart-chain-2",
  "--chart-chain-3",
  "--chart-chain-4",
] as const;

type ChainColor = (typeof CHAIN_COLORS)[number];

export type ChartColor = (typeof CHART_COLORS)[keyof typeof CHART_COLORS] | ChainColor;

/** Chains and categories cycle through four hues; their legends name each one. */
export function chainColor(index: number): ChainColor {
  switch (index % CHAIN_COLORS.length) {
    case 0:
      return CHAIN_COLORS[0];
    case 1:
      return CHAIN_COLORS[1];
    case 2:
      return CHAIN_COLORS[2];
    default:
      return CHAIN_COLORS[3];
  }
}

export const cssColor = (token: ChartColor) => `var(${token})`;

/** Canvas needs concrete colours; the theme owns them as custom properties. */
export function resolveColor(element: Element, token: ChartColor): string {
  return getComputedStyle(element).getPropertyValue(token).trim();
}
