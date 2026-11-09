/** Signatures for the jStat distribution functions used by authored-law plots. */
declare module "jstat" {
  interface DistributionFunctions {
    readonly pdf: (value: number, first: number, second: number) => number;
    readonly cdf: (value: number, first: number, second: number) => number;
    readonly inv: (probability: number, first: number, second: number) => number;
  }

  export const jStat: {
    readonly normal: DistributionFunctions;
    readonly beta: DistributionFunctions;
    readonly gamma: DistributionFunctions;
    readonly lognormal: DistributionFunctions;
    readonly uniform: DistributionFunctions;
    readonly studentt: {
      readonly inv: (probability: number, degreesOfFreedom: number) => number;
    };
    readonly lowRegGamma: (shape: number, value: number) => number;
    readonly ibeta: (value: number, first: number, second: number) => number;
    readonly exponential: {
      readonly pdf: (value: number, rate: number) => number;
      readonly inv: (probability: number, rate: number) => number;
    };
  };
}
