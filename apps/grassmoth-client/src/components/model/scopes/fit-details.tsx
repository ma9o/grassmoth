import { ChainLegend, StepSizeChart } from "@/components/charts/chain-charts";
import { useInferenceReport } from "@/lib/hooks/use-inference-report";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { humanize } from "@/lib/model-asset/selection";
import { formatSignificant } from "@/lib/utils/format";
import { Hint, KeyValue, Section, StatusIcon } from "../scope-primitives";
import { FitCalibration } from "./fit-calibration";
import { PosteriorPairs } from "./law-sections";

export function FitOutcome({ context }: { context: ScopeContext }) {
  const fit = context.modelSnapshot.fit;
  if (!fit) return <Hint>No inference report recorded.</Hint>;
  const report = fit;
  const { convergence } = report;
  const mcmc = report.inference_diagnostics;
  const settings = report.inference_metadata.segments.at(-1)?.settings;
  return (
    <Section title="Chain convergence">
      <Hint>
        {settings
          ? {
              particle_mgrad: "Particle-mGRAD",
              twisted_particle_agrad: "Twisted Particle-aGRAD",
              auxiliary_kalman: "Auxiliary Kalman",
            }[settings.path_update.method]
          : "Unavailable"}{" "}
        · {mcmc?.num_chains ?? "Unavailable"} chains ×{" "}
        {mcmc?.num_samples_per_chain ?? "Unavailable"} draws ·{" "}
        {report.inference_metadata.duration_seconds.toLocaleString()} s
      </Hint>
      <div className="flex items-center gap-2">
        <StatusIcon status={convergence.status} />
        <span>Chain convergence: {humanize(convergence.status)}.</span>
      </div>
      {convergence.warnings.map((message) => (
        <p key={message} className="text-amber-700">
          Warning: {message}
        </p>
      ))}
      {convergence.messages.map((message) => (
        <p key={message}>{message}</p>
      ))}
    </Section>
  );
}

/** Model-wide fitted evidence; individual laws own their chains. */
export function FitDetails({ context }: { context: ScopeContext }) {
  const fit = context.modelSnapshot.fit;
  const detail = useInferenceReport(context.modelSnapshot);
  if (!fit)
    return (
      <Section title="Fit">
        <Hint>No inference report recorded at this version.</Hint>
      </Section>
    );
  const report = fit;
  const plot = detail.data?.detail;
  const mcmc = report.inference_diagnostics;
  const gibbs = report.inference_metadata.sampler_diagnostics;
  const weights = gibbs?.path_weights;
  const stages = weights?.filters.map((stage, index) => ({
    label: `Step ${index}`,
    weights: stage,
  }));
  const lowestWeightStages = stages
    ?.toSorted(
      (left, right) =>
        (left.weights.metrics.ess_fraction?.mean ?? Infinity) -
        (right.weights.metrics.ess_fraction?.mean ?? Infinity),
    )
    .slice(0, 8);
  return (
    <>
      <FitCalibration context={context} />
      {weights && lowestWeightStages && (
        <Section title="Particle weights">
          <Hint>
            {humanize(weights.phase)} · up to eight stages with the lowest mean particle ESS
            fraction. This measures weight concentration, not MCMC effective sample size.
          </Hint>
          <KeyValue
            rows={lowestWeightStages.map((stage) => [
              stage.label,
              `${stage.weights.candidates} candidates · ESS fraction ${
                stage.weights.metrics.ess_fraction?.mean == null
                  ? "Unavailable"
                  : formatSignificant(stage.weights.metrics.ess_fraction.mean)
              }`,
            ])}
          />
        </Section>
      )}
      {(gibbs || plot?.final_latent_delta) && (
        <Section title="Latent mixing" wide>
          <Hint>
            Movement telemetry complements the latent R-hat and ESS reported for each construct.
          </Hint>
          <KeyValue
            rows={[
              ["Latent moves accepted", gibbs?.latent_update_fraction],
              ["Latent coordinates frozen", gibbs?.latent_frozen_fraction],
              ["Mean latent move", gibbs?.latent_move_rms_mean],
            ]
              .filter((row): row is [string, number] => typeof row[1] === "number")
              .map(([label, value]): [string, string] => [label, formatSignificant(value)])}
          />
          {plot && <StepSizeChart detail={plot} />}
          {typeof mcmc?.num_chains === "number" && <ChainLegend chains={mcmc.num_chains} />}
        </Section>
      )}
      <PosteriorPairs context={context} />
    </>
  );
}
