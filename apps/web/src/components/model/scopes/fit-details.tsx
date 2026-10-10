import { useInferenceReport } from "@/lib/hooks/use-inference-report";
import type { ScopeContext } from "@/lib/model-asset/scope";
import { formatSignificant } from "@/lib/utils/format";
import { humanize } from "@/lib/model-asset/selection";
import { Hint, KeyValue, Section, StatusIcon } from "../scope-primitives";
import { ChainLegend, StepSizeChart } from "@/components/charts/chain-charts";
import { PosteriorPairs } from "./law-sections";
import { FitCalibration } from "./fit-calibration";

export function FitOutcome({ context }: { context: ScopeContext }) {
  const fit = context.modelSnapshot.fit;
  if (!fit) return <Hint>No inference report recorded.</Hint>;
  const report = fit;
  const { convergence } = report;
  const mcmc = report.inference_diagnostics;
  return (
    <Section title="Chain convergence">
      <Hint>
        Marginal particle Gibbs · {mcmc?.num_chains ?? "Unavailable"} chains ×{" "}
        {mcmc?.num_samples_per_chain ?? "Unavailable"} draws ·{" "}
        {report.inference_metadata.duration_seconds.toLocaleString()} s
      </Hint>
      <div className="flex items-center gap-2">
        <StatusIcon status={convergence.status} />
        <span>Chain convergence: {humanize(convergence.status)}.</span>
      </div>
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
  const tempering = gibbs?.tempering;
  return (
    <>
      <FitCalibration context={context} />
      {tempering && (
        <Section title="Parallel tempering" wide>
          <Hint>
            Each temperature ladder contributes one cold posterior chain. Local movement telemetry
            below is recorded before replica exchanges.
          </Hint>
          <KeyValue
            rows={[
              ["Temperatures per chain", String(tempering.inverse_temperatures.length)],
              ["Ladder frozen at warmup step", String(tempering.ladder_frozen_at_warmup_step)],
              [
                "Inverse temperatures",
                tempering.inverse_temperatures.map((beta) => formatSignificant(beta)).join(", "),
              ],
            ]}
          />
          {tempering.swaps.map((counts, chain) => (
            <p key={chain}>
              Chain {chain + 1} swaps by neighboring pair:{" "}
              {counts.map(({ accepted, attempted }) => `${accepted}/${attempted}`).join(" · ")}
            </p>
          ))}
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
