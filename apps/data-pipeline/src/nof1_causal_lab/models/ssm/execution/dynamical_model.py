"""Compile scientific drift and measurement semantics into Dynestyx models.

Dynestyx owns state evolution, initial distributions, and numerical interpretation.
These adapters supply the application's causal vector field and heterogeneous
measurement semantics. Their parameters are explicit pytree leaves, so a model
can be carried through JAX transformations without closing over traced arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import dynestyx as dsx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.dynamics.intervention import Intervention
from nof1_causal_lab.models.ssm.dynamics.spec import (
    compile_dynamics,
    pack_component_params_from_samples,
)
from nof1_causal_lab.models.ssm.dynamics.vector_field import VectorField, VectorFieldArgs
from nof1_causal_lab.models.ssm.execution.contracts import MeasurementParams
from nof1_causal_lab.models.ssm.execution.observation_model import compile_observation_model
from nof1_causal_lab.models.ssm.execution.supported_normal import SupportedNormal

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dynestyx import StochasticContinuousTimeStateEvolution
    from jax.typing import ArrayLike
    from numpyro.distributions import MultivariateNormal

    from nof1_causal_lab.models.ssm.compile.inputs import CompiledDynamicalModel
    from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
    from nof1_causal_lab.models.ssm.execution.contracts import (
        ObservationLaws,
    )
    from nof1_causal_lab.models.ssm.execution.observation_dispatch import ObservationSampleFn
    from nof1_causal_lab.models.ssm.execution.observation_model import ObservationKernel


def continuous_state_evolution(
    vector_field: VectorField,
    vf_params: tuple[Mapping[str, jax.Array], ...],
    diffusion: jax.Array,
    *,
    intervention: Intervention | None = None,
) -> dsx.StochasticContinuousTimeStateEvolution:
    """Declare the nonlinear SDE without selecting a numerical approximation."""
    intervention = Intervention.none() if intervention is None else intervention
    for clamp in intervention.variable_overrides():
        diffusion = diffusion.at[clamp.index].set(0.0)
    return vector_field.evolution(
        VectorFieldArgs(params=vf_params, intervention=intervention),
        diffusion=dsx.FullDiffusion(diffusion),
    )


class HeterogeneousObservation(dsx.ObservationModel):
    """Measurement mapping with the application's family/link and missingness rules."""

    measurement: MeasurementParams
    laws: ObservationLaws

    @override
    def __call__(
        self, x: jax.Array, u: jax.Array | None, t: float | int | jax.Array
    ) -> _ObservationDistribution:
        del u, t
        return self.at_predictor(self.linear_predictor(x))

    def linear_predictor(self, state: jax.Array) -> jax.Array:
        """Map latent states to observation predictors through loadings and manifest intercepts."""
        return self.measurement.lambda_mat @ state + self.measurement.manifest_means

    def at_predictor(self, predictor: jax.Array) -> _ObservationDistribution:
        """Use the same observation law with cached predictors and interval projections."""
        return _ObservationDistribution(predictor, self)


class _ObservationDistribution(dist.Distribution):
    """NumPyro distribution for one heterogeneous measurement row."""

    _predictor: jax.Array
    _measurement: MeasurementParams
    _kernel: ObservationKernel
    _sampler: ObservationSampleFn
    _laws: ObservationLaws

    @property
    @override
    def support(self) -> dist.constraints.Constraint:
        return dist.constraints.real_vector

    def __init__(self, predictor: jax.Array, observation: HeterogeneousObservation) -> None:
        self._predictor = predictor
        self._laws = observation.laws
        self._measurement = observation.measurement
        compiled = compile_observation_model(
            observation.laws,
            manifest_cov=self._measurement.manifest_cov,
        )
        self._kernel = compiled.kernel
        self._sampler = compiled.point_sampler.sample_point
        super().__init__(
            batch_shape=(),
            event_shape=(int(self._measurement.lambda_mat.shape[0]),),
            validate_args=False,
        )

    @property
    def response(self) -> jax.Array:
        """Declared response/location for observation projection."""
        return self._kernel.response_fn(self._predictor)

    @property
    def mean(self) -> jax.Array:
        """Actual scalar moments; a category law's mean is its expected category code."""
        from nof1_causal_lab.artifacts.likelihood import CategoricalLawSpec, OrderedLogisticLawSpec
        from nof1_causal_lab.models.ssm.execution.observation_distributions import (
            evaluate_law,
            law_moments,
            law_response,
        )

        means = []
        for index, law in enumerate(self._laws):
            native = evaluate_law(
                law, self._predictor[index], jnp.sqrt(self._measurement.manifest_cov[index, index])
            )
            means.append(
                law_response(native)
                if isinstance(native, (CategoricalLawSpec, OrderedLogisticLawSpec))
                else law_moments(native)[0]
            )
        return jnp.stack(means)

    def sample(self, key: jax.Array | None, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        assert key is not None  # This stochastic distribution requires a NumPyro PRNG key.
        if not sample_shape:
            return self._sampler(key, self._predictor)
        from math import prod

        keys = jax.random.split(key, prod(sample_shape))
        flat = jax.vmap(lambda k: self._sampler(k, self._predictor))(keys)
        return flat.reshape((*sample_shape, *self.event_shape))

    def log_prob(self, value: ArrayLike, intermediates: list[object] | None = None) -> jax.Array:
        del intermediates
        observation = jnp.asarray(value)
        mask = ~jnp.isnan(observation)
        filled = jnp.nan_to_num(observation, nan=0.0)
        measurement = self._measurement
        return jnp.asarray(
            self._kernel.log_prob_fn(
                filled,
                self._predictor,
                measurement.manifest_cov,
                mask.astype(self._predictor.dtype),
            ),
            dtype=self._predictor.dtype,
        )


def initial_state_distribution(
    compiled_dynamical_model: CompiledDynamicalModel,
    mean: jax.Array,
    covariance: jax.Array,
    *,
    input_values: jax.Array | None = None,
) -> MultivariateNormal:
    """Embed the modeled initial law in a state vector with lawless inputs."""
    if input_values is not None:
        mean = jnp.where(numeric.input_mask(compiled_dynamical_model), input_values[0], mean)
    return SupportedNormal(mean, covariance, compiled_dynamical_model.state_space.initial)


def assemble_likelihood_inputs(
    samples: dict[str, jnp.ndarray],
    compiled_dynamical_model: CompiledDynamicalModel,
    *,
    dynamics: DynamicsSpec | None = None,
    intervention: Intervention | None = None,
    input_values: jax.Array | None = None,
) -> tuple[
    StochasticContinuousTimeStateEvolution,
    MeasurementParams,
    MultivariateNormal,
    ObservationLaws,
]:
    """Consume the canonical deterministic values from NumPyro's prior replay."""
    from nof1_causal_lab.models.ssm.compile.observations import materialize_observation_laws

    dynamics_spec = compiled_dynamical_model.dynamics.spec if dynamics is None else dynamics
    compiled = (
        compiled_dynamical_model.dynamics if dynamics is None else compile_dynamics(dynamics_spec)
    )
    diffusion_chol = samples["diffusion"]
    evolution = continuous_state_evolution(
        vector_field=compiled.vector_field,
        vf_params=pack_component_params_from_samples(dynamics_spec, samples),
        diffusion=diffusion_chol,
        intervention=intervention,
    )
    measurement = MeasurementParams(
        lambda_mat=samples["lambda"],
        manifest_means=samples["manifest_means"],
        manifest_cov=samples["manifest_cov"],
    )
    initial = initial_state_distribution(
        compiled_dynamical_model, samples["t0_means"], samples["t0_cov"], input_values=input_values
    )
    laws = materialize_observation_laws(compiled_dynamical_model, samples)
    return evolution, measurement, initial, laws


def build_dynamical_model(
    compiled_dynamical_model: CompiledDynamicalModel,
    values: dict[str, jnp.ndarray],
    *,
    t0: jax.Array,
    dynamics: DynamicsSpec | None = None,
) -> dsx.DynamicalModel:
    """Construct Dynestyx's model from the compiled model and one parameter draw.

    Values include the matrices derived by assemble_model_matrices or by
    NumPyro's prior replay. An execution-only dynamics override supports paired
    edge-off admission contrasts while retaining the same measurement and initial laws.
    """
    evolution, measurement, initial, laws = assemble_likelihood_inputs(
        values, compiled_dynamical_model, dynamics=dynamics
    )
    return dsx.DynamicalModel(
        initial_condition=initial,
        state_evolution=evolution,
        observation_model=HeterogeneousObservation(
            measurement,
            laws,
        ),
        t0=t0,
    )
