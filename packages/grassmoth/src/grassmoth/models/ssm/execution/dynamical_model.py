"""Compile scientific drift and measurement semantics into Dynestyx models.

Dynestyx owns state evolution, initial distributions, and numerical interpretation.
These adapters supply the application's causal vector field and heterogeneous
measurement semantics. Their parameters are explicit pytree leaves, so a model
can be carried through JAX transformations without closing over traced arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from grassmoth.models.ssm import compiled_access as numeric
from grassmoth.models.ssm.dynamics.intervention import Intervention
from grassmoth.models.ssm.dynamics.spec import (
    compile_dynamics,
    pack_component_params_from_samples,
)
from grassmoth.models.ssm.dynamics.vector_field import VectorField, VectorFieldArgs
from grassmoth.models.ssm.execution.contracts import MeasurementParams
from grassmoth.models.ssm.execution.observation_dispatch import (
    build_point_observation_sampler,
)
from grassmoth.models.ssm.execution.observation_groups import LawGroup, compile_law_groups
from grassmoth.models.ssm.execution.observation_model import grouped_density, grouped_response
from grassmoth.models.ssm.execution.supported_normal import SupportedNormal

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dynestyx import StochasticContinuousTimeStateEvolution
    from jax.typing import ArrayLike
    from numpyro.distributions import MultivariateNormal

    from grassmoth.models.ssm.compile.inputs import CompiledDynamicalModel
    from grassmoth.models.ssm.dynamics.spec import DynamicsSpec
    from grassmoth.models.ssm.execution.contracts import ObservationLaws


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
    group_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)

    def __init__(self, measurement: MeasurementParams, laws: ObservationLaws) -> None:
        """Own the channel grouping once, with numerical law operands retained as array leaves."""
        self.measurement = measurement
        self.laws = laws
        self.group_indices = tuple(group.indices for group in compile_law_groups(laws))

    @override
    def __call__(
        self, x: jax.Array, u: jax.Array | None, t: float | int | jax.Array
    ) -> _ObservationDistribution:
        del u, t
        return self.at_predictor(self.linear_predictor(x))

    def select_channels(self, channels: tuple[int, ...]) -> HeterogeneousObservation:
        """Marginalize unselected channels before building their exact density programs."""
        indices = jnp.asarray(channels, dtype=jnp.int32)
        measurement = self.measurement
        return HeterogeneousObservation(
            MeasurementParams(
                measurement.lambda_mat[indices],
                measurement.manifest_means[indices],
                measurement.manifest_cov[jnp.ix_(indices, indices)],
            ),
            tuple(self.laws[index] for index in channels),
        )

    def linear_predictor(self, state: jax.Array) -> jax.Array:
        """Map latent states to observation predictors through loadings and manifest intercepts."""
        return self.measurement.lambda_mat @ state + self.measurement.manifest_means

    def density_groups(self) -> tuple[LawGroup, ...]:
        """Bind the already-owned channel grouping to this context's numerical laws."""
        return tuple(
            LawGroup(indices, tuple(self.laws[index] for index in indices))
            for indices in self.group_indices
        )

    def at_predictor(self, predictor: jax.Array) -> _ObservationDistribution:
        """Use the same observation law with cached predictors and interval projections."""
        return _ObservationDistribution(predictor, self)


class _ObservationDistribution(dist.Distribution):
    """NumPyro distribution for one heterogeneous measurement row."""

    _predictor: jax.Array
    _measurement: MeasurementParams
    _groups: tuple[LawGroup, ...]
    _laws: ObservationLaws

    @property
    @override
    def support(self) -> dist.constraints.Constraint:
        return dist.constraints.real_vector

    def __init__(self, predictor: jax.Array, observation: HeterogeneousObservation) -> None:
        self._predictor = predictor
        self._laws = observation.laws
        self._measurement = observation.measurement
        self._groups = observation.density_groups()
        super().__init__(
            batch_shape=(),
            event_shape=(int(self._measurement.lambda_mat.shape[0]),),
            validate_args=False,
        )

    @property
    def response(self) -> jax.Array:
        """Declared response/location for observation projection."""
        return grouped_response(self._groups, self._predictor, self._measurement.manifest_cov)

    @property
    def mean(self) -> jax.Array:
        """Actual scalar moments; a category law's mean is its expected category code."""
        from grassmoth.artifacts.model.likelihood import (
            CategoricalLawSpec,
            OrderedLogisticLawSpec,
        )
        from grassmoth.models.ssm.execution.observation_distributions import (
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
        sampler = build_point_observation_sampler(
            self._laws, self._measurement.manifest_cov, groups=self._groups
        )
        if not sample_shape:
            return sampler.sample_point(key, self._predictor)
        from math import prod

        keys = jax.random.split(key, prod(sample_shape))
        flat = jax.vmap(lambda k: sampler.sample_point(k, self._predictor))(keys)
        return flat.reshape((*sample_shape, *self.event_shape))

    def log_prob(self, value: ArrayLike, intermediates: list[object] | None = None) -> jax.Array:
        del intermediates
        observation = jnp.asarray(value)
        mask = ~jnp.isnan(observation)
        filled = jnp.nan_to_num(observation, nan=0.0)
        measurement = self._measurement
        return jnp.asarray(
            grouped_density(
                self._groups,
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
    from grassmoth.models.ssm.compile.observations import materialize_observation_laws

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
