"""Interpret nof1's scientific model through Dynestyx's public distributions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

import dynestyx as dsx
import equinox as eqx
import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.compile.interval_tree import compile_interval_tree
from nof1_causal_lab.models.ssm.execution.dynamical_model import build_dynamical_model
from nof1_causal_lab.models.ssm.execution.observation_operator import compile_observation_operator
from nof1_causal_lab.models.ssm.execution.state_transition import SupportedEulerMaruyama
from nof1_causal_lab.models.ssm.inference.conditioning import (
    ExactStateConstraints,
    compile_exact_state_constraints,
)
from nof1_causal_lab.models.ssm.inference.parameter_transform import ParameterTransform
from nof1_causal_lab.models.ssm.inference.targets.intervals import ParticleIntervalObservations
from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleTarget
from nof1_causal_lab.models.ssm.inference.utils import (
    prepare_model_parameters,
)
from nof1_causal_lab.models.ssm.transition_kinds import LATENT_TRANSITION_EULER_MARUYAMA
from nof1_causal_lab.utils.immutability import freeze_fields

if TYPE_CHECKING:
    import jax

    from nof1_causal_lab.models.ssm.autoreparam import Strategy
    from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleContext
    from nof1_causal_lab.models.ssm.inference.utils import SiteInfo
    from nof1_causal_lab.models.ssm.parameterization import PriorRuntimeBundle
    from nof1_causal_lab.models.ssm.runtime import BoundPanel


@dataclass(frozen=True)
class ParticleProblem:
    """Exact model target together with parameter and reporting metadata."""

    runtime: ParticleTarget
    site_info: SiteInfo
    public_sites: frozenset[str]
    exact_constraints: ExactStateConstraints | None = None

    def __post_init__(self) -> None:
        """Own immutable collection fields for the compiled particle-sampling problem."""
        freeze_fields(self)


def build_particle_problem(
    priors: PriorRuntimeBundle,
    panel: BoundPanel,
    *,
    scheme: str,
    trace_key: jax.Array,
    reparam: Strategy | None,
) -> ParticleProblem:
    """Prepare the parameter transform and the exact discrete model for sampling."""
    observations, times = panel.observations, panel.times
    compiled_dynamical_model = panel.compiled_dynamical_model
    if scheme != LATENT_TRANSITION_EULER_MARUYAMA:
        raise ValueError(f"Particle inference requires 'euler_maruyama'; got {scheme!r}.")
    exact_constraints = compile_exact_state_constraints(
        compiled_dynamical_model, observations, input_values=panel.input_values
    )
    if DistributionFamily.STUDENT_T in numeric.diffusion_families(compiled_dynamical_model):
        raise ValueError(
            "Particle inference currently requires Gaussian latent diffusion for every state."
        )
    scientific_parameters, site_info, public_sites = prepare_model_parameters(
        priors, panel, trace_key, reparam
    )
    parameter_size = scientific_parameters.initial_position.size
    state_space = compiled_dynamical_model.state_space
    free_static = tuple(
        index
        for index in state_space.static
        if index in state_space.initial
        and (exact_constraints is None or bool(exact_constraints.free_mask[0, index]))
    )

    def continuous_model(position: jax.Array, runtime_times: jax.Array) -> dsx.DynamicalModel:
        return build_dynamical_model(
            compiled_dynamical_model,
            scientific_parameters.constrain(position[:parameter_size]),
            t0=runtime_times[0],
        )

    initial_model = continuous_model(scientific_parameters.initial_position, times)
    static_indices = jnp.asarray(free_static, dtype=jnp.int32)
    parameters = ParameterTransform(
        jnp.concatenate(
            (
                scientific_parameters.initial_position,
                initial_model.initial_condition.mean[static_indices],
            )
        ),
        lambda position: scientific_parameters.unravel(position[:parameter_size]),
        lambda position: scientific_parameters.constrain(position[:parameter_size]),
        # The joint initial-state factor retains the static prior, including its
        # correlation with dynamic initial states. It must be paid exactly once.
        lambda position: scientific_parameters.log_prior(position[:parameter_size]),
    )

    # Partition once to retain static metadata outside the sampler state. Every
    # parameter-dependent value in the declared model is an explicit array leaf.
    _, static_model = eqx.partition(
        continuous_model(parameters.initial_position, times), eqx.is_array
    )

    def context_fn(position: jax.Array, runtime_times: jax.Array) -> ParticleContext:
        model = continuous_model(position, runtime_times)
        static_values = (
            jnp.asarray(model.initial_condition.mean)
            .at[static_indices]
            .set(position[parameter_size:])
        )
        if exact_constraints is not None:
            static_values = jnp.where(
                exact_constraints.free_mask[0], static_values, exact_constraints.values[0]
            )
        dynamical_model_arrays: dsx.DynamicalModel = eqx.filter(model, eqx.is_array)
        return dynamical_model_arrays, runtime_times, static_values

    def declared_model(context: ParticleContext) -> dsx.DynamicalModel:
        model = eqx.combine(context[0], static_model)
        return dsx.DynamicalModel(
            initial_condition=model.initial_condition,
            state_evolution=SupportedEulerMaruyama(
                cast("dsx.StochasticContinuousTimeStateEvolution", model.state_evolution),
                compiled_dynamical_model.state_space.transition,
            ),
            observation_model=model.observation_model,
            t0=model.t0,
        )

    operator = compile_observation_operator(panel.observation_support)
    runtime = ParticleTarget(
        parameters,
        context_fn,
        declared_model,
        observations,
        times,
        state_space,
        exact_constraints=exact_constraints,
    )
    intervals = (
        ParticleIntervalObservations(
            declared_model,
            operator,
            compile_interval_tree(
                operator.window_plan,
                panel.observation_support.summary_operators,
                np.asarray(observations),
            ),
            runtime.project_state,
        )
        if operator.requires_interval_summary_handling
        else None
    )
    runtime = replace(runtime, intervals=intervals)
    return ParticleProblem(runtime, site_info, public_sites, exact_constraints)
