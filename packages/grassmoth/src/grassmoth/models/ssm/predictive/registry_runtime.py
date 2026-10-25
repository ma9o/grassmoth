"""Compile-stable predictive runtime for the model's current uncertainty.

Builds prior predictive samples directly from compiled prior semantics or
native NumPyro priors without tracing back through ``numpyro_model()``.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

from grassmoth.artifacts.model.expressions import expression_coefficients
from grassmoth.artifacts.model.parameter import SiteKind
from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.compute.batching import compile_batched, compiled_memory
from grassmoth.models.ssm import compiled_access as numeric
from grassmoth.models.ssm.dynamics.cache_identity import dynamics_spec_to_dict
from grassmoth.models.ssm.dynamics.simulator import (
    IndexedBrownianSpec,
    ProcessNoise,
    SimulationConfig,
    simulate_model_path,
)
from grassmoth.models.ssm.dynamics.spec import compile_dynamics
from grassmoth.models.ssm.execution.dynamical_model import build_dynamical_model
from grassmoth.models.ssm.predictive.observations import sample_model_observations
from grassmoth.models.ssm.predictive.types import PredictiveDraws, PredictiveTrajectory

if TYPE_CHECKING:
    from _hashlib import HASH
    from collections.abc import Callable, Mapping

    import dynestyx as dsx
    from jax.stages import Compiled
    from jax.typing import ArrayLike

    from grassmoth.compute.budget import ComputeBudget, MemoryDemand
    from grassmoth.models.ssm.compile.inputs import CompiledDynamicalModel
    from grassmoth.models.ssm.dynamics.spec import CompiledDynamics, DynamicsSpec


class PredictiveKeys(NamedTuple):
    """Independent random streams for one predictive simulation."""

    parameters: jax.Array
    latents: jax.Array
    observations: jax.Array


def predictive_keys(seed: int) -> PredictiveKeys:
    """Derive independent parameter, latent, and observation streams."""
    parameter_key, latent_key, observation_key = random.split(random.PRNGKey(seed), 3)
    return PredictiveKeys(parameter_key, latent_key, observation_key)


def _ensure_gaussian_process_diffusion(compiled_dynamical_model: CompiledDynamicalModel) -> None:
    non_gaussian = [
        dist.value
        for dist in numeric.diffusion_families(compiled_dynamical_model)
        if dist != DistributionFamily.GAUSSIAN
    ]
    if non_gaussian:
        raise ValueError(
            f"Forward simulation currently requires Gaussian process diffusion; got {non_gaussian}."
        )


def _predictive_models(
    compiled_dynamical_model: CompiledDynamicalModel,
    samples,
    times,
    *,
    dynamics: DynamicsSpec | None = None,
) -> dsx.DynamicalModel:
    """Batch the same model constructor used by the particle target."""
    build_draws: Callable[[Mapping[str, jnp.ndarray]], dsx.DynamicalModel] = eqx.filter_vmap(
        lambda draw: build_dynamical_model(
            compiled_dynamical_model, draw, t0=times[0], dynamics=dynamics
        )
    )
    return build_draws(samples)


# Refine the SDE step using declared relaxation rates, including fixed rates.
# This controls the linear restoring contribution; it is not a global stability
# certificate for a composed nonlinear drift. Diffrax evaluates the full
# expression at every step. Scientific coefficient metadata identifies rates;
# numerical sample-site names carry no meaning.
_SDE_CFL_SAFETY = 0.25
_SDE_MAX_STEPS = 16384
_LATENT_CACHE_MAX_ENTRIES = 2
_LATENT_CACHE_ENGINE_VERSION = 4
_latent_cache: OrderedDict[str, jax.Array] = OrderedDict()
_latent_cache_lock = threading.Lock()


def _update_array_digest(digest: HASH, label: str, value: ArrayLike) -> None:
    array = np.asarray(jax.device_get(value))
    digest.update(label.encode())
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))


def _prior_predictive_latent_cache_key(
    dynamics: DynamicsSpec,
    vf_params: object,
    samples: Mapping[str, jnp.ndarray | np.ndarray],
    times: jnp.ndarray,
    rng_key: jax.Array,
) -> str:
    """Fingerprint only inputs that can change the latent trajectories."""
    digest = hashlib.sha256()
    digest.update(f"latent-cache-v{_LATENT_CACHE_ENGINE_VERSION}".encode())
    digest.update(
        json.dumps(
            dynamics_spec_to_dict(dynamics),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    digest.update(jax.default_backend().encode())
    for leaf_index, leaf in enumerate(jax.tree.leaves(vf_params)):
        _update_array_digest(digest, f"vf:{leaf_index}", leaf)
    for name in ("t0_cov", "t0_means", "diffusion"):
        _update_array_digest(digest, name, samples[name])
    _update_array_digest(digest, "times", times)
    _update_array_digest(digest, "rng-key", rng_key)
    return digest.hexdigest()


def _cached_latents(key: str) -> jax.Array | None:
    with _latent_cache_lock:
        cached = _latent_cache.pop(key, None)
        if cached is not None:
            _latent_cache[key] = cached
        return cached


def _cache_latents(key: str, latents: jax.Array) -> None:
    with _latent_cache_lock:
        _latent_cache[key] = latents
        _latent_cache.move_to_end(key)
        while len(_latent_cache) > _LATENT_CACHE_MAX_ENTRIES:
            _latent_cache.popitem(last=False)


def _predictive_max_rates(
    compiled: CompiledDynamics, samples: Mapping[str, jnp.ndarray | np.ndarray]
) -> jnp.ndarray:
    """Read the fastest declared relaxation rate per draw from coefficient metadata."""
    n_draws = int(next(iter(samples.values())).shape[0])
    rates = [
        jnp.max(jnp.abs(samples[site.name]).reshape(n_draws, -1), axis=1)
        for site in compiled.site_registry
        if site.site_kind == SiteKind.DYNAMICS_DECAY
    ]
    for component in compiled.spec.components:
        rates.extend(
            jnp.full(n_draws, operand.value)
            for operand in expression_coefficients(component.expression)
            if operand.meaning.quantity == SiteKind.DYNAMICS_DECAY
            and isinstance(operand.value, (int, float))
        )
    return jnp.max(jnp.stack(rates), axis=0) if rates else jnp.zeros(n_draws)


def _predictive_sde_step(max_rate: jnp.ndarray, span: float) -> jnp.ndarray:
    # Traced (not host) arithmetic: the per-draw step size stays a jnp scalar
    # so every draw reuses ONE compiled program — a host float here bakes into
    # the XLA graph as a constant and forces a retrace + recompile per draw.
    # This is a maximum step. The solver also lands on every prepared time; its
    # work budget must never enlarge this limit or skip a prepared boundary.
    return jnp.minimum(span / 200.0, _SDE_CFL_SAFETY / jnp.maximum(max_rate, 1e-30))


def _predictive_sde_config(max_rate: jnp.ndarray, span: float) -> SimulationConfig:
    if span <= 0.0:
        return SimulationConfig()
    return SimulationConfig(
        brownian=IndexedBrownianSpec(step_size=_predictive_sde_step(max_rate, span)),
        max_steps=_SDE_MAX_STEPS + 16,
    )


def _predictive_draw_order(max_rates: jnp.ndarray, span: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Order draws by expected solver work and return the inverse permutation."""
    n_draws = int(max_rates.shape[0])
    if span <= 0.0:
        identity = jnp.arange(n_draws)
        return identity, identity

    sde_dt = _predictive_sde_step(max_rates, span)
    step_counts = jnp.ceil(span / sde_dt)

    order = jnp.argsort(step_counts, stable=True)
    return order, jnp.argsort(order)


def _simulate_model_predictive_latent_draw(
    dynamical_model: dsx.DynamicalModel,
    times: jnp.ndarray,
    key: jnp.ndarray,
    span: float,
    max_rate: jnp.ndarray,
) -> jnp.ndarray:
    """Draw the native initial law and execute the model's exact state evolution."""
    key_init, key_latent = random.split(key)
    return simulate_model_path(
        dynamical_model,
        jnp.asarray(dynamical_model.initial_condition.sample(key_init)),
        times,
        config=_predictive_sde_config(max_rate, span),
        key=key_latent,
    )


def _simulate_model_predictive_draws_microbatched(
    dynamical_models: dsx.DynamicalModel,
    times: jnp.ndarray,
    keys: jnp.ndarray,
    span: float,
    max_rates: jnp.ndarray,
    *,
    batch_size: int,
) -> jnp.ndarray:
    arrays, structure = eqx.partition(dynamical_models, eqx.is_array)

    def simulate_one(args) -> jnp.ndarray:
        model_arrays, key, max_rate = args
        return _simulate_model_predictive_latent_draw(
            eqx.combine(model_arrays, structure),
            times,
            key,
            span,
            max_rate,
        )

    result: jnp.ndarray = jax.lax.map(
        simulate_one,
        (arrays, keys, max_rates),
        batch_size=batch_size,
    )
    return result


class _CompiledPrediction(Protocol):
    compiled: Compiled

    def __call__(self, *args: object, **kwargs: object) -> jnp.ndarray: ...


class _LoweredPrediction(Protocol):
    def compile(self) -> _CompiledPrediction: ...


class _PredictiveCompiler(Protocol):
    def lower(self, *args: object, **kwargs: object) -> _LoweredPrediction: ...


# Equinox's annotation preserves the wrapped function signature and omits its
# documented lower/compile surface; describe that surface at this library seam.
_simulate_model_predictive_draws = cast(
    "_PredictiveCompiler", eqx.filter_jit(_simulate_model_predictive_draws_microbatched)
)


def _simulate_vector_field_predictive_latents(
    compiled_dynamical_model: CompiledDynamicalModel,
    samples: Mapping[str, jnp.ndarray | np.ndarray],
    times: jnp.ndarray,
    *,
    compute_budget: ComputeBudget,
    rng_key: jax.Array,
    dynamics: DynamicsSpec | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    _ensure_gaussian_process_diffusion(compiled_dynamical_model)
    dynamics = compiled_dynamical_model.dynamics.spec if dynamics is None else dynamics
    compiled = compile_dynamics(dynamics)
    n_draws = int(next(iter(samples.values())).shape[0])
    draw_keys = random.split(rng_key, n_draws)
    dynamical_models = _predictive_models(
        compiled_dynamical_model, samples, times, dynamics=dynamics
    )
    span = float(times[-1] - times[0]) if int(times.shape[0]) > 1 else 0.0
    cache_key = _prior_predictive_latent_cache_key(
        dynamics,
        dynamical_models.state_evolution,
        samples,
        times,
        rng_key,
    )
    latents = _cached_latents(cache_key)
    if latents is None:
        max_rates = _predictive_max_rates(compiled, samples)
        order, inverse_order = _predictive_draw_order(max_rates, span)
        sorted_models = jax.tree.map(
            lambda leaf: leaf[order] if eqx.is_array(leaf) else leaf, dynamical_models
        )
        args = (sorted_models, times, draw_keys[order], span, max_rates[order])

        def compile_batch(size: int) -> tuple[Callable[..., jnp.ndarray], MemoryDemand]:
            executable = _simulate_model_predictive_draws.lower(*args, batch_size=size).compile()
            return executable, compiled_memory(executable.compiled)

        batch = compile_batched(compile_batch, max_batch_size=n_draws, budget=compute_budget)
        sorted_latents = batch.executable(*args, batch_size=batch.batch_size)
        latents = sorted_latents[inverse_order]
        _cache_latents(cache_key, latents)
    predict: Callable[[dsx.DynamicalModel, jnp.ndarray], jnp.ndarray] = eqx.filter_vmap(
        lambda dynamical_model, path: jax.vmap(dynamical_model.observation_model.linear_predictor)(
            path
        )
    )
    return latents, predict(dynamical_models, latents)


def sample_predictive_emissions(
    compiled_dynamical_model: CompiledDynamicalModel,
    samples: Mapping[str, jnp.ndarray | np.ndarray],
    linear_predictors: jnp.ndarray,
    times: jnp.ndarray,
    *,
    observation_support,
    observation_mask: jnp.ndarray | None,
    num_samples: int,
    rng_key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample the observation layer conditional on cached latent predictors."""
    indices = jnp.linspace(
        0, linear_predictors.shape[0] - 1, min(num_samples, linear_predictors.shape[0])
    ).astype(int)
    observations, mask, means = sample_model_observations(
        _predictive_models(
            compiled_dynamical_model,
            {name: values[indices] for name, values in samples.items()},
            times,
        ),
        linear_predictors[indices],
        times,
        rng_key=rng_key,
        observation_support=observation_support,
        observation_mask=observation_mask,
        manifest_names=list(numeric.observation_names(compiled_dynamical_model)),
        held_channels=tuple(
            index
            for index, indicator in enumerate(compiled_dynamical_model.observations)
            if compiled_dynamical_model.states[indicator.state_index].is_input
        ),
    )
    return observations, mask, means


def simulate_predictive_draws(
    compiled_dynamical_model: CompiledDynamicalModel,
    samples: Mapping[str, jnp.ndarray | np.ndarray],
    times: jnp.ndarray,
    *,
    compute_budget: ComputeBudget,
    observation_support=None,
    observation_mask: jnp.ndarray | None = None,
    seed: int = 0,
    initial_states: jax.Array | None = None,
    interventions=(),
    input_events=(),
) -> PredictiveDraws:
    """Generate one shared path/observation batch from aligned parameter draws."""
    _ensure_gaussian_process_diffusion(compiled_dynamical_model)
    n_draws = int(next(iter(samples.values())).shape[0])
    keys = predictive_keys(seed)
    latents, linear_predictors, reference_latents = simulate_latent_histories(
        compiled_dynamical_model,
        samples,
        times,
        random.fold_in(keys.latents, 0),
        initial_states,
        interventions,
        input_events,
        compute_budget=compute_budget,
    )
    observations, observations_mask, expected_observations = sample_predictive_emissions(
        compiled_dynamical_model,
        samples,
        linear_predictors,
        times,
        observation_support=observation_support,
        observation_mask=observation_mask,
        num_samples=n_draws,
        rng_key=keys.observations,
    )
    trajectory = PredictiveTrajectory(
        latents=latents,
        linear_predictors=linear_predictors,
        observations=observations,
        observations_mask=observations_mask,
        expected_observations=expected_observations,
    )
    reference = None
    if reference_latents is not None:
        dynamical_models = _predictive_models(compiled_dynamical_model, samples, times)
        reference_predictors = eqx.filter_vmap(
            lambda dynamical_model, path: jax.vmap(
                dynamical_model.observation_model.linear_predictor
            )(path)
        )(dynamical_models, reference_latents)
        reference_observations, reference_mask, reference_means = sample_predictive_emissions(
            compiled_dynamical_model,
            samples,
            reference_predictors,
            times,
            observation_support=observation_support,
            observation_mask=observation_mask,
            num_samples=n_draws,
            rng_key=keys.observations,
        )
        reference = PredictiveTrajectory(
            latents=reference_latents,
            linear_predictors=reference_predictors,
            observations=reference_observations,
            observations_mask=reference_mask,
            expected_observations=reference_means,
        )
    return PredictiveDraws(
        parameters=dict(samples),
        trajectory=trajectory,
        reference=reference,
    )


def simulate_latent_histories(
    compiled_dynamical_model: CompiledDynamicalModel,
    samples,
    times,
    key,
    initial_states,
    interventions,
    input_events=(),
    *,
    compute_budget: ComputeBudget,
):
    """Execute the same nonlinear field with explicit starts, noise and paired do-operations."""
    if numeric.input_mask(compiled_dynamical_model).any() and not input_events:
        raise ValueError("Exogenous inputs require replayed panel readings")
    if initial_states is None and not interventions and not input_events:
        # The initial-law case retains caching and microbatching of the same exact solver.
        latents, predictors = _simulate_vector_field_predictive_latents(
            compiled_dynamical_model, samples, times, rng_key=key, compute_budget=compute_budget
        )
        return latents, predictors, None

    from grassmoth.models.ssm.counterfactual.orchestration import (
        vmap_simulate_interventions_from_state,
    )
    from grassmoth.models.ssm.dynamics.draws import dynamics_from_samples

    dynamical_models = _predictive_models(compiled_dynamical_model, samples, times)
    draw_keys = random.split(key, next(iter(samples.values())).shape[0])
    if initial_states is None:
        initial_states = eqx.filter_vmap(
            lambda dynamical_model, k: dynamical_model.initial_condition.sample(random.split(k)[0])
        )(dynamical_models, draw_keys)
    dynamics = dynamics_from_samples(compiled_dynamical_model, samples, n_draws=draw_keys.shape[0])
    span = float(times[-1] - times[0])
    max_rates = _predictive_max_rates(
        compile_dynamics(compiled_dynamical_model.dynamics.spec), samples
    )
    # Paired paths use identical step sizes and random streams in each segment.
    reference, latents, _ = vmap_simulate_interventions_from_state(
        dynamics,
        initial_states,
        list(interventions),
        time_grid=times,
        config=_predictive_sde_config(jnp.max(max_rates), span),
        noise=ProcessNoise(
            key=jax.vmap(lambda k: random.split(k)[1])(draw_keys),
            diffusion=samples["diffusion"],
        ),
        input_events=tuple(input_events),
    )
    predictors = eqx.filter_vmap(
        lambda dynamical_model, path: jax.vmap(dynamical_model.observation_model.linear_predictor)(
            path
        )
    )(dynamical_models, latents)
    return latents, predictors, reference if interventions else None
