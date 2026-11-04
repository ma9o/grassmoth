"""Small computations behind the stochastic volatility walkthrough's figures.

Launch with JAX_PLATFORMS=cpu and JAX_ENABLE_X64=true, as the notebook does. The kernels and
densities live in sv_kernels and sv_model; this module only asks them questions.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from arviz_stats.base import array_stats
from jax.scipy.special import logsumexp
from numpyro.infer.hmc import HMCState, hmc

from notebooks.support import sv_kernels as kernels
from notebooks.support import sv_model as model
from notebooks.support import sv_runs

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

DAYS = 1000
SEED = 721
_HERE = Path(__file__).resolve().parent
ARTIFACTS = _HERE.parent / "artifacts" / "sv_methods"


def _fingerprint(recipe: tuple[Callable[..., object], ...], inputs: object) -> str:
    """Digest of the model, the kernels, this computation's own code and its numerical inputs."""
    digest = hashlib.sha256()
    for name in ("sv_model.py", "sv_kernels.py"):
        digest.update((_HERE / name).read_bytes())
    for function in recipe:
        digest.update(inspect.getsource(function).encode())
    leaves, structure = jax.tree.flatten(inputs)
    digest.update(str(structure).encode())
    for leaf in leaves:
        digest.update(np.asarray(leaf).tobytes() if hasattr(leaf, "shape") else repr(leaf).encode())
    return digest.hexdigest()[:16]


def cached_arrays(
    name: str,
    compute: Callable[[], Mapping[str, np.ndarray]],
    inputs: object,
    *,
    helpers: tuple[Callable[..., object], ...] = (),
) -> Mapping[str, np.ndarray]:
    """Load arrays saved for this exact recipe and these inputs, or compute and save them once.

    The identity covers the model and kernel modules, the source of compute and of the helpers
    it relies on, and every numerical input. A sidecar records what produced the arrays.
    """
    fingerprint = _fingerprint((compute, *helpers), inputs)
    path = ARTIFACTS / f"{name}-{fingerprint}.npz"
    if path.exists():
        with np.load(path) as saved:
            return {key: saved[key] for key in saved.files}
    arrays = {key: np.asarray(value) for key, value in compute().items()}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)  # ty: ignore[invalid-argument-type] - NumPy types **kwds as if they could be allow_pickle.
    temporary.replace(path)
    path.with_suffix(".json").write_text(
        json.dumps({"name": name, "fingerprint": fingerprint, "arrays": sorted(arrays)}, indent=2)
        + "\n"
    )
    return arrays


def reference_posterior(
    r2: jax.Array, *, chains: int = 4, warmup: int = 1000, draws: int = 3000, thin: int = 3
) -> Mapping[str, np.ndarray]:
    """Long joint NUTS in daily-shock coordinates: the notebook's reference posterior."""

    def _compute() -> dict[str, np.ndarray]:
        starts = jnp.zeros(r2.shape)
        eta = model.to_eta(0.0, 0.05, 0.3)
        spec = kernels.JointNUTSSpec(model.INNOVATIONS)

        def _chain(key: jax.Array) -> kernels.MethodRun:
            return kernels.run_method(spec, r2, key, eta, starts, warmup=warmup, draws=draws)

        run = jax.jit(jax.vmap(_chain))(jax.random.split(jax.random.key(5), chains))
        return {
            "etas": np.asarray(run.parameters)[:, ::thin],
            "paths": np.asarray(run.paths)[:, ::thin],
        }

    return cached_arrays("reference", _compute, (r2, chains, warmup, draws, thin))


def conditional_density(
    coordinates: model.Coordinates,
    eta: jax.Array,
    held: jax.Array,
    r2: jax.Array,
    parameter: int,
    grid: np.ndarray,
) -> np.ndarray:
    """Density of one sampled parameter with the stored path and the other two held."""
    log_density = jax.vmap(
        lambda value: model.held_log_density(coordinates, eta.at[parameter].set(value), held, r2)
    )(jnp.asarray(grid))
    values = np.nan_to_num(np.asarray(log_density), nan=-np.inf)
    density = np.exp(values - values.max())
    return density / np.trapezoid(density, grid)


def conditional_sd(density: np.ndarray, grid: np.ndarray) -> float:
    weights = density / np.trapezoid(density, grid)
    mean = np.trapezoid(weights * grid, grid)
    return float(np.sqrt(np.trapezoid(weights * (grid - mean) ** 2, grid)))


class WidthRow(NamedTuple):
    """Conditional sd over marginal sd, per parameter, across reference draws."""

    mean: np.ndarray
    spread: np.ndarray


def coordinatewise_widths(
    etas: np.ndarray,
    paths: np.ndarray,
    r2: jax.Array,
    coordinates: Mapping[str, model.Coordinates],
    *,
    draws: int = 60,
    half_width: float = 8.0,
    points: int = 641,
    seed: int = 0,
) -> dict[str, WidthRow]:
    """sd(eta_j | stored path, eta_-j, y) / sd(eta_j | y), one parameter at a time."""
    marginal = etas.std(axis=0)
    picks = np.random.default_rng(seed).choice(etas.shape[0], draws, replace=False)
    table: dict[str, WidthRow] = {}
    for name, coordinate in coordinates.items():
        rows = np.zeros((draws, etas.shape[1]))
        for row, pick in enumerate(picks):
            eta = jnp.asarray(etas[pick])
            held = model.stored_path(coordinate, eta, jnp.asarray(paths[pick]))
            for parameter in range(etas.shape[1]):
                centre = etas[pick, parameter]
                span = half_width * marginal[parameter]
                grid = np.linspace(centre - span, centre + span, points)
                density = conditional_density(coordinate, eta, held, r2, parameter, grid)
                rows[row, parameter] = conditional_sd(density, grid) / marginal[parameter]
        table[name] = WidthRow(rows.mean(axis=0), rows.std(axis=0))
    return table


def reading_factor(grid: np.ndarray, squared_return: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact log factor over level + path, and its Gaussian fit at the peak; peak = 0."""
    peak = np.log(squared_return)
    exact = -0.5 * (grid + squared_return * np.exp(-grid))
    return exact - exact.max(), -0.25 * (grid - peak) ** 2


def iact(values: np.ndarray) -> float:
    """Integrated autocorrelation time of one chain, with Sokal's automatic window."""
    centred = values - values.mean()
    length = centred.size
    spectrum = np.fft.rfft(centred, 2 * length)
    autocovariance = np.fft.irfft(spectrum * np.conj(spectrum))[:length]
    if autocovariance[0] <= 0.0:
        return float("inf")
    autocorrelation = autocovariance / autocovariance[0]
    tau = 1.0
    for lag in range(1, length):
        tau += 2.0 * autocorrelation[lag]
        if lag >= 5.0 * tau:
            break
    return float(tau)


class LaterReading(NamedTuple):
    """Mean change by day when one late reading changes, under common random numbers."""

    proposed: np.ndarray
    returned: np.ndarray


def later_reading_experiment(
    eta: jax.Array,
    r2: jax.Array,
    reference: jax.Array,
    *,
    late_day: int,
    late_reading: float,
    repeats: int = 300,
    delta: float = 0.1,
    particles: int = 32,
) -> dict[str, LaterReading]:
    """Same random numbers, one reading replaced. How far back does the change reach?"""
    perturbed = r2.at[late_day].set(late_reading)
    keys = jax.random.split(jax.random.key(11), repeats)
    before = kernels.sv_target(eta[None], r2)
    after = kernels.sv_target(eta[None], perturbed)

    @jax.jit
    def _kalman(key: jax.Array) -> tuple[jax.Array, jax.Array]:
        first = kernels.aux_kalman_trace(key, before, reference, delta=delta)
        second = kernels.aux_kalman_trace(key, after, reference, delta=delta)
        return second.proposal_mean - first.proposal_mean, second.path - first.path

    @jax.jit
    def _bank(key: jax.Array) -> tuple[jax.Array, jax.Array]:
        first = kernels.mgrad_trace(key, before, reference, particles=particles, delta=delta)
        second = kernels.mgrad_trace(key, after, reference, particles=particles, delta=delta)
        return (
            jnp.mean(second.proposal_means - first.proposal_means, axis=-1),
            second.path - first.path,
        )

    def _collect(function: Callable[[jax.Array], tuple[jax.Array, jax.Array]]) -> LaterReading:
        proposed, returned = jax.lax.map(function, keys)
        return LaterReading(np.asarray(proposed).mean(axis=0), np.asarray(returned).mean(axis=0))

    def _compute() -> dict[str, np.ndarray]:
        kalman = _collect(_kalman)
        bank = _collect(_bank)
        return {
            "kalman_proposed": kalman.proposed,
            "kalman_returned": kalman.returned,
            "bank_proposed": bank.proposed,
            "bank_returned": bank.returned,
        }

    arrays = cached_arrays(
        "later-reading",
        _compute,
        (eta, r2, reference, late_day, late_reading, repeats, delta, particles),
        helpers=(later_reading_experiment,),
    )
    return {
        "Auxiliary Kalman": LaterReading(arrays["kalman_proposed"], arrays["kalman_returned"]),
        "Particle-mGRAD": LaterReading(arrays["bank_proposed"], arrays["bank_returned"]),
    }


class KalmanScan(NamedTuple):
    days: np.ndarray
    deltas: np.ndarray
    whole_path: np.ndarray
    block_length: int
    blocks: np.ndarray


def kalman_scan(
    eta: jax.Array,
    r2: jax.Array,
    path: jax.Array,
    *,
    days: tuple[int, ...] = (50, 100, 200, 400, 800, 1000),
    deltas: tuple[float, ...] = (0.03, 0.1, 0.3),
    block_length: int = 128,
    moves: int = 96,
) -> KalmanScan:
    """Mean acceptance probability of one verdict, starting from an exact posterior path."""

    def _whole(
        target: kernels.PathTarget, start: jax.Array, delta: float, keys: jax.Array
    ) -> float:
        def _accept(key: jax.Array) -> jax.Array:
            return kernels.aux_kalman_trace(key, target, start, delta=delta).acceptance_probability

        return float(jnp.mean(jax.jit(jax.vmap(_accept))(keys)))

    def _blocked(target: kernels.PathTarget, start: jax.Array, keys: jax.Array) -> float:
        def _accept(key: jax.Array) -> jax.Array:
            return kernels.blocked_kalman_trace(
                key, target, start, delta=max(deltas), block_length=block_length
            ).acceptance_probability

        return float(jnp.mean(jax.jit(jax.vmap(_accept))(keys)))

    def _compute() -> dict[str, np.ndarray]:
        keys = jax.random.split(jax.random.key(2), moves)
        whole = np.zeros((len(deltas), len(days)))
        blocks = np.zeros(len(days))
        for column, length in enumerate(days):
            target = kernels.sv_target(eta[None], r2[:length])
            start = path[:length]
            for row, delta in enumerate(deltas):
                whole[row, column] = _whole(target, start, delta, keys)
            blocks[column] = _blocked(target, start, keys)
        return {"whole_path": whole, "blocks": blocks}

    arrays = cached_arrays(
        "kalman-scan",
        _compute,
        (eta, r2, path, days, deltas, block_length, moves),
        helpers=(_whole, _blocked),
    )
    return KalmanScan(
        np.asarray(days),
        np.asarray(deltas),
        arrays["whole_path"],
        block_length,
        arrays["blocks"],
    )


class KernelRace(NamedTuple):
    """Three kernels on one conditional, from the same start, with the path held."""

    traces: Mapping[str, np.ndarray]
    conditional_mean: np.ndarray
    conditional_sd: np.ndarray
    marginal_mean: np.ndarray
    marginal_sd: np.ndarray


def parameter_kernel_race(
    etas: np.ndarray,
    paths: np.ndarray,
    r2: jax.Array,
    coordinates: model.Coordinates,
    *,
    name: str,
    steps: int = 120,
    tune_draws: int = 1500,
) -> KernelRace:
    """Tune on the conditional at one held path, then restart everything from far away."""
    standard = (etas - etas.mean(axis=0)) / etas.std(axis=0)
    held_index = int(np.argmin(np.abs(standard[:, 2] - 1.0) + np.abs(standard[:, 0])))
    start_index = int(np.argmin(np.abs(standard[:, 2] + 1.5) + np.abs(standard[:, 0] + 1.0)))
    eta_held = jnp.asarray(etas[held_index])
    held = model.stored_path(coordinates, eta_held, jnp.asarray(paths[held_index]))
    start = jnp.asarray(etas[start_index])

    def _density(eta: jax.Array) -> jax.Array:
        return model.held_log_density(coordinates, eta, held, r2)

    def _compute() -> dict[str, np.ndarray]:
        tuned = kernels.nuts_chain(
            jax.random.key(21),
            _density,
            eta_held,
            warmup=800,
            draws=tune_draws,
            settings=kernels.NUTSSettings(dense_mass=True),
        )
        draws = np.asarray(tuned.positions)
        covariance = np.cov(draws, rowvar=False)
        scale = np.linalg.cholesky(covariance)
        walk = np.asarray(
            kernels.parameter_mh_chain(
                jax.random.key(22),
                _density,
                start,
                jnp.asarray(scale * 2.38 / np.sqrt(3.0)),
                sweeps=steps,
            )
        )
        langevin = np.asarray(
            kernels.parameter_mh_chain(
                jax.random.key(23),
                _density,
                start,
                jnp.asarray(scale * 1.2),
                sweeps=steps,
                langevin=True,
            )
        )
        initialize, transition = hmc(
            potential_fn=lambda value: -_density(value),
            kinetic_fn=kernels.gaussian_kinetic_energy,
            algo="NUTS",
        )
        state = initialize(
            start,
            num_warmup=0,
            step_size=tuned.tuning.step_size,
            inverse_mass_matrix=tuned.tuning.inverse_mass,
            adapt_step_size=False,
            adapt_mass_matrix=False,
            dense_mass=True,
            rng_key=jax.random.key(24),
        )

        def _advance(current: HMCState, _: None) -> tuple[HMCState, jax.Array]:
            updated = transition(current)
            return updated, updated.z

        _, restarted = jax.lax.scan(_advance, state, None, length=steps)
        return {
            "random_walk": np.concatenate((np.asarray(start)[None], walk)),
            "mala": np.concatenate((np.asarray(start)[None], langevin)),
            "nuts": np.concatenate((np.asarray(start)[None], np.asarray(restarted))),
            "conditional_mean": draws.mean(axis=0),
            "conditional_sd": draws.std(axis=0),
        }

    arrays = cached_arrays(
        f"kernel-race-{name}",
        _compute,
        (etas, paths, r2, coordinates, steps, tune_draws),
        helpers=(parameter_kernel_race,),
    )
    return KernelRace(
        {
            "Random walk": arrays["random_walk"],
            "MALA": arrays["mala"],
            "NUTS": arrays["nuts"],
        },
        arrays["conditional_mean"],
        arrays["conditional_sd"],
        etas.mean(axis=0),
        etas.std(axis=0),
    )


class DecodeLedger(NamedTuple):
    """Physical path after a parameter change, and the change in each density term."""

    path: np.ndarray
    terms: np.ndarray


def decode_ledger(
    coordinates: model.Coordinates,
    eta: jax.Array,
    path: jax.Array,
    moved: jax.Array,
    r2: jax.Array,
) -> DecodeLedger:
    held = model.stored_path(coordinates, eta, path)
    before = model.held_terms(coordinates, eta, held, r2)
    after = model.held_terms(coordinates, moved, held, r2)
    return DecodeLedger(
        np.asarray(model.physical_path(coordinates, moved, held)),
        np.asarray([float(new - old) for new, old in zip(after, before, strict=True)]),
    )


class BlockWidths(NamedTuple):
    table: dict[str, WidthRow]
    fewest_effective_draws: float


def block_conditional_widths(
    etas: np.ndarray,
    paths: np.ndarray,
    r2: jax.Array,
    coordinates: Mapping[str, model.Coordinates],
    *,
    draws: int = 16,
    warmup: int = 500,
    samples: int = 1500,
    seed: int = 1,
) -> BlockWidths:
    """sd(eta_j | stored path, y) / sd(eta_j | y): what a three-parameter kernel can reach."""
    marginal = etas.std(axis=0)
    picks = np.random.default_rng(seed).choice(etas.shape[0], draws, replace=False)

    def _spreads(coordinate: model.Coordinates) -> tuple[np.ndarray, np.ndarray]:
        def _one(key: jax.Array, eta: jax.Array, path: jax.Array) -> jax.Array:
            held = model.stored_path(coordinate, eta, path)

            def _density(value: jax.Array) -> jax.Array:
                return model.held_log_density(coordinate, value, held, r2)

            return kernels.nuts_chain(
                key,
                _density,
                eta,
                warmup=warmup,
                draws=samples,
                settings=kernels.NUTSSettings(dense_mass=True),
            ).positions

        positions = np.asarray(
            jax.jit(jax.vmap(_one))(
                jax.random.split(jax.random.key(seed), draws),
                jnp.asarray(etas[picks]),
                jnp.asarray(paths[picks]),
            )
        )
        effective = np.asarray(
            [
                [samples / iact(run[:, column]) for column in range(run.shape[1])]
                for run in positions
            ]
        )
        return positions.std(axis=1), effective

    def _compute() -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for name, coordinate in coordinates.items():
            arrays[f"{name}|spread"], arrays[f"{name}|effective"] = _spreads(coordinate)
        return arrays

    arrays = cached_arrays(
        "block-widths",
        _compute,
        (etas, paths, r2, dict(coordinates), draws, warmup, samples, seed),
        helpers=(block_conditional_widths, iact),
    )
    table = {
        name: WidthRow(
            (arrays[f"{name}|spread"] / marginal).mean(axis=0),
            (arrays[f"{name}|spread"] / marginal).std(axis=0),
        )
        for name in coordinates
    }
    return BlockWidths(table, float(min(arrays[f"{name}|effective"].min() for name in coordinates)))


def pmmh_noise(
    eta: jax.Array,
    r2: jax.Array,
    *,
    days: tuple[int, ...] = (100, 300, 1000),
    particles: tuple[int, ...] = (16, 64, 256, 1024),
    repeats: int = 48,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spread of the bootstrap filter's log-likelihood estimate, by length and particles."""

    def _compute() -> dict[str, np.ndarray]:
        spread = np.zeros((len(days), len(particles)))
        keys = jax.random.split(jax.random.key(3), repeats)
        for row, length in enumerate(days):
            for column, count in enumerate(particles):
                estimate = jax.jit(
                    jax.vmap(
                        lambda key, size=count, data=r2[:length]: (
                            kernels.bootstrap_filter(key, eta, data, particles=size).log_likelihood
                        )
                    )
                )
                spread[row, column] = float(jnp.std(estimate(keys)))
        return {"spread": spread}

    arrays = cached_arrays("pmmh-noise", _compute, (eta, r2, days, particles, repeats))
    return np.asarray(days), np.asarray(particles), arrays["spread"]


def coordinatewise_widths_cached(
    etas: np.ndarray,
    paths: np.ndarray,
    r2: jax.Array,
    coordinates: Mapping[str, model.Coordinates],
) -> dict[str, WidthRow]:
    def _compute() -> dict[str, np.ndarray]:
        table = coordinatewise_widths(etas, paths, r2, coordinates)
        return {
            **{f"{name}|mean": row.mean for name, row in table.items()},
            **{f"{name}|spread": row.spread for name, row in table.items()},
        }

    arrays = cached_arrays(
        "coordinate-widths",
        _compute,
        (etas, paths, r2, dict(coordinates)),
        helpers=(coordinatewise_widths, conditional_density, conditional_sd),
    )
    return {
        name: WidthRow(arrays[f"{name}|mean"], arrays[f"{name}|spread"]) for name in coordinates
    }


class MassMatrixChecks(NamedTuple):
    trajectory_difference: float
    slope_difference: float
    cholesky_difference: float
    slope_from_chart: np.ndarray
    slope_from_covariance: np.ndarray
    slope_from_cholesky: np.ndarray
    raw_block: np.ndarray


def mass_matrix_checks(r2: jax.Array, eta: jax.Array, *, days: int = 100) -> MassMatrixChecks:
    """Fact 1: linear coordinates against a dense metric. Fact 2: the slope in a Gaussian."""
    readings = r2[:days]
    path = model.simulate(days, SEED).h

    def _density(value: jax.Array) -> jax.Array:
        return model.held_log_density(model.PHYSICAL, value, path, readings)

    factor = jnp.asarray([[0.2, 0.0, 0.0], [0.05, 0.4, 0.0], [-0.02, 0.03, 0.1]])
    inverse_mass = factor @ factor.T
    key = jax.random.key(13)
    physical = kernels.hmc_trace(
        key, _density, eta, step_size=0.1, steps=20, inverse_mass=inverse_mass
    )
    whitened = kernels.hmc_trace(
        key,
        lambda value: _density(factor @ value),
        jax.scipy.linalg.solve_triangular(factor, eta, lower=True),
        step_size=0.1,
        steps=20,
    )
    trajectory_difference = float(
        jnp.max(jnp.abs(whitened.positions @ factor.T - physical.positions))
    )

    decay, vol, reading_sd, level = 0.05, 0.3, 0.5, -0.2
    twin = model.linear_gaussian_twin(
        days, decay=decay, vol=vol, level=level, reading_sd=reading_sd
    )
    noise_key, reading_key = jax.random.split(jax.random.key(17))
    latent = model.gaussian_sample(
        noise_key,
        model.gaussian_from_precision(
            *_prior_precision(twin),
        ),
    )
    observed = level + latent + reading_sd * jax.random.normal(reading_key, (days,))

    def _conditional_mean(value: jax.Array) -> jax.Array:
        moved = model.linear_gaussian_twin(
            days, decay=decay, vol=vol, level=value, reading_sd=reading_sd
        )
        return model.linear_gaussian_posterior(moved, observed).mean

    from_chart = np.asarray(jax.jacfwd(_conditional_mean)(jnp.asarray(level)))
    diagonal, off_diagonal, _ = _prior_precision(twin)
    precision = np.zeros((days + 1, days + 1))
    precision[0, 0] = 1.0 / 4.0 + days / reading_sd**2
    precision[0, 1:] = precision[1:, 0] = 1.0 / reading_sd**2
    body = np.diag(np.asarray(diagonal) + 1.0 / reading_sd**2)
    body += np.diag(np.asarray(off_diagonal), 1) + np.diag(np.asarray(off_diagonal), -1)
    precision[1:, 1:] = body
    covariance = np.linalg.inv(precision)
    from_covariance = covariance[1:, 0] / covariance[0, 0]
    # With the parameter ordered first, the inverse mass's Cholesky factor holds B in its first
    # column, after dividing by the parameter's own entry: L_h,eta = B * L_eta,eta.
    factor_joint = np.linalg.cholesky(covariance)
    from_cholesky = factor_joint[1:, 0] / factor_joint[0, 0]
    return MassMatrixChecks(
        trajectory_difference,
        float(np.max(np.abs(from_chart - from_covariance))),
        float(np.max(np.abs(from_cholesky - from_covariance))),
        from_chart,
        from_covariance,
        from_cholesky,
        covariance[1:, 0],
    )


def _prior_precision(twin: model.LinearGaussianModel) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Precision of the twin's path before any readings, with zero information."""
    precision = 1.0 / twin.transition_variances
    diagonal = jnp.zeros_like(twin.observation_variances).at[0].add(1.0 / twin.initial_variance)
    diagonal = diagonal.at[:-1].add(twin.transition_slopes**2 * precision).at[1:].add(precision)
    return (
        diagonal,
        -twin.transition_slopes * precision,
        jnp.zeros_like(twin.observation_variances),
    )


def metric_versus_chart(
    r2: jax.Array, *, days: int = 300, chains: int = 8, warmup: int = 1000, draws: int = 6000
) -> list[dict[str, str | float]]:
    """Joint NUTS on a prefix, for each combination of coordinates and metric."""
    readings = r2[:days]

    def _compute() -> dict[str, np.ndarray]:
        eta = model.to_eta(0.0, 0.05, 0.3)
        start = model.refine_path(eta, jnp.zeros(days), readings)
        keys = jax.random.split(jax.random.key(31), chains)

        def _run(spec: kernels.JointNUTSSpec, chain_keys: jax.Array) -> kernels.MethodRun:
            def _chain(key: jax.Array) -> kernels.MethodRun:
                return kernels.run_method(
                    spec, readings, key, eta, start, warmup=warmup, draws=draws
                )

            return jax.jit(jax.vmap(_chain))(chain_keys)

        pilot = _run(kernels.JointNUTSSpec(model.INNOVATIONS), keys)
        centre = jnp.mean(pilot.parameters, axis=(0, 1))
        chart = jax.jit(model.fit_chart)(centre, jnp.mean(pilot.paths, axis=(0, 1)), readings)
        arrays: dict[str, np.ndarray] = {}
        for name, coordinate in (
            ("h", model.PHYSICAL),
            ("daily shocks", model.INNOVATIONS),
            ("chart", chart),
        ):
            for dense in (False, True):
                metric = "dense" if dense else "diagonal"
                run = _run(
                    kernels.JointNUTSSpec(coordinate, kernels.NUTSSettings(dense_mass=dense)), keys
                )
                assert isinstance(run.statistics, kernels.NUTSStatistics)
                arrays[f"{name}|{metric}|parameters"] = np.asarray(run.parameters)
                arrays[f"{name}|{metric}|steps"] = np.asarray(run.statistics.leapfrog_steps)
        return arrays

    arrays = cached_arrays("metric-versus-chart", _compute, (r2, days, chains, warmup, draws))
    rows: list[dict[str, str | float]] = []
    for name in ("h", "daily shocks", "chart"):
        for metric in ("diagonal", "dense"):
            parameters = arrays[f"{name}|{metric}|parameters"]
            steps = float(arrays[f"{name}|{metric}|steps"].mean())
            ess = np.asarray(array_stats.ess(parameters, method="mean", chain_axis=0, draw_axis=1))
            rhat = np.asarray(array_stats.rhat(parameters, chain_axis=0, draw_axis=1))
            sweeps = chains * draws / ess
            rows.append(
                {
                    "Coordinates": name,
                    "Mass matrix": metric,
                    "Leapfrog steps per draw": round(steps, 1),
                    "Sweeps per effective sample: level": round(float(sweeps[0]), 1),
                    "Sweeps per effective sample: log σ": round(float(sweeps[2]), 1),
                    "Gradients per effective sample: log σ": round(float(sweeps[2]) * steps),
                    "Worst R̂": round(float(rhat.max()), 3),
                }
            )
    return rows


class Estimate(NamedTuple):
    mean: np.ndarray | float
    error: np.ndarray | float


class BarkerReference(NamedTuple):
    """How often the new candidate wins, with standard errors across independent chains."""

    particles: np.ndarray
    mpgibbs: Estimate
    pmmh: Estimate
    ideal: Estimate
    ideal_metropolis: Estimate
    held: Estimate


def _gaussian_bootstrap_log_likelihood(
    key: jax.Array, eta: jax.Array, readings: jax.Array, reading_variance: float, particles: int
) -> jax.Array:
    """Unbiased (in exponential) bootstrap estimate on the Gaussian oracle."""

    def _step(
        carry: tuple[jax.Array, jax.Array], values: tuple[jax.Array, jax.Array]
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        old, log_weights = carry
        index, time_key = values
        ancestor_key, noise_key = jax.random.split(time_key)
        ancestors = jax.random.categorical(ancestor_key, log_weights, shape=(particles,))
        mean = jnp.where(index == 0, 0.0, (1.0 - jnp.exp(eta[1])) * old[ancestors])
        deviation = jnp.where(index == 0, 1.0, jnp.exp(eta[2]))
        current = mean + deviation * jax.random.normal(noise_key, (particles,))
        weights = model.normal_logpdf(readings[index], eta[0] + current, reading_variance)
        return (current, weights), logsumexp(weights) - jnp.log(particles)

    days = readings.shape[0]
    _, increments = jax.lax.scan(
        _step,
        (jnp.zeros(particles), jnp.zeros(particles)),
        (jnp.arange(days), jax.random.split(key, days)),
    )
    return jnp.sum(increments)


def barker_reference(
    *,
    days: int = 100,
    particle_counts: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128),
    chains: int = 8,
    sweeps: int = 3000,
    step: float = 0.15,
) -> BarkerReference:
    """How often the new candidate wins, on a Gaussian oracle where every reference is exact."""
    decay, vol, reading_sd, level = 0.1, 0.3, 0.5, -0.2
    truth = model.to_eta(level, decay, vol)
    twin = model.linear_gaussian_twin(
        days, decay=decay, vol=vol, level=level, reading_sd=reading_sd
    )
    latent_key, reading_key = jax.random.split(jax.random.key(19))
    latent = model.gaussian_sample(
        latent_key, model.gaussian_from_precision(*_prior_precision(twin))
    )
    observed = level + latent + reading_sd * jax.random.normal(reading_key, (days,))
    reading_variance = reading_sd**2
    proposal = step * jnp.eye(3)

    def _twin(eta: jax.Array) -> model.LinearGaussianModel:
        values = model.from_eta(eta)
        return model.linear_gaussian_twin(
            days,
            decay=values.decay,
            vol=values.vol,
            level=values.level,
            reading_sd=reading_sd,
        )

    def _marginal(eta: jax.Array) -> jax.Array:
        return model.kalman_filter(_twin(eta), observed).log_likelihood + model.log_prior(eta)

    def _candidate(key: jax.Array, eta: jax.Array) -> jax.Array:
        return eta + proposal @ jax.random.normal(key, (3,))

    def _compute() -> dict[str, np.ndarray]:
        keys = jax.random.split(jax.random.key(23), chains)

        @jax.jit
        def _ideal(chain_key: jax.Array) -> tuple[jax.Array, jax.Array]:
            def _step(
                eta: jax.Array, key: jax.Array
            ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
                proposal_key, flip_key = jax.random.split(key)
                candidate = _candidate(proposal_key, eta)
                difference = _marginal(candidate) - _marginal(eta)
                win = jax.nn.sigmoid(difference)
                chosen = jnp.where(jax.random.uniform(flip_key) < win, candidate, eta)
                return chosen, (win, jnp.exp(jnp.minimum(0.0, difference)))

            _, (wins, accepts) = jax.lax.scan(_step, truth, jax.random.split(chain_key, sweeps))
            return jnp.mean(wins[sweeps // 5 :]), jnp.mean(accepts[sweeps // 5 :])

        ideal, ideal_metropolis = jax.vmap(_ideal)(keys)

        @jax.jit
        def _held(chain_key: jax.Array) -> jax.Array:
            def _step(eta: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
                path_key, proposal_key, flip_key = jax.random.split(key, 3)
                path = model.gaussian_sample(
                    path_key, model.linear_gaussian_posterior(_twin(eta), observed)
                )
                candidate = _candidate(proposal_key, eta)

                def _complete(value: jax.Array) -> jax.Array:
                    return model.linear_gaussian_log_joint(
                        _twin(value), path, observed
                    ) + model.log_prior(value)

                win = jax.nn.sigmoid(_complete(candidate) - _complete(eta))
                chosen = jnp.where(jax.random.uniform(flip_key) < win, candidate, eta)
                return chosen, win

            _, wins = jax.lax.scan(_step, truth, jax.random.split(chain_key, sweeps))
            return jnp.mean(wins[sweeps // 5 :])

        held = jax.vmap(_held)(keys)
        start = model.linear_gaussian_posterior(_twin(truth), observed).mean
        mpgibbs = np.zeros((chains, len(particle_counts)))
        pmmh = np.zeros((chains, len(particle_counts)))
        for column, count in enumerate(particle_counts):

            def _mpgibbs(chain_key: jax.Array, size: int = count) -> jax.Array:
                def _step(
                    carry: tuple[jax.Array, jax.Array], key: jax.Array
                ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
                    eta, path = carry
                    trace = kernels.ensemble_trace(
                        key,
                        eta,
                        path,
                        lambda parameters: kernels.gaussian_target(
                            parameters, observed, reading_variance=reading_variance
                        ),
                        kernels.BootstrapSpec(size),
                        proposal,
                    )
                    return (trace.eta, trace.stored_path), trace.final_label_probabilities[1]

                _, wins = jax.lax.scan(_step, (truth, start), jax.random.split(chain_key, sweeps))
                return jnp.mean(wins[sweeps // 5 :])

            def _pmmh(chain_key: jax.Array, size: int = count) -> jax.Array:
                def _step(
                    carry: tuple[jax.Array, jax.Array], key: jax.Array
                ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
                    eta, estimate = carry
                    proposal_key, filter_key, accept_key = jax.random.split(key, 3)
                    candidate = _candidate(proposal_key, eta)
                    new_estimate = _gaussian_bootstrap_log_likelihood(
                        filter_key, candidate, observed, reading_variance, size
                    ) + model.log_prior(candidate)
                    accept = jnp.exp(jnp.minimum(0.0, new_estimate - estimate))
                    taken = jax.random.uniform(accept_key) < accept
                    return (
                        jnp.where(taken, candidate, eta),
                        jnp.where(taken, new_estimate, estimate),
                    ), accept

                first = _gaussian_bootstrap_log_likelihood(
                    chain_key, truth, observed, reading_variance, size
                ) + model.log_prior(truth)
                _, accepts = jax.lax.scan(
                    _step,
                    (truth, first),
                    jax.random.split(jax.random.fold_in(chain_key, 1), sweeps),
                )
                return jnp.mean(accepts[sweeps // 5 :])

            mpgibbs[:, column] = np.asarray(jax.jit(jax.vmap(_mpgibbs))(keys))
            pmmh[:, column] = np.asarray(jax.jit(jax.vmap(_pmmh))(keys))
        return {
            "mpgibbs": mpgibbs,
            "pmmh": pmmh,
            "ideal": np.asarray(ideal),
            "ideal_metropolis": np.asarray(ideal_metropolis),
            "held": np.asarray(held),
        }

    arrays = cached_arrays(
        "barker-reference",
        _compute,
        (days, particle_counts, chains, sweeps, step),
        helpers=(barker_reference, _gaussian_bootstrap_log_likelihood, _prior_precision),
    )

    def _estimate(values: np.ndarray) -> Estimate:
        return Estimate(values.mean(axis=0), values.std(axis=0) / np.sqrt(values.shape[0]))

    return BarkerReference(
        np.asarray(particle_counts),
        _estimate(arrays["mpgibbs"]),
        _estimate(arrays["pmmh"]),
        _estimate(arrays["ideal"]),
        _estimate(arrays["ideal_metropolis"]),
        _estimate(arrays["held"]),
    )


class Regime(NamedTuple):
    """Fixed-parameter chains in one regime: an exploratory scan, then an independent check."""

    deltas: np.ndarray
    lines: Mapping[str, np.ndarray]
    rows: list[dict[str, str | float]]
    bootstrap: float
    mgrad: float
    kalman: float


class ChainScores(NamedTuple):
    mean_path: float
    middle_day: float
    days_changed: float
    milliseconds: float

    def row(self, label: str) -> dict[str, str | float]:
        return {
            "Path move": label,
            "Sweeps per effective sample, mean h": round(self.mean_path, 1),
            "Sweeps per effective sample, middle day": round(self.middle_day, 1),
            "Fraction of days changed per sweep": round(self.days_changed, 2),
            "ms per sweep": round(self.milliseconds, 2),
            "ms per effective sample, mean h": round(self.mean_path * self.milliseconds, 1),
        }


def _chain_scores(
    spec: kernels.SmootherSpec,
    target: kernels.PathTarget,
    start: jax.Array,
    sweeps: int,
    seed: int,
) -> np.ndarray:
    """Scores of one chain from an exact posterior start; compilation is not timed."""

    def _run(key: jax.Array) -> kernels.PathRun:
        return kernels.run_path_chain(spec, key, target, start, sweeps=sweeps)

    key = jax.random.key(seed)
    compiled = jax.jit(_run).lower(key).compile()
    begun = time.perf_counter()
    run = compiled(key)
    jax.block_until_ready(run.stored_paths)
    seconds = time.perf_counter() - begun
    paths = np.asarray(run.stored_paths)
    return np.asarray(
        (
            iact(paths.mean(axis=1)),
            iact(paths[:, paths.shape[1] // 2]),
            float(np.mean(np.asarray(run.change_rate))),
            1000.0 * seconds / sweeps,
        )
    )


def _regime(
    name: str,
    target: kernels.PathTarget,
    start: jax.Array,
    deltas: tuple[float, ...],
    *,
    particles: int,
    block_length: int,
    hmc_spec: kernels.PathHMCSpec,
    scan_sweeps: int,
    check_sweeps: int,
) -> Regime:
    makers: dict[str, Callable[[float], kernels.SmootherSpec]] = {
        "mgrad": lambda delta: kernels.MGradSpec(particles, delta),
        "whole": kernels.AuxKalmanSpec,
        "blocks": lambda delta: kernels.BlockedKalmanSpec(delta, block_length),
    }

    def _compute() -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {
            "bootstrap": _chain_scores(
                kernels.BootstrapSpec(particles), target, start, check_sweeps, 2
            ),
            "hmc": _chain_scores(hmc_spec, target, start, check_sweeps, 2),
        }
        for key, make in makers.items():
            scan = np.stack(
                [_chain_scores(make(delta), target, start, scan_sweeps, 1) for delta in deltas]
            )
            arrays[f"{key}|scan"] = scan
            # An independent seed and a longer run for the value chosen on the scan.
            finite = np.where(np.isfinite(scan[:, 0]), scan[:, 0], np.inf)
            best = int(np.argmin(finite))
            arrays[f"{key}|best"] = np.asarray(best)
            arrays[f"{key}|check"] = _chain_scores(
                make(deltas[best]), target, start, check_sweeps, 2
            )
        return arrays

    arrays = cached_arrays(
        f"regime-{name}",
        _compute,
        (target, start, deltas, particles, block_length, hmc_spec, scan_sweeps, check_sweeps),
        helpers=(_regime, _chain_scores, iact),
    )
    names = {
        "mgrad": "Particle-mGRAD",
        "whole": "Auxiliary Kalman, whole path",
        "blocks": "Auxiliary Kalman, blocks",
    }
    rows = [
        ChainScores(*arrays["bootstrap"].tolist()).row(
            f"Bootstrap conditional SMC, N = {particles}"
        )
    ]
    for key, label in names.items():
        best = int(arrays[f"{key}|best"])
        rows.append(
            ChainScores(*arrays[f"{key}|check"].tolist()).row(f"{label}, δ = {deltas[best]:g}")
        )
    rows.append(
        ChainScores(*arrays["hmc"].tolist()).row(
            f"HMC, step {hmc_spec.step_size:g}, {hmc_spec.steps} steps"
        )
    )
    return Regime(
        np.asarray(deltas),
        {label: arrays[f"{key}|scan"][:, 0] for key, label in names.items()},
        rows,
        float(arrays["bootstrap"][0]),
        float(arrays["mgrad|check"][0]),
        float(arrays["whole|check"][0]),
    )


def fixed_parameter_regimes(
    eta: jax.Array,
    r2: jax.Array,
    path: jax.Array,
    *,
    sharp_reading_sd: float = 0.001,
    particles: int = 32,
    scan_sweeps: int = 2000,
    check_sweeps: int = 10000,
) -> dict[str, Regime]:
    """Our model, whose readings are weak, and a Gaussian twin with very sharp readings."""
    days = r2.shape[0]
    weak = _regime(
        "weak",
        kernels.sv_target(eta[None], r2),
        path,
        (0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
        particles=particles,
        block_length=128,
        hmc_spec=kernels.PathHMCSpec(0.05, 30),
        scan_sweeps=scan_sweeps,
        check_sweeps=check_sweeps,
    )
    values = model.from_eta(eta)
    twin = model.linear_gaussian_twin(
        days, decay=values.decay, vol=values.vol, level=values.level, reading_sd=sharp_reading_sd
    )
    latent_key, reading_key, start_key = jax.random.split(jax.random.key(29), 3)
    latent = model.gaussian_sample(
        latent_key, model.gaussian_from_precision(*_prior_precision(twin))
    )
    observed = (
        jnp.asarray(values.level)
        + latent
        + sharp_reading_sd * jax.random.normal(reading_key, (days,))
    )
    start = model.gaussian_sample(start_key, model.linear_gaussian_posterior(twin, observed))
    variance = sharp_reading_sd**2
    sharp = _regime(
        "sharp",
        kernels.gaussian_target(eta[None], observed, reading_variance=variance),
        start,
        tuple(factor * variance for factor in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)),
        particles=particles,
        block_length=128,
        hmc_spec=kernels.PathHMCSpec(0.5 * sharp_reading_sd, 60),
        scan_sweeps=scan_sweeps,
        check_sweeps=check_sweeps,
    )
    return {
        "Our model: each reading is weak": weak,
        f"A Gaussian twin: readings with sd {sharp_reading_sd:g}": sharp,
    }


class IdealWalk(NamedTuple):
    """Barker's rule on a three-dimensional standard Gaussian: MPGibbs after a perfect path draw."""

    scales: np.ndarray
    flip_rates: np.ndarray
    sweeps: np.ndarray


def ideal_walk(
    scales: tuple[float, ...] = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0),
    *,
    dimension: int = 3,
    chains: int = 8,
    sweeps: int = 100_000,
    seed: int = 61,
) -> IdealWalk:
    """Flip rate and sweeps per effective sample of the walk, per proposal sd in whitened coordinates."""

    def _compute() -> dict[str, np.ndarray]:
        def _chain(key: jax.Array, scale: jax.Array) -> tuple[jax.Array, jax.Array]:
            def _step(
                state: jax.Array, step_key: jax.Array
            ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
                move_key, flip_key = jax.random.split(step_key)
                candidate = state + scale * jax.random.normal(move_key, state.shape)
                log_ratio = 0.5 * (state @ state - candidate @ candidate)
                win = jax.random.uniform(flip_key) < jax.nn.sigmoid(log_ratio)
                moved = jnp.where(win, candidate, state)
                return moved, (moved[0], win)

            start = jax.random.normal(jax.random.fold_in(key, 1), (dimension,))
            return jax.lax.scan(_step, start, jax.random.split(key, sweeps))[1]

        run = jax.jit(jax.vmap(_chain, in_axes=(0, None)))
        keys = jax.random.split(jax.random.key(seed), chains)
        draws, wins = zip(*(run(keys, jnp.asarray(value)) for value in scales), strict=True)
        return {"draws": np.stack(draws), "wins": np.stack(wins)}

    arrays = cached_arrays("ideal-walk", _compute, (scales, dimension, chains, sweeps, seed))
    ess = np.asarray(
        [array_stats.ess(row, method="mean", chain_axis=0, draw_axis=1) for row in arrays["draws"]]
    )
    return IdealWalk(np.asarray(scales), arrays["wins"].mean(axis=(1, 2)), chains * sweeps / ess)


class Runs(NamedTuple):
    """Saved full-method runs, read once at the notebook boundary."""

    directory: Path
    days: int
    chains: int
    names: tuple[str, ...]
    metadata: Mapping[str, dict[str, Any]]
    reference_means: np.ndarray
    reference_errors: np.ndarray
    reference_draws: int
    cloud: np.ndarray
    chart: model.Chart
    tuning: Mapping[str, Any]
    timing: Mapping[str, float]

    def substeps(self, name: str, *, sweeps: int = 40) -> np.ndarray:
        """Stages for the first sweeps of one chain: (sweeps, stages, 6)."""
        with np.load(self.directory / f"{name}.npz") as saved:
            return saved["substep_summaries"][0, :sweeps]

    @property
    def means(self) -> np.ndarray:
        return np.asarray([self.metadata[name]["diagnostics"]["mean"] for name in self.names])

    @property
    def errors(self) -> np.ndarray:
        return np.asarray([self.metadata[name]["diagnostics"]["mcse"] for name in self.names])

    def sweeps_per_effective_sample(self, name: str) -> np.ndarray:
        ess = np.asarray(self.metadata[name]["diagnostics"]["ess_mean"])
        config = self.metadata[name]["config"]
        return int(config["chains"]) * int(config["draws"]) / ess

    def milliseconds_per_sweep(self, name: str) -> float:
        """Wall time per chain update from the separate one-method-at-a-time timing run."""
        return float(self.timing[name])

    def milliseconds_per_effective_sample(self, name: str, column: int) -> float:
        """Amortized: ms per sweep times sweeps per effective sample, so warmup's fixed cost isn't charged."""
        return self.milliseconds_per_sweep(name) * float(
            self.sweeps_per_effective_sample(name)[column]
        )

    @property
    def reference_gaps(self) -> np.ndarray:
        """Each run's distance from the reference, in combined Monte Carlo errors: (runs, quantities)."""
        return (self.means - self.reference_means) / np.hypot(self.errors, self.reference_errors)

    def flip_rates(
        self, labels: Mapping[str, str]
    ) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
        """Each smoother's flip rate by candidate scale, from the pilots, chart off and on."""
        panels: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        for coordinate, title in (("path", "Chart off"), ("chart", "Chart on")):
            panels[title] = {}
            for method, label in labels.items():
                scales = self.tuning["parameter_scale"][f"mpgibbs_{method}_{coordinate}"]
                scan = scales["screen"]["scan"]
                panels[title][label] = (
                    np.asarray([row["scale"] for row in scan]),
                    np.asarray([row["label_change_rate"] for row in scan]),
                )
        return panels

    def walk_rows(self, labels: Mapping[str, str], walk: IdealWalk) -> list[dict[str, str | float]]:
        """Each MPGibbs run next to an ideal walk at the candidate scale it used."""
        ideal = dict(zip(walk.scales.tolist(), walk.sweeps.tolist(), strict=True))
        rows: list[dict[str, str | float]] = []
        for name, label in labels.items():
            if not name.startswith("mpgibbs"):
                continue
            scale = float(self.metadata[name]["parameter_scale"])
            sweeps = self.sweeps_per_effective_sample(name)
            rows.append(
                {
                    "Method": label,
                    "Candidate scale": scale,
                    "Ideal walk at that scale": round(ideal[scale], 1),
                    "Measured: level": round(float(sweeps[0]), 1),
                    "Measured: log σ": round(float(sweeps[2]), 1),
                }
            )
        return rows

    def pairing_rows(
        self, smoothers: Mapping[str, str], column: int
    ) -> list[dict[str, str | float]]:
        """NUTS + smoother next to MPGibbs + the same smoother, chart off and on, for one quantity."""
        rows: list[dict[str, str | float]] = []
        for coordinate, chart in (("path", "off"), ("chart", "on")):
            for method, label in smoothers.items():
                nuts, mpgibbs = f"nuts_{method}_{coordinate}", f"mpgibbs_{method}_{coordinate}"
                rows.append(
                    {
                        "Smoother": label,
                        "Chart": chart,
                        "Sweeps: NUTS": round(
                            float(self.sweeps_per_effective_sample(nuts)[column]), 1
                        ),
                        "Sweeps: MPGibbs": round(
                            float(self.sweeps_per_effective_sample(mpgibbs)[column]), 1
                        ),
                        "ms: NUTS": round(self.milliseconds_per_effective_sample(nuts, column), 1),
                        "ms: MPGibbs": round(
                            self.milliseconds_per_effective_sample(mpgibbs, column), 1
                        ),
                    }
                )
        return rows

    def example_rows(
        self, labels: Mapping[str, str], names: Sequence[str]
    ) -> list[dict[str, str | float]]:
        """Sweeps per effective sample of level and log σ, for a few runs of the running example."""
        return [
            {
                "Sampler": labels[name],
                "Sweeps: level": round(float(self.sweeps_per_effective_sample(name)[0]), 1),
                "Sweeps: log σ": round(float(self.sweeps_per_effective_sample(name)[2]), 1),
            }
            for name in names
        ]

    def choice_rows(
        self, labels: Mapping[str, str], changes: Sequence[tuple[str, str]]
    ) -> list[dict[str, str | float]]:
        """One row per (choice changed, run), with sweeps and time per effective sample."""
        return [
            {
                "Choice changed": choice,
                "Sampler": labels[name],
                "Sweeps: level": round(float(self.sweeps_per_effective_sample(name)[0]), 1),
                "Sweeps: log σ": round(float(self.sweeps_per_effective_sample(name)[2]), 1),
                "ms: log σ": round(self.milliseconds_per_effective_sample(name, 2), 1),
            }
            for choice, name in changes
        ]

    def chart_gain_rows(self, smoothers: Mapping[str, str]) -> list[dict[str, str | float]]:
        """How many times fewer sweeps per effective sample each pairing needs with the chart on."""

        def _gain(name: str, column: int) -> str:
            off = self.sweeps_per_effective_sample(f"{name}_path")[column]
            on = self.sweeps_per_effective_sample(f"{name}_chart")[column]
            return f"{off / on:.0f}×"

        return [
            {
                "Path move": label,
                "NUTS: level": _gain(f"nuts_{method}", 0),
                "NUTS: log σ": _gain(f"nuts_{method}", 2),
                "MPGibbs: level": _gain(f"mpgibbs_{method}", 0),
                "MPGibbs: log σ": _gain(f"mpgibbs_{method}", 2),
            }
            for method, label in smoothers.items()
        ]

    def tree_limit_share(self, name: str) -> float:
        """Share of retained sweeps where NUTS hit its limit of 511 leapfrog steps."""
        with np.load(self.directory / f"{name}.npz") as saved:
            return float(np.mean(saved["leapfrog_steps"] >= 511))

    def divergences(self, name: str) -> int:
        with np.load(self.directory / f"{name}.npz") as saved:
            return int(np.sum(saved["divergences"]))

    def leapfrog_steps(self, name: str) -> float | None:
        with np.load(self.directory / f"{name}.npz") as saved:
            if "leapfrog_steps" not in saved.files:
                return None
            return float(saved["leapfrog_steps"].mean())

    def speed_rows(self, labels: Mapping[str, str]) -> list[dict[str, str | float]]:
        rows: list[dict[str, str | float]] = []
        for name, label in labels.items():
            if name not in self.metadata:
                continue
            sweeps = self.sweeps_per_effective_sample(name)
            rows.append(
                {
                    "Method": label,
                    "δ": self.metadata[name].get("delta") or "",
                    "Candidate scale": self.metadata[name].get("parameter_scale") or "",
                    "Level": round(float(sweeps[0]), 1),
                    "log κ": round(float(sweeps[1]), 1),
                    "log σ": round(float(sweeps[2]), 1),
                    "Mean of h": round(float(sweeps[3]), 1),
                    "RMS residual": round(float(sweeps[4]), 1),
                    "Middle-day h": round(float(sweeps[5]), 1),
                    "Retained sweeps": int(self.metadata[name]["config"]["draws"]),
                    "ms per sweep": round(self.milliseconds_per_sweep(name), 2),
                    "ms per effective sample, log σ": round(
                        self.milliseconds_per_effective_sample(name, 2), 1
                    ),
                    "Worst R̂": round(max(self.metadata[name]["diagnostics"]["rhat"]), 3),
                }
            )
        return rows


def load_runs() -> Runs:
    manifest = json.loads((ARTIFACTS / "latest.json").read_text())
    directory = Path(manifest["directory"])
    names = tuple(sorted(manifest["arms"]))
    metadata = {name: json.loads((directory / f"{name}.json").read_text()) for name in names}
    config = manifest["config"]
    reference = json.loads((directory / "reference.json").read_text())
    reference_means = np.asarray(reference["diagnostics"]["mean"])
    reference_errors = np.asarray(reference["diagnostics"]["mcse"])
    with np.load(directory / "pilot.npz") as pilot:
        summaries = pilot["summaries"]
        chart = model.Chart(
            jnp.asarray(pilot["chart_center"]),
            jnp.asarray(pilot["chart_slopes"]),
            jnp.asarray(pilot["chart_exponents"]),
            jnp.asarray(pilot["chart_reference"]),
        )
    flat = summaries.reshape(-1, summaries.shape[-1])
    return Runs(
        directory,
        int(config["days"]),
        int(config["chains"]),
        names,
        metadata,
        reference_means,
        reference_errors,
        int(reference["config"]["draws"]),
        flat[:: max(1, flat.shape[0] // 1500)],
        chart,
        json.loads((directory / "tuning.json").read_text()),
        json.loads((directory / "timing.json").read_text()),
    )


def repeat_run_gaps(runs: Runs, r2: jax.Array, *, seed: int = 151) -> np.ndarray:
    """A separately seeded repeat of the one run with a gap near 3, against the reference."""
    name = "mpgibbs_ak_whole_chart"

    def _compute() -> dict[str, np.ndarray]:
        metadata = runs.metadata[name]
        config = metadata["config"]
        spec = kernels.MPGibbsSpec(
            runs.chart,
            kernels.AuxKalmanSpec(metadata["delta"]),
            metadata["parameter_scale"] * jnp.asarray(runs.tuning["posterior_cholesky"]),
        )
        with np.load(runs.directory / "pilot.npz") as pilot:
            etas, paths = jnp.asarray(pilot["last_eta"]), jnp.asarray(pilot["last_path"])
        keys = jax.random.split(
            jax.random.fold_in(jax.random.key(seed), metadata["fold_in"]), config["chains"]
        )
        run = jax.jit(
            jax.vmap(
                partial(kernels.run_method, spec, warmup=config["warmup"], draws=config["draws"]),
                in_axes=(None, 0, 0, 0),
            )
        )(r2, keys, etas, paths)
        summaries = np.asarray(run.substep_summaries[:, :, -1])
        return {
            "mean": summaries.mean(axis=(0, 1)),
            "mcse": np.asarray(array_stats.mcse(summaries, chain_axis=0, draw_axis=1)),
        }

    arrays = cached_arrays("repeat-run", _compute, (str(runs.directory), r2, seed))
    return (arrays["mean"] - runs.reference_means) / np.hypot(arrays["mcse"], runs.reference_errors)


def markdown_table(rows: Sequence[Mapping[str, str | float]]) -> str:
    """Rows as a static table in the notebook's own table style, with numbers right-aligned."""
    headers = tuple(rows[0])

    def _numeric(value: str | float) -> bool:
        return isinstance(value, int | float)

    def _cell(value: str | float) -> str:
        return f"{value:,}" if _numeric(value) else str(value)

    lines = (
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---:" if _numeric(rows[0][name]) else ":---" for name in headers) + "|",
        *("| " + " | ".join(_cell(row[name]) for name in headers) + " |" for row in rows),
    )
    return "\n".join(lines)


def kernel_source(*names: str) -> str:
    """The running kernels' own source, so text and code can't drift."""
    return "\n\n\n".join(inspect.getsource(getattr(kernels, name)).rstrip() for name in names)


def schedule_source() -> str:
    """The loop that builds every saved run from its choices."""
    return inspect.getsource(sv_runs._arm_specs).rstrip()


class JointTrajectory(NamedTuple):
    """Summaries along one fixed-length HMC trajectory, with a tuned step and metric."""

    summaries: np.ndarray
    steps: int
    acceptance: float


def joint_trajectories(
    runs: Runs,
    r2: jax.Array,
    coordinates: Mapping[str, model.Coordinates],
    *,
    seed: int = 41,
) -> dict[str, JointTrajectory]:
    """One trajectory per coordinate system, as long as a typical NUTS move in that system."""

    def _compute() -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for name, coordinate in coordinates.items():
            with np.load(runs.directory / f"{name}.npz") as saved:
                step = float(np.ravel(saved["nuts_step_size"])[0])
                inverse_mass = jnp.asarray(saved["nuts_inverse_mass"][0])
                eta = jnp.asarray(saved["last_eta"][0])
                path = jnp.asarray(saved["last_path"][0])
                steps = round(float(saved["leapfrog_steps"].mean()))
            arrays.update(_one_trajectory(name, coordinate, eta, path, step, steps, inverse_mass))
        return arrays

    def _one_trajectory(
        name: str,
        coordinate: model.Coordinates,
        eta: jax.Array,
        path: jax.Array,
        step: float,
        steps: int,
        inverse_mass: jax.Array,
    ) -> dict[str, np.ndarray]:
        def _density(position: jax.Array) -> jax.Array:
            return model.held_log_density(coordinate, position[:3], position[3:], r2)

        start = jnp.concatenate((eta, model.stored_path(coordinate, eta, path)))
        trace = kernels.hmc_trace(
            jax.random.key(seed),
            _density,
            start,
            step_size=step,
            steps=steps,
            inverse_mass=inverse_mass,
        )

        def _summary(position: jax.Array) -> jax.Array:
            physical = model.physical_path(coordinate, position[:3], position[3:])
            return kernels.path_summary(position[:3], physical)

        return {
            f"{name}|summaries": np.asarray(jax.vmap(_summary)(trace.positions)),
            f"{name}|acceptance": np.asarray(trace.acceptance_probability),
        }

    with np.load(runs.directory / "pilot.npz") as pilot:
        run_identity = pilot["chart_center"]
    arrays = cached_arrays(
        "joint-trajectories",
        _compute,
        (str(runs.directory), run_identity, r2, dict(coordinates), seed),
        helpers=(joint_trajectories,),
    )
    return {
        name: JointTrajectory(
            arrays[f"{name}|summaries"],
            arrays[f"{name}|summaries"].shape[0] - 1,
            float(arrays[f"{name}|acceptance"]),
        )
        for name in coordinates
    }


def sweep_rows(
    substeps: np.ndarray, sweep: int, *, parameters_first: bool
) -> list[dict[str, str | float]]:
    """One sweep's states, stage by stage, for a small table."""
    stages = substeps[sweep]
    names = (
        ("Start", "After the parameter move", "After the path move")
        if parameters_first
        else ("Start", "After the path move, at the old θ", "After the flip")
    )
    return [
        {
            "State": names[row],
            "Level": round(float(stages[row, 0]), 3),
            "log κ": round(float(stages[row, 1]), 3),
            "log σ": round(float(stages[row, 2]), 3),
            "Mean of h": round(float(stages[row, 3]), 3),
            "RMS residual": round(float(stages[row, 4]), 3),
            "Middle-day h": round(float(stages[row, 5]), 3),
        }
        for row in range(stages.shape[0])
    ]


def predicted_rows(
    widths: Mapping[str, WidthRow], runs: Runs, *, name_off: str, name_on: str
) -> list[dict[str, str | float]]:
    """The scalar-Gaussian rule from the block conditional widths, next to measured mixing."""
    off = widths["Hold the path"].mean
    on = widths["Chart, both"].mean
    measured_off = runs.sweeps_per_effective_sample(name_off)
    measured_on = runs.sweeps_per_effective_sample(name_on)
    return [
        {
            "Parameter": label,
            "Chart off, scalar Gaussian rule": round(2.0 / off[column] ** 2 - 1.0, 1),
            "Chart off, measured": round(float(measured_off[column]), 1),
            "Chart on, scalar Gaussian rule": round(2.0 / on[column] ** 2 - 1.0, 1),
            "Chart on, measured": round(float(measured_on[column]), 1),
        }
        for column, label in ((0, "Level"), (1, "log κ"), (2, "log σ"))
    ]
