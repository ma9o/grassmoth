"""Independent dense controls for positive path curvature and singular recurrences."""

from dataclasses import replace

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.models.ssm.inference.parameter_transform import ParameterTransform
from grassmoth.models.ssm.inference.warmup.computed_chart import computed_chart
from grassmoth_test_support.interval_particle import (
    computed_translation_slopes,
    gaussian_interval_case,
)
from packages.grassmoth.tests.models.ssm.inference.test_guided_recurrences import WINDOWS, _target
from packages.grassmoth.tests.models.ssm.inference.test_joint_coordinates import _correlated_target
from packages.grassmoth.tests.models.ssm.inference.test_translation_chart import _gaussian_joint

pytestmark = pytest.mark.inference(concern="warmup")


def test_fixed_parameter_target_has_identity_translation_and_no_scale_exponents():
    base, _, _ = _gaussian_joint()
    position = jnp.empty((0,), dtype=jnp.float32)
    target = replace(
        base,
        parameters=ParameterTransform(
            position, lambda _: {}, lambda _: {}, lambda _: jnp.asarray(0.0)
        ),
        context=lambda _, times: base.context(jnp.zeros(1), times),
    )
    path = jnp.array([[0.2], [-0.1], [0.4]])
    chart = jax.jit(lambda: computed_chart(target, position[None], path[None]))()
    slopes = jax.jit(lambda: computed_translation_slopes(target, position, path))()
    exponents = jax.jit(target.diffusion_scale_exponents)(position)
    assert slopes.shape == (3, 1, 0)
    assert exponents.shape == (1, 0)
    np.testing.assert_array_equal(chart.encode(position, path), path)
    np.testing.assert_array_equal(chart.decode(position, path), path)
    translated = target.in_coordinates(chart)
    np.testing.assert_allclose(
        translated.log_posterior(position, path, target.observations, target.times),
        target.log_posterior(position, path, target.observations, target.times),
    )


@pytest.mark.parametrize("nonlinear", [False, True])
def test_point_gauss_newton_slopes_match_dense_positive_curvature(nonlinear):
    base, _, _ = _gaussian_joint()
    path = jnp.array([[0.2], [-0.1], [0.4]])
    position = jnp.array([0.3])

    def mean(x):
        return 0.7 * x - nonlinear * 0.2 * jnp.tanh(x) ** 3

    def model(context):
        return eqx.tree_at(
            lambda value: value.state_evolution,
            base.dynamical_model(context),
            dsx.GaussianStateEvolution(lambda x, *_: mean(x), lambda *_: jnp.array([[0.3]])),
        )

    target = replace(base, dynamical_model=model)
    actual = jax.jit(lambda p, x: computed_translation_slopes(target, p, x))(position, path)
    residual_jacobian = np.eye(3)
    residual_jacobian[1, 0] = -float(jax.grad(lambda x: mean(x))(path[0, 0]))
    residual_jacobian[2, 1] = -float(jax.grad(lambda x: mean(x))(path[1, 0]))
    precision = (
        residual_jacobian.T @ np.diag([1.0, 1 / 0.3, 1 / 0.3]) @ residual_jacobian + np.eye(3) / 0.2
    )
    expected = -np.linalg.solve(precision, np.ones((3, 1)) / 0.2)
    np.testing.assert_allclose(actual[:, 0], expected, rtol=3e-6, atol=1e-6)
    assert np.linalg.eigvalsh(precision).min() > 0


def test_overlapping_intervals_match_dense_joint_gaussian_curvature():
    target, _, _ = gaussian_interval_case(steps=9)
    position = jnp.array([0.3])
    path = jnp.linspace(-0.2, 0.4, 9)[:, None]
    actual = jax.jit(lambda p, x: computed_translation_slopes(target, p, x))(position, path)
    precision = np.asarray(
        jax.hessian(
            lambda joint: (
                -target.log_posterior(joint[:1], joint[1:, None], target.observations, target.times)
            )
        )(jnp.concatenate((position, path[:, 0])))
    )
    expected = -np.linalg.solve(precision[1:, 1:], precision[1:, :1])
    np.testing.assert_allclose(actual[:, 0], expected, rtol=3e-5, atol=3e-6)
    assert np.linalg.norm(actual) > 0.1


def test_deterministic_stages_and_interval_readings_contribute_to_translation():
    target = _target(False)
    position = jnp.array([0.15])
    path = jnp.array([[0.2, 0.03, -0.015], [0.1, 0.1, 0.0], [-0.1, 0.12, 0.03], [0.15, 0.08, 0.05]])
    actual = jax.jit(lambda p, x: computed_translation_slopes(target, p, x))(position, path)
    matrix = np.array([[0.65, 0.0, 0.25], [0.4 + 0.2 * 0.15, 0.8, 0.0], [0.0, 0.3, 0.5]])
    noise_map = np.zeros((12, 4))
    prior_slope = np.zeros((4, 3))
    prior_slope[0] = [0.2, 0.2, -0.1]
    noise_map[:3, 0] = [np.sqrt(0.65), 0.0, 0.0]
    for t in range(1, 4):
        noise_map[3 * t : 3 * t + 3] = matrix @ noise_map[3 * t - 3 : 3 * t]
        noise_map[3 * t, t] = np.sqrt(0.3)
        prior_slope[t] = matrix @ prior_slope[t - 1] + [0.0, 0.2 * path[t - 1, 0], 0.0]
    covariance = noise_map @ noise_map.T
    rows = []
    for t in range(4):
        for channel in range(3):
            if not np.isfinite(target.observations[t, channel]):
                continue
            row = np.zeros((4, 3))
            if channel == 2:
                row[:, 2] = WINDOWS[t]
            else:
                row[t, 0 if channel == 0 else 2] = 1.0
            rows.append(row.ravel())
    design = np.stack(rows)
    expected = prior_slope.ravel() - covariance @ design.T @ np.linalg.solve(
        design @ covariance @ design.T + 0.45 * np.eye(len(rows)), design @ prior_slope.ravel()
    )
    np.testing.assert_allclose(actual[..., 0], expected.reshape(4, 3), rtol=3e-5, atol=3e-6)
    assert np.linalg.norm(actual[:, 1:]) > 0.1
    point_only = replace(
        target, intervals=None, observations=target.observations.at[:, 2].set(jnp.nan)
    )
    point_slopes = jax.jit(lambda: computed_translation_slopes(point_only, position, path))()
    assert np.linalg.norm(actual - point_slopes) > 0.001


def test_covariance_mean_response_matches_dense_gaussian_with_pins_and_windows():
    base = _correlated_target()
    position = base.initial_position
    context = base.context(position, base.times)
    initial = base.dynamical_model(context).initial_condition
    # The declared chart surrogate freezes the initial covariance, while transition
    # variances AND correlations vary. This checks regression and pin precision.
    target = replace(
        base,
        dynamical_model=lambda ctx: eqx.tree_at(
            lambda model: model.initial_condition, base.dynamical_model(ctx), initial
        ),
    )
    path = target.initial_path(context, exact_constraints=target.exact_constraints)
    free = np.asarray(target.independent_mask & target.exact_constraints.free_mask)
    indices = jnp.asarray(np.flatnonzero(free.ravel()))
    origin = jnp.zeros(len(indices), dtype=path.dtype)

    def dense_mean(parameter):
        def density(values):
            physical = path.ravel().at[indices].set(values).reshape(path.shape)
            return target.log_posterior(parameter, physical, target.observations, target.times)

        information = -jax.hessian(density)(origin)
        return jnp.linalg.solve(information, jax.grad(density)(origin))

    actual = jax.jit(lambda p, x: computed_translation_slopes(target, p, x))(position, path)
    expected = jax.jit(jax.jacfwd(dense_mean))(position)
    np.testing.assert_allclose(np.asarray(actual)[free], expected, rtol=3e-4, atol=3e-6)
    assert np.linalg.norm(actual) > 0.05
    control = _saved_scale_gate(target, position, path, np.eye(1))
    np.testing.assert_allclose(control["mean_response_norm"], np.linalg.norm(expected), rtol=3e-4)
    np.testing.assert_array_equal(np.asarray(actual)[~free], 0.0)
    chart = jax.jit(lambda p, x: computed_chart(target, p[None], x[None]))(position, path)
    np.testing.assert_allclose(
        np.asarray(chart.center)[free], dense_mean(position), rtol=3e-5, atol=3e-6
    )
    np.testing.assert_array_equal(np.asarray(chart.exponents)[~free], 0.0)


def _compare_scale_rules(precision, derivatives, structural, shift, shift_derivatives, metric):
    """CPU dense control of diagonal Gaussian transports, including full mean response."""
    from time import perf_counter

    from scipy.linalg import cho_factor, cho_solve

    started = perf_counter()
    factor = cho_factor(precision)
    covariance = cho_solve(factor, np.eye(len(precision)))
    covariance_seconds = perf_counter() - started
    mean = cho_solve(factor, shift)
    slopes = cho_solve(
        factor,
        np.column_stack(
            [db - dh @ mean for db, dh in zip(shift_derivatives, derivatives, strict=True)]
        ),
    )
    products = tuple(np.asarray(dh @ covariance).T for dh in derivatives)
    scores = np.column_stack([np.diag(product) for product in products])
    constant = np.array([[0.5 * np.sum(left * right.T) for right in products] for left in products])
    started = perf_counter()
    system = covariance * precision + np.eye(len(precision))
    optimum = cho_solve(cho_factor(system), -scores)
    optimum_seconds = covariance_seconds + perf_counter() - started
    started = perf_counter()
    marginal = np.column_stack(
        [-0.5 * np.sum(product * covariance, axis=1) / np.diag(covariance) for product in products]
    )
    marginal_seconds = covariance_seconds + perf_counter() - started
    started = perf_counter()
    conditional = np.column_stack(
        [-0.5 * dh.diagonal() / precision.diagonal() for dh in derivatives]
    )
    conditional_seconds = perf_counter() - started
    candidates = {
        "unscaled_full_mean": (np.zeros_like(structural), 0.0),
        "structural": (structural, 0.0),
        "marginal_sd": (marginal, marginal_seconds),
        "conditional_precision": (conditional, conditional_seconds),
        "optimum": (optimum, optimum_seconds),
    }
    rows = {}
    for name, (exponents, seconds) in candidates.items():
        cross = exponents.T @ scores
        residual = constant + cross + cross.T + exponents.T @ system @ exponents
        residual = (residual + residual.T) / 2
        projected = metric.T @ residual @ metric
        rows[name] = {
            "residual": residual.tolist(),
            "proposal_residual": projected.tolist(),
            "proposal_max_eigenvalue": float(np.linalg.eigvalsh(projected)[-1]),
            "coefficient_seconds": seconds,
            "exponent_range": [float(exponents.min()), float(exponents.max())],
        }
    mean_score = slopes.T @ precision @ slopes
    today = constant + mean_score
    projected_today = metric.T @ today @ metric
    rows["translation_only"] = {
        "residual": today.tolist(),
        "proposal_residual": projected_today.tolist(),
        "proposal_max_eigenvalue": float(np.linalg.eigvalsh(projected_today)[-1]),
        "coefficient_seconds": 0.0,
        "exponent_range": [0.0, 0.0],
        "omitted_mean_score": mean_score.tolist(),
    }
    best = np.diag(rows["optimum"]["residual"])
    for row in rows.values():
        row["excess"] = (np.diag(row["residual"]) - best).tolist()
    return {
        "dimension": len(precision),
        "mean_response_norm": float(np.linalg.norm(slopes)),
        "rules": rows,
    }


def _gaussian_scale_gate():
    """Independent Gaussian cases for the pre-integration scale-rule decision."""
    from scipy.sparse import csr_array

    rows = {}
    steps = 32
    difference = np.eye(steps) - np.eye(steps, k=-1)
    transition = difference[1:].T @ difference[1:]
    initial = np.diag(np.r_[1.0, np.zeros(steps - 1)])
    point = np.eye(steps)[[0, 8, 16, 24, 31]]
    windows = np.stack((np.r_[np.ones(22), np.zeros(10)], np.r_[np.zeros(10), np.ones(22)]))
    cases = (
        ("diffuse_walk", 1e6, np.zeros((0, steps)), 1.0),
        ("tight_initial", 1e-3, np.zeros((0, steps)), 1.0),
        ("ordinary_initial", 1.0, np.zeros((0, steps)), 1.0),
        ("weak_points", 1.0, point, 0.1),
        ("points", 1.0, point, 10.0),
        ("precise_points", 1.0, point, 1e5),
        ("precise_sum", 1e-8, np.ones((1, steps)), 1e5),
        ("overlap_weak", 1.0, windows, 1e-4),
        ("overlap", 1.0, windows, 1.0),
        ("overlap_precise", 1.0, windows, 1e5),
    )
    for name, variance, readings, strength in cases:
        information = strength * readings.T @ readings
        selected = np.arange(1 if name == "precise_sum" else 0, steps)
        rows[name] = _compare_scale_rules(
            (transition + initial / variance + information)[np.ix_(selected, selected)],
            (csr_array(-2 * transition[np.ix_(selected, selected)]),),
            np.ones((len(selected), 1)),
            (strength * readings.T @ np.linspace(-1, 1, len(readings)))[selected],
            (np.zeros(len(selected)),),
            np.eye(1),
        )
    steps = 16
    dynamics = np.array([[0.95, 0.2], [0.1, 0.9]])
    residuals = np.eye(2 * steps) - np.kron(np.eye(steps, k=-1), dynamics)
    contributions = tuple(residuals[2 + d :: 2].T @ residuals[2 + d :: 2] for d in range(2))
    readings = np.eye(2 * steps)[::5]
    precision = sum(contributions) + np.diag(np.r_[1.0, 1.0, np.zeros(2 * steps - 2)])
    precision = precision + 4 * readings.T @ readings
    rows["coupled_two_scales"] = _compare_scale_rules(
        precision,
        tuple(csr_array(-2 * value) for value in contributions),
        np.tile(np.eye(2), (steps, 1)),
        readings.T @ np.linspace(-2, 2, len(readings)),
        (np.zeros(2 * steps), np.zeros(2 * steps)),
        np.array([[0.2, 0.0], [0.07, 0.1]]),
    )
    return rows


def test_scale_gate_optimum_and_precise_window_counterexample():
    cases = _gaussian_scale_gate()
    for case in cases.values():
        for row in case["rules"].values():
            assert min(row["excess"]) > -1e-5
            assert np.linalg.eigvalsh(row["residual"]).min() > -1e-5
    precise = cases["precise_sum"]["rules"]
    assert (
        precise["conditional_precision"]["residual"][0][0]
        > 25 * precise["structural"]["residual"][0][0]
    )
    diffuse = cases["diffuse_walk"]["rules"]
    assert diffuse["marginal_sd"]["residual"][0][0] > 20 * diffuse["structural"]["residual"][0][0]


def _saved_scale_gate(target, position, path, parameter_chol):
    """Pull the library's frozen augmented curvature onto its independent path coordinates."""
    from time import perf_counter

    from scipy.sparse import block_diag, csr_array

    from grassmoth.models.ssm.inference.warmup.path_curvature import linearize_path

    started = perf_counter()
    context = target.context(position, target.times)
    path = target.replay_path(context, path)
    curvature = linearize_path(target, context, path)
    free = np.asarray(target.independent_mask)
    if target.exact_constraints is not None:
        free = free & np.asarray(target.exact_constraints.free_mask)
    steps, dimension = free.shape
    active = np.flatnonzero(free.ravel())
    lookup = {int(flat): index for index, flat in enumerate(active)}
    structural = np.asarray(target.diffusion_scale_exponents(position))
    directions = np.flatnonzero(np.any(structural != 0, axis=0))
    assert len(directions) > 0, directions
    structural = np.broadcast_to(structural, (steps, *structural.shape)).reshape(-1, len(position))[
        active
    ][:, directions]

    def covariance_at(value):
        ctx = target.context(value, target.times)
        return jax.vmap(lambda previous, t: target.transition_moments(ctx, previous, t)[1])(
            path[:-1], jnp.arange(1, steps)
        )

    covariances = np.asarray(covariance_at(position), dtype=float)
    changes = np.asarray(jax.jacfwd(covariance_at)(position), dtype=float)[..., directions]
    initial_mean, initial_covariance = target.initial_moments(context)
    tail_means = jax.vmap(lambda previous, t: target.transition_moments(context, previous, t)[0])(
        path[:-1], jnp.arange(1, steps)
    )
    augmented = np.concatenate((np.asarray(path), np.asarray(curvature.memory)), axis=-1)
    matrices = np.asarray(curvature.matrices, dtype=float)
    prior_offsets = np.concatenate((np.asarray(initial_mean)[None], np.asarray(tail_means)))
    # The reference augmented trajectory supplies affine offsets for deterministic memory.
    offsets = augmented - np.einsum(
        "tij,tj->ti", matrices, np.concatenate((np.zeros_like(augmented[:1]), augmented[:-1]))
    )
    offsets[:, :dimension] = prior_offsets - np.einsum(
        "tij,tj->ti",
        matrices[:, :dimension],
        np.concatenate((np.zeros_like(augmented[:1]), augmented[:-1])),
    )
    size = augmented.shape[1]
    previous_map = np.zeros((size, len(active)))
    previous_offset = np.zeros(size)
    residual_rows, residual_offsets, weights = [], [], []
    weight_changes = [[] for _ in directions]
    reading_rows = []
    reading_shift = np.zeros(len(active))
    for t in range(steps):
        mapping = matrices[t] @ previous_map
        offset = matrices[t] @ previous_offset + offsets[t]
        fixed = np.asarray(curvature.fixed[t])
        mapping[fixed] = 0
        offset[fixed] = augmented[t, fixed]
        for d in np.flatnonzero(free[t]):
            mapping[d] = 0
            mapping[d, lookup[t * dimension + d]] = 1
            offset[d] = 0
        coordinates = np.asarray(
            target.state_space.initial if t == 0 else target.state_space.transition
        )
        q = np.asarray(initial_covariance, dtype=float) if t == 0 else covariances[t - 1]
        w = np.linalg.inv(q[np.ix_(coordinates, coordinates)])
        jacobian = (mapping - matrices[t] @ previous_map)[coordinates]
        residual = (offset - matrices[t] @ previous_offset - offsets[t])[coordinates]
        residual_rows.append(csr_array(jacobian))
        residual_offsets.append(residual)
        weights.append(w)
        for j in range(len(directions)):
            dq = np.zeros_like(q) if t == 0 else changes[t - 1, :, :, j]
            weight_changes[j].append(-w @ dq[np.ix_(coordinates, coordinates)] @ w)
        jacobian = np.asarray(curvature.reading_jacobian[t], dtype=float)
        precision = np.asarray(curvature.reading_precision[t], dtype=float)
        values, vectors = np.linalg.eigh(precision)
        positive = values > 0
        if np.any(positive):
            root = np.sqrt(values[positive])[:, None] * vectors[:, positive].T
            reading_rows.append(root @ jacobian @ mapping)
        reading_shift += (
            mapping.T
            @ jacobian.T
            @ (
                np.asarray(curvature.reading_gradient[t])
                + precision @ jacobian @ (augmented[t] - offset)
            )
        )
        previous_map, previous_offset = mapping, offset
    from scipy.sparse import vstack

    residual = vstack(residual_rows, format="csr")
    offset = np.concatenate(residual_offsets)
    weight = block_diag(weights, format="csr")
    derivatives = tuple(
        (residual.T @ block_diag(dw, format="csr") @ residual).tocsr() for dw in weight_changes
    )
    shift_derivatives = tuple(
        -residual.T @ (block_diag(dw, format="csr") @ offset) for dw in weight_changes
    )
    reading = np.concatenate(reading_rows)
    precision = (residual.T @ weight @ residual).toarray() + reading.T @ reading
    shift = -residual.T @ (weight @ offset) + reading_shift
    from scipy.linalg import cho_factor, cho_solve

    dense_mean = cho_solve(cho_factor(precision), shift)
    smoother_mean = np.asarray(curvature.mean(target, context, path)).ravel()[active]
    mean_error = float(np.max(np.abs(dense_mean - smoother_mean)))
    np.testing.assert_allclose(dense_mean, smoother_mean, rtol=2e-4, atol=2e-4)
    proposal = np.asarray(parameter_chol) @ np.asarray(parameter_chol).T
    metric = np.linalg.cholesky(proposal[np.ix_(directions, directions)])
    assembly_seconds = perf_counter() - started
    result = _compare_scale_rules(
        precision, derivatives, structural, shift, shift_derivatives, metric
    )
    result.update(
        directions=directions.tolist(),
        assembly_seconds=assembly_seconds,
        proposal_covariance=proposal[np.ix_(directions, directions)].tolist(),
        mean_control_error=mean_error,
    )
    return result
