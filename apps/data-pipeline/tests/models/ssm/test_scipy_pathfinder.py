import time
from typing import Any, ClassVar

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import LbfgsInvHessProduct, OptimizeResult

import nof1_causal_lab.models.ssm.inference.warmup.scipy_pathfinder as scipy_pathfinder_module
from nof1_causal_lab.compute.budget import ComputeBudget, SharedMemoryBudget
from nof1_causal_lab.models.ssm.inference.warmup.scipy_pathfinder import scipy_pathfinder

pytestmark = pytest.mark.inference(concern="warmup")


def test_pathfinder_batched_scores_preserve_every_candidate_and_partial_chunk():
    candidates = jnp.arange(38, dtype=jnp.float32).reshape(19, 2) / 10
    observations = jnp.array([0.2, -0.4, 1.2])
    times = jnp.array([0.0, 0.5, 1.8])

    def log_posterior(z, observations, times, solver_state):
        del solver_state
        return -0.5 * jnp.sum((observations - z[0] - z[1] * times) ** 2)

    scores = scipy_pathfinder_module._scipy_pathfinder_value_batch_runtime(
        candidates, observations, times, runtime_log_posterior_fn=log_posterior, batch_size=8
    )
    expected = np.array(
        [
            -0.5 * np.sum((np.asarray(observations) - intercept - slope * np.asarray(times)) ** 2)
            for intercept, slope in np.asarray(candidates)
        ]
    )
    np.testing.assert_allclose(scores, expected, rtol=1e-6)


def test_scipy_pathfinder_uses_accepted_iterates_for_custom_history(monkeypatch):
    call_counts = {"value_batch": 0, "value_and_grad": 0}
    batch_sizes: list[int] = []

    def log_post_batch_fn(x: np.ndarray) -> np.ndarray:
        call_counts["value_batch"] += 1
        x_np = np.asarray(x, dtype=np.float64)
        batch_sizes.append(int(x_np.shape[0]))
        return -0.5 * np.sum(x_np * x_np, axis=1)

    def log_post_and_grad_fn(x: np.ndarray) -> tuple[float, np.ndarray]:
        call_counts["value_and_grad"] += 1
        x_np = np.asarray(x, dtype=np.float64)
        return float(-0.5 * np.dot(x_np, x_np)), -x_np

    def fake_minimize(fun, x0, jac, method, callback=None, options=None):
        del jac, options
        assert method == "L-BFGS-B"
        assert callback is not None
        x0_np = np.asarray(x0, dtype=np.float64)
        accepted_1 = np.array([0.5], dtype=np.float64)
        accepted_2 = np.array([0.1], dtype=np.float64)

        fun(x0_np)
        fun(np.array([0.75], dtype=np.float64))
        fun(accepted_1)
        callback(accepted_1)
        fun(np.array([0.2], dtype=np.float64))
        final_fun, _final_grad = fun(accepted_2)
        callback(accepted_2)
        return OptimizeResult(
            x=accepted_2,
            fun=float(final_fun),
            nit=2,
            nfev=5,
            status=0,
            success=True,
            hess_inv=LbfgsInvHessProduct(np.zeros((0, 1)), np.zeros((0, 1))),
        )

    monkeypatch.setattr(scipy_pathfinder_module.scipy.optimize, "minimize", fake_minimize)

    result = scipy_pathfinder(
        log_post_batch_fn,
        log_post_and_grad_fn,
        [np.array([1.0], dtype=np.float64)],
        maxiter=5,
        elbo_samples=8,
        elbo_candidate_batch_size=4,
        parallel_workers=1,
        seed=0,
        clock=time.monotonic,
    )

    per_start = result.diagnostics.per_start[0]
    assert per_start.n_trajectory_points == 3
    assert per_start.n_lbfgs_iterations == 2
    assert per_start.n_valid_iterates == 2
    assert per_start.n_elbo_candidates == 3
    assert per_start.n_elbo_batch_evaluations == 1
    assert call_counts["value_batch"] == 1
    assert batch_sizes == [32]
    assert call_counts["value_and_grad"] > call_counts["value_batch"]
    assert np.isfinite(result.best_elbo)


def test_scipy_pathfinder_submits_multistarts_to_thread_pool(monkeypatch):
    class _ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _RecordingExecutor:
        instances: ClassVar[list[Any]] = []

        def __init__(self, max_workers):
            self.max_workers = max_workers
            self.submissions = []
            self.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            self.submissions.append((fn, args, kwargs))
            return _ImmediateFuture(fn(*args, **kwargs))

    def log_post_and_grad_fn(x: np.ndarray) -> tuple[float, np.ndarray]:
        x_np = np.asarray(x, dtype=np.float64)
        return float(-0.5 * np.dot(x_np, x_np)), -x_np

    def log_post_batch_fn(x: np.ndarray) -> np.ndarray:
        x_np = np.asarray(x, dtype=np.float64)
        return -0.5 * np.sum(x_np * x_np, axis=1)

    def fake_minimize(fun, x0, jac, method, callback=None, options=None):
        del jac, options
        assert method == "L-BFGS-B"
        assert callback is not None
        accepted = np.asarray(x0, dtype=np.float64) * 0.5
        final_fun, _final_grad = fun(accepted)
        callback(accepted)
        return OptimizeResult(
            x=accepted,
            fun=float(final_fun),
            nit=1,
            nfev=1,
            status=0,
            success=True,
            hess_inv=LbfgsInvHessProduct(np.zeros((0, 1)), np.zeros((0, 1))),
        )

    monkeypatch.setattr(scipy_pathfinder_module, "ThreadPoolExecutor", _RecordingExecutor)
    monkeypatch.setattr(scipy_pathfinder_module.scipy.optimize, "minimize", fake_minimize)

    result = scipy_pathfinder(
        log_post_batch_fn,
        log_post_and_grad_fn,
        [
            np.array([1.0], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
            np.array([3.0], dtype=np.float64),
        ],
        maxiter=2,
        elbo_samples=4,
        elbo_candidate_batch_size=2,
        parallel_workers=2,
        seed=0,
        clock=time.monotonic,
    )

    executor = _RecordingExecutor.instances[0]
    assert executor.max_workers == 2
    assert len(executor.submissions) == 3
    assert result.diagnostics.parallel_workers == 2
    assert [item.start_idx for item in result.diagnostics.per_start] == [0, 1, 2]


def test_scipy_pathfinder_finds_valid_custom_iterates_on_gaussian_target():
    precision = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)

    def log_post_batch_fn(x: np.ndarray) -> np.ndarray:
        x_np = np.asarray(x, dtype=np.float64)
        return -0.5 * np.einsum("bi,ij,bj->b", x_np, precision, x_np)

    def log_post_and_grad_fn(x: np.ndarray) -> tuple[float, np.ndarray]:
        x_np = np.asarray(x, dtype=np.float64)
        return float(-0.5 * x_np @ precision @ x_np), -(precision @ x_np)

    result = scipy_pathfinder(
        log_post_batch_fn,
        log_post_and_grad_fn,
        [np.array([3.0, -2.0], dtype=np.float64)],
        maxiter=10,
        elbo_samples=20,
        elbo_candidate_batch_size=4,
        parallel_workers=1,
        seed=0,
        clock=time.monotonic,
    )

    per_start = result.diagnostics.per_start[0]
    assert per_start.n_valid_iterates > 0
    np.testing.assert_allclose(result.mean, np.zeros((2,), dtype=np.float64), atol=1e-3)


def test_candidate_batch_width_preserves_scores_and_the_next_random_stream():
    means = [np.array([float(index), 0.5]) for index in range(5)]
    chols = [np.eye(2) for _ in means]

    def score(values):
        return -np.sum(values**2, axis=1)

    results = []
    for width in (1, 4, 8):
        rng = np.random.default_rng(27)
        elbos, _ = scipy_pathfinder_module._estimate_elbo_candidate_batch(
            score, rng, means, chols, elbo_samples=6, dim=2, candidate_batch_size=width
        )
        results.append((elbos, rng.standard_normal(10)))
    for elbos, following in results[1:]:
        np.testing.assert_array_equal(elbos, results[0][0])
        np.testing.assert_array_equal(following, results[0][1])


@pytest.mark.parametrize(
    ("memory_bytes", "expected_workers", "expected_candidates"),
    [(16 * 1024**2, 2, 4), (2000, 1, 1)],
)
def test_compiled_pathfinder_programs_honor_concurrency_and_both_elbo_shapes(
    memory_bytes, expected_workers, expected_candidates
):
    budget = ComputeBudget(cpu_workers=2, memory=SharedMemoryBudget(available_bytes=memory_bytes))

    def objective(z, observations, times, solver_state):
        del observations, times, solver_state
        return -jnp.sum(z**2)

    observations, times = jnp.zeros((2, 1)), jnp.arange(2.0)
    execution = scipy_pathfinder_module._compile_pathfinder_execution(
        objective,
        jnp.zeros(2),
        observations,
        times,
        compute_budget=budget,
        n_starts=8,
        requested_workers=3,
        maxiter=2,
        elbo_samples=5,
        screen_samples=2,
    )
    assert execution.parallel_workers == expected_workers
    assert execution.candidate_batch_size == expected_candidates
    assert tuple(count for count, _ in execution.values) == (
        2 * expected_candidates,
        5 * expected_candidates,
    )
    value, gradient = execution.gradient(jnp.ones(2), observations, times)
    assert float(value) == -2
    np.testing.assert_array_equal(gradient, [-2, -2])
    for count, program in execution.values:
        values = program.executable(jnp.ones((count, 2)), observations, times)
        np.testing.assert_array_equal(values, np.full(count, -2))
