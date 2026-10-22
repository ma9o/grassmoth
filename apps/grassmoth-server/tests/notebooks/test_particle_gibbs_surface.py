"""The displayed toy laws and production targets describe the same probability model."""

import jax.numpy as jnp
import numpy as np
import pytest

from notebooks.support.particle_gibbs_plots import LIGHT, smoother_echo, sweep_figure
from notebooks.support.particle_gibbs_surface import (
    BoundedScale,
    Ridge,
    dsmc_sweep,
    log_joint,
    production_target,
    surface,
)

pytestmark = pytest.mark.inference(concern="sampling")


@pytest.mark.parametrize("world", [Ridge(0.15, 0.8), BoundedScale(2.0, prior_sd=2.0)])
def test_display_law_matches_native_target_and_integrated_shadow(world):
    target = production_target(world)
    for theta, latent in ((-2.0, -0.1), (0.3, 0.8), (1.1, -1.2)):
        actual = target.log_posterior(
            jnp.array([theta]), jnp.array([[latent]]), target.observations, target.times
        )
        np.testing.assert_allclose(actual, log_joint(world, theta, latent), rtol=2e-6, atol=2e-6)

    grid = surface(world)
    integrated = np.trapezoid(grid.density, grid.x, axis=0)
    integrated /= np.trapezoid(integrated, grid.theta)
    error = np.trapezoid(np.abs(integrated - grid.theta_marginal), grid.theta)
    assert error < 0.01, "The plotted surface must integrate to its declared parameter shadow"


@pytest.mark.parametrize("world", [Ridge(0.15, 0.8), BoundedScale(2.0, prior_sd=2.0)])
def test_smoother_display_matches_drifted_gaussian_conditional(world):
    mean, variance = world.x_given_theta(0.3)
    delta, kappa = 0.1, 0.75
    sweep = dsmc_sweep(
        world, 0.3, float(mean - np.sqrt(variance)), 7, particles=4, delta=delta, kappa=kappa
    )
    figure = sweep_figure(world, sweep, LIGHT, delta=delta, color=LIGHT.dsmc, x_range=world.x_box())
    conditional = next(trace for trace in figure.data if trace.name.startswith("π(x | z)"))
    grid = np.asarray(conditional.x)
    slope = 1 - kappa * delta / (2 * variance)
    conditional_variance = 1 / (1 / variance + 2 * slope**2 / delta)
    conditional_mean = mean + conditional_variance * 2 * slope / delta * (sweep.aux - mean)
    expected = np.exp(-0.5 * (grid - conditional_mean) ** 2 / conditional_variance)
    expected /= np.trapezoid(expected, grid)
    np.testing.assert_allclose(conditional.y, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        smoother_echo(float(variance), delta, kappa=kappa), 1 - conditional_variance / variance
    )
    assert sweep.center == sweep.aux
