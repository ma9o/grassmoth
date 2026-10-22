"""Prior display is an ephemeral read of a native law, never authored science."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from pydantic import TypeAdapter

from grassmoth.artifacts.probability.native import (
    interval_effect_to_rate,
    persistence_to_decay,
)
from grassmoth.artifacts.probability.serialization import NumPyroDistribution
from grassmoth.study.projections.priors import prior_density, quantity_prior_densities
from tests.support.model import construct_named, load_model_fixture


def _quantity_curves_put_authored_laws_on_the_posterior_scale_complete_test_model() -> (
    DynamicalModelSpec
):
    dynamical_model_spec = load_model_fixture(
        "snapshots/fitted_snapshot_keeps_joint_arrays_lazy_and_workspace_bound_complete_test_model.json"
    )
    x = construct_named(dynamical_model_spec, "X")
    y = construct_named(dynamical_model_spec, "Y")
    x_to_y = next(
        edge
        for edge in dynamical_model_spec.edges
        if edge.cause.id == x.id and edge.effect.id == y.id
    )
    return dynamical_model_spec.with_entities(edges=(x_to_y.revised(description="X"),))


if TYPE_CHECKING:
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec


pytestmark = [
    pytest.mark.inference(concern="sampling"),
    pytest.mark.inference(concern="predictive"),
]


def test_prior_curves_preserve_native_gamma_and_transforms_without_mutating_the_law():
    prior = dist.Gamma(2.0, 3.0)
    adapter = TypeAdapter(NumPyroDistribution)
    before = adapter.dump_json(prior)
    for law, rate in ((prior, 3.0), (interval_effect_to_rate(prior, 2.0), 6.0)):
        curve = prior_density(law)
        assert len(curve.x) == 100
        assert all(left < right for left, right in pairwise(curve.x))
        np.testing.assert_allclose(
            curve.density,
            [rate**2 * x * math.exp(-rate * x) for x in curve.x],
            rtol=2e-6,
        )
        assert prior_density(law) == curve
    assert adapter.dump_json(prior) == before
    for law in (
        dist.Delta(1.0),
        dist.Bernoulli(0.5),
        dist.Normal(jnp.zeros(2), 1.0),
        dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2)),
    ):
        assert not prior_density(law).x


def test_quantity_curves_put_authored_laws_on_the_posterior_scale():
    dynamical_model_spec = (
        _quantity_curves_put_authored_laws_on_the_posterior_scale_complete_test_model()
    )
    curves = quantity_prior_densities(
        dynamical_model_spec,
        frozenset(parameter.id for parameter in dynamical_model_spec.parameters),
    )
    transforms = {parameter.transform.kind for parameter in dynamical_model_spec.parameters}
    assert "dt_persistence_to_ct_decay" in transforms
    for parameter in dynamical_model_spec.parameters:
        law = dynamical_model_spec.distribution_for(parameter.id)
        match parameter.transform.kind:
            case "dt_persistence_to_ct_decay":
                # Posteriors report decay rates, so persistence priors move to that axis.
                native = persistence_to_decay(law, 1.0)
            case "dt_effect_to_ct_rate":
                native = interval_effect_to_rate(law, 1.0)
            case "identity" if law is not None:
                native = law
            case _:
                continue
        np.testing.assert_allclose(
            tuple(zip(curves[parameter.id].x, curves[parameter.id].density, strict=True)),
            tuple(zip(prior_density(native).x, prior_density(native).density, strict=True)),
            rtol=1e-6,
            err_msg=parameter.name,
        )
