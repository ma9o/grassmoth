"""The persistence boundary preserves native laws, dimensions, and scientific evidence."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from pydantic import TypeAdapter, ValidationError

from grassmoth.artifacts.identity import DistributionId
from grassmoth.artifacts.model.parameter_spec import PersistenceTransformSpec
from grassmoth.artifacts.probability.native import persistence_to_decay
from grassmoth.artifacts.probability.serialization import NumPyroDistribution
from grassmoth.models.parameters import require_priors
from grassmoth.models.structure import StructuralSelection
from grassmoth_test_support.model import (
    parameter_laws,
    parameter_named,
    replace_parameters,
    x_model,
)


def _parameter_changes_distribution_without_keeping_authoring_history_model_with_prior_payloads() -> (
    DynamicalModelSpec
):
    dynamical_model_spec = x_model()
    rho_x = parameter_named(dynamical_model_spec, "rho_X")
    return dynamical_model_spec.with_entities(
        parameters=replace_parameters(
            dynamical_model_spec.parameters,
            rho_x.revised(transform=rho_x.transform.revised(interval_days=7.0)),
        ),
        distributions=parameter_laws(
            dynamical_model_spec,
            {
                rho_x.id: dist.Normal(
                    loc=jnp.array(0.4000000059604645, dtype=jnp.float32),
                    scale=jnp.array(0.20000000298023224, dtype=jnp.float32),
                    validate_args=True,
                )
            },
        ),
    )


if TYPE_CHECKING:
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec


_ADAPTER = TypeAdapter(NumPyroDistribution)


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize(
    ("law", "value"),
    [
        (dist.Normal(0.2, 0.7), 0.3),
        (dist.LogNormal(0.1, 0.4), 0.6),
        (dist.TruncatedNormal(0.0, 1.0, low=-0.2, high=1.0), 0.3),
        (dist.Normal(0.0, 1.0).expand((2, 3)), jnp.zeros((2, 3))),
        (dist.Normal(jnp.zeros(2), jnp.ones(2)).to_event(1), jnp.array([0.2, 0.4])),
        (
            dist.MultivariateNormal(
                jnp.zeros(2), covariance_matrix=jnp.array([[1.0, 0.2], [0.2, 2.0]])
            ),
            jnp.array([0.2, 0.4]),
        ),
        (
            dist.MixtureGeneral(
                dist.Categorical(probs=jnp.array([0.3, 0.7])),
                [dist.Normal(-1.0, 0.5), dist.StudentT(4.0, 1.0, 0.8)],
            ),
            0.3,
        ),
        (persistence_to_decay(dist.Beta(2.0, 3.0), 7.0), 0.2),
    ],
)
def test_native_json_roundtrip_preserves_law_and_dimensions(law, value):
    encoded = _ADAPTER.dump_json(law)
    restored = _ADAPTER.validate_json(encoded)
    assert type(restored) is type(law)
    assert restored.batch_shape == law.batch_shape
    assert restored.event_shape == law.event_shape
    np.testing.assert_allclose(restored.log_prob(value), law.log_prob(value), atol=2e-6)
    assert json.loads(_ADAPTER.dump_json(restored)) == json.loads(encoded)


@pytest.mark.contract
@pytest.mark.parametrize(
    "payload",
    [
        {"distribution": "NotANumPyroDistribution", "params": {}},
        {"distribution": "Normal", "params": {"loc": 0.0, "scale": -1.0}},
        {
            "distribution": "Normal",
            "params": {"loc": 0.0, "scale": -1.0, "validate_args": False},
        },
        {"distribution": "Normal", "params": {"loc": 0.0, "scale": 1.0, "unknown": 2.0}},
    ],
)
def test_invalid_native_constructors_are_rejected(payload):
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python(payload)


@pytest.mark.contract
def test_parameter_changes_distribution_without_keeping_authoring_history():
    dynamical_model_spec = x_model()
    parameter = dynamical_model_spec.parameters[0]
    specified = _parameter_changes_distribution_without_keeping_authoring_history_model_with_prior_payloads()
    restored = type(dynamical_model_spec).model_validate_json(specified.model_dump_json())
    assert restored == specified
    assert restored.parameter(parameter.id).id == parameter.id
    assert isinstance(restored.distribution_for(parameter.id), dist.Normal)
    transform = restored.parameter(parameter.id).transform
    assert isinstance(transform, PersistenceTransformSpec)
    assert transform.interval_days == 7.0
    distribution = restored.parameter(parameter.id).distribution
    assert distribution is not None
    revised = restored.with_entities(
        distributions={
            **restored.distributions,
            distribution: dist.Normal(0.3, 0.1),
        }
    )
    assert (
        revised.parameter(parameter.id).distribution
        == restored.parameter(parameter.id).distribution
    )
    before_law = restored.distribution_for(parameter.id)
    after_law = revised.distribution_for(parameter.id)
    assert isinstance(before_law, dist.Normal)
    assert isinstance(after_law, dist.Normal)
    assert float(before_law.loc) == pytest.approx(0.4)
    assert float(after_law.loc) == pytest.approx(0.3)
    with pytest.raises(ValidationError):
        parameter.revised(distribution=dist.Normal(0, 1))


@pytest.mark.contract
def test_parameter_tool_boundary_validates_the_reference_interval():
    dynamical_model_spec = x_model()
    with pytest.raises(ValidationError):
        dynamical_model_spec.parameters[0].revised(
            transform={"kind": "dt_persistence_to_ct_decay", "interval_days": -7.0},
        )


@pytest.mark.contract
def test_completed_model_requires_a_prior_on_each_parameter():

    from grassmoth.compilation_errors import IncompleteModelError

    science = x_model()
    draft = science.with_entities(
        distributions={},
        parameters=tuple(parameter.revised(distribution=None) for parameter in science.parameters),
    )
    with pytest.raises(IncompleteModelError, match="prior"):
        require_priors(StructuralSelection(draft, None))
    completed = x_model()
    require_priors(StructuralSelection(completed, None))
    assert [p.id for p in completed.parameters] == [p.id for p in draft.parameters]
    assert all(p.distribution is not None for p in completed.parameters)
    assert all(p.distribution is None for p in draft.parameters)


@pytest.mark.contract
def test_law_memberships_reject_dangling_unused_and_accidentally_shared_scalar_laws():
    dynamical_model_spec = x_model()
    first, second = dynamical_model_spec.parameters[:2]
    assert first.distribution != second.distribution
    with pytest.raises(ValidationError, match="every reference must exist"):
        dynamical_model_spec.with_entities(distributions={})
    with pytest.raises(ValidationError, match="must be referenced"):
        dynamical_model_spec.with_entities(
            distributions={
                **dynamical_model_spec.distributions,
                DistributionId("distribution:unused"): dist.Normal(0, 1),
            }
        )
    with pytest.raises(ValidationError, match="exactly one parameter"):
        dynamical_model_spec.with_entities(
            parameters=tuple(
                p.revised(distribution=first.distribution) if p.id == second.id else p
                for p in dynamical_model_spec.parameters
            ),
            distributions={
                k: v
                for k, v in dynamical_model_spec.distributions.items()
                if k != second.distribution
            },
        )
