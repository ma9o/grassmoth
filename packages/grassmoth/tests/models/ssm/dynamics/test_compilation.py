"""Scientific expression compilation, free parameters, and surgical interventions."""

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace

from grassmoth.models.ssm.dynamics.intervention import (
    ConstantValueFn,
    EdgeInputOverride,
    Intervention,
)
from grassmoth.models.ssm.dynamics.spec import (
    DynamicsSpec,
    compile_dynamics,
    pack_component_params_from_samples,
)
from grassmoth.models.ssm.dynamics.vector_field import VectorFieldArgs
from grassmoth_test_support.dynamics import decay_term, hill_term, interaction_term, linear_term

pytestmark = pytest.mark.contract


def test_composed_expression_field_preserves_nonlinearity_and_edge_surgery():
    spec = DynamicsSpec(
        5,
        (
            *(decay_term(i) for i in range(5)),
            interaction_term(0, 1, 2, weight=0.5),
            linear_term(2, 3, weight=0.7),
            hill_term(
                3,
                4,
                emax=2,
                ec50=1,
                n=2,
            ),
        ),
    )
    compiled = compile_dynamics(spec)
    params = seed(lambda: compiled.sample_params(lambda _: dist.Delta(jnp.array(1.0))), 0)()
    x = jnp.array([2.0, 3.0, 4.0, 2.0, 1.0])
    natural = compiled.vector_field(jnp.array(0.0), x, VectorFieldArgs(params, Intervention.none()))
    np.testing.assert_allclose(natural, [-2.0, -3.0, -1.0, 0.8, 0.6], atol=1e-6)
    intervention = Intervention((EdgeInputOverride(2, 3, ConstantValueFn(jnp.array(0.0))),))
    changed = compiled.vector_field(jnp.array(0.0), x, VectorFieldArgs(params, intervention))
    np.testing.assert_allclose(changed, [-2.0, -3.0, -1.0, -2.0, 0.6], atol=1e-6)


def test_free_operands_sample_once_and_pack_without_resampling():
    spec = DynamicsSpec(
        2,
        (
            decay_term(0),
            hill_term(0, 1, ec50=2, n=1),
        ),
    )
    compiled = compile_dynamics(spec, prefix="mechanism")

    def model():
        return compiled.sample_params(lambda _: dist.HalfNormal(1.0))

    traced = trace(seed(model, 3)).get_trace()
    samples = {name: site["value"] for name, site in traced.items() if site["type"] == "sample"}
    assert len(samples) == 2
    assert set(samples) == {site.name for site in compiled.site_registry}
    packed = pack_component_params_from_samples(spec, samples, prefix="mechanism")
    actual = seed(model, 3)()
    for expected_component, actual_component in zip(packed, actual, strict=True):
        for name, value in expected_component.items():
            np.testing.assert_array_equal(value, actual_component[name])


def test_empty_expression_field_has_zero_drift():
    compiled = compile_dynamics(DynamicsSpec(2))
    result = compiled.vector_field(
        jnp.array(0.0), jnp.ones(2), VectorFieldArgs((), Intervention.none())
    )
    np.testing.assert_array_equal(result, np.zeros(2))
