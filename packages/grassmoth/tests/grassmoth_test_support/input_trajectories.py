"""Authored deterministic inputs for library and server tests."""

import jax.numpy as jnp
import numpyro.distributions as dist

from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.model.expressions import state
from grassmoth.artifacts.model.likelihood import DeltaLawSpec, LikelihoodSpec
from grassmoth.artifacts.probability.joint_layout import JointLawLayout
from grassmoth_test_support.model import x_y_model


def deterministic_input_model(*, measured=False, time_origin="relative"):
    dynamical_model_spec = x_y_model()
    source = dynamical_model_spec.constructs[0]
    layout = JointLawLayout(
        parameters=(),
        constructs=(source.id,),
        time_points=(0.0, 0.5, 2.5),
        time_origin=time_origin,
        labels={},
    )
    source = source.revised(
        role="exogenous",
        dynamics=(),
        coefficients=(),
        distribution=layout.distribution_id,
        indicators=(
            source.indicators[0].revised(
                likelihood=LikelihoodSpec(
                    law=DeltaLawSpec(v=state(source.id)), reasoning="Exact known input"
                ),
            ),
        )
        if measured
        else (),
    )
    target = dynamical_model_spec.constructs[1]
    target = target.revised(
        indicators=tuple(
            indicator.revised(observation=indicator.observation.revised(aggregation="last"))
            for indicator in target.indicators
        )
    )
    parameters = tuple(
        p for p in dynamical_model_spec.parameters if p.name not in {"rho_X", "sigma_X"}
    )
    return dynamical_model_spec.with_entities(
        constructs=(source, target),
        edges=replace_constructs(dynamical_model_spec.edges, (source, target)),
        parameters=parameters,
        distributions={
            **{
                key: value
                for key, value in dynamical_model_spec.distributions.items()
                if key in {p.distribution for p in parameters}
            },
            layout.distribution_id: dist.Delta(jnp.array([1.0, 3.0, 2.0]), event_dim=1),
        },
        law_layouts={layout.distribution_id: layout},
    )
