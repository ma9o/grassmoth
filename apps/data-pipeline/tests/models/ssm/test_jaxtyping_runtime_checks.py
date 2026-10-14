"""Runtime shape/dtype checking via the jaxtyping + beartype import hook.

The ``--jaxtyping-packages=...,beartype.beartype`` pytest flag (see
``pyproject.toml``) instruments the listed modules so their jaxtyping
annotations are enforced at call time. These tests confirm the hook rejects
named-axis and dtype violations at the function boundary (before the body runs).
Valid inputs are covered by the numerical tests in ``test_covariance_utils.py``.
The checks also gate the convention that an instrumented module omits
``from __future__ import annotations`` so beartype can resolve its annotations.
"""

import jax.numpy as jnp
import jaxtyping
import pytest

from nof1_causal_lab.models.ssm.covariance import symmetrize

pytestmark = pytest.mark.contract


def test_covariance_named_axis_mismatch_raises():
    # A covariance must use the same named axis for both matrix dimensions.
    with pytest.raises(jaxtyping.TypeCheckError):
        symmetrize(jnp.ones((3, 2)))


def test_covariance_dtype_mismatch_raises():
    # Float[Array, "*batch N N"] rejects an integer-dtyped covariance.
    with pytest.raises(jaxtyping.TypeCheckError):
        symmetrize(jnp.eye(3, dtype=jnp.int32))
