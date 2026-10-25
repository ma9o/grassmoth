"""Package imports stay lightweight; computational entry points own process policy."""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.contract


def test_numerical_policy_requires_explicit_initialization() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import grassmoth
from grassmoth_server.bootstrap import initialize_numerical_runtime

assert "numpyro" not in sys.modules
assert "jax" not in sys.modules
initialize_numerical_runtime()

import numpy as np
import numpyro.distributions as dist
import jax
import jax.numpy as jnp
assert (jnp.ones(2) + jnp.asarray(np.ones(2))).dtype == jnp.float32
assert jax.config.jax_threefry_gpu_kernel_lowering is True
assert dist.Normal(0.0, 1.0)._validate_args is False
assert dist.Normal(0.0, 1.0, validate_args=True)._validate_args is True
""",
        ],
        env={**os.environ, "JAX_ENABLE_X64": "true"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
