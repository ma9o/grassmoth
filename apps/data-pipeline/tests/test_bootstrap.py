"""Package imports stay lightweight; computational entry points own process policy."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.contract


def test_numerical_policy_requires_explicit_initialization() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import nof1_causal_lab
from nof1_causal_service.bootstrap import initialize_numerical_runtime

assert "numpyro" not in sys.modules
assert "jax" not in sys.modules
initialize_numerical_runtime()

import numpyro.distributions as dist
import jax
assert jax.config.jax_threefry_gpu_kernel_lowering is True
assert dist.Normal(0.0, 1.0)._validate_args is False
assert dist.Normal(0.0, 1.0, validate_args=True)._validate_args is True
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_numerical_startup_uses_float32_with_explicit_float64_reference_scope():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""\
                import json

                import jax
                import jax.numpy as jnp
                import numpy as np

                from nof1_causal_service.bootstrap import initialize_numerical_runtime

                initialize_numerical_runtime()

                def array_dtype():
                    return str((jnp.ones(2) + jnp.asarray(np.ones(2))).dtype)

                production = array_dtype()
                with jax.enable_x64():
                    reference = array_dtype()
                print(json.dumps([production, reference, array_dtype()]))
                """),
        ],
        env={**os.environ, "JAX_ENABLE_X64": "true"},
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == ["float32", "float64", "float32"]
