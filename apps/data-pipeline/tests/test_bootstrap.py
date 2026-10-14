"""Package imports stay lightweight; computational entry points own process policy."""

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
import nof1_causal_lab
from nof1_causal_lab.bootstrap import initialize_numerical_runtime

assert "numpyro" not in sys.modules
assert "jax" not in sys.modules
initialize_numerical_runtime()

import numpyro.distributions as dist
assert dist.Normal(0.0, 1.0)._validate_args is False
assert dist.Normal(0.0, 1.0, validate_args=True)._validate_args is True
""",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
