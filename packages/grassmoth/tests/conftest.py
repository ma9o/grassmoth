"""Numerical process policy for library tests."""

import jax
import numpyro


def pytest_configure() -> None:  # noqa: V103 - pytest calls this configuration hook before collection.
    """Set numerical precision before collecting library fixtures."""
    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_threefry_gpu_kernel_lowering", True)
    numpyro.enable_validation(False)
