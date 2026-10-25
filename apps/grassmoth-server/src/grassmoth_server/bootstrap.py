"""Explicit process policy for numerical entry points."""


def initialize_numerical_runtime() -> None:
    """Use float32 arrays, native GPU Threefry, and opt-in NumPyro validation.

    Authored and decoded laws choose their own validation. Internal selectors
    retain exact zero weights as negative infinite logits, and stored laws keep
    the validation setting under which they were built.
    """
    import jax
    import numpyro

    jax.config.update("jax_enable_x64", False)
    jax.config.update("jax_threefry_gpu_kernel_lowering", True)
    numpyro.enable_validation(False)
