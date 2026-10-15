"""Explicit process policy for numerical entry points."""


def initialize_numerical_runtime() -> None:
    """Use float32 production arrays and opt-in NumPyro argument validation.

    Reference calculations can explicitly opt into ``jax.enable_x64()`` afterward.
    Authored and decoded laws choose their own validation. Internal selectors
    retain exact zero weights as negative infinite logits, and stored laws keep
    the validation setting under which they were built.
    """
    import jax
    import numpyro

    jax.config.update("jax_enable_x64", False)
    numpyro.enable_validation(False)
