"""Explicit process policy for numerical entry points."""


def initialize_numerical_runtime() -> None:
    """Keep NumPyro argument validation opt-in for internal numerical laws.

    Authored and decoded laws choose their own validation. Internal selectors
    retain exact zero weights as negative infinite logits, and stored laws keep
    the validation setting under which they were built.
    """
    import numpyro

    numpyro.enable_validation(False)
