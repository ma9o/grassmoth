"""Resolve owned numerical views and index embedded law values for storage."""

from collections.abc import Mapping

from nof1_causal_lab.artifacts.arrays import NumericalArray


def owned_arrays(value: object) -> Mapping[str, NumericalArray]:
    """Index embedded numerical owners only at the persistence boundary."""
    if isinstance(value, Mapping):
        if set(value) == {"npy"}:
            array = NumericalArray.model_validate(value)
            return {array.identity: array}
        return {
            ref: array for child in value.values() for ref, array in owned_arrays(child).items()
        }
    if isinstance(value, (list, tuple)):
        return {ref: array for child in value for ref, array in owned_arrays(child).items()}
    return dict[str, NumericalArray]()
