"""Resolve owned numerical views and index embedded law values for storage."""

from collections.abc import Mapping

from nof1_causal_lab.artifacts.arrays import NumericalArray


def owned_arrays(value: object) -> Mapping[str, NumericalArray]:
    """Index each shared buffer once, preserving the codec's restored ownership."""
    buffers: dict[int, NumericalArray] = {}

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            if set(item) == {"npy"}:
                identity = id(item["npy"])
                if identity not in buffers:
                    buffers[identity] = NumericalArray.model_validate(item)
            else:
                for child in item.values():
                    visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return {array.identity: array for array in buffers.values()}
