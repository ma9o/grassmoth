"""Owned collections retain no mutable collection or NumPy storage aliases."""

from types import MappingProxyType

import numpy as np
import pytest
from pydantic import ValidationError

from nof1_causal_lab.artifacts.immutability import freeze
from nof1_causal_service.study.artifact_files import ArtifactFileSpec

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.dtype("U3")])
def test_nested_collections_and_strided_buffers_are_detached(dtype) -> None:
    builder = np.arange(12).astype(dtype).reshape(3, 4)[:, ::2]
    expected = builder.copy()
    source = {"draws": [{"buffer": builder}], "labels": {"a", "b"}}
    owned = freeze(source)
    assert isinstance(owned, MappingProxyType)
    draws = owned["draws"]
    assert isinstance(draws, tuple)
    assert isinstance(draws[0], MappingProxyType)
    array = draws[0]["buffer"]
    assert isinstance(array, np.ndarray)
    builder[...] = 0
    source["draws"].clear()
    np.testing.assert_array_equal(array, expected)
    assert array.shape == expected.shape
    assert array.dtype == expected.dtype
    assert owned["labels"] == frozenset({"a", "b"})
    with pytest.raises(ValueError, match="read-only"):
        array[0, 0] = 1
    with pytest.raises(ValueError, match="WRITEABLE"):
        array.setflags(write=True)
    assert isinstance(array.base, np.ndarray)
    with pytest.raises(ValueError, match="WRITEABLE"):
        array.base.setflags(write=True)


def test_object_arrays_cannot_claim_immutable_storage() -> None:
    with pytest.raises(TypeError, match="non-object dtype"):
        freeze(np.array([{"mutable": []}], dtype=object))


def test_artifact_file_layout_owns_inputs_and_default_collections() -> None:
    json_files = {"metadata": "metadata.json"}
    parquet_files = {"panel": "panel.parquet"}
    layout = ArtifactFileSpec(json_files=json_files, parquet_files=parquet_files)
    json_files.clear()
    parquet_files.clear()

    assert layout.json_files == {"metadata": "metadata.json"}
    assert layout.parquet_files == {"panel": "panel.parquet"}
    for mapping in (layout.json_files, layout.parquet_files, ArtifactFileSpec().json_files):
        assert isinstance(mapping, MappingProxyType)
    for field_name in ("json_files", "parquet_files"):
        with pytest.raises(ValidationError, match="frozen"):
            setattr(layout, field_name, {})
