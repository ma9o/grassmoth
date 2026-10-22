"""Owned collections retain no mutable collection or NumPy storage aliases."""

from types import MappingProxyType

import pytest
from pydantic import ValidationError

from grassmoth_server.study.artifact_files import ArtifactFileSpec

pytestmark = pytest.mark.contract


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
