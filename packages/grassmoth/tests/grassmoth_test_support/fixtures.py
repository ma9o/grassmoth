"""JSON fixtures owned by library tests."""

from pathlib import Path


def model_fixture_json(name: str) -> str:
    return (Path(__file__).parent.parent / "fixtures/models" / name).read_text()
