"""JSON fixtures owned by server tests."""

from pathlib import Path


def study_fixture_json(name: str) -> str:
    return (Path(__file__).parent.parent / "fixtures/models" / name).read_text()
