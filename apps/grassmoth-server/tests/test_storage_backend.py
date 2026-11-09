"""Deployment storage roots remain independent of scientific requests."""

import pytest

from grassmoth_server.storage.backend import get_base_uri

pytestmark = pytest.mark.contract


def test_local_data_root_is_explicit_and_does_not_override_remote_storage(tmp_path, monkeypatch):
    monkeypatch.delenv("DEPLOYMENT_ENV", raising=False)
    monkeypatch.setenv("GRASSMOTH_DATA_ROOT", str(tmp_path))
    assert get_base_uri() == str(tmp_path)
    monkeypatch.setenv("DEPLOYMENT_ENV", "production")
    monkeypatch.setenv("R2_BUCKET", "test-bucket")
    monkeypatch.setenv("R2_PREFIX", "test-prefix")
    assert get_base_uri() == "s3://test-bucket/test-prefix"


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.parametrize("previous", [None, "old"])
def test_text_publication_exposes_only_complete_records(tmp_path, monkeypatch, fails, previous):
    from pathlib import Path

    from grassmoth_server.storage.backend import write_text
    from grassmoth_server.storage.records import read_records

    monkeypatch.delenv("DEPLOYMENT_ENV", raising=False)
    directory = tmp_path / "records"
    target = directory / "record.json"
    if previous is not None:
        directory.mkdir()
        target.write_text(previous)
    replace = Path.replace
    before = [] if previous is None else [(target.name, previous)]

    def publish(temporary, destination):
        assert temporary.read_text() == "complete"
        assert list(read_records(str(directory))) == before
        if fails:
            raise OSError("publication failed")
        return replace(temporary, destination)

    monkeypatch.setattr(Path, "replace", publish)
    if fails:
        with pytest.raises(OSError, match="publication failed"):
            write_text(str(target), "complete")
    else:
        write_text(str(target), "complete")
    assert list(read_records(str(directory))) == (before if fails else [(target.name, "complete")])
    assert not list(directory.glob("*.partial"))
