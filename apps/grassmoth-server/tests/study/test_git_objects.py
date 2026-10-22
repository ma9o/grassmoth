"""Git blob writes preserve the bytes and identities consumed by libgit2 readers."""

import hashlib

import pygit2
import pytest

from grassmoth_server.study.persistence.git_objects import read_blob, write_blob

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    "payload",
    [b"", b"binary\x00\xff\r\n", bytes(range(256)) * 4096],
    ids=["empty", "binary", "one_mib"],
)
def test_blob_writer_preserves_native_git_identity_and_bytes(tmp_path, payload):
    repository = pygit2.init_repository(str(tmp_path / "study with spaces.git"), bare=True)
    expected = hashlib.sha1(
        b"blob " + str(len(payload)).encode() + b"\x00" + payload, usedforsecurity=False
    ).hexdigest()
    identity = write_blob(repository, payload)
    assert str(identity) == expected
    assert read_blob(repository, identity) == payload
    assert repository.create_blob(payload) == identity
    assert write_blob(repository, payload) == identity
