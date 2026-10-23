"""Local Git object storage shared by scientific payloads and study commits."""

import subprocess
from pathlib import Path

import pygit2

from grassmoth_server.storage import paths as data_module


def open_repository(workspace_id: str, path: Path | None = None) -> pygit2.Repository:
    """Open or initialize local study history, rejecting incompatible repository formats."""
    if path is None and "://" in data_module.study_dir(workspace_id):
        raise ValueError("Study history requires local storage")
    destination = path or Path(data_module.study_dir(workspace_id)) / "history.git"
    if destination.exists():
        repository = pygit2.Repository(str(destination))
        if (
            "grassmoth.format" not in repository.config
            or repository.config.get_int("grassmoth.format") != 29
        ):
            raise ValueError(
                "This study uses an incompatible storage format. Create a new study for format 29."
            )
        return repository
    destination.parent.mkdir(parents=True, exist_ok=True)
    repository = pygit2.init_repository(str(destination), bare=True, initial_head="main")
    repository.config["grassmoth.format"] = 29
    repository.config["user.name"] = "grassmoth"
    repository.config["user.email"] = "study@local"
    return repository


def write_blob(repository: pygit2.Repository, payload: bytes) -> pygit2.Oid:
    """Write identical Git objects without holding the worker GIL during compression.

    libgit2's Python blob writer retains the GIL for the whole compression,
    starving concurrent Temporal activations even from an activity thread.
    The Git process owns that CPU work; the caller retains its publication order.
    """
    written = subprocess.run(
        ["git", "--git-dir", repository.path, "hash-object", "-w", "--stdin"],
        input=payload,
        capture_output=True,
        check=True,
    )
    return pygit2.Oid(hex=written.stdout.decode().strip())


def write_tree(repository: pygit2.Repository, files: dict[str, bytes]) -> pygit2.Oid:
    """Write path-keyed bytes as nested Git trees and return the root tree identity."""
    builder = repository.TreeBuilder()
    directories: dict[str, dict[str, bytes]] = {}
    for name, content in files.items():
        directory, separator, rest = name.partition("/")
        if separator:
            directories.setdefault(directory, {})[rest] = content
        else:
            builder.insert(name, write_blob(repository, content), pygit2.GIT_FILEMODE_BLOB)
    for name, children in directories.items():
        builder.insert(name, write_tree(repository, children), pygit2.GIT_FILEMODE_TREE)
    oid: pygit2.Oid = builder.write()
    return oid


def read_blob(repository: pygit2.Repository, identity: pygit2.Oid) -> bytes:
    """Decode one exact blob directly, without the object lookup's extra inflation."""
    kind, payload = repository.odb.read(identity)
    if kind != pygit2.enums.ObjectType.BLOB:
        raise ValueError("The selected object is not a stored payload blob")
    return payload


def object_tree(repository: pygit2.Repository, revision: str) -> pygit2.Tree:
    """Resolve an exact commit or tree identity, rejecting missing objects and non-tree content."""
    from grassmoth_server.study.errors import StudyLookupError

    oid = pygit2.Oid(hex=revision)
    if oid not in repository:
        raise StudyLookupError(f"Unknown study revision: {revision}")
    obj = repository[oid]
    if not isinstance(obj, (pygit2.Tree, pygit2.Commit)):
        raise StudyLookupError("The selected revision does not contain a tree")
    return obj.peel(pygit2.Tree)


def read_file(repository: pygit2.Repository, revision: str, path: str) -> bytes:
    """Read blob bytes at an exact revision and path, rejecting unavailable stored files."""
    from grassmoth_server.study.errors import StudyLookupError

    tree = object_tree(repository, revision)
    if path not in tree:
        raise StudyLookupError(f"No stored file {path!r} at {revision}")
    blob = tree[path].peel(pygit2.Blob)
    return blob.data
