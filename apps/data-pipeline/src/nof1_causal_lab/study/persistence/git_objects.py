"""Local Git object storage shared by scientific payloads and study commits."""

from pathlib import Path

import pygit2

from nof1_causal_lab.storage import paths as data_module


def open_repository(workspace_id: str, path: Path | None = None) -> pygit2.Repository:
    """Open or initialize local study history, rejecting incompatible repository formats."""
    if path is None and "://" in data_module.study_dir(workspace_id):
        raise ValueError("Study history requires local storage")
    destination = path or Path(data_module.study_dir(workspace_id)) / "history.git"
    if destination.exists():
        repository = pygit2.Repository(str(destination))
        if "nof1.format" not in repository.config or repository.config.get_int("nof1.format") != 28:
            raise ValueError(
                "This study uses an incompatible storage format. Create a new study for format 28."
            )
        return repository
    destination.parent.mkdir(parents=True, exist_ok=True)
    repository = pygit2.init_repository(str(destination), bare=True, initial_head="main")
    repository.config["nof1.format"] = 28
    repository.config["user.name"] = "nof1-causal-lab"
    repository.config["user.email"] = "study@local"
    signature = pygit2.Signature("nof1-causal-lab", "study@local", 0, 0)
    root = repository.create_commit(
        None, signature, signature, "Study root", repository.TreeBuilder().write(), []
    )
    with repository.transaction() as transaction:
        transaction.lock_ref("refs/heads/main")
        if "refs/heads/main" not in repository.references:
            transaction.set_target("refs/heads/main", root)
    return repository


def write_tree(repository: pygit2.Repository, files: dict[str, bytes]) -> pygit2.Oid:
    """Write path-keyed bytes as nested Git trees and return the root tree identity."""
    builder = repository.TreeBuilder()
    directories: dict[str, dict[str, bytes]] = {}
    for name, content in files.items():
        directory, separator, rest = name.partition("/")
        if separator:
            directories.setdefault(directory, {})[rest] = content
        else:
            builder.insert(name, repository.create_blob(content), pygit2.GIT_FILEMODE_BLOB)
    for name, children in directories.items():
        builder.insert(name, write_tree(repository, children), pygit2.GIT_FILEMODE_TREE)
    oid: pygit2.Oid = builder.write()
    return oid


def object_tree(repository: pygit2.Repository, revision: str) -> pygit2.Tree:
    """Resolve an exact commit or tree identity, rejecting missing objects and non-tree content."""
    from nof1_causal_lab.study.errors import StudyLookupError

    oid = pygit2.Oid(hex=revision)
    if oid not in repository:
        raise StudyLookupError(f"Unknown study revision: {revision}")
    obj = repository[oid]
    if not isinstance(obj, (pygit2.Tree, pygit2.Commit)):
        raise StudyLookupError("The selected revision does not contain a tree")
    return obj.peel(pygit2.Tree)


def read_file(repository: pygit2.Repository, revision: str, path: str) -> bytes:
    """Read blob bytes at an exact revision and path, rejecting unavailable stored files."""
    from nof1_causal_lab.study.errors import StudyLookupError

    tree = object_tree(repository, revision)
    if path not in tree:
        raise StudyLookupError(f"No stored file {path!r} at {revision}")
    blob = tree[path].peel(pygit2.Blob)
    return blob.data
