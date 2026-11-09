"""Capture immutable staged sources once and reuse their exact Git tree."""

from pathlib import Path

import pygit2

from grassmoth_server.storage import backend as storage
from grassmoth_server.study.persistence.git_objects import write_directory


def capture_sources(repository: pygit2.Repository, directory: Path) -> pygit2.Oid:
    """Memoize a call's source tree for raw-data ownership and final publication.

    Source files are staged before execution and remain immutable until the call
    is swept. Calls rejected before raw-data creation capture them at publication.
    """
    identity = directory / "sources.tree"
    if identity.exists():
        return pygit2.Oid(hex=identity.read_text())
    tree = write_directory(repository, directory / "sources")
    storage.write_text(str(identity), str(tree))
    return tree
