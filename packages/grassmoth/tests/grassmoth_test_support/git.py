"""Synthetic library revision identities."""

from grassmoth.artifacts.identity import GitOid


def git_oid(label: int) -> GitOid:
    return GitOid(f"{label:040x}")
