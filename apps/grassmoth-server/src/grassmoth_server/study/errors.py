"""Storage failures shared by persistence and the action executor."""


class StudyLookupError(Exception):
    """A requested revision, artifact, or stored path does not exist."""


class IncompatibleStudyFormatError(ValueError):
    """The selected study uses an unsupported storage format."""
