"""Translate the public study boundary's expected failures into HTTP responses."""

from typing import NoReturn

from fastapi import HTTPException

from grassmoth_server.actions.contracts.polling import StudyError


def raise_study_error(error: StudyError) -> NoReturn:
    """Map typed library failures to the existing HTTP contract."""
    status = {
        "invalid_request": 400,
        "read_only": 403,
        "invalid_inputs": 422,
        "unknown_call": 404,
    }
    raise HTTPException(status[error.code], error.detail)
