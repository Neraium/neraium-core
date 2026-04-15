"""Read-only mode enforcement utilities."""

from typing import Any

from fastapi import HTTPException, status


class ReadOnlyModeError(HTTPException):
    """Exception raised when attempting write operations in read-only mode."""

    def __init__(self) -> None:
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="System is in read-only mode. Write operations are not permitted.",
        )


def enforce_read_only(read_only: bool) -> None:
    """
    Enforce read-only mode by raising an HTTPException if read-only is enabled.

    Args:
        read_only: Whether the system is in read-only mode.

    Raises:
        ReadOnlyModeError: If read-only mode is enabled.
    """
    if read_only:
        raise ReadOnlyModeError()
