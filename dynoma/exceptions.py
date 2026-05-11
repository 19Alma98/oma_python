from __future__ import annotations


class ModalIdentificationError(Exception):
    """Modal identification error."""

    def __init__(self, detail: str | None = None) -> None:
        """Initialize a ModalIdentificationError instance."""
        super().__init__(detail)
