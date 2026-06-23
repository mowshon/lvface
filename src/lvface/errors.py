"""Public exceptions raised by lvface."""


class AlignmentError(ValueError):
    """Raised when a face cannot be aligned from its landmarks."""


class NoFaceError(ValueError):
    """Raised when an operation requires a face but none was found."""
