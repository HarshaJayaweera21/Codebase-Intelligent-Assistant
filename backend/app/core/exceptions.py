class RepositoryCloneError(Exception):
    """Raised when a GitHub repository cannot be cloned."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

class RepositoryScanError(Exception):
    """Raised when a cloned repository cannot be scanned."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)