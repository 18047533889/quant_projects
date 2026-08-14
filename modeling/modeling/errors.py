"""
Modeling package error hierarchy.
"""


class ModelingError(Exception):
    """Base error for modeling package."""
    pass


class FutureLeakageError(ModelingError):
    """Raised when future information would leak into training data."""
    pass


class FitWindowError(ModelingError):
    """Raised when fit window constraints are violated."""
    pass


class SplitError(ModelingError):
    """Raised when split specification is invalid."""
    pass


class ContractViolation(ModelingError):
    """Raised when a contract is violated."""
    pass


class InsufficientDataError(ModelingError):
    """Raised when insufficient data is available for operation."""
    pass


class AdapterError(ModelingError):
    """Raised when adapter cannot translate to underlying implementation."""
    pass
