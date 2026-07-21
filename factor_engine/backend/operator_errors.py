"""Typed operator failures used by production execution paths."""


class OperatorExecutionError(RuntimeError):
    """Base class for operator execution failures."""


class OperatorParameterError(OperatorExecutionError, ValueError):
    """Operator parameters violate the declared contract."""


class OperatorShapeError(OperatorExecutionError, ValueError):
    """Operator output does not preserve its declared panel shape."""


class OperatorDomainError(OperatorExecutionError, ValueError):
    """Operator input is outside its mathematical domain."""
