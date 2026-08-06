"""Typed operator failures used by production execution paths."""


class OperatorExecutionError(RuntimeError):
    """Base class for operator execution failures."""


class OperatorParameterError(OperatorExecutionError, ValueError):
    """Operator parameters violate the declared contract."""


class OperatorShapeError(OperatorExecutionError, ValueError):
    """Operator output does not preserve its declared panel shape."""


class OperatorDomainError(OperatorExecutionError, ValueError):
    """Operator input is outside its mathematical domain."""


class FutureReferenceError(OperatorParameterError):
    """A parameter (e.g. a negative lag) would reference future data.

    Negative lags must be rejected at validation time instead of silently
    returning an all-NaN panel, which turns an illegal formula into a factor
    that only happens to have zero backtest coverage.
    """
