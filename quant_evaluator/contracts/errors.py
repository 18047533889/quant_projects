"""Contract errors for quant_evaluator."""


class InvalidContractError(Exception):
    """Raised when a contract validation fails."""
    pass


class UnsupportedMetricError(Exception):
    """Raised when an unsupported metric is requested."""
    pass
