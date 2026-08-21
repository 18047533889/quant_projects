"""Canonical normalization for label values and validity masks."""

from typing import Optional, Tuple

import numpy as np

from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.contracts.label_bundle import LabelBundle


def normalize_label_panel(
    label_bundle: LabelBundle,
    num_assets: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return label values and validity with a shared ``(T, N)`` shape."""
    values = np.asarray(label_bundle.values)
    if values.ndim == 1:
        values = np.broadcast_to(values[:, np.newaxis], (values.shape[0], num_assets))
    elif values.ndim == 2:
        if values.shape[1] != num_assets:
            raise InvalidContractError(
                f"Label asset axis ({values.shape[1]}) does not match "
                f"factor asset axis ({num_assets})"
            )
    else:
        raise InvalidContractError("Label values must be 1D or 2D")

    validity = label_bundle.validity
    if validity is None:
        return values, None
    validity = np.asarray(validity, dtype=bool)
    if validity.ndim == 1:
        validity = np.broadcast_to(validity[:, np.newaxis], (values.shape[0], num_assets))
    elif validity.ndim == 2:
        if validity.shape != values.shape:
            raise InvalidContractError(
                f"Label validity shape {validity.shape} does not match "
                f"normalized label shape {values.shape}"
            )
    else:
        raise InvalidContractError("Label validity must be 1D or 2D")
    return values, validity


__all__ = ["normalize_label_panel"]
