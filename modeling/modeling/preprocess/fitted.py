"""
Fitted transforms for modeling.

These transforms require a fit() step on training data before they can
be applied to validation/test data. Critical for preventing leakage.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np

from modeling.contracts import FitWindow
from modeling.errors import FitWindowError, FutureLeakageError, InsufficientDataError


class FittedTransform(ABC):
    """
    Base class for fitted transforms.

    All fitted transforms must:
    1. Record fit_window during fit()
    2. Validate temporal ordering during transform()
    3. Store minimal state (parameters only, not data)
    """

    def __init__(self):
        self.fit_window: Optional[FitWindow] = None
        self._fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, fit_window: FitWindow) -> "FittedTransform":
        """
        Fit transform on training data.

        Args:
            X: Training data (T, N) where T=time, N=features
            fit_window: Temporal window of training data

        Returns:
            Self for chaining
        """
        pass

    @abstractmethod
    def transform(self, X: np.ndarray, apply_start_time: Any) -> np.ndarray:
        """
        Apply fitted transform to new data (OUT-OF-SAMPLE ONLY).

        Args:
            X: Data to transform (T, N)
            apply_start_time: Start time of data being transformed (REQUIRED for OOS validation).
                This parameter is REQUIRED and cannot be None. For in-sample transformation
                during training, use fit_transform() instead.

        Returns:
            Transformed data with same shape

        Raises:
            TypeError: If apply_start_time is None (use fit_transform() for in-sample)
            FitWindowError: If transform not fitted
            FutureLeakageError: If apply_start_time is before fit_window.fit_end
        """
        pass

    def fit_transform(self, X: np.ndarray, fit_window: FitWindow) -> np.ndarray:
        """
        Fit and transform in one step - ONLY valid for in-sample (training) data.

        This method should ONLY be used when transforming the SAME data that was
        used for fitting (i.e., training data). For out-of-sample data (validation/test),
        use fit() then transform() separately with explicit apply_start_time.

        Args:
            X: Training data to fit and transform (T, N)
            fit_window: Temporal window of the training data

        Returns:
            Transformed training data with same shape

        Warning:
            This method is for in-sample transformation only. Using it on different
            data than what was fitted would bypass leakage detection. For OOS data,
            always use fit() then transform() with apply_start_time.
        """
        self.fit(X, fit_window)
        # FM2-P0-012~015: Internal in-sample path uses fit_window.fit_end as apply_start_time
        # This is safe because we're transforming the same data we just fit on.
        return self.transform(X, apply_start_time=fit_window.fit_end)

    def _validate_temporal_ordering(self, apply_start_time: Any):
        """Validate that fit happened before application (no future leakage)."""
        if not self._fitted:
            raise FitWindowError("Transform must be fitted before applying")

        if self.fit_window is None:
            raise FitWindowError("fit_window not recorded during fit")

        if apply_start_time is not None:
            # Check for future leakage
            if self.fit_window.fit_end > apply_start_time:
                raise FutureLeakageError(
                    f"Fit window ends at {self.fit_window.fit_end} but "
                    f"applying to data starting at {apply_start_time}"
                )


class CrossSectionalScaler(FittedTransform):
    """
    Fit cross-sectional mean/std on training period, apply to any period.

    This is a simple example - for production, use factor_preprocess via adapter.
    """

    def __init__(self, method: str = "zscore"):
        super().__init__()
        self.method = method
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, fit_window: FitWindow) -> "CrossSectionalScaler":
        """Fit scaler on training data."""
        if X.size == 0:
            raise InsufficientDataError("Empty training data")

        self.fit_window = fit_window

        if self.method == "zscore":
            # Compute mean/std across time for each feature
            self.mean_ = np.nanmean(X, axis=0)
            self.std_ = np.nanstd(X, axis=0, ddof=1)
            # Avoid division by zero
            self.std_ = np.where(self.std_ < 1e-10, 1.0, self.std_)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._fitted = True
        return self

    def transform(self, X: np.ndarray, apply_start_time: Any) -> np.ndarray:
        """Apply fitted scaling (OUT-OF-SAMPLE ONLY).

        Args:
            X: Data to transform
            apply_start_time: Start time of data (REQUIRED for OOS validation).
                Cannot be None. For in-sample use, call fit_transform() instead.

        Raises:
            TypeError: If apply_start_time is None
            FitWindowError: If not fitted
            FutureLeakageError: If apply_start_time is before fit_window.fit_end
        """
        # FM2-P0-012~015: Enforce apply_start_time requirement (no silent None bypass)
        if apply_start_time is None:
            raise TypeError(
                "transform() requires apply_start_time for OOS safety. "
                "For in-sample (training) data, use fit_transform() instead."
            )

        if not self._fitted:
            raise FitWindowError("Transform must be fitted before applying")

        if self.fit_window is None:
            raise FitWindowError("fit_window not recorded during fit")

        # Always validate temporal ordering when apply_start_time is provided
        self._validate_temporal_ordering(apply_start_time)

        if self.method == "zscore":
            return (X - self.mean_) / self.std_
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def get_state(self) -> Dict[str, Any]:
        """Export fitted state for serialization."""
        return {
            "method": self.method,
            "mean": self.mean_,
            "std": self.std_,
            "fit_window": {
                "fit_start": self.fit_window.fit_start if self.fit_window else None,
                "fit_end": self.fit_window.fit_end if self.fit_window else None,
            },
        }


def create_fitted_scaler(method: str = "zscore") -> FittedTransform:
    """
    Factory function for fitted scalers.

    Args:
        method: Scaling method ('zscore', etc.)

    Returns:
        FittedTransform instance
    """
    if method == "zscore":
        return CrossSectionalScaler(method="zscore")
    else:
        raise ValueError(f"Unknown scaling method: {method}")
