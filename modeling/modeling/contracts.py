"""
Core contracts for model input preparation.

Defines temporal boundaries, split specifications, and output formats.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal
from enum import Enum

from modeling.errors import FitWindowError, SplitError, ContractViolation


@dataclass(frozen=True)
class FitWindow:
    """
    Temporal window for fitting preprocessing transforms.

    Critical for preventing full-sample leakage.
    """
    fit_start: datetime
    fit_end: datetime

    # Optional: universe/data version binding
    universe_ref: Optional[str] = None
    data_snapshot_ref: Optional[str] = None

    def __post_init__(self):
        if self.fit_start >= self.fit_end:
            raise FitWindowError(f"fit_start must be before fit_end: {self.fit_start} >= {self.fit_end}")

    def is_valid_for_application(self, apply_start: datetime, apply_end: datetime) -> bool:
        """Check if this fit window is valid for applying to given period."""
        # Fit window must end before or at application start (causal)
        return self.fit_end <= apply_start

    def duration_days(self) -> int:
        """Return fit window duration in days."""
        return (self.fit_end - self.fit_start).days


@dataclass(frozen=True)
class SplitSpec:
    """
    Specification for train/validation/test splits.

    Defines temporal boundaries for each split to prevent leakage.
    """
    split_id: str

    # Train period
    train_start: datetime
    train_end: datetime

    # Validation period (optional)
    val_start: Optional[datetime] = None
    val_end: Optional[datetime] = None

    # Test period (optional)
    test_start: Optional[datetime] = None
    test_end: Optional[datetime] = None

    # Gap between train and val/test (in days) to prevent lookahead
    gap_days: int = 0

    def __post_init__(self):
        """Validate split specification."""
        if self.train_start >= self.train_end:
            raise SplitError("train_start must be before train_end")

        # Validate validation period if specified
        if self.val_start is not None or self.val_end is not None:
            if self.val_start is None or self.val_end is None:
                raise SplitError("Both val_start and val_end must be specified together")
            if self.val_start >= self.val_end:
                raise SplitError("val_start must be before val_end")
            if self.val_start < self.train_end:
                raise SplitError("val_start must be >= train_end (no overlap)")

        # Validate test period if specified
        if self.test_start is not None or self.test_end is not None:
            if self.test_start is None or self.test_end is None:
                raise SplitError("Both test_start and test_end must be specified together")
            if self.test_start >= self.test_end:
                raise SplitError("test_start must be before test_end")
            if self.val_end is not None and self.test_start < self.val_end:
                raise SplitError("test_start must be >= val_end (no overlap)")
            elif self.val_end is None and self.test_start < self.train_end:
                raise SplitError("test_start must be >= train_end (no overlap)")

    def get_fit_window(self, for_split: Literal["train", "val", "test"]) -> FitWindow:
        """
        Get the appropriate fit window for a given split.

        For train: fit on train period
        For val/test: fit on train period (no future leakage)
        """
        if for_split == "train":
            return FitWindow(fit_start=self.train_start, fit_end=self.train_end)
        elif for_split in ("val", "test"):
            # Always fit on train data only - no future leakage
            return FitWindow(fit_start=self.train_start, fit_end=self.train_end)
        else:
            raise ValueError(f"Invalid split: {for_split}")


@dataclass(frozen=True)
class OutOfFoldSpec:
    """
    Specification for out-of-fold (OOF) cross-validation.

    Defines multiple folds with proper temporal ordering.
    """
    oof_id: str
    folds: List[SplitSpec]

    # Strategy
    strategy: Literal["rolling", "expanding", "anchored"] = "rolling"

    def __post_init__(self):
        """Validate OOF specification."""
        if not self.folds:
            raise SplitError("At least one fold required")

        # Validate temporal ordering of folds
        for i in range(len(self.folds) - 1):
            current_fold = self.folds[i]
            next_fold = self.folds[i + 1]

            # For expanding window, train_start can be the same but train_end must increase
            # For rolling window, train_start must increase
            if self.strategy == "expanding":
                # Expanding: train_start can be same/earlier, but train_end must increase
                if next_fold.train_end <= current_fold.train_end:
                    raise SplitError(
                        f"Fold {i+1} train_end must be after fold {i} train_end for expanding strategy"
                    )
            else:
                # Rolling/anchored: train_start must increase
                if next_fold.train_start <= current_fold.train_start:
                    raise SplitError(f"Fold {i+1} train_start must be after fold {i} train_start")

    def num_folds(self) -> int:
        """Return number of folds."""
        return len(self.folds)


class TransformMode(Enum):
    """Transform mode for preprocessing."""
    STATELESS = "stateless"
    FITTED = "fitted"


@dataclass(frozen=True)
class PreprocessContract:
    """
    Contract for preprocessing pipeline.

    Defines what transforms to apply and their temporal constraints.
    """
    contract_id: str

    # Transform specifications
    transforms: List[Dict[str, Any]] = field(default_factory=list)

    # Mode
    mode: TransformMode = TransformMode.STATELESS

    # If fitted, must specify fit window
    fit_window: Optional[FitWindow] = None

    # Requirements - explicit exposure types
    requires_universe: bool = False
    requires_industry: bool = False
    requires_size: bool = False

    def __post_init__(self):
        """Validate contract."""
        if self.mode == TransformMode.FITTED and self.fit_window is None:
            raise ContractViolation("FITTED mode requires fit_window")

        if self.mode == TransformMode.STATELESS and self.fit_window is not None:
            raise ContractViolation("STATELESS mode should not have fit_window")


@dataclass
class ModelReadyData:
    """
    Output format for model-ready data.

    Contains features, metadata, and provenance.
    """
    # Core data
    features: Any  # DataFrame or array-like
    feature_names: List[str]

    # Temporal metadata
    data_start: datetime
    data_end: datetime

    # Preprocessing metadata
    preprocess_contract_id: str
    fit_window: Optional[FitWindow] = None

    # Optional auxiliary channels
    missing_indicators: Optional[Any] = None
    exposure_residuals: Optional[Any] = None

    # Provenance
    created_at: Optional[datetime] = None
    producer: str = "modeling"
    producer_version: str = "0.1.0"

    def __post_init__(self):
        """Validate model ready data."""
        if self.data_start >= self.data_end:
            raise ContractViolation("data_start must be before data_end")
