"""Schema validation contracts using Pydantic.

Provides runtime schema validation for:
- FactorBatch: computed factor values with metadata
- FeatureBundle: feature vectors for model training
- LabelBundle: labels for supervised learning
- PredictionBatch: model predictions

All validators support strict mode for production environments.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np

try:
    from pydantic import BaseModel, Field, field_validator, ConfigDict
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore


__all__ = [
    "ValidationConfig",
    "FactorBatchSchema",
    "FeatureBundleSchema",
    "LabelBundleSchema",
    "PredictionBatchSchema",
    "validate_factor_batch",
    "validate_feature_bundle",
    "validate_label_bundle",
    "validate_prediction_batch",
]


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for data validation behavior."""

    strict: bool = False
    allow_nan: bool = True
    allow_inf: bool = False
    max_nan_fraction: float = 0.5
    check_sorted: bool = True
    check_duplicates: bool = True
    check_dtypes: bool = True


def _check_pydantic() -> None:
    if not PYDANTIC_AVAILABLE:
        raise ImportError(
            "pydantic>=2.0 is required for validation contracts. "
            "Install with: pip install pydantic>=2.0"
        )


if PYDANTIC_AVAILABLE:
    class FactorBatchSchema(BaseModel):
        """Schema for factor batch data.

        Validates:
        - values: numeric array with shape (n_timestamps, n_instruments)
        - timestamps: sorted sequence of dates
        - instruments: sorted sequence of stock codes
        - factor_name: non-empty identifier
        - metadata: optional computation metadata
        """

        model_config = ConfigDict(arbitrary_types_allowed=True)

        factor_name: str = Field(..., min_length=1, max_length=256)
        timestamps: list[str | date] = Field(..., min_length=1)
        instruments: list[str] = Field(..., min_length=1)
        values: Any = Field(...)  # np.ndarray validated separately
        metadata: dict[str, Any] = Field(default_factory=dict)
        computation_time: datetime | None = None
        semantic_version: str = Field(default="1.0.0")

        @field_validator("factor_name")
        @classmethod
        def validate_factor_name(cls, v: str) -> str:
            if not v or v.startswith("_"):
                raise ValueError("factor_name must be non-empty and not start with underscore")
            return v

        @field_validator("timestamps")
        @classmethod
        def validate_timestamps(cls, v: list) -> list:
            if not v:
                raise ValueError("timestamps must not be empty")
            # Check sortedness
            str_vals = [str(x) for x in v]
            if str_vals != sorted(str_vals):
                raise ValueError("timestamps must be sorted")
            # Check duplicates
            if len(str_vals) != len(set(str_vals)):
                raise ValueError("timestamps must not contain duplicates")
            return v

        @field_validator("instruments")
        @classmethod
        def validate_instruments(cls, v: list[str]) -> list[str]:
            if not v:
                raise ValueError("instruments must not be empty")
            # Check sortedness
            if v != sorted(v):
                raise ValueError("instruments must be sorted")
            # Check duplicates
            if len(v) != len(set(v)):
                raise ValueError("instruments must not contain duplicates")
            return v

        @field_validator("values")
        @classmethod
        def validate_values(cls, v: Any) -> Any:
            arr = np.asarray(v)
            if arr.ndim != 2:
                raise ValueError(f"values must be 2D array, got {arr.ndim}D")
            if not np.issubdtype(arr.dtype, np.number):
                raise ValueError(f"values must be numeric, got dtype {arr.dtype}")
            return arr

        def validate_shape(self) -> None:
            """Validate that values shape matches timestamps × instruments."""
            arr = np.asarray(self.values)
            expected_shape = (len(self.timestamps), len(self.instruments))
            if arr.shape != expected_shape:
                raise ValueError(
                    f"values shape {arr.shape} does not match "
                    f"timestamps×instruments {expected_shape}"
                )

        def validate_finiteness(self, config: ValidationConfig) -> None:
            """Validate numeric properties according to config."""
            arr = np.asarray(self.values)

            if not config.allow_inf and np.isinf(arr).any():
                raise ValueError("values contain inf but allow_inf=False")

            if not config.allow_nan:
                if np.isnan(arr).any():
                    raise ValueError("values contain NaN but allow_nan=False")
            else:
                nan_fraction = np.isnan(arr).sum() / arr.size
                if nan_fraction > config.max_nan_fraction:
                    raise ValueError(
                        f"NaN fraction {nan_fraction:.2%} exceeds max "
                        f"{config.max_nan_fraction:.2%}"
                    )


    class FeatureBundleSchema(BaseModel):
        """Schema for feature bundle.

        Validates feature vectors for model training with metadata.
        """

        model_config = ConfigDict(arbitrary_types_allowed=True)

        canonical: str = Field(..., min_length=1)
        operator_semantic_version: str = Field(default="1.0.0")
        params: dict[str, Any] = Field(default_factory=dict)
        normalized_ast_hash: str = Field(..., min_length=1)
        source_snapshot: str = Field(..., min_length=1)
        semantic_version: str = Field(default="1.0.0")
        values: Any | None = None
        row_ids: list[str] = Field(default_factory=list)

        @field_validator("canonical")
        @classmethod
        def validate_canonical(cls, v: str) -> str:
            if not v or v.isspace():
                raise ValueError("canonical must be non-empty")
            return v

        @field_validator("normalized_ast_hash", "source_snapshot")
        @classmethod
        def validate_hash_fields(cls, v: str) -> str:
            if not v or len(v) < 8:
                raise ValueError("hash field must be at least 8 characters")
            return v

        @field_validator("values")
        @classmethod
        def validate_values(cls, v: Any) -> Any:
            if v is None:
                return v
            arr = np.asarray(v)
            if not np.issubdtype(arr.dtype, np.number):
                raise ValueError(f"values must be numeric, got dtype {arr.dtype}")
            return arr

        def validate_alignment(self) -> None:
            """Validate that values align with row_ids."""
            if self.values is not None and self.row_ids:
                arr = np.asarray(self.values)
                if len(arr) != len(self.row_ids):
                    raise ValueError(
                        f"values length {len(arr)} does not match "
                        f"row_ids length {len(self.row_ids)}"
                    )


    class LabelBundleSchema(BaseModel):
        """Schema for label bundle.

        Validates labels for supervised learning with horizon and basis.
        """

        model_config = ConfigDict(arbitrary_types_allowed=True)

        label_name: str = Field(..., min_length=1)
        horizon_bars: int = Field(..., ge=1)
        return_basis: str = Field(default="vwap_to_vwap")
        values: Any | None = None
        row_ids: list[str] = Field(default_factory=list)
        timestamps: list[str | date] = Field(default_factory=list)
        availability_time_rule: str = Field(default="label matured at t+H close")
        overlapping: bool = False

        @field_validator("label_name")
        @classmethod
        def validate_label_name(cls, v: str) -> str:
            if not v or v.isspace():
                raise ValueError("label_name must be non-empty")
            return v

        @field_validator("values")
        @classmethod
        def validate_values(cls, v: Any) -> Any:
            if v is None:
                return v
            arr = np.asarray(v)
            if not np.issubdtype(arr.dtype, np.number):
                raise ValueError(f"values must be numeric, got dtype {arr.dtype}")
            return arr

        def validate_alignment(self) -> None:
            """Validate that values align with row_ids."""
            if self.values is not None and self.row_ids:
                arr = np.asarray(self.values)
                if len(arr) != len(self.row_ids):
                    raise ValueError(
                        f"values length {len(arr)} does not match "
                        f"row_ids length {len(self.row_ids)}"
                    )


    class PredictionBatchSchema(BaseModel):
        """Schema for prediction batch.

        Validates model predictions with status tracking.
        """

        model_config = ConfigDict(arbitrary_types_allowed=True)

        values: Any
        row_ids: list[str] = Field(..., min_length=1)
        timestamps: list[str | date] = Field(default_factory=list)
        status: str = Field(default="ok")
        model_version: str | None = None
        artifact_id: str | None = None

        @field_validator("values")
        @classmethod
        def validate_values(cls, v: Any) -> Any:
            arr = np.asarray(v)
            if arr.ndim != 1:
                raise ValueError(f"prediction values must be 1D, got {arr.ndim}D")
            if not np.issubdtype(arr.dtype, np.number):
                raise ValueError(f"values must be numeric, got dtype {arr.dtype}")
            return arr

        @field_validator("status")
        @classmethod
        def validate_status(cls, v: str) -> str:
            valid_statuses = {"ok", "missing_feature", "out_of_universe",
                            "no_active_artifact", "numerical_failure", "clock_violation"}
            if v not in valid_statuses:
                raise ValueError(f"status must be one of {valid_statuses}, got {v!r}")
            return v

        def validate_alignment(self) -> None:
            """Validate that values align with row_ids."""
            arr = np.asarray(self.values)
            if len(arr) != len(self.row_ids):
                raise ValueError(
                    f"values length {len(arr)} does not match "
                    f"row_ids length {len(self.row_ids)}"
                )
            if self.timestamps and len(self.timestamps) != len(self.row_ids):
                raise ValueError(
                    f"timestamps length {len(self.timestamps)} does not match "
                    f"row_ids length {len(self.row_ids)}"
                )

        def validate_finiteness(self, allow_nan: bool = False) -> None:
            """Validate that predictions are finite."""
            arr = np.asarray(self.values)
            if not allow_nan and not np.isfinite(arr).all():
                raise ValueError("prediction values must be finite when allow_nan=False")


else:
    # Stub classes when pydantic is not available
    class FactorBatchSchema:  # type: ignore
        pass

    class FeatureBundleSchema:  # type: ignore
        pass

    class LabelBundleSchema:  # type: ignore
        pass

    class PredictionBatchSchema:  # type: ignore
        pass


def validate_factor_batch(
    factor_name: str,
    timestamps: list,
    instruments: list[str],
    values: Any,
    metadata: dict[str, Any] | None = None,
    config: ValidationConfig | None = None,
) -> FactorBatchSchema:
    """Validate a factor batch against schema and quality constraints.

    Parameters
    ----------
    factor_name : str
        Factor identifier
    timestamps : list
        Sorted sequence of dates
    instruments : list[str]
        Sorted sequence of stock codes
    values : array-like
        Factor values with shape (n_timestamps, n_instruments)
    metadata : dict, optional
        Computation metadata
    config : ValidationConfig, optional
        Validation configuration

    Returns
    -------
    FactorBatchSchema
        Validated schema instance

    Raises
    ------
    ValueError
        If validation fails
    ImportError
        If pydantic is not installed
    """
    _check_pydantic()

    config = config or ValidationConfig()
    batch = FactorBatchSchema(
        factor_name=factor_name,
        timestamps=timestamps,
        instruments=instruments,
        values=values,
        metadata=metadata or {},
    )

    # Additional validations
    batch.validate_shape()
    if config.strict:
        batch.validate_finiteness(config)

    return batch


def validate_feature_bundle(
    canonical: str,
    operator_semantic_version: str,
    params: dict[str, Any],
    normalized_ast_hash: str,
    source_snapshot: str,
    semantic_version: str = "1.0.0",
    values: Any | None = None,
    row_ids: list[str] | None = None,
) -> FeatureBundleSchema:
    """Validate a feature bundle.

    Parameters
    ----------
    canonical : str
        Operator canonical name
    operator_semantic_version : str
        Operator version
    params : dict
        Operator parameters
    normalized_ast_hash : str
        AST hash for identity
    source_snapshot : str
        Source data snapshot identifier
    semantic_version : str
        Feature semantic version
    values : array-like, optional
        Feature values
    row_ids : list[str], optional
        Row identifiers

    Returns
    -------
    FeatureBundleSchema
        Validated schema instance
    """
    _check_pydantic()

    bundle = FeatureBundleSchema(
        canonical=canonical,
        operator_semantic_version=operator_semantic_version,
        params=params,
        normalized_ast_hash=normalized_ast_hash,
        source_snapshot=source_snapshot,
        semantic_version=semantic_version,
        values=values,
        row_ids=row_ids or [],
    )

    bundle.validate_alignment()
    return bundle


def validate_label_bundle(
    label_name: str,
    horizon_bars: int,
    return_basis: str = "vwap_to_vwap",
    values: Any | None = None,
    row_ids: list[str] | None = None,
    timestamps: list | None = None,
) -> LabelBundleSchema:
    """Validate a label bundle.

    Parameters
    ----------
    label_name : str
        Label identifier
    horizon_bars : int
        Prediction horizon in bars
    return_basis : str
        Return calculation basis
    values : array-like, optional
        Label values
    row_ids : list[str], optional
        Row identifiers
    timestamps : list, optional
        Timestamps for each label

    Returns
    -------
    LabelBundleSchema
        Validated schema instance
    """
    _check_pydantic()

    bundle = LabelBundleSchema(
        label_name=label_name,
        horizon_bars=horizon_bars,
        return_basis=return_basis,
        values=values,
        row_ids=row_ids or [],
        timestamps=timestamps or [],
    )

    bundle.validate_alignment()
    return bundle


def validate_prediction_batch(
    values: Any,
    row_ids: list[str],
    timestamps: list | None = None,
    status: str = "ok",
    model_version: str | None = None,
    artifact_id: str | None = None,
    allow_nan: bool = False,
) -> PredictionBatchSchema:
    """Validate a prediction batch.

    Parameters
    ----------
    values : array-like
        Prediction values
    row_ids : list[str]
        Row identifiers
    timestamps : list, optional
        Timestamps for each prediction
    status : str
        Prediction status
    model_version : str, optional
        Model version identifier
    artifact_id : str, optional
        Artifact identifier
    allow_nan : bool
        Whether to allow NaN values

    Returns
    -------
    PredictionBatchSchema
        Validated schema instance
    """
    _check_pydantic()

    batch = PredictionBatchSchema(
        values=values,
        row_ids=row_ids,
        timestamps=timestamps or [],
        status=status,
        model_version=model_version,
        artifact_id=artifact_id,
    )

    batch.validate_alignment()
    batch.validate_finiteness(allow_nan=allow_nan)
    return batch
