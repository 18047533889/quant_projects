"""Runtime data validation for factor computation pipelines.

Provides three complementary validation layers:

1. Schema Contracts (validation.contracts)
   - Pydantic-based validation for FactorBatch, FeatureBundle, LabelBundle, PredictionBatch
   - Requires pydantic>=2.0

2. Quality Checks (validation.checks)
   - Statistical validation: NaN rates, outliers, distribution checks
   - Requires only numpy

3. Input Sanitization (validation.sanitize)
   - Defensive data cleaning and normalization
   - Requires only numpy

All validation is opt-in with configurable strict mode for production.

Quick Start
-----------

Schema validation:
    >>> from validation import validate_factor_batch, ValidationConfig
    >>> batch = validate_factor_batch(
    ...     factor_name="momentum_20d",
    ...     timestamps=["2024-01-01", "2024-01-02"],
    ...     instruments=["000001.SZ", "000002.SZ"],
    ...     values=np.random.randn(2, 2),
    ... )

Quality checks:
    >>> from validation import check_data_quality
    >>> report = check_data_quality(values, max_nan_fraction=0.5)
    >>> print(report.summary())

Input sanitization:
    >>> from validation import sanitize_factor_inputs
    >>> result = sanitize_factor_inputs(
    ...     values,
    ...     replace_inf="clip",
    ...     winsorize=(0.01, 0.99),
    ...     normalize=True,
    ... )
    >>> clean_values = result.sanitized

See validation/README.md for comprehensive documentation.
"""

# Re-export all public APIs

# Quality checks (core, no extra dependencies)
from validation.checks import (
    DataQualityReport,
    OutlierStats,
    check_data_quality,
    check_distribution,
    check_nan_rates,
    check_outliers,
)

# Input sanitization (core, no extra dependencies)
from validation.sanitize import (
    SanitizationResult,
    sanitize_dataframe,
    sanitize_factor_inputs,
    sanitize_numeric_array,
)

# Schema contracts (requires pydantic>=2.0)
try:
    from validation.contracts import (
        FactorBatchSchema,
        FeatureBundleSchema,
        LabelBundleSchema,
        PredictionBatchSchema,
        ValidationConfig,
        validate_factor_batch,
        validate_feature_bundle,
        validate_label_bundle,
        validate_prediction_batch,
    )
    _CONTRACTS_AVAILABLE = True
except ImportError:
    _CONTRACTS_AVAILABLE = False
    # Provide helpful error message if contracts are imported
    def _contracts_unavailable(*args, **kwargs):
        raise ImportError(
            "Schema validation requires pydantic>=2.0. "
            "Install with: pip install pydantic>=2.0 or pip install factor-engine[validation]"
        )

    FactorBatchSchema = _contracts_unavailable
    FeatureBundleSchema = _contracts_unavailable
    LabelBundleSchema = _contracts_unavailable
    PredictionBatchSchema = _contracts_unavailable
    ValidationConfig = _contracts_unavailable
    validate_factor_batch = _contracts_unavailable
    validate_feature_bundle = _contracts_unavailable
    validate_label_bundle = _contracts_unavailable
    validate_prediction_batch = _contracts_unavailable


__all__ = [
    # Quality checks
    "DataQualityReport",
    "OutlierStats",
    "check_data_quality",
    "check_distribution",
    "check_nan_rates",
    "check_outliers",
    # Sanitization
    "SanitizationResult",
    "sanitize_dataframe",
    "sanitize_factor_inputs",
    "sanitize_numeric_array",
    # Schema contracts (may not be available)
    "FactorBatchSchema",
    "FeatureBundleSchema",
    "LabelBundleSchema",
    "PredictionBatchSchema",
    "ValidationConfig",
    "validate_factor_batch",
    "validate_feature_bundle",
    "validate_label_bundle",
    "validate_prediction_batch",
]

__version__ = "0.1.0"
