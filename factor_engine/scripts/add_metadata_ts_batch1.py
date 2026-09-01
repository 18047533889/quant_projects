#!/usr/bin/env python3
"""Script to add metadata to all operators in ts_batch1.py"""

import re

# Define metadata for each operator based on its signature
OPERATOR_METADATA = {
    "TSMeanIfPolarsNative": {
        "name": "ts_mean_if",
        "description": "Conditional rolling mean - mean of values where condition is True",
        "params": ["feature", "condition", "window"],
        "param_specs": {"window": "ParamSpec(dtype=int, min=1, param_role=ParamRole.WINDOW)"},
    },
    "TSStdIfPolarsNative": {
        "name": "ts_std_if",
        "description": "Conditional rolling standard deviation",
        "params": ["feature", "condition", "window"],
        "param_specs": {"window": "ParamSpec(dtype=int, min=1, param_role=ParamRole.WINDOW)"},
    },
    "TSSumIfPolarsNative": {
        "name": "ts_sum_if",
        "description": "Conditional rolling sum",
        "params": ["feature", "condition", "window"],
        "param_specs": {"window": "ParamSpec(dtype=int, min=1, param_role=ParamRole.WINDOW)"},
    },
    "TSMinIfPolarsNative": {
        "name": "ts_min_if",
        "description": "Conditional rolling minimum",
        "params": ["feature", "condition", "window"],
        "param_specs": {"window": "ParamSpec(dtype=int, min=1, param_role=ParamRole.WINDOW)"},
    },
    "TSMaxIfPolarsNative": {
        "name": "ts_max_if",
        "description": "Conditional rolling maximum",
        "params": ["feature", "condition", "window"],
        "param_specs": {"window": "ParamSpec(dtype=int, min=1, param_role=ParamRole.WINDOW)"},
    },
    "TSMedianPolarsNative": {
        "name": "ts_median",
        "description": "Rolling median",
        "params": ["feature", "window"],
        "param_specs": {"window": "ParamSpec(dtype=int, min=1, param_role=ParamRole.WINDOW)"},
    },
    "TSQuantilePolarsNative": {
        "name": "ts_quantile",
        "description": "Rolling quantile",
        "params": ["feature", "window", "quantile"],
        "param_specs": {
            "window": "ParamSpec(dtype=int, min=1, param_role=ParamRole.WINDOW)",
            "quantile": "ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5, param_role=ParamRole.SCALAR)",
        },
    },
    "TSQuantileIfPolarsNative": {
        "name": "ts_quantile_if",
        "description": "Conditional rolling quantile",
        "params": ["feature", "condition", "window", "quantile"],
        "param_specs": {
            "window": "ParamSpec(dtype=int, min=1, param_role=ParamRole.WINDOW)",
            "quantile": "ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5, param_role=ParamRole.SCALAR)",
        },
    },
    "TSRankIfPolarsNative": {
        "name": "ts_rank_if",
        "description": "Conditional rolling rank (current value rank in window)",
        "params": ["feature", "condition", "window"],
        "param_specs": {"window": "ParamSpec(dtype=int, min=1, param_role=ParamRole.WINDOW)"},
    },
    "TSCorrPolarsNative": {
        "name": "ts_corr",
        "description": "Rolling correlation between two series",
        "params": ["x", "y", "window"],
        "param_specs": {"window": "ParamSpec(dtype=int, min=2, param_role=ParamRole.WINDOW)"},
    },
    "TSCovPolarsNative": {
        "name": "ts_cov",
        "description": "Rolling covariance between two series",
        "params": ["x", "y", "window"],
        "param_specs": {"window": "ParamSpec(dtype=int, min=2, param_role=ParamRole.WINDOW)"},
    },
    "TSCorrIfPolarsNative": {
        "name": "ts_corr_if",
        "description": "Conditional rolling correlation",
        "params": ["x", "y", "condition", "window"],
        "param_specs": {"window": "ParamSpec(dtype=int, min=2, param_role=ParamRole.WINDOW)"},
    },
    "TSCovIfPolarsNative": {
        "name": "ts_cov_if",
        "description": "Conditional rolling covariance",
        "params": ["x", "y", "condition", "window"],
        "param_specs": {"window": "ParamSpec(dtype=int, min=2, param_role=ParamRole.WINDOW)"},
    },
    "TSNthValuePolarsNative": {
        "name": "ts_nth_value",
        "description": "Get nth value from current position (negative=past, positive=future)",
        "params": ["feature", "n"],
        "param_specs": {"n": "ParamSpec(dtype=int, param_role=ParamRole.SCALAR)"},
    },
    "TSLastIfPolarsNative": {
        "name": "ts_last_if",
        "description": "Get last value where condition was True within window",
        "params": ["feature", "condition", "window"],
        "param_specs": {"window": "ParamSpec(dtype=int, min=1, param_role=ParamRole.WINDOW)"},
    },
    "TSCountIfPolarsNative": {
        "name": "ts_count_if",
        "description": "Count of True values in rolling window",
        "params": ["condition", "window"],
        "param_specs": {"window": "ParamSpec(dtype=int, min=1, param_role=ParamRole.WINDOW)"},
    },
    "TSValidCountPolarsNative": {
        "name": "ts_valid_count",
        "description": "Count of non-null values in rolling window",
        "params": ["feature", "window"],
        "param_specs": {"window": "ParamSpec(dtype=int, min=1, param_role=ParamRole.WINDOW)"},
    },
    "TSCoverageRatioPolarsNative": {
        "name": "ts_coverage_ratio",
        "description": "Ratio of non-null values in rolling window",
        "params": ["feature", "window"],
        "param_specs": {"window": "ParamSpec(dtype=int, min=1, param_role=ParamRole.WINDOW)"},
    },
    "TSPositiveRatioPolarsNative": {
        "name": "ts_positive_ratio",
        "description": "Ratio of positive values in rolling window",
        "params": ["feature", "window"],
        "param_specs": {"window": "ParamSpec(dtype=int, min=1, param_role=ParamRole.WINDOW)"},
    },
    "TSNegativeRatioPolarsNative": {
        "name": "ts_negative_ratio",
        "description": "Ratio of negative values in rolling window",
        "params": ["feature", "window"],
        "param_specs": {"window": "ParamSpec(dtype=int, min=1, param_role=ParamRole.WINDOW)"},
    },
    "TSZeroRatioPolarsNative": {
        "name": "ts_zero_ratio",
        "description": "Ratio of zero values in rolling window",
        "params": ["feature", "window"],
        "param_specs": {"window": "ParamSpec(dtype=int, min=1, param_role=ParamRole.WINDOW)"},
    },
}

# Continue with more operators...
print("Metadata definitions created. Now need to apply them to the file.")
