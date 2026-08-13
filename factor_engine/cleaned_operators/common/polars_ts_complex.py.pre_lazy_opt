# -*- coding: utf-8 -*-
"""Complex time-series operators - Polars native implementations (SKELETONS).

This module contains skeleton implementations for advanced time-series operators.
Complex algorithms are marked with TODO and return placeholder values.
"""
from __future__ import annotations

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.base import ParamRole, ParamSpec

_SKIP = frozenset({"date", "stock_code"})
_SRC = "factor_dsl_polars_native"


def _numeric_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _SKIP]


def _with_meta(result: pl.DataFrame, source: pl.DataFrame) -> pl.DataFrame:
    if "date" in source.columns and "date" not in result.columns:
        result = result.with_columns(source["date"])
    return result


# ==============================================================================
# Kalman Filter Operators
# ==============================================================================

@register_operator(
    name="ts_kalman_filter",
    category="time_series",
    business_category="state_space",
    canonical="ts_kalman_filter",
    source=_SRC,
    backend="polars",
)
class TSKalmanFilterNative(SeriesOperator):
    """Kalman filter state estimate."""

    metadata = OperatorMetadata(
        name="ts_kalman_filter",
        category="time_series",
        description="卡尔曼滤波",
        param_names=["x", "process_noise", "measurement_noise"],
        return_type="series",
        tags=["time_series", "kalman", "polars", "native"],
        param_specs={
            "process_noise": ParamSpec(dtype=float, min=0.0, default=0.01, searchable=True, param_role=ParamRole.NUMERICAL),
            "measurement_noise": ParamSpec(dtype=float, min=0.0, default=0.1, searchable=True, param_role=ParamRole.NUMERICAL),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, process_noise: float = 0.01, measurement_noise: float = 0.1, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar
        q = strict_finite_scalar(process_noise, "process_noise", minimum=0.0)
        r = strict_finite_scalar(measurement_noise, "measurement_noise", minimum=0.0)

        # TODO: Implement Kalman filter algorithm
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_kalman_gain",
    category="time_series",
    business_category="state_space",
    canonical="ts_kalman_gain",
    source=_SRC,
    backend="polars",
)
class TSKalmanGainNative(SeriesOperator):
    """Kalman gain sequence."""

    metadata = OperatorMetadata(
        name="ts_kalman_gain",
        category="time_series",
        description="卡尔曼增益",
        param_names=["x", "process_noise", "measurement_noise"],
        return_type="series",
        tags=["time_series", "kalman", "polars", "native"],
        param_specs={
            "process_noise": ParamSpec(dtype=float, min=0.0, default=0.01, searchable=True, param_role=ParamRole.NUMERICAL),
            "measurement_noise": ParamSpec(dtype=float, min=0.0, default=0.1, searchable=True, param_role=ParamRole.NUMERICAL),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, process_noise: float = 0.01, measurement_noise: float = 0.1, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar
        q = strict_finite_scalar(process_noise, "process_noise", minimum=0.0)
        r = strict_finite_scalar(measurement_noise, "measurement_noise", minimum=0.0)

        # TODO: Implement Kalman gain calculation
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_kalman_innovation",
    category="time_series",
    business_category="state_space",
    canonical="ts_kalman_innovation",
    source=_SRC,
    backend="polars",
)
class TSKalmanInnovationNative(SeriesOperator):
    """Kalman filter innovation sequence."""

    metadata = OperatorMetadata(
        name="ts_kalman_innovation",
        category="time_series",
        description="卡尔曼新息",
        param_names=["x", "process_noise", "measurement_noise"],
        return_type="series",
        tags=["time_series", "kalman", "polars", "native"],
        param_specs={
            "process_noise": ParamSpec(dtype=float, min=0.0, default=0.01, searchable=True, param_role=ParamRole.NUMERICAL),
            "measurement_noise": ParamSpec(dtype=float, min=0.0, default=0.1, searchable=True, param_role=ParamRole.NUMERICAL),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, process_noise: float = 0.01, measurement_noise: float = 0.1, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar
        q = strict_finite_scalar(process_noise, "process_noise", minimum=0.0)
        r = strict_finite_scalar(measurement_noise, "measurement_noise", minimum=0.0)

        # TODO: Implement Kalman innovation
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_kalman_smoothed",
    category="time_series",
    business_category="state_space",
    canonical="ts_kalman_smoothed",
    source=_SRC,
    backend="polars",
)
class TSKalmanSmoothedNative(SeriesOperator):
    """Kalman smoother (backward pass)."""

    metadata = OperatorMetadata(
        name="ts_kalman_smoothed",
        category="time_series",
        description="卡尔曼平滑",
        param_names=["x", "process_noise", "measurement_noise"],
        return_type="series",
        tags=["time_series", "kalman", "polars", "native"],
        param_specs={
            "process_noise": ParamSpec(dtype=float, min=0.0, default=0.01, searchable=True, param_role=ParamRole.NUMERICAL),
            "measurement_noise": ParamSpec(dtype=float, min=0.0, default=0.1, searchable=True, param_role=ParamRole.NUMERICAL),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, process_noise: float = 0.01, measurement_noise: float = 0.1, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar
        q = strict_finite_scalar(process_noise, "process_noise", minimum=0.0)
        r = strict_finite_scalar(measurement_noise, "measurement_noise", minimum=0.0)

        # TODO: Implement Kalman smoother
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


# ==============================================================================
# GARCH / Volatility Models
# ==============================================================================

@register_operator(
    name="ts_garch_volatility",
    category="time_series",
    business_category="volatility",
    canonical="ts_garch_volatility",
    source=_SRC,
    backend="polars",
)
class TSGarchVolatilityNative(SeriesOperator):
    """GARCH(1,1) conditional volatility."""

    metadata = OperatorMetadata(
        name="ts_garch_volatility",
        category="time_series",
        description="GARCH波动率",
        param_names=["x", "omega", "alpha", "beta"],
        return_type="series",
        tags=["time_series", "garch", "polars", "native"],
        param_specs={
            "omega": ParamSpec(dtype=float, min=0.0, default=0.01, searchable=True, param_role=ParamRole.NUMERICAL),
            "alpha": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.1, searchable=True, param_role=ParamRole.NUMERICAL),
            "beta": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.85, searchable=True, param_role=ParamRole.NUMERICAL),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, omega: float = 0.01, alpha: float = 0.1, beta: float = 0.85, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar
        w = strict_finite_scalar(omega, "omega", minimum=0.0)
        a = strict_finite_scalar(alpha, "alpha", minimum=0.0, maximum=1.0)
        b = strict_finite_scalar(beta, "beta", minimum=0.0, maximum=1.0)

        # TODO: Implement GARCH(1,1) algorithm
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_garch_standardized_resid",
    category="time_series",
    business_category="volatility",
    canonical="ts_garch_standardized_resid",
    source=_SRC,
    backend="polars",
)
class TSGarchStandardizedResidNative(SeriesOperator):
    """GARCH standardized residuals."""

    metadata = OperatorMetadata(
        name="ts_garch_standardized_resid",
        category="time_series",
        description="GARCH标准化残差",
        param_names=["x", "omega", "alpha", "beta"],
        return_type="series",
        tags=["time_series", "garch", "polars", "native"],
        param_specs={
            "omega": ParamSpec(dtype=float, min=0.0, default=0.01, searchable=True, param_role=ParamRole.NUMERICAL),
            "alpha": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.1, searchable=True, param_role=ParamRole.NUMERICAL),
            "beta": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.85, searchable=True, param_role=ParamRole.NUMERICAL),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, omega: float = 0.01, alpha: float = 0.1, beta: float = 0.85, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar
        w = strict_finite_scalar(omega, "omega", minimum=0.0)
        a = strict_finite_scalar(alpha, "alpha", minimum=0.0, maximum=1.0)
        b = strict_finite_scalar(beta, "beta", minimum=0.0, maximum=1.0)

        # TODO: Implement GARCH standardized residuals
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_gjr_garch_volatility",
    category="time_series",
    business_category="volatility",
    canonical="ts_gjr_garch_volatility",
    source=_SRC,
    backend="polars",
)
class TSGJRGarchVolatilityNative(SeriesOperator):
    """GJR-GARCH asymmetric volatility."""

    metadata = OperatorMetadata(
        name="ts_gjr_garch_volatility",
        category="time_series",
        description="GJR-GARCH波动率",
        param_names=["x", "omega", "alpha", "gamma", "beta"],
        return_type="series",
        tags=["time_series", "garch", "polars", "native"],
        param_specs={
            "omega": ParamSpec(dtype=float, min=0.0, default=0.01, searchable=True, param_role=ParamRole.NUMERICAL),
            "alpha": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.05, searchable=True, param_role=ParamRole.NUMERICAL),
            "gamma": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.1, searchable=True, param_role=ParamRole.NUMERICAL),
            "beta": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.85, searchable=True, param_role=ParamRole.NUMERICAL),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, omega: float = 0.01, alpha: float = 0.05,
                         gamma: float = 0.1, beta: float = 0.85, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar
        w = strict_finite_scalar(omega, "omega", minimum=0.0)
        a = strict_finite_scalar(alpha, "alpha", minimum=0.0, maximum=1.0)
        g = strict_finite_scalar(gamma, "gamma", minimum=0.0, maximum=1.0)
        b = strict_finite_scalar(beta, "beta", minimum=0.0, maximum=1.0)

        # TODO: Implement GJR-GARCH algorithm
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_har_volatility",
    category="time_series",
    business_category="volatility",
    canonical="ts_har_volatility",
    source=_SRC,
    backend="polars",
)
class TSHARVolatilityNative(SeriesOperator):
    """HAR (Heterogeneous AutoRegressive) volatility."""

    metadata = OperatorMetadata(
        name="ts_har_volatility",
        category="time_series",
        description="HAR波动率",
        param_names=["x", "daily_window", "weekly_window", "monthly_window"],
        return_type="series",
        tags=["time_series", "volatility", "polars", "native"],
        param_specs={
            "daily_window": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
            "weekly_window": ParamSpec(dtype=int, min=1, default=5, searchable=True, param_role=ParamRole.HORIZON),
            "monthly_window": ParamSpec(dtype=int, min=1, default=22, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, daily_window: int = 1, weekly_window: int = 5,
                         monthly_window: int = 22, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        dw = strict_integer(daily_window, "daily_window", minimum=1)
        ww = strict_integer(weekly_window, "weekly_window", minimum=1)
        mw = strict_integer(monthly_window, "monthly_window", minimum=1)

        # TODO: Implement HAR volatility
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


# ==============================================================================
# Spectral / Wavelet Operators
# ==============================================================================

@register_operator(
    name="ts_spectral_density",
    category="time_series",
    business_category="spectral",
    canonical="ts_spectral_density",
    source=_SRC,
    backend="polars",
)
class TSSpectralDensityNative(SeriesOperator):
    """Power spectral density at dominant frequency."""

    metadata = OperatorMetadata(
        name="ts_spectral_density",
        category="time_series",
        description="谱密度",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "spectral", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=8, default=64, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 64, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=8)

        # TODO: Implement spectral density calculation
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_spectral_centroid",
    category="time_series",
    business_category="spectral",
    canonical="ts_spectral_centroid",
    source=_SRC,
    backend="polars",
)
class TSSpectralCentroidNative(SeriesOperator):
    """Spectral centroid (center of mass of spectrum)."""

    metadata = OperatorMetadata(
        name="ts_spectral_centroid",
        category="time_series",
        description="谱质心",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "spectral", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=8, default=64, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 64, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=8)

        # TODO: Implement spectral centroid
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_spectral_entropy",
    category="time_series",
    business_category="spectral",
    canonical="ts_spectral_entropy",
    source=_SRC,
    backend="polars",
)
class TSSpectralEntropyNative(SeriesOperator):
    """Spectral entropy (frequency domain disorder)."""

    metadata = OperatorMetadata(
        name="ts_spectral_entropy",
        category="time_series",
        description="谱熵",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "spectral", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=8, default=64, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 64, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=8)

        # TODO: Implement spectral entropy
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_wavelet_energy",
    category="time_series",
    business_category="wavelet",
    canonical="ts_wavelet_energy",
    source=_SRC,
    backend="polars",
)
class TSWaveletEnergyNative(SeriesOperator):
    """Wavelet decomposition energy at specified scale."""

    metadata = OperatorMetadata(
        name="ts_wavelet_energy",
        category="time_series",
        description="小波能量",
        param_names=["x", "scale", "window"],
        return_type="series",
        tags=["time_series", "wavelet", "polars", "native"],
        param_specs={
            "scale": ParamSpec(dtype=int, min=1, default=2, searchable=True, param_role=ParamRole.SUPPORT_POLICY),
            "window": ParamSpec(dtype=int, min=16, default=64, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, scale: int = 2, window: int = 64, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        s = strict_integer(scale, "scale", minimum=1)
        w = strict_integer(window, "window", minimum=16)

        # TODO: Implement wavelet energy
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_wavelet_variance",
    category="time_series",
    business_category="wavelet",
    canonical="ts_wavelet_variance",
    source=_SRC,
    backend="polars",
)
class TSWaveletVarianceNative(SeriesOperator):
    """Wavelet variance at specified scale."""

    metadata = OperatorMetadata(
        name="ts_wavelet_variance",
        category="time_series",
        description="小波方差",
        param_names=["x", "scale", "window"],
        return_type="series",
        tags=["time_series", "wavelet", "polars", "native"],
        param_specs={
            "scale": ParamSpec(dtype=int, min=1, default=2, searchable=True, param_role=ParamRole.SUPPORT_POLICY),
            "window": ParamSpec(dtype=int, min=16, default=64, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, scale: int = 2, window: int = 64, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        s = strict_integer(scale, "scale", minimum=1)
        w = strict_integer(window, "window", minimum=16)

        # TODO: Implement wavelet variance
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


# ==============================================================================
# Entropy / Information Theory Operators
# ==============================================================================

@register_operator(
    name="ts_sample_entropy",
    category="time_series",
    business_category="entropy",
    canonical="ts_sample_entropy",
    source=_SRC,
    backend="polars",
)
class TSSampleEntropyNative(SeriesOperator):
    """Sample entropy measure of time series regularity."""

    metadata = OperatorMetadata(
        name="ts_sample_entropy",
        category="time_series",
        description="样本熵",
        param_names=["x", "m", "r", "window"],
        return_type="series",
        tags=["time_series", "entropy", "polars", "native"],
        param_specs={
            "m": ParamSpec(dtype=int, min=1, default=2, searchable=True, param_role=ParamRole.SUPPORT_POLICY),
            "r": ParamSpec(dtype=float, min=0.0, default=0.2, searchable=True, param_role=ParamRole.NUMERICAL),
            "window": ParamSpec(dtype=int, min=10, default=100, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, m: int = 2, r: float = 0.2, window: int = 100, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer, strict_finite_scalar
        m_val = strict_integer(m, "m", minimum=1)
        r_val = strict_finite_scalar(r, "r", minimum=0.0)
        w = strict_integer(window, "window", minimum=10)

        # TODO: Implement sample entropy algorithm
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_approximate_entropy",
    category="time_series",
    business_category="entropy",
    canonical="ts_approximate_entropy",
    source=_SRC,
    backend="polars",
)
class TSApproximateEntropyNative(SeriesOperator):
    """Approximate entropy (ApEn)."""

    metadata = OperatorMetadata(
        name="ts_approximate_entropy",
        category="time_series",
        description="近似熵",
        param_names=["x", "m", "r", "window"],
        return_type="series",
        tags=["time_series", "entropy", "polars", "native"],
        param_specs={
            "m": ParamSpec(dtype=int, min=1, default=2, searchable=True, param_role=ParamRole.SUPPORT_POLICY),
            "r": ParamSpec(dtype=float, min=0.0, default=0.2, searchable=True, param_role=ParamRole.NUMERICAL),
            "window": ParamSpec(dtype=int, min=10, default=100, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, m: int = 2, r: float = 0.2, window: int = 100, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer, strict_finite_scalar
        m_val = strict_integer(m, "m", minimum=1)
        r_val = strict_finite_scalar(r, "r", minimum=0.0)
        w = strict_integer(window, "window", minimum=10)

        # TODO: Implement approximate entropy
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_permutation_entropy",
    category="time_series",
    business_category="entropy",
    canonical="ts_permutation_entropy",
    source=_SRC,
    backend="polars",
)
class TSPermutationEntropyNative(SeriesOperator):
    """Permutation entropy based on ordinal patterns."""

    metadata = OperatorMetadata(
        name="ts_permutation_entropy",
        category="time_series",
        description="排列熵",
        param_names=["x", "order", "window"],
        return_type="series",
        tags=["time_series", "entropy", "polars", "native"],
        param_specs={
            "order": ParamSpec(dtype=int, min=2, default=3, searchable=True, param_role=ParamRole.SUPPORT_POLICY),
            "window": ParamSpec(dtype=int, min=10, default=100, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, order: int = 3, window: int = 100, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        ord_val = strict_integer(order, "order", minimum=2)
        w = strict_integer(window, "window", minimum=10)

        # TODO: Implement permutation entropy
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_mutual_information",
    category="time_series",
    business_category="entropy",
    canonical="ts_mutual_information",
    source=_SRC,
    backend="polars",
)
class TSMutualInformationNative(SeriesOperator):
    """Mutual information between time series and its lag."""

    metadata = OperatorMetadata(
        name="ts_mutual_information",
        category="time_series",
        description="互信息",
        param_names=["x", "lag", "window", "bins"],
        return_type="series",
        tags=["time_series", "entropy", "polars", "native"],
        param_specs={
            "lag": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
            "window": ParamSpec(dtype=int, min=20, default=100, searchable=True, param_role=ParamRole.HORIZON),
            "bins": ParamSpec(dtype=int, min=2, default=10, searchable=True, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, lag: int = 1, window: int = 100, bins: int = 10, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        lag_val = strict_integer(lag, "lag", minimum=1)
        w = strict_integer(window, "window", minimum=20)
        b = strict_integer(bins, "bins", minimum=2)

        # TODO: Implement mutual information
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_transfer_entropy",
    category="time_series",
    business_category="entropy",
    canonical="ts_transfer_entropy",
    source=_SRC,
    backend="polars",
)
class TSTransferEntropyNative(SeriesOperator):
    """Transfer entropy from y to x."""

    metadata = OperatorMetadata(
        name="ts_transfer_entropy",
        category="time_series",
        description="转移熵",
        param_names=["x", "y", "lag", "window"],
        return_type="series",
        tags=["time_series", "entropy", "polars", "native"],
        param_specs={
            "lag": ParamSpec(dtype=int, min=1, default=1, searchable=True, param_role=ParamRole.HORIZON),
            "window": ParamSpec(dtype=int, min=50, default=200, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame | None = None, lag: int = 1, window: int = 200, **kwargs) -> pl.DataFrame:
        if y is None:
            raise ValueError("ts_transfer_entropy requires y")
        from cleaned_operators.parameter_validation import strict_integer
        lag_val = strict_integer(lag, "lag", minimum=1)
        w = strict_integer(window, "window", minimum=50)

        # TODO: Implement transfer entropy
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


# ==============================================================================
# Extreme Value Theory Operators
# ==============================================================================

@register_operator(
    name="ts_gpd_shape",
    category="time_series",
    business_category="extreme_value",
    canonical="ts_gpd_shape",
    source=_SRC,
    backend="polars",
)
class TSGPDShapeNative(SeriesOperator):
    """Generalized Pareto Distribution shape parameter."""

    metadata = OperatorMetadata(
        name="ts_gpd_shape",
        category="time_series",
        description="GPD形状参数",
        param_names=["x", "threshold", "window"],
        return_type="series",
        tags=["time_series", "extreme_value", "polars", "native"],
        param_specs={
            "threshold": ParamSpec(dtype=float, min=0.0, default=0.9, searchable=True, param_role=ParamRole.NUMERICAL),
            "window": ParamSpec(dtype=int, min=50, default=252, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = 0.9, window: int = 252, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar, strict_integer
        thresh = strict_finite_scalar(threshold, "threshold", minimum=0.0)
        w = strict_integer(window, "window", minimum=50)

        # TODO: Implement GPD shape estimation
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_gpd_scale",
    category="time_series",
    business_category="extreme_value",
    canonical="ts_gpd_scale",
    source=_SRC,
    backend="polars",
)
class TSGPDScaleNative(SeriesOperator):
    """GPD scale parameter."""

    metadata = OperatorMetadata(
        name="ts_gpd_scale",
        category="time_series",
        description="GPD尺度参数",
        param_names=["x", "threshold", "window"],
        return_type="series",
        tags=["time_series", "extreme_value", "polars", "native"],
        param_specs={
            "threshold": ParamSpec(dtype=float, min=0.0, default=0.9, searchable=True, param_role=ParamRole.NUMERICAL),
            "window": ParamSpec(dtype=int, min=50, default=252, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = 0.9, window: int = 252, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar, strict_integer
        thresh = strict_finite_scalar(threshold, "threshold", minimum=0.0)
        w = strict_integer(window, "window", minimum=50)

        # TODO: Implement GPD scale estimation
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_evt_var",
    category="time_series",
    business_category="extreme_value",
    canonical="ts_evt_var",
    source=_SRC,
    backend="polars",
)
class TSEVTVaRNative(SeriesOperator):
    """Value-at-Risk from extreme value theory."""

    metadata = OperatorMetadata(
        name="ts_evt_var",
        category="time_series",
        description="极值VaR",
        param_names=["x", "confidence", "threshold", "window"],
        return_type="series",
        tags=["time_series", "extreme_value", "polars", "native"],
        param_specs={
            "confidence": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.95, searchable=True, param_role=ParamRole.NUMERICAL),
            "threshold": ParamSpec(dtype=float, min=0.0, default=0.9, searchable=True, param_role=ParamRole.NUMERICAL),
            "window": ParamSpec(dtype=int, min=50, default=252, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, confidence: float = 0.95, threshold: float = 0.9, window: int = 252, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar, strict_integer
        conf = strict_finite_scalar(confidence, "confidence", minimum=0.0, maximum=1.0)
        thresh = strict_finite_scalar(threshold, "threshold", minimum=0.0)
        w = strict_integer(window, "window", minimum=50)

        # TODO: Implement EVT VaR
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_hill_estimator",
    category="time_series",
    business_category="extreme_value",
    canonical="ts_hill_estimator",
    source=_SRC,
    backend="polars",
)
class TSHillEstimatorNative(SeriesOperator):
    """Hill estimator for tail index."""

    metadata = OperatorMetadata(
        name="ts_hill_estimator",
        category="time_series",
        description="Hill尾指数",
        param_names=["x", "k", "window"],
        return_type="series",
        tags=["time_series", "extreme_value", "polars", "native"],
        param_specs={
            "k": ParamSpec(dtype=int, min=5, default=20, searchable=True, param_role=ParamRole.SUPPORT_POLICY),
            "window": ParamSpec(dtype=int, min=50, default=252, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, k: int = 20, window: int = 252, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        k_val = strict_integer(k, "k", minimum=5)
        w = strict_integer(window, "window", minimum=50)

        # TODO: Implement Hill estimator
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_pickands_estimator",
    category="time_series",
    business_category="extreme_value",
    canonical="ts_pickands_estimator",
    source=_SRC,
    backend="polars",
)
class TSPickandsEstimatorNative(SeriesOperator):
    """Pickands estimator for extreme value index."""

    metadata = OperatorMetadata(
        name="ts_pickands_estimator",
        category="time_series",
        description="Pickands估计量",
        param_names=["x", "k", "window"],
        return_type="series",
        tags=["time_series", "extreme_value", "polars", "native"],
        param_specs={
            "k": ParamSpec(dtype=int, min=5, default=20, searchable=True, param_role=ParamRole.SUPPORT_POLICY),
            "window": ParamSpec(dtype=int, min=50, default=252, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, k: int = 20, window: int = 252, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        k_val = strict_integer(k, "k", minimum=5)
        w = strict_integer(window, "window", minimum=50)

        # TODO: Implement Pickands estimator
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


# ==============================================================================
# Nonlinear Dynamics / Chaos Operators
# ==============================================================================

@register_operator(
    name="ts_lyapunov_exponent",
    category="time_series",
    business_category="nonlinear",
    canonical="ts_lyapunov_exponent",
    source=_SRC,
    backend="polars",
)
class TSLyapunovExponentNative(SeriesOperator):
    """Largest Lyapunov exponent (chaos indicator)."""

    metadata = OperatorMetadata(
        name="ts_lyapunov_exponent",
        category="time_series",
        description="Lyapunov指数",
        param_names=["x", "emb_dim", "window"],
        return_type="series",
        tags=["time_series", "chaos", "polars", "native"],
        param_specs={
            "emb_dim": ParamSpec(dtype=int, min=2, default=3, searchable=True, param_role=ParamRole.SUPPORT_POLICY),
            "window": ParamSpec(dtype=int, min=100, default=500, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, emb_dim: int = 3, window: int = 500, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        dim = strict_integer(emb_dim, "emb_dim", minimum=2)
        w = strict_integer(window, "window", minimum=100)

        # TODO: Implement Lyapunov exponent calculation
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_dfa_exponent",
    category="time_series",
    business_category="nonlinear",
    canonical="ts_dfa_exponent",
    source=_SRC,
    backend="polars",
)
class TSDFAExponentNative(SeriesOperator):
    """Detrended Fluctuation Analysis scaling exponent."""

    metadata = OperatorMetadata(
        name="ts_dfa_exponent",
        category="time_series",
        description="DFA标度指数",
        param_names=["x", "min_window", "max_window"],
        return_type="series",
        tags=["time_series", "fractal", "polars", "native"],
        param_specs={
            "min_window": ParamSpec(dtype=int, min=4, default=10, searchable=True, param_role=ParamRole.HORIZON),
            "max_window": ParamSpec(dtype=int, min=10, default=100, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, min_window: int = 10, max_window: int = 100, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        min_w = strict_integer(min_window, "min_window", minimum=4)
        max_w = strict_integer(max_window, "max_window", minimum=10)

        # TODO: Implement DFA algorithm
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_hurst_exponent",
    category="time_series",
    business_category="nonlinear",
    canonical="ts_hurst_exponent",
    source=_SRC,
    backend="polars",
)
class TSHurstExponentNative(SeriesOperator):
    """Hurst exponent (long-range dependence)."""

    metadata = OperatorMetadata(
        name="ts_hurst_exponent",
        category="time_series",
        description="Hurst指数",
        param_names=["x", "window"],
        return_type="series",
        tags=["time_series", "fractal", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=20, default=100, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 100, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=20)

        # TODO: Implement Hurst exponent
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_fractal_dimension",
    category="time_series",
    business_category="nonlinear",
    canonical="ts_fractal_dimension",
    source=_SRC,
    backend="polars",
)
class TSFractalDimensionNative(SeriesOperator):
    """Fractal dimension (Higuchi method)."""

    metadata = OperatorMetadata(
        name="ts_fractal_dimension",
        category="time_series",
        description="分形维数",
        param_names=["x", "kmax", "window"],
        return_type="series",
        tags=["time_series", "fractal", "polars", "native"],
        param_specs={
            "kmax": ParamSpec(dtype=int, min=2, default=10, searchable=True, param_role=ParamRole.SUPPORT_POLICY),
            "window": ParamSpec(dtype=int, min=50, default=200, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, kmax: int = 10, window: int = 200, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        k = strict_integer(kmax, "kmax", minimum=2)
        w = strict_integer(window, "window", minimum=50)

        # TODO: Implement fractal dimension (Higuchi)
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_correlation_dimension",
    category="time_series",
    business_category="nonlinear",
    canonical="ts_correlation_dimension",
    source=_SRC,
    backend="polars",
)
class TSCorrelationDimensionNative(SeriesOperator):
    """Correlation dimension (Grassberger-Procaccia)."""

    metadata = OperatorMetadata(
        name="ts_correlation_dimension",
        category="time_series",
        description="关联维数",
        param_names=["x", "emb_dim", "window"],
        return_type="series",
        tags=["time_series", "chaos", "polars", "native"],
        param_specs={
            "emb_dim": ParamSpec(dtype=int, min=2, default=5, searchable=True, param_role=ParamRole.SUPPORT_POLICY),
            "window": ParamSpec(dtype=int, min=100, default=500, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, emb_dim: int = 5, window: int = 500, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        dim = strict_integer(emb_dim, "emb_dim", minimum=2)
        w = strict_integer(window, "window", minimum=100)

        # TODO: Implement correlation dimension
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


# ==============================================================================
# Regime / State Space Operators
# ==============================================================================

@register_operator(
    name="ts_markov_regime_prob",
    category="time_series",
    business_category="regime",
    canonical="ts_markov_regime_prob",
    source=_SRC,
    backend="polars",
)
class TSMarkovRegimeProbNative(SeriesOperator):
    """Markov regime switching probability."""

    metadata = OperatorMetadata(
        name="ts_markov_regime_prob",
        category="time_series",
        description="马尔科夫状态概率",
        param_names=["x", "n_regimes", "window"],
        return_type="series",
        tags=["time_series", "regime", "polars", "native"],
        param_specs={
            "n_regimes": ParamSpec(dtype=int, min=2, default=2, searchable=True, param_role=ParamRole.SUPPORT_POLICY),
            "window": ParamSpec(dtype=int, min=50, default=252, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, n_regimes: int = 2, window: int = 252, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        n = strict_integer(n_regimes, "n_regimes", minimum=2)
        w = strict_integer(window, "window", minimum=50)

        # TODO: Implement Markov regime probability
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_regime_volatility",
    category="time_series",
    business_category="regime",
    canonical="ts_regime_volatility",
    source=_SRC,
    backend="polars",
)
class TSRegimeVolatilityNative(SeriesOperator):
    """Volatility of current regime."""

    metadata = OperatorMetadata(
        name="ts_regime_volatility",
        category="time_series",
        description="状态波动率",
        param_names=["x", "n_regimes", "window"],
        return_type="series",
        tags=["time_series", "regime", "polars", "native"],
        param_specs={
            "n_regimes": ParamSpec(dtype=int, min=2, default=2, searchable=True, param_role=ParamRole.SUPPORT_POLICY),
            "window": ParamSpec(dtype=int, min=50, default=252, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, n_regimes: int = 2, window: int = 252, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        n = strict_integer(n_regimes, "n_regimes", minimum=2)
        w = strict_integer(window, "window", minimum=50)

        # TODO: Implement regime volatility
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_two_state_filter",
    category="time_series",
    business_category="regime",
    canonical="ts_two_state_filter",
    source=_SRC,
    backend="polars",
)
class TSTwoStateFilterNative(SeriesOperator):
    """Two-state filter (high/low volatility)."""

    metadata = OperatorMetadata(
        name="ts_two_state_filter",
        category="time_series",
        description="双状态滤波",
        param_names=["x", "threshold", "window"],
        return_type="series",
        tags=["time_series", "regime", "polars", "native"],
        param_specs={
            "threshold": ParamSpec(dtype=float, min=0.0, default=1.0, searchable=True, param_role=ParamRole.NUMERICAL),
            "window": ParamSpec(dtype=int, min=20, default=60, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = 1.0, window: int = 60, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar, strict_integer
        thresh = strict_finite_scalar(threshold, "threshold", minimum=0.0)
        w = strict_integer(window, "window", minimum=20)

        # TODO: Implement two-state filter
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


# ==============================================================================
# Recurrence / Persistence Operators
# ==============================================================================

@register_operator(
    name="ts_recurrence_rate",
    category="time_series",
    business_category="recurrence",
    canonical="ts_recurrence_rate",
    source=_SRC,
    backend="polars",
)
class TSRecurrenceRateNative(SeriesOperator):
    """Recurrence rate from recurrence plot."""

    metadata = OperatorMetadata(
        name="ts_recurrence_rate",
        category="time_series",
        description="递归率",
        param_names=["x", "threshold", "window"],
        return_type="series",
        tags=["time_series", "recurrence", "polars", "native"],
        param_specs={
            "threshold": ParamSpec(dtype=float, min=0.0, default=0.1, searchable=True, param_role=ParamRole.NUMERICAL),
            "window": ParamSpec(dtype=int, min=50, default=200, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = 0.1, window: int = 200, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar, strict_integer
        thresh = strict_finite_scalar(threshold, "threshold", minimum=0.0)
        w = strict_integer(window, "window", minimum=50)

        # TODO: Implement recurrence rate
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_determinism",
    category="time_series",
    business_category="recurrence",
    canonical="ts_determinism",
    source=_SRC,
    backend="polars",
)
class TSDeterminismNative(SeriesOperator):
    """Determinism from RQA (recurrence quantification)."""

    metadata = OperatorMetadata(
        name="ts_determinism",
        category="time_series",
        description="确定性",
        param_names=["x", "threshold", "min_line", "window"],
        return_type="series",
        tags=["time_series", "recurrence", "polars", "native"],
        param_specs={
            "threshold": ParamSpec(dtype=float, min=0.0, default=0.1, searchable=True, param_role=ParamRole.NUMERICAL),
            "min_line": ParamSpec(dtype=int, min=2, default=2, searchable=True, param_role=ParamRole.SUPPORT_POLICY),
            "window": ParamSpec(dtype=int, min=50, default=200, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = 0.1, min_line: int = 2, window: int = 200, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar, strict_integer
        thresh = strict_finite_scalar(threshold, "threshold", minimum=0.0)
        ml = strict_integer(min_line, "min_line", minimum=2)
        w = strict_integer(window, "window", minimum=50)

        # TODO: Implement determinism (RQA)
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_laminarity",
    category="time_series",
    business_category="recurrence",
    canonical="ts_laminarity",
    source=_SRC,
    backend="polars",
)
class TSLaminarityNative(SeriesOperator):
    """Laminarity from RQA."""

    metadata = OperatorMetadata(
        name="ts_laminarity",
        category="time_series",
        description="层流性",
        param_names=["x", "threshold", "min_vert", "window"],
        return_type="series",
        tags=["time_series", "recurrence", "polars", "native"],
        param_specs={
            "threshold": ParamSpec(dtype=float, min=0.0, default=0.1, searchable=True, param_role=ParamRole.NUMERICAL),
            "min_vert": ParamSpec(dtype=int, min=2, default=2, searchable=True, param_role=ParamRole.SUPPORT_POLICY),
            "window": ParamSpec(dtype=int, min=50, default=200, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = 0.1, min_vert: int = 2, window: int = 200, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar, strict_integer
        thresh = strict_finite_scalar(threshold, "threshold", minimum=0.0)
        mv = strict_integer(min_vert, "min_vert", minimum=2)
        w = strict_integer(window, "window", minimum=50)

        # TODO: Implement laminarity (RQA)
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_trapping_time",
    category="time_series",
    business_category="recurrence",
    canonical="ts_trapping_time",
    source=_SRC,
    backend="polars",
)
class TSTrappingTimeNative(SeriesOperator):
    """Average trapping time from RQA."""

    metadata = OperatorMetadata(
        name="ts_trapping_time",
        category="time_series",
        description="捕获时间",
        param_names=["x", "threshold", "window"],
        return_type="series",
        tags=["time_series", "recurrence", "polars", "native"],
        param_specs={
            "threshold": ParamSpec(dtype=float, min=0.0, default=0.1, searchable=True, param_role=ParamRole.NUMERICAL),
            "window": ParamSpec(dtype=int, min=50, default=200, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, threshold: float = 0.1, window: int = 200, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar, strict_integer
        thresh = strict_finite_scalar(threshold, "threshold", minimum=0.0)
        w = strict_integer(window, "window", minimum=50)

        # TODO: Implement trapping time
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


# ==============================================================================
# Pattern / Motif Discovery Operators
# ==============================================================================

@register_operator(
    name="ts_matrix_profile_min",
    category="time_series",
    business_category="pattern",
    canonical="ts_matrix_profile_min",
    source=_SRC,
    backend="polars",
)
class TSMatrixProfileMinNative(SeriesOperator):
    """Matrix profile minimum distance."""

    metadata = OperatorMetadata(
        name="ts_matrix_profile_min",
        category="time_series",
        description="矩阵轮廓最小距离",
        param_names=["x", "subsequence_length", "window"],
        return_type="series",
        tags=["time_series", "pattern", "polars", "native"],
        param_specs={
            "subsequence_length": ParamSpec(dtype=int, min=3, default=10, searchable=True, param_role=ParamRole.SUPPORT_POLICY),
            "window": ParamSpec(dtype=int, min=50, default=200, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, subsequence_length: int = 10, window: int = 200, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        subseq = strict_integer(subsequence_length, "subsequence_length", minimum=3)
        w = strict_integer(window, "window", minimum=50)

        # TODO: Implement matrix profile
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_motif_count",
    category="time_series",
    business_category="pattern",
    canonical="ts_motif_count",
    source=_SRC,
    backend="polars",
)
class TSMotifCountNative(SeriesOperator):
    """Count of repeated motifs."""

    metadata = OperatorMetadata(
        name="ts_motif_count",
        category="time_series",
        description="模式计数",
        param_names=["x", "subsequence_length", "threshold", "window"],
        return_type="series",
        tags=["time_series", "pattern", "polars", "native"],
        param_specs={
            "subsequence_length": ParamSpec(dtype=int, min=3, default=10, searchable=True, param_role=ParamRole.SUPPORT_POLICY),
            "threshold": ParamSpec(dtype=float, min=0.0, default=0.1, searchable=True, param_role=ParamRole.NUMERICAL),
            "window": ParamSpec(dtype=int, min=50, default=200, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, subsequence_length: int = 10, threshold: float = 0.1, window: int = 200, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer, strict_finite_scalar
        subseq = strict_integer(subsequence_length, "subsequence_length", minimum=3)
        thresh = strict_finite_scalar(threshold, "threshold", minimum=0.0)
        w = strict_integer(window, "window", minimum=50)

        # TODO: Implement motif count
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_ordinal_pattern_distribution",
    category="time_series",
    business_category="pattern",
    canonical="ts_ordinal_pattern_distribution",
    source=_SRC,
    backend="polars",
)
class TSOrdinalPatternDistributionNative(SeriesOperator):
    """Distribution entropy of ordinal patterns."""

    metadata = OperatorMetadata(
        name="ts_ordinal_pattern_distribution",
        category="time_series",
        description="序数模式分布",
        param_names=["x", "order", "window"],
        return_type="series",
        tags=["time_series", "pattern", "polars", "native"],
        param_specs={
            "order": ParamSpec(dtype=int, min=2, default=3, searchable=True, param_role=ParamRole.SUPPORT_POLICY),
            "window": ParamSpec(dtype=int, min=20, default=100, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, order: int = 3, window: int = 100, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        ord_val = strict_integer(order, "order", minimum=2)
        w = strict_integer(window, "window", minimum=20)

        # TODO: Implement ordinal pattern distribution
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


# ==============================================================================
# Support / Resistance Operators
# ==============================================================================

@register_operator(
    name="ts_support_level",
    category="time_series",
    business_category="technical",
    canonical="ts_support_level",
    source=_SRC,
    backend="polars",
)
class TSSupportLevelNative(SeriesOperator):
    """Support level from local minima."""

    metadata = OperatorMetadata(
        name="ts_support_level",
        category="time_series",
        description="支撑位",
        param_names=["x", "window", "n_levels"],
        return_type="series",
        tags=["time_series", "technical", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=10, default=60, searchable=True, param_role=ParamRole.HORIZON),
            "n_levels": ParamSpec(dtype=int, min=1, default=3, searchable=True, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 60, n_levels: int = 3, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=10)
        n = strict_integer(n_levels, "n_levels", minimum=1)

        # TODO: Implement support level detection
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_resistance_level",
    category="time_series",
    business_category="technical",
    canonical="ts_resistance_level",
    source=_SRC,
    backend="polars",
)
class TSResistanceLevelNative(SeriesOperator):
    """Resistance level from local maxima."""

    metadata = OperatorMetadata(
        name="ts_resistance_level",
        category="time_series",
        description="阻力位",
        param_names=["x", "window", "n_levels"],
        return_type="series",
        tags=["time_series", "technical", "polars", "native"],
        param_specs={
            "window": ParamSpec(dtype=int, min=10, default=60, searchable=True, param_role=ParamRole.HORIZON),
            "n_levels": ParamSpec(dtype=int, min=1, default=3, searchable=True, param_role=ParamRole.SUPPORT_POLICY),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, window: int = 60, n_levels: int = 3, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_integer
        w = strict_integer(window, "window", minimum=10)
        n = strict_integer(n_levels, "n_levels", minimum=1)

        # TODO: Implement resistance level detection
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_pivot_point",
    category="time_series",
    business_category="technical",
    canonical="ts_pivot_point",
    source=_SRC,
    backend="polars",
)
class TSPivotPointNative(SeriesOperator):
    """Pivot point from high/low/close."""

    metadata = OperatorMetadata(
        name="ts_pivot_point",
        category="time_series",
        description="枢轴点",
        param_names=["high", "low", "close"],
        return_type="series",
        tags=["time_series", "technical", "polars", "native"],
    )

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame | None = None, close: pl.DataFrame | None = None, **kwargs) -> pl.DataFrame:
        if low is None or close is None:
            raise ValueError("ts_pivot_point requires high, low, close")

        # TODO: Implement pivot point calculation
        cols = _numeric_cols(high)
        return high.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


# ==============================================================================
# Quantile / Expectile Operators
# ==============================================================================

@register_operator(
    name="ts_quantile_tracking",
    category="time_series",
    business_category="quantile",
    canonical="ts_quantile_tracking",
    source=_SRC,
    backend="polars",
)
class TSQuantileTrackingNative(SeriesOperator):
    """Rolling quantile with exponential smoothing."""

    metadata = OperatorMetadata(
        name="ts_quantile_tracking",
        category="time_series",
        description="分位数追踪",
        param_names=["x", "quantile", "alpha", "window"],
        return_type="series",
        tags=["time_series", "quantile", "polars", "native"],
        param_specs={
            "quantile": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5, searchable=True, param_role=ParamRole.NUMERICAL),
            "alpha": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.05, searchable=True, param_role=ParamRole.NUMERICAL),
            "window": ParamSpec(dtype=int, min=20, default=100, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, quantile: float = 0.5, alpha: float = 0.05, window: int = 100, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar, strict_integer
        q = strict_finite_scalar(quantile, "quantile", minimum=0.0, maximum=1.0)
        a = strict_finite_scalar(alpha, "alpha", minimum=0.0, maximum=1.0)
        w = strict_integer(window, "window", minimum=20)

        # TODO: Implement quantile tracking
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_quantile_crossing",
    category="time_series",
    business_category="quantile",
    canonical="ts_quantile_crossing",
    source=_SRC,
    backend="polars",
)
class TSQuantileCrossingNative(SeriesOperator):
    """Indicator of crossing a rolling quantile."""

    metadata = OperatorMetadata(
        name="ts_quantile_crossing",
        category="time_series",
        description="分位数穿越",
        param_names=["x", "quantile", "window"],
        return_type="series",
        tags=["time_series", "quantile", "polars", "native"],
        param_specs={
            "quantile": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.75, searchable=True, param_role=ParamRole.NUMERICAL),
            "window": ParamSpec(dtype=int, min=20, default=100, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, quantile: float = 0.75, window: int = 100, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar, strict_integer
        q = strict_finite_scalar(quantile, "quantile", minimum=0.0, maximum=1.0)
        w = strict_integer(window, "window", minimum=20)

        # TODO: Implement quantile crossing
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


@register_operator(
    name="ts_expectile",
    category="time_series",
    business_category="quantile",
    canonical="ts_expectile",
    source=_SRC,
    backend="polars",
)
class TSExpectileNative(SeriesOperator):
    """Expectile (asymmetric least squares analogue of quantile)."""

    metadata = OperatorMetadata(
        name="ts_expectile",
        category="time_series",
        description="期望分位数",
        param_names=["x", "tau", "window"],
        return_type="series",
        tags=["time_series", "quantile", "polars", "native"],
        param_specs={
            "tau": ParamSpec(dtype=float, min=0.0, max=1.0, default=0.5, searchable=True, param_role=ParamRole.NUMERICAL),
            "window": ParamSpec(dtype=int, min=20, default=100, searchable=True, param_role=ParamRole.HORIZON),
        },
    )

    def _calculate_series(self, x: pl.DataFrame, tau: float = 0.5, window: int = 100, **kwargs) -> pl.DataFrame:
        from cleaned_operators.parameter_validation import strict_finite_scalar, strict_integer
        t = strict_finite_scalar(tau, "tau", minimum=0.0, maximum=1.0)
        w = strict_integer(window, "window", minimum=20)

        # TODO: Implement expectile calculation
        cols = _numeric_cols(x)
        return x.with_columns([pl.col(c).fill_nan(None).alias(c) for c in cols])


# ==============================================================================
# NOTE: This module contains skeleton implementations for ~600+ ts_* operators.
# The remaining operators follow the same pattern:
# - Proper class structure with OperatorMetadata
# - ParamSpec with ParamRole annotations
# - Registration with backend="polars"
# - Placeholder implementation returning fill_nan(None)
# - TODO comment for complex algorithm
#
# Additional operator families not fully implemented here include:
# - More Kalman variants (ts_kalman_*, ~10 operators)
# - GARCH extensions (ts_egarch_*, ts_tgarch_*, ~10 operators)
# - Spectral analysis (ts_spectral_*, ~15 operators)
# - Wavelet decomposition (ts_wavelet_*, ~10 operators)
# - Information theory (ts_conditional_entropy, ts_joint_entropy, ~10 operators)
# - EVT extensions (ts_evt_*, ~10 operators)
# - Chaos theory (ts_kolmogorov_complexity, ts_attractor_*, ~15 operators)
# - Fractal analysis (ts_multifractal_*, ~10 operators)
# - Regime detection (ts_markov_*, ts_hmm_*, ~15 operators)
# - RQA extensions (ts_rqa_*, ~10 operators)
# - Pattern mining (ts_shapelets, ts_discord_*, ~15 operators)
# - Technical indicators (ts_ichimoku_*, ts_bollinger_*, ~30 operators)
# - Microstructure (ts_bid_ask_*, ts_trade_*, ~20 operators)
# - Order flow (ts_order_imbalance_*, ~10 operators)
# - And many more domain-specific operators
#
# Total: ~600+ additional operators can be added following this skeleton pattern.
# ==============================================================================
