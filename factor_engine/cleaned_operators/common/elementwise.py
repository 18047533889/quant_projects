# -*- coding: utf-8 -*-
"""
逐元素数学、逻辑比较与矩阵/信号处理算子。

语义
----
- **元素级**：``abs``、``log``、``sign``、``power``、``sqrt``、``min``/``max``（两序列逐点）；
- **比较**：``gt``、``lt``、``eq`` 等 — ``dsl_parser`` 会把 ``a > b``  lowering 为这些算子；
- **逻辑**：``and``、``or``、``not``（若启用）；
- **高级**：矩阵乘、FFT 等（见文件后部，用于特定 alpha 模板）。

**逐点**计算：不做 rolling，也不做截面 rank；时间/截面维度保持不变。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from cleaned_operators._causal import (
    causal_convolve_column,
    causal_correlate_column,
    expanding_eig_diagonal,
    expanding_geometric_mean,
    expanding_harmonic_mean,
    expanding_svd_diagonal,
    expanding_unwrap_panel,
    rolling_panel_irfft,
    rolling_panel_rfft,
    rolling_wavelet_panel,
)
from cleaned_operators.base import (
    Operator,
    OperatorMetadata,
    SeriesOperator,
    ScalarOperator,
    TwoVarOperator,
    register_operator,
)

import numpy as np
import pandas as pd
try:
    from scipy import stats as sp_stats
except ImportError:
    sp_stats = None  # type: ignore

# canonical=abs backend=pandas_numpy selected=abs source=math/elementary.py
@register_operator(name="abs", category="math", business_category="elementwise_math", canonical="abs", source="factor_dsl_np")
class Abs(SeriesOperator):
    """绝对值"""

    metadata = OperatorMetadata(
        name="abs",
        category="math",
        description="计算绝对值",
        examples=["abs(Return)", "abs(close - open)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "unary", "absolute"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.abs()

# aliases: ABS



# canonical=acos backend=pandas_numpy selected=acos source=math/trigonometric.py
@register_operator(name="acos", category="math", business_category="elementwise_math", canonical="acos", source="factor_dsl_np")
class Acos(SeriesOperator):
    """反余弦"""

    metadata = OperatorMetadata(
        name="acos",
        category="math",
        description="计算反余弦值（返回弧度）",
        examples=["acos(value)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "trigonometric", "inverse"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.arccos(x)



# canonical=arg backend=pandas_numpy selected=arg source=math/complex_ops.py
@register_operator(name="arg", category="math", business_category="elementwise_math", canonical="arg", source="factor_dsl_np")
class Arg(SeriesOperator):
    """复数相位（弧度）"""
    metadata = OperatorMetadata(
        name="arg",
        category="math",
        description="复数相位（弧度）",
        examples=["arg(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "complex", "phase"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        result = np.angle(x.values)
        return pd.DataFrame(result, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)



# canonical=asin backend=pandas_numpy selected=asin source=math/trigonometric.py
@register_operator(name="asin", category="math", business_category="elementwise_math", canonical="asin", source="factor_dsl_np")
class Asin(SeriesOperator):
    """反正弦"""

    metadata = OperatorMetadata(
        name="asin",
        category="math",
        description="计算反正弦值（返回弧度）",
        examples=["asin(value)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "trigonometric", "inverse"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.arcsin(x)



# canonical=atan backend=pandas_numpy selected=atan source=math/trigonometric.py
@register_operator(name="atan", category="math", business_category="elementwise_math", canonical="atan", source="factor_dsl_np")
class Atan(SeriesOperator):
    """反正切"""

    metadata = OperatorMetadata(
        name="atan",
        category="math",
        description="计算反正切值（返回弧度）",
        examples=["atan(value)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "trigonometric", "inverse"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.arctan(x)



# canonical=atan2 backend=pandas_numpy selected=atan2 source=math/trigonometric.py
@register_operator(name="atan2", category="math", business_category="elementwise_math", canonical="atan2", source="factor_dsl_np")
class Atan2(SeriesOperator):
    """坐标反正切"""

    metadata = OperatorMetadata(
        name="atan2",
        category="math",
        description="计算坐标(y,x)的反正切值",
        examples=["atan2(y, x)"],
        param_names=["y", "x"],
        return_type="series",
        tags=["math", "trigonometric", "inverse"]
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.arctan2(y, x)



# canonical=blom_transform backend=pandas_numpy selected=blom_transform source=math/utility_ops.py
@register_operator(name="blom_transform", category="math", business_category="elementwise_math", canonical="blom_transform", source="factor_dsl_np")
class BlomTransform(SeriesOperator):
    """Blom变换（秩的逆正态CDF）"""
    metadata = OperatorMetadata(
        name="blom_transform",
        category="math",
        description="Blom变换（秩的逆正态CDF）",
        examples=["blom_transform(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "utility", "transform", "normal"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ranked = x.rank(axis=1, method='average')
        n = x.count(axis=1).to_numpy(dtype=float)[:, None]
        result = sp_stats.norm.ppf((ranked - 3 / 8) / (n + 1 / 4))
        return pd.DataFrame(result, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)



# canonical=cbrt backend=pandas_numpy selected=cbrt source=math/elementary.py
@register_operator(name="cbrt", category="math", business_category="elementwise_math", canonical="cbrt", source="factor_dsl_np")
class Cbrt(SeriesOperator):
    """立方根"""

    metadata = OperatorMetadata(
        name="cbrt",
        category="math",
        description="计算立方根",
        examples=["cbrt(volume)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "root", "cube"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.cbrt(x)



# canonical=ceil backend=pandas_numpy selected=ceil source=math/rounding.py
@register_operator(name="ceil", category="math", business_category="elementwise_math", canonical="ceil", source="factor_dsl_np")
class Ceil(SeriesOperator):
    """向上取整"""

    metadata = OperatorMetadata(
        name="ceil",
        category="math",
        description="向上取整",
        examples=["ceil(price)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "rounding", "up"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.ceil(x)



# canonical=cap backend=pandas_numpy selected=clip source=math/utility_ops.py
@register_operator(name="cap", category="math", business_category="elementwise_math", canonical="clip", source="factor_dsl_np")
class Cap(SeriesOperator):
    """裁剪到 [lo, hi] 区间"""
    metadata = OperatorMetadata(
        name="cap",
        category="math",
        description="裁剪到 [lo, hi] 区间",
        examples=["cap(x, -3, 3)"],
        param_names=["x", "lo", "hi"],
        return_type="series",
        tags=["math", "utility", "clip", "clamp"]
    )

    def _calculate_series(self, x: pd.DataFrame, lo: float = -3.0, hi: float = 3.0, **kwargs) -> pd.DataFrame:
        min_val = float(kwargs.get("min", lo))
        max_val = float(kwargs.get("max", hi))
        return x.clip(lower=min_val, upper=max_val).replace([np.inf, -np.inf], np.nan)



# canonical=complex backend=pandas_numpy selected=complex source=math/complex_ops.py
@register_operator(name="complex", category="math", business_category="elementwise_math", canonical="complex", source="factor_dsl_np")
class Complex(SeriesOperator):
    """构造复数DataFrame"""
    metadata = OperatorMetadata(
        name="complex",
        category="math",
        description="构造复数DataFrame",
        examples=["complex(real_part, imag_part)"],
        param_names=["real", "imag"],
        return_type="series",
        tags=["math", "complex"]
    )

    def _calculate_series(self, real: pd.DataFrame, imag: pd.DataFrame, **kwargs) -> pd.DataFrame:
        result = real + 1j * imag
        return pd.DataFrame(result, index=real.index, columns=real.columns)



# canonical=conj backend=pandas_numpy selected=conj source=math/complex_ops.py
@register_operator(name="conj", category="math", business_category="elementwise_math", canonical="conj", source="factor_dsl_np")
class Conj(SeriesOperator):
    """复数共轭"""
    metadata = OperatorMetadata(
        name="conj",
        category="math",
        description="复数共轭",
        examples=["conj(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "complex", "conjugate"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        result = np.conj(x.values)
        return pd.DataFrame(result, index=x.index, columns=x.columns)



# canonical=constant backend=pandas_numpy selected=constant source=math/utility_ops.py
@register_operator(name="constant", category="math", business_category="elementwise_math", canonical="constant", source="factor_dsl_np")
class Constant(ScalarOperator):
    """返回常量值"""
    metadata = OperatorMetadata(
        name="constant",
        category="math",
        description="返回常量值",
        examples=["constant(1.0)"],
        param_names=["c"],
        return_type="scalar",
        tags=["math", "utility", "scalar"]
    )

    def _calculate_scalar(self, c: float = 0.0, **kwargs):
        return c



# canonical=convolve backend=pandas_numpy selected=convolve source=math/fourier_ops.py
@register_operator(name="convolve", category="math", business_category="elementwise_math", canonical="convolve", source="factor_dsl_np")
class Convolve(SeriesOperator):
    """卷积"""
    metadata = OperatorMetadata(
        name="convolve",
        category="math",
        description="卷积",
        examples=["convolve(x, y)"],
        param_names=["x", "y"],
        return_type="series",
        tags=["math", "fourier", "convolve"]
    )

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> pd.DataFrame:
        try:
            x_vals = x.values.astype(float)
            y_vals = y.values.astype(float)
            result = np.full(x_vals.shape, np.nan, dtype=float)
            for col_idx in range(x_vals.shape[1]):
                result[:, col_idx] = causal_convolve_column(
                    x_vals[:, col_idx], y_vals[:, col_idx]
                )
            return pd.DataFrame(result, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)
        except Exception:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=correlate backend=pandas_numpy selected=correlate source=math/fourier_ops.py
@register_operator(name="correlate", category="math", business_category="elementwise_math", canonical="correlate", source="factor_dsl_np")
class Correlate(SeriesOperator):
    """互相关"""
    metadata = OperatorMetadata(
        name="correlate",
        category="math",
        description="互相关",
        examples=["correlate(x, y)"],
        param_names=["x", "y"],
        return_type="series",
        tags=["math", "fourier", "correlate"]
    )

    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> pd.DataFrame:
        try:
            x_vals = x.values.astype(float)
            y_vals = y.values.astype(float)
            result = np.full(x_vals.shape, np.nan, dtype=float)
            for col_idx in range(x_vals.shape[1]):
                result[:, col_idx] = causal_correlate_column(
                    x_vals[:, col_idx], y_vals[:, col_idx]
                )
            return pd.DataFrame(result, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)
        except Exception:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=cos backend=pandas_numpy selected=cos source=math/trigonometric.py
@register_operator(name="cos", category="math", business_category="elementwise_math", canonical="cos", source="factor_dsl_np")
class Cos(SeriesOperator):
    """余弦"""

    metadata = OperatorMetadata(
        name="cos",
        category="math",
        description="计算余弦值（弧度）",
        examples=["cos(angle)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "trigonometric"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.cos(x)



# canonical=cosh backend=pandas_numpy selected=cosh source=math/trigonometric.py
@register_operator(name="cosh", category="math", business_category="elementwise_math", canonical="cosh", source="factor_dsl_np")
class Cosh(SeriesOperator):
    """双曲余弦"""

    metadata = OperatorMetadata(
        name="cosh",
        category="math",
        description="计算双曲余弦值",
        examples=["cosh(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "hyperbolic"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.cosh(x)



# canonical=cot backend=pandas_numpy selected=cot source=math/trigonometric.py
@register_operator(name="cot", category="math", business_category="elementwise_math", canonical="cot", source="factor_dsl_np")
class Cot(SeriesOperator):
    """余切"""

    metadata = OperatorMetadata(
        name="cot",
        category="math",
        description="计算余切值 (1/tan)",
        examples=["cot(angle)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "trigonometric"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return 1 / np.tan(x)



# canonical=csc backend=pandas_numpy selected=csc source=math/trigonometric.py
@register_operator(name="csc", category="math", business_category="elementwise_math", canonical="csc", source="factor_dsl_np")
class Csc(SeriesOperator):
    """余割"""

    metadata = OperatorMetadata(
        name="csc",
        category="math",
        description="计算余割值 (1/sin)",
        examples=["csc(angle)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "trigonometric"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return 1 / np.sin(x)



# canonical=cube backend=pandas_numpy selected=cube source=math/utility_ops.py
@register_operator(name="cube", category="math", business_category="elementwise_math", canonical="cube", source="factor_dsl_np")
class Cube(SeriesOperator):
    """返回x的立方"""
    metadata = OperatorMetadata(
        name="cube",
        category="math",
        description="返回x的立方",
        examples=["cube(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "utility"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return (x ** 3).replace([np.inf, -np.inf], np.nan)



# canonical=cumulative_max backend=pandas_numpy selected=cumulative_max source=math/utility_ops.py
@register_operator(name="cumulative_max", category="math", business_category="elementwise_math", canonical="cumulative_max", source="factor_dsl_np")
class CumulativeMax(SeriesOperator):
    """累计最大值"""
    metadata = OperatorMetadata(
        name="cumulative_max",
        category="math",
        description="累计最大值",
        examples=["cumulative_max(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "utility", "cumulative", "max"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.expanding(min_periods=1).max().replace([np.inf, -np.inf], np.nan)



# canonical=cumulative_mean backend=pandas_numpy selected=cumulative_mean source=math/utility_ops.py
@register_operator(name="cumulative_mean", category="math", business_category="elementwise_math", canonical="cumulative_mean", source="factor_dsl_np")
class CumulativeMean(SeriesOperator):
    """累计均值"""
    metadata = OperatorMetadata(
        name="cumulative_mean",
        category="math",
        description="累计均值",
        examples=["cumulative_mean(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "utility", "cumulative", "mean"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.expanding(min_periods=1).mean().replace([np.inf, -np.inf], np.nan)



# canonical=cumulative_min backend=pandas_numpy selected=cumulative_min source=math/utility_ops.py
@register_operator(name="cumulative_min", category="math", business_category="elementwise_math", canonical="cumulative_min", source="factor_dsl_np")
class CumulativeMin(SeriesOperator):
    """累计最小值"""
    metadata = OperatorMetadata(
        name="cumulative_min",
        category="math",
        description="累计最小值",
        examples=["cumulative_min(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "utility", "cumulative", "min"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.expanding(min_periods=1).min().replace([np.inf, -np.inf], np.nan)



# canonical=decimate backend=pandas_numpy selected=decimate source=math/fourier_ops.py
@register_operator(name="decimate", category="math", business_category="elementwise_math", canonical="decimate", source="factor_dsl_np")
class Decimate(SeriesOperator):
    """降采样"""
    metadata = OperatorMetadata(
        name="decimate",
        category="math",
        description="降采样",
        examples=["decimate(x, 2)"],
        param_names=["x", "factor"],
        return_type="series",
        tags=["math", "fourier", "decimate", "downsample"]
    )

    def _calculate_series(self, x: pd.DataFrame, factor: int = 2, **kwargs) -> pd.DataFrame:
        try:
            from scipy.signal import butter, lfilter

            factor = max(int(factor), 2)
            result = np.full(x.values.shape, np.nan, dtype=float)
            for col_idx in range(x.values.shape[1]):
                col = x.values[:, col_idx]
                if not np.isfinite(col).any():
                    continue
                col_clean = np.where(np.isfinite(col), col, 0.0)
                b, a = butter(2, min(0.99, 0.8 / factor), btype="low")
                filtered = lfilter(b, a, col_clean)
                result[::factor, col_idx] = filtered[::factor]
            return pd.DataFrame(result, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)
        except Exception:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=dft backend=pandas_numpy selected=dft source=math/fourier_ops.py
@register_operator(name="dft", category="math", business_category="elementwise_math", canonical="dft", source="factor_dsl_np")
class Dft(SeriesOperator):
    """离散傅里叶变换"""
    metadata = OperatorMetadata(
        name="dft",
        category="math",
        description="离散傅里叶变换",
        examples=["dft(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "fourier", "dft"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        try:
            return rolling_panel_rfft(x).replace([np.inf, -np.inf], np.nan)
        except Exception:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=eig backend=pandas_numpy selected=eig source=math/matrix_ops.py
@register_operator(name="eig", category="math", business_category="elementwise_math", canonical="eig", source="factor_dsl_np")
class Eig(SeriesOperator):
    """特征值分解"""
    metadata = OperatorMetadata(
        name="eig",
        category="math",
        description="特征值分解",
        examples=["eig(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "matrix", "eigenvalue", "decomposition"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        try:
            return expanding_eig_diagonal(x).replace([np.inf, -np.inf], np.nan)
        except Exception:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=exp backend=pandas_numpy selected=exp source=math/elementary.py
@register_operator(name="exp", category="math", business_category="elementwise_math", canonical="exp", source="factor_dsl_np")
class Exp(SeriesOperator):
    """指数函数"""

    metadata = OperatorMetadata(
        name="exp",
        category="math",
        description="计算e的x次方",
        examples=["exp(Return)", "exp(close - mean(close, 20))"],
        param_names=["x"],
        return_type="series",
        tags=["math", "exponential", "e"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.exp(x)



# canonical=exp_neg backend=pandas_numpy selected=exp_neg source=math/utility_ops.py
@register_operator(name="exp_neg", category="math", business_category="elementwise_math", canonical="exp_neg", source="factor_dsl_np")
class ExpNeg(SeriesOperator):
    """返回exp(-x)"""
    metadata = OperatorMetadata(
        name="exp_neg",
        category="math",
        description="返回exp(-x)",
        examples=["exp_neg(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "utility"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.exp(-x).replace([np.inf, -np.inf], np.nan)



# canonical=fft backend=pandas_numpy selected=fft source=math/fourier_ops.py
@register_operator(name="fft", category="math", business_category="elementwise_math", canonical="fft", source="factor_dsl_np")
class Fft(SeriesOperator):
    """快速傅里叶变换"""
    metadata = OperatorMetadata(
        name="fft",
        category="math",
        description="快速傅里叶变换",
        examples=["fft(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "fourier", "fft"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        try:
            return rolling_panel_rfft(x).replace([np.inf, -np.inf], np.nan)
        except Exception:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=filter_bandpass backend=pandas_numpy selected=filter_bandpass source=math/fourier_ops.py
@register_operator(name="filter_bandpass", category="math", business_category="elementwise_math", canonical="filter_bandpass", source="factor_dsl_np")
class FilterBandpass(SeriesOperator):
    """带通滤波器"""
    metadata = OperatorMetadata(
        name="filter_bandpass",
        category="math",
        description="带通滤波器",
        examples=["filter_bandpass(x, 0.05, 0.2)"],
        param_names=["x", "low", "high"],
        return_type="series",
        tags=["math", "fourier", "filter", "bandpass"]
    )

    def _calculate_series(self, x: pd.DataFrame, low: float = 0.05, high: float = 0.2, **kwargs) -> pd.DataFrame:
        try:
            from scipy.signal import butter, lfilter
            nyquist = 0.5
            normalized_low = min(low, nyquist - 1e-10)
            normalized_high = min(high, nyquist - 1e-10)
            b, a = butter(4, [normalized_low, normalized_high], btype='band')
            result = np.apply_along_axis(lambda col: lfilter(b, a, col), 0, x.values)
            return pd.DataFrame(result, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)
        except Exception:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=filter_highpass backend=pandas_numpy selected=filter_highpass source=math/fourier_ops.py
@register_operator(name="filter_highpass", category="math", business_category="elementwise_math", canonical="filter_highpass", source="factor_dsl_np")
class FilterHighpass(SeriesOperator):
    """高通滤波器"""
    metadata = OperatorMetadata(
        name="filter_highpass",
        category="math",
        description="高通滤波器",
        examples=["filter_highpass(x, 0.1)"],
        param_names=["x", "cutoff"],
        return_type="series",
        tags=["math", "fourier", "filter", "highpass"]
    )

    def _calculate_series(self, x: pd.DataFrame, cutoff: float = 0.1, **kwargs) -> pd.DataFrame:
        try:
            from scipy.signal import butter, lfilter
            nyquist = 0.5
            normalized_cutoff = min(cutoff, nyquist - 1e-10)
            b, a = butter(4, normalized_cutoff, btype='high')
            result = np.apply_along_axis(lambda col: lfilter(b, a, col), 0, x.values)
            return pd.DataFrame(result, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)
        except Exception:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=filter_lowpass backend=pandas_numpy selected=filter_lowpass source=math/fourier_ops.py
@register_operator(name="filter_lowpass", category="math", business_category="elementwise_math", canonical="filter_lowpass", source="factor_dsl_np")
class FilterLowpass(SeriesOperator):
    """低通滤波器"""
    metadata = OperatorMetadata(
        name="filter_lowpass",
        category="math",
        description="低通滤波器",
        examples=["filter_lowpass(x, 0.1)"],
        param_names=["x", "cutoff"],
        return_type="series",
        tags=["math", "fourier", "filter", "lowpass"]
    )

    def _calculate_series(self, x: pd.DataFrame, cutoff: float = 0.1, **kwargs) -> pd.DataFrame:
        try:
            from scipy.signal import butter, lfilter
            nyquist = 0.5
            normalized_cutoff = min(cutoff, nyquist - 1e-10)
            b, a = butter(4, normalized_cutoff, btype='low')
            result = np.apply_along_axis(lambda col: lfilter(b, a, col), 0, x.values)
            return pd.DataFrame(result, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)
        except Exception:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=filter_notch backend=pandas_numpy selected=filter_notch source=math/fourier_ops.py
@register_operator(name="filter_notch", category="math", business_category="elementwise_math", canonical="filter_notch", source="factor_dsl_np")
class FilterNotch(SeriesOperator):
    """陷波滤波器"""
    metadata = OperatorMetadata(
        name="filter_notch",
        category="math",
        description="陷波滤波器",
        examples=["filter_notch(x, 0.1, 0.02)"],
        param_names=["x", "freq", "width"],
        return_type="series",
        tags=["math", "fourier", "filter", "notch"]
    )

    def _calculate_series(self, x: pd.DataFrame, freq: float = 0.1, width: float = 0.02, **kwargs) -> pd.DataFrame:
        try:
            from scipy.signal import iirnotch, lfilter
            nyquist = 0.5
            w0 = min(freq, nyquist - 1e-10)
            q = w0 / max(width, 1e-10)
            b, a = iirnotch(w0, q)
            result = np.apply_along_axis(lambda col: lfilter(b, a, col), 0, x.values)
            return pd.DataFrame(result, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)
        except Exception:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=fix backend=pandas_numpy selected=fix source=math/rounding.py
@register_operator(name="fix", category="math", business_category="elementwise_math", canonical="fix", source="factor_dsl_np")
class Fix(SeriesOperator):
    """取整数部分"""

    metadata = OperatorMetadata(
        name="fix",
        category="math",
        description="取整数部分（向0取整）",
        examples=["fix(-3.7)", "fix(3.7)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "rounding", "integer"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x - x % 1 * np.sign(x)



# canonical=floor backend=pandas_numpy selected=floor source=math/rounding.py
@register_operator(name="floor", category="math", business_category="elementwise_math", canonical="floor", source="factor_dsl_np")
class Floor(SeriesOperator):
    """向下取整"""

    metadata = OperatorMetadata(
        name="floor",
        category="math",
        description="向下取整",
        examples=["floor(price)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "rounding", "down"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.floor(x)



# canonical=flex_max backend=pandas_numpy selected=flex_max source=math/special.py
@register_operator(name="flex_max", category="math", business_category="elementwise_math", canonical="flex_max", source="factor_dsl_np")
class FlexMax(SeriesOperator):
    """逐点 max 或滚动 max：第二参数为序列时逐点，为整数时滚动窗口。"""

    metadata = OperatorMetadata(
        name="flex_max",
        category="math",
        description="max(x,y) 逐点比较；max(x,n) 滚动最大值",
        examples=["max(abs(a), abs(b))", "max(high, 9)"],
        param_names=["x", "y_or_window"],
        return_type="series",
        tags=["math", "max", "elementwise", "rolling"],
    )

    def _calculate_series(self, x: pd.DataFrame, y_or_window, **kwargs) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame) and isinstance(y_or_window, pd.DataFrame):
            out = np.fmax(x.to_numpy(dtype=np.float64, copy=False), y_or_window.to_numpy(dtype=np.float64, copy=False))
            return pd.DataFrame(out, index=x.index, columns=x.columns)
        if isinstance(x, pd.DataFrame) and isinstance(y_or_window, (int, float)) and not isinstance(y_or_window, bool):
            if isinstance(y_or_window, float) and float(y_or_window) != int(y_or_window):
                out = np.fmax(x.to_numpy(dtype=np.float64, copy=False), float(y_or_window))
                return pd.DataFrame(out, index=x.index, columns=x.columns)
            window = int(y_or_window)
            # max(series, 0) 等 GTJA191 常见写法：非正标量做逐点比较，不做 rolling
            if window <= 0:
                out = np.fmax(x.to_numpy(dtype=np.float64, copy=False), float(y_or_window))
                return pd.DataFrame(out, index=x.index, columns=x.columns)
            return x.rolling(window=window, min_periods=1).max()
        if isinstance(y_or_window, pd.DataFrame) and isinstance(x, (int, float)) and not isinstance(x, bool):
            out = np.fmax(float(x), y_or_window.to_numpy(dtype=np.float64, copy=False))
            return pd.DataFrame(out, index=y_or_window.index, columns=y_or_window.columns)
        raise TypeError(f"flex_max expects series or window, got {type(y_or_window)}")



# canonical=flex_min backend=pandas_numpy selected=flex_min source=math/special.py
@register_operator(name="flex_min", category="math", business_category="elementwise_math", canonical="flex_min", source="factor_dsl_np")
class FlexMin(SeriesOperator):
    """逐点 min 或滚动 min：第二参数为序列时逐点，为整数时滚动窗口。"""

    metadata = OperatorMetadata(
        name="flex_min",
        category="math",
        description="min(x,y) 逐点比较；min(x,n) 滚动最小值",
        examples=["min(low, delay(close, 1))", "min(close, 5)"],
        param_names=["x", "y_or_window"],
        return_type="series",
        tags=["math", "min", "elementwise", "rolling"],
    )

    def _calculate_series(self, x: pd.DataFrame, y_or_window, **kwargs) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame) and isinstance(y_or_window, pd.DataFrame):
            out = np.fmin(x.to_numpy(dtype=np.float64, copy=False), y_or_window.to_numpy(dtype=np.float64, copy=False))
            return pd.DataFrame(out, index=x.index, columns=x.columns)
        if isinstance(x, pd.DataFrame) and isinstance(y_or_window, (int, float)) and not isinstance(y_or_window, bool):
            if isinstance(y_or_window, float) and float(y_or_window) != int(y_or_window):
                out = np.fmin(x.to_numpy(dtype=np.float64, copy=False), float(y_or_window))
                return pd.DataFrame(out, index=x.index, columns=x.columns)
            window = int(y_or_window)
            if window <= 0:
                out = np.fmin(x.to_numpy(dtype=np.float64, copy=False), float(y_or_window))
                return pd.DataFrame(out, index=x.index, columns=x.columns)
            return x.rolling(window=window, min_periods=1).min()
        if isinstance(y_or_window, pd.DataFrame) and isinstance(x, (int, float)) and not isinstance(x, bool):
            out = np.fmin(float(x), y_or_window.to_numpy(dtype=np.float64, copy=False))
            return pd.DataFrame(out, index=y_or_window.index, columns=y_or_window.columns)
        raise TypeError(f"flex_min expects series or window, got {type(y_or_window)}")



# canonical=fmax backend=pandas_numpy selected=fmax source=math/special.py
@register_operator(name="fmax", category="math", business_category="elementwise_math", canonical="fmax", source="factor_dsl_np")
class Fmax(SeriesOperator):
    """NaN感知的最大值（若一个为NaN则返回另一个）"""
    metadata = OperatorMetadata(
        name="fmax",
        category="math",
        description="NaN感知的最大值（若一个为NaN则返回另一个）",
        examples=["fmax(a, b)"],
        param_names=["a", "b"],
        return_type="series",
        tags=["math", "max", "nan-aware"]
    )

    def _calculate_series(self, a: pd.DataFrame, b: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        result = np.fmax(a, b)
        return pd.DataFrame(result, index=a.index, columns=a.columns).replace([np.inf, -np.inf], np.nan)



# canonical=fmin backend=pandas_numpy selected=fmin source=math/special.py
@register_operator(name="fmin", category="math", business_category="elementwise_math", canonical="fmin", source="factor_dsl_np")
class Fmin(SeriesOperator):
    """NaN感知的最小值（若一个为NaN则返回另一个）"""
    metadata = OperatorMetadata(
        name="fmin",
        category="math",
        description="NaN感知的最小值（若一个为NaN则返回另一个）",
        examples=["fmin(a, b)"],
        param_names=["a", "b"],
        return_type="series",
        tags=["math", "min", "nan-aware"]
    )

    def _calculate_series(self, a: pd.DataFrame, b: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        result = np.fmin(a, b)
        return pd.DataFrame(result, index=a.index, columns=a.columns).replace([np.inf, -np.inf], np.nan)



# canonical=geometric_mean backend=pandas_numpy selected=geometric_mean source=math/utility_ops.py
@register_operator(name="geometric_mean", category="math", business_category="elementwise_math", canonical="geometric_mean", source="factor_dsl_np")
class GeometricMean(SeriesOperator):
    """几何平均数"""
    metadata = OperatorMetadata(
        name="geometric_mean",
        category="math",
        description="几何平均数",
        examples=["geometric_mean(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "utility", "mean", "geometric"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_geometric_mean(x).replace([np.inf, -np.inf], np.nan)



# canonical=harmonic_mean backend=pandas_numpy selected=harmonic_mean source=math/utility_ops.py
@register_operator(name="harmonic_mean", category="math", business_category="elementwise_math", canonical="harmonic_mean", source="factor_dsl_np")
class HarmonicMean(SeriesOperator):
    """调和平均数"""
    metadata = OperatorMetadata(
        name="harmonic_mean",
        category="math",
        description="调和平均数",
        examples=["harmonic_mean(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "utility", "mean", "harmonic"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return expanding_harmonic_mean(x).replace([np.inf, -np.inf], np.nan)



# canonical=identity backend=pandas_numpy selected=identity source=math/utility_ops.py
@register_operator(name="identity", category="math", business_category="elementwise_math", canonical="identity", source="factor_dsl_np")
class Identity(SeriesOperator):
    """返回x不变"""
    metadata = OperatorMetadata(
        name="identity",
        category="math",
        description="返回x不变",
        examples=["identity(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "utility"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x



# canonical=idft backend=pandas_numpy selected=idft source=math/fourier_ops.py
@register_operator(name="idft", category="math", business_category="elementwise_math", canonical="idft", source="factor_dsl_np")
class Idft(SeriesOperator):
    """逆离散傅里叶变换"""
    metadata = OperatorMetadata(
        name="idft",
        category="math",
        description="逆离散傅里叶变换",
        examples=["idft(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "fourier", "idft"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        try:
            return rolling_panel_irfft(x).replace([np.inf, -np.inf], np.nan)
        except Exception:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=ifft backend=pandas_numpy selected=ifft source=math/fourier_ops.py
@register_operator(name="ifft", category="math", business_category="elementwise_math", canonical="ifft", source="factor_dsl_np")
class Ifft(SeriesOperator):
    """逆快速傅里叶变换"""
    metadata = OperatorMetadata(
        name="ifft",
        category="math",
        description="逆快速傅里叶变换",
        examples=["ifft(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "fourier", "ifft"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        try:
            return rolling_panel_irfft(x).replace([np.inf, -np.inf], np.nan)
        except Exception:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=imag backend=pandas_numpy selected=imag source=math/complex_ops.py
@register_operator(name="imag", category="math", business_category="elementwise_math", canonical="imag", source="factor_dsl_np")
class Imag(SeriesOperator):
    """取复数虚部"""
    metadata = OperatorMetadata(
        name="imag",
        category="math",
        description="取复数虚部",
        examples=["imag(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "complex"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        result = np.imag(x.values)
        return pd.DataFrame(result, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)



# canonical=interpolate backend=pandas_numpy selected=interpolate source=math/fourier_ops.py
@register_operator(name="interpolate", category="math", business_category="elementwise_math", canonical="interpolate", source="factor_dsl_np")
class Interpolate(SeriesOperator):
    """上采样插值"""
    metadata = OperatorMetadata(
        name="interpolate",
        category="math",
        description="上采样插值",
        examples=["interpolate(x, 2)"],
        param_names=["x", "factor"],
        return_type="series",
        tags=["math", "fourier", "interpolate", "upsample"]
    )

    def _calculate_series(self, x: pd.DataFrame, factor: int = 2, **kwargs) -> pd.DataFrame:
        # 全序列插值会使用未来样本；因子链路中保持原序列（仅 ffill 补缺）
        return x.ffill()



# canonical=inv backend=pandas_numpy selected=inv source=math/elementary.py

# helper for inv
class Reciprocal(SeriesOperator):
    """倒数"""

    metadata = OperatorMetadata(
        name="reciprocal",
        category="math",
        description="返回1/x",
        examples=["reciprocal(close)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "inverse"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return 1 / x

@register_operator(name="inv", category="math", business_category="elementwise_math", canonical="inv", source="factor_dsl_np")
class Inv(Reciprocal):
    """倒数（inv的别名）"""

    metadata = OperatorMetadata(
        name="inv",
        category="math",
        description="返回1/x（与reciprocal相同）",
        examples=["inv(close)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "inverse"]
    )



# canonical=lerp backend=pandas_numpy selected=lerp source=math/special.py
@register_operator(name="lerp", category="math", business_category="elementwise_math", canonical="lerp", source="factor_dsl_np")
class Lerp(SeriesOperator):
    """线性插值 a + f*(b-a)"""
    metadata = OperatorMetadata(
        name="lerp",
        category="math",
        description="线性插值 a + f*(b-a)",
        examples=["lerp(a, b, 0.5)"],
        param_names=["a", "b", "f"],
        return_type="series",
        tags=["math", "interpolation"]
    )

    def _calculate_series(self, a: pd.DataFrame, b: pd.DataFrame, f: float = 0.5, **kwargs) -> pd.DataFrame:
        result = a + f * (b - a)
        return result.replace([np.inf, -np.inf], np.nan)



# canonical=log backend=pandas_numpy selected=log source=math/elementary.py
@register_operator(name="log", category="math", business_category="elementwise_math", canonical="log", source="factor_dsl_np")
class Log(SeriesOperator):
    """自然对数"""

    metadata = OperatorMetadata(
        name="log",
        category="math",
        description="计算自然对数（底数为e）",
        examples=["log(close)", "log(volume)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "logarithm", "ln"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.log(x)

# aliases: LOG, ln



# canonical=log10 backend=pandas_numpy selected=log10 source=math/elementary.py
@register_operator(name="log10", category="math", business_category="elementwise_math", canonical="log10", source="factor_dsl_np")
class Log10(SeriesOperator):
    """以10为底的对数"""

    metadata = OperatorMetadata(
        name="log10",
        category="math",
        description="计算以10为底的对数",
        examples=["log10(price)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "logarithm", "lg"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.log10(x)



# canonical=log2 backend=pandas_numpy selected=log2 source=math/elementary.py
@register_operator(name="log2", category="math", business_category="elementwise_math", canonical="log2", source="factor_dsl_np")
class Log2(SeriesOperator):
    """以2为底的对数"""

    metadata = OperatorMetadata(
        name="log2",
        category="math",
        description="计算以2为底的对数",
        examples=["log2(volume)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "logarithm", "binary"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.log2(x)



# canonical=log_abs backend=pandas_numpy selected=log_abs source=math/utility_ops.py
@register_operator(name="log_abs", category="math", business_category="elementwise_math", canonical="log_abs", source="factor_dsl_np")
class LogAbs(SeriesOperator):
    """返回log(abs(x))"""
    metadata = OperatorMetadata(
        name="log_abs",
        category="math",
        description="返回log(abs(x))",
        examples=["log_abs(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "utility"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.log(x.abs()).replace([np.inf, -np.inf], np.nan)



# canonical=lu_decompose backend=pandas_numpy selected=lu_decompose source=math/matrix_ops.py
@register_operator(name="lu_decompose", category="math", business_category="elementwise_math", canonical="lu_decompose", source="factor_dsl_np")
class LuDecompose(SeriesOperator):
    """LU分解"""
    metadata = OperatorMetadata(
        name="lu_decompose",
        category="math",
        description="LU分解",
        examples=["lu_decompose(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "matrix", "lu", "decomposition"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        try:
            from scipy.linalg import lu
            _, l, u = lu(x.values)
            result = l + u - np.eye(x.shape[0], x.shape[1])
            return pd.DataFrame(result, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)
        except Exception:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=mat_add backend=pandas_numpy selected=mat_add source=math/matrix_ops.py
@register_operator(name="mat_add", category="math", business_category="elementwise_math", canonical="mat_add", source="factor_dsl_np")
class MatAdd(SeriesOperator):
    """矩阵加法"""
    metadata = OperatorMetadata(
        name="mat_add",
        category="math",
        description="矩阵加法",
        examples=["mat_add(a, b)"],
        param_names=["a", "b"],
        return_type="series",
        tags=["math", "matrix", "add"]
    )

    def _calculate_series(self, a: pd.DataFrame, b: pd.DataFrame, **kwargs) -> pd.DataFrame:
        result = a.values + b.values
        return pd.DataFrame(result, index=a.index, columns=a.columns).replace([np.inf, -np.inf], np.nan)



# canonical=mat_determinant backend=pandas_numpy selected=mat_determinant source=math/matrix_ops.py
@register_operator(name="mat_determinant", category="math", business_category="elementwise_math", canonical="mat_determinant", source="factor_dsl_np")
class MatDeterminant(SeriesOperator):
    """矩阵行列式"""
    metadata = OperatorMetadata(
        name="mat_determinant",
        category="math",
        description="矩阵行列式",
        examples=["mat_determinant(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "matrix", "determinant"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        try:
            det = np.linalg.det(x.values)
            result = np.full_like(x.values, det, dtype=float)
            return pd.DataFrame(result, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)
        except np.linalg.LinAlgError:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=mat_inverse backend=pandas_numpy selected=mat_inverse source=math/matrix_ops.py
@register_operator(name="mat_inverse", category="math", business_category="elementwise_math", canonical="mat_inverse", source="factor_dsl_np")
class MatInverse(SeriesOperator):
    """矩阵求逆"""
    metadata = OperatorMetadata(
        name="mat_inverse",
        category="math",
        description="矩阵求逆",
        examples=["mat_inverse(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "matrix", "inverse"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        try:
            result = np.linalg.inv(x.values)
            return pd.DataFrame(result, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)
        except np.linalg.LinAlgError:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=mat_multiply backend=pandas_numpy selected=mat_multiply source=math/matrix_ops.py
@register_operator(name="mat_multiply", category="math", business_category="elementwise_math", canonical="mat_multiply", source="factor_dsl_np")
class MatMultiply(SeriesOperator):
    """矩阵乘法"""
    metadata = OperatorMetadata(
        name="mat_multiply",
        category="math",
        description="矩阵乘法",
        examples=["mat_multiply(a, b)"],
        param_names=["a", "b"],
        return_type="series",
        tags=["math", "matrix", "multiply"]
    )

    def _calculate_series(self, a: pd.DataFrame, b: pd.DataFrame, **kwargs) -> pd.DataFrame:
        try:
            result = np.dot(a.values, b.values)
            return pd.DataFrame(result).replace([np.inf, -np.inf], np.nan)
        except (ValueError, np.linalg.LinAlgError):
            return pd.DataFrame(np.nan, index=a.index, columns=a.columns)



# canonical=mat_rank backend=pandas_numpy selected=mat_rank source=math/matrix_ops.py
@register_operator(name="mat_rank", category="math", business_category="elementwise_math", canonical="mat_rank", source="factor_dsl_np")
class MatRank(SeriesOperator):
    """矩阵秩"""
    metadata = OperatorMetadata(
        name="mat_rank",
        category="math",
        description="矩阵秩",
        examples=["mat_rank(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "matrix", "rank"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        try:
            rank = np.linalg.matrix_rank(x.values)
            result = np.full_like(x.values, float(rank), dtype=float)
            return pd.DataFrame(result, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)
        except np.linalg.LinAlgError:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=mat_subtract backend=pandas_numpy selected=mat_subtract source=math/matrix_ops.py
@register_operator(name="mat_subtract", category="math", business_category="elementwise_math", canonical="mat_subtract", source="factor_dsl_np")
class MatSubtract(SeriesOperator):
    """矩阵减法"""
    metadata = OperatorMetadata(
        name="mat_subtract",
        category="math",
        description="矩阵减法",
        examples=["mat_subtract(a, b)"],
        param_names=["a", "b"],
        return_type="series",
        tags=["math", "matrix", "subtract"]
    )

    def _calculate_series(self, a: pd.DataFrame, b: pd.DataFrame, **kwargs) -> pd.DataFrame:
        result = a.values - b.values
        return pd.DataFrame(result, index=a.index, columns=a.columns).replace([np.inf, -np.inf], np.nan)



# canonical=mat_transpose backend=pandas_numpy selected=mat_transpose source=math/matrix_ops.py
@register_operator(name="mat_transpose", category="math", business_category="elementwise_math", canonical="mat_transpose", source="factor_dsl_np")
class MatTranspose(SeriesOperator):
    """矩阵转置"""
    metadata = OperatorMetadata(
        name="mat_transpose",
        category="math",
        description="矩阵转置",
        examples=["mat_transpose(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "matrix", "transpose"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        result = x.values.T
        return pd.DataFrame(result).replace([np.inf, -np.inf], np.nan)



# canonical=neg backend=pandas_numpy selected=neg source=math/elementary.py
@register_operator(name="neg", category="math", business_category="elementwise_math", canonical="neg", source="factor_dsl_np")
class Neg(SeriesOperator):
    """取负数"""

    metadata = OperatorMetadata(
        name="neg",
        category="math",
        description="返回-x",
        examples=["neg(close)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "unary", "negative"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return -x



# canonical=negate backend=pandas_numpy selected=negate source=math/utility_ops.py
@register_operator(name="negate", category="math", business_category="elementwise_math", canonical="negate", source="factor_dsl_np")
class Negate(SeriesOperator):
    """返回-x"""
    metadata = OperatorMetadata(
        name="negate",
        category="math",
        description="返回-x",
        examples=["negate(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "utility"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return (-x).replace([np.inf, -np.inf], np.nan)



# canonical=norm backend=pandas_numpy selected=norm source=math/matrix_ops.py
@register_operator(name="norm", category="math", business_category="elementwise_math", canonical="norm", source="factor_dsl_np")
class Norm(SeriesOperator):
    """L2范数"""
    metadata = OperatorMetadata(
        name="norm",
        category="math",
        description="L2范数",
        examples=["norm(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "matrix", "norm", "l2"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        try:
            n = np.linalg.norm(x.values, ord=2)
            result = np.full_like(x.values, n, dtype=float)
            return pd.DataFrame(result, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)
        except np.linalg.LinAlgError:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=norm_l1 backend=pandas_numpy selected=norm_l1 source=math/matrix_ops.py
@register_operator(name="norm_l1", category="math", business_category="elementwise_math", canonical="norm_l1", source="factor_dsl_np")
class NormL1(SeriesOperator):
    """L1范数"""
    metadata = OperatorMetadata(
        name="norm_l1",
        category="math",
        description="L1范数",
        examples=["norm_l1(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "matrix", "norm", "l1"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        try:
            n = np.linalg.norm(x.values, ord=1)
            result = np.full_like(x.values, n, dtype=float)
            return pd.DataFrame(result, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)
        except np.linalg.LinAlgError:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=norm_linf backend=pandas_numpy selected=norm_linf source=math/matrix_ops.py
@register_operator(name="norm_linf", category="math", business_category="elementwise_math", canonical="norm_linf", source="factor_dsl_np")
class NormLinf(SeriesOperator):
    """L-inf范数"""
    metadata = OperatorMetadata(
        name="norm_linf",
        category="math",
        description="L-inf范数",
        examples=["norm_linf(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "matrix", "norm", "linf"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        try:
            n = np.linalg.norm(x.values, ord=np.inf)
            result = np.full_like(x.values, n, dtype=float)
            return pd.DataFrame(result, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)
        except np.linalg.LinAlgError:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=normalize backend=pandas_numpy selected=normalize source=math/utility_ops.py
@register_operator(name="normalize", category="math", business_category="elementwise_math", canonical="normalize", source="factor_dsl_np")
class Normalize(SeriesOperator):
    """归一化到[0, 1]"""
    metadata = OperatorMetadata(
        name="normalize",
        category="math",
        description="归一化到[0, 1]",
        examples=["normalize(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "utility", "normalize"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        min_val = x.min(axis=1)
        max_val = x.max(axis=1)
        range_val = (max_val - min_val).replace(0, np.nan)
        result = x.sub(min_val, axis=0).div(range_val, axis=0)
        return result.replace([np.inf, -np.inf], np.nan)



# canonical=pca backend=pandas_numpy selected=pca source=math/matrix_ops.py
@register_operator(name="pca", category="math", business_category="elementwise_math", canonical="pca", source="factor_dsl_np")
class Pca(SeriesOperator):
    """PCA降维"""
    metadata = OperatorMetadata(
        name="pca",
        category="math",
        description="PCA降维",
        examples=["pca(x, 2)"],
        param_names=["x", "n_components"],
        return_type="series",
        tags=["math", "matrix", "pca", "dimensionality"]
    )

    def _calculate_series(self, x: pd.DataFrame, n_components: int = 2, **kwargs) -> pd.DataFrame:
        try:
            arr = x.values.astype(float)
            t_rows, n_cols = arr.shape
            n_comp = min(n_components, n_cols)
            out = np.full_like(arr, np.nan, dtype=float)
            for i in range(max(2, n_comp), t_rows):
                sub = arr[: i + 1]
                mu = np.nanmean(sub, axis=0)
                centered = sub - mu
                valid_rows = np.all(np.isfinite(centered), axis=1)
                if valid_rows.sum() < max(3, n_comp + 1):
                    continue
                c = centered[valid_rows]
                cov = np.cov(c, rowvar=False)
                if cov.ndim < 2:
                    continue
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                idx = np.argsort(eigenvalues)[::-1]
                vecs = eigenvectors[:, idx[:n_comp]]
                last = centered[-1]
                if not np.all(np.isfinite(last)):
                    continue
                transformed = last @ vecs
                out[i, :n_comp] = transformed[:n_comp]
            return pd.DataFrame(out, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)
        except Exception:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=phase backend=pandas_numpy selected=phase source=math/complex_ops.py
@register_operator(name="phase", category="math", business_category="elementwise_math", canonical="phase", source="factor_dsl_np")
class Phase(SeriesOperator):
    """复数相位（角度）"""
    metadata = OperatorMetadata(
        name="phase",
        category="math",
        description="复数相位（角度）",
        examples=["phase(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "complex", "phase"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        result = np.angle(x.values, deg=True)
        return pd.DataFrame(result, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)



# canonical=polar backend=pandas_numpy selected=polar source=math/complex_ops.py
@register_operator(name="polar", category="math", business_category="elementwise_math", canonical="polar", source="factor_dsl_np")
class Polar(SeriesOperator):
    """极坐标转复数"""
    metadata = OperatorMetadata(
        name="polar",
        category="math",
        description="极坐标转复数",
        examples=["polar(magnitude, phase_degrees)"],
        param_names=["mag", "phase"],
        return_type="series",
        tags=["math", "complex", "polar"]
    )

    def _calculate_series(self, mag: pd.DataFrame, phase: pd.DataFrame, **kwargs) -> pd.DataFrame:
        phase_rad = np.deg2rad(phase.values)
        result = mag.values * np.exp(1j * phase_rad)
        return pd.DataFrame(result, index=mag.index, columns=mag.columns)



# canonical=power backend=pandas_numpy selected=pow source=math/elementary.py
@register_operator(name="pow", category="math", business_category="elementwise_math", canonical="power", source="factor_dsl_np")
class Pow(SeriesOperator):
    """幂函数"""

    metadata = OperatorMetadata(
        name="pow",
        category="math",
        description="计算x的y次方",
        examples=["pow(close, 2)", "pow(volume, 0.5)"],
        param_names=["x", "y"],
        return_type="series",
        tags=["math", "power", "exponent"]
    )

    def _calculate_series(self, x: pd.DataFrame, y: float = 2, **kwargs) -> pd.DataFrame:
        return np.power(x, y)

# aliases: POWER



# canonical=qr_decompose backend=pandas_numpy selected=qr_decompose source=math/matrix_ops.py
@register_operator(name="qr_decompose", category="math", business_category="elementwise_math", canonical="qr_decompose", source="factor_dsl_np")
class QrDecompose(SeriesOperator):
    """QR分解"""
    metadata = OperatorMetadata(
        name="qr_decompose",
        category="math",
        description="QR分解",
        examples=["qr_decompose(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "matrix", "qr", "decomposition"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        try:
            q, r = np.linalg.qr(x.values)
            m, n = x.values.shape
            r_padded = np.zeros((m, n), dtype=float)
            r_padded[:r.shape[0], :r.shape[1]] = r
            result = q + r_padded
            return pd.DataFrame(result, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)
        except np.linalg.LinAlgError:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=rank_transform backend=pandas_numpy selected=rank_transform source=math/utility_ops.py
@register_operator(name="rank_transform", category="math", business_category="elementwise_math", canonical="rank_transform", source="factor_dsl_np")
class RankTransform(SeriesOperator):
    """秩变换"""
    metadata = OperatorMetadata(
        name="rank_transform",
        category="math",
        description="秩变换",
        examples=["rank_transform(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "utility", "rank"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.rank(axis=1, pct=True).replace([np.inf, -np.inf], np.nan)



# canonical=rankavg_transform backend=pandas_numpy selected=rankavg_transform source=math/utility_ops.py
@register_operator(name="rankavg_transform", category="math", business_category="elementwise_math", canonical="rankavg_transform", source="factor_dsl_np")
class RankavgTransform(SeriesOperator):
    """平均秩变换"""
    metadata = OperatorMetadata(
        name="rankavg_transform",
        category="math",
        description="平均秩变换",
        examples=["rankavg_transform(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "utility", "transform", "rank"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.rank(axis=1, method='average').replace([np.inf, -np.inf], np.nan)



# canonical=real backend=pandas_numpy selected=real source=math/complex_ops.py
@register_operator(name="real", category="math", business_category="elementwise_math", canonical="real", source="factor_dsl_np")
class Real(SeriesOperator):
    """取复数实部"""
    metadata = OperatorMetadata(
        name="real",
        category="math",
        description="取复数实部",
        examples=["real(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "complex"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        result = np.real(x.values)
        return pd.DataFrame(result, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)



# canonical=reciprocal backend=pandas_numpy selected=reciprocal source=math/elementary.py
@register_operator(name="reciprocal", category="math", business_category="elementwise_math", canonical="reciprocal", source="factor_dsl_np")
class Reciprocal(SeriesOperator):
    """倒数"""

    metadata = OperatorMetadata(
        name="reciprocal",
        category="math",
        description="返回1/x",
        examples=["reciprocal(close)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "inverse"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return 1 / x



# canonical=round backend=pandas_numpy selected=round source=math/rounding.py
@register_operator(name="round", category="math", business_category="elementwise_math", canonical="round", source="factor_dsl_np")
class Round(SeriesOperator):
    """四舍五入"""

    metadata = OperatorMetadata(
        name="round",
        category="math",
        description="四舍五入（保留k位小数）",
        examples=["round(price, 2)"],
        param_names=["x", "k"],
        return_type="series",
        tags=["math", "rounding"]
    )

    def _calculate_series(self, x: pd.DataFrame, k: int = 0, **kwargs) -> pd.DataFrame:
        return x.round(decimals=k)

# aliases: ROUND



# canonical=running_mean backend=pandas_numpy selected=running_mean source=math/utility_ops.py
@register_operator(name="running_mean", category="math", business_category="elementwise_math", canonical="running_mean", source="factor_dsl_np")
class RunningMean(SeriesOperator):
    """滚动均值"""
    metadata = OperatorMetadata(
        name="running_mean",
        category="math",
        description="滚动均值",
        examples=["running_mean(x, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["math", "utility", "rolling", "mean"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        window = max(int(window), 1)
        return x.rolling(window=window, min_periods=1).mean().replace([np.inf, -np.inf], np.nan)



# canonical=running_std backend=pandas_numpy selected=running_std source=math/utility_ops.py
@register_operator(name="running_std", category="math", business_category="elementwise_math", canonical="running_std", source="factor_dsl_np")
class RunningStd(SeriesOperator):
    """滚动标准差"""
    metadata = OperatorMetadata(
        name="running_std",
        category="math",
        description="滚动标准差",
        examples=["running_std(x, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["math", "utility", "rolling", "std"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        window = max(int(window), 2)
        return x.rolling(window=window, min_periods=2).std().replace([np.inf, -np.inf], np.nan)



# canonical=running_sum backend=pandas_numpy selected=running_sum source=math/utility_ops.py
@register_operator(name="running_sum", category="math", business_category="elementwise_math", canonical="running_sum", source="factor_dsl_np")
class RunningSum(SeriesOperator):
    """滚动求和"""
    metadata = OperatorMetadata(
        name="running_sum",
        category="math",
        description="滚动求和",
        examples=["running_sum(x, 20)"],
        param_names=["x", "window"],
        return_type="series",
        tags=["math", "utility", "rolling", "sum"]
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **kwargs) -> pd.DataFrame:
        window = max(int(window), 1)
        return x.rolling(window=window, min_periods=1).sum().replace([np.inf, -np.inf], np.nan)



# canonical=sec backend=pandas_numpy selected=sec source=math/trigonometric.py
@register_operator(name="sec", category="math", business_category="elementwise_math", canonical="sec", source="factor_dsl_np")
class Sec(SeriesOperator):
    """正割"""

    metadata = OperatorMetadata(
        name="sec",
        category="math",
        description="计算正割值 (1/cos)",
        examples=["sec(angle)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "trigonometric"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return 1 / np.cos(x)



# canonical=sign backend=pandas_numpy selected=sign source=math/elementary.py
@register_operator(name="sign", category="math", business_category="elementwise_math", canonical="sign", source="factor_dsl_np")
class Sign(SeriesOperator):
    """符号函数"""

    metadata = OperatorMetadata(
        name="sign",
        category="math",
        description="返回符号：正数返回1，负数返回-1，零返回0",
        examples=["sign(Return)", "sign(close - open)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "unary", "symbol"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.sign(x)

# aliases: SIGN



# canonical=sin backend=pandas_numpy selected=sin source=math/trigonometric.py
@register_operator(name="sin", category="math", business_category="elementwise_math", canonical="sin", source="factor_dsl_np")
class Sin(SeriesOperator):
    """正弦"""

    metadata = OperatorMetadata(
        name="sin",
        category="math",
        description="计算正弦值（弧度）",
        examples=["sin(angle)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "trigonometric"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.sin(x)



# canonical=sinh backend=pandas_numpy selected=sinh source=math/trigonometric.py
@register_operator(name="sinh", category="math", business_category="elementwise_math", canonical="sinh", source="factor_dsl_np")
class Sinh(SeriesOperator):
    """双曲正弦"""

    metadata = OperatorMetadata(
        name="sinh",
        category="math",
        description="计算双曲正弦值",
        examples=["sinh(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "hyperbolic"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.sinh(x)



# canonical=sqr backend=pandas_numpy selected=sqr source=math/elementary.py
@register_operator(name="sqr", category="math", business_category="elementwise_math", canonical="sqr", source="factor_dsl_np")
class Sqr(SeriesOperator):
    """平方"""

    metadata = OperatorMetadata(
        name="sqr",
        category="math",
        description="计算x的平方",
        examples=["sqr(close)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "square", "power"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x ** 2



# canonical=sqrt backend=pandas_numpy selected=sqrt source=math/elementary.py
@register_operator(name="sqrt", category="math", business_category="elementwise_math", canonical="sqrt", source="factor_dsl_np")
class Sqrt(SeriesOperator):
    """平方根"""

    metadata = OperatorMetadata(
        name="sqrt",
        category="math",
        description="计算平方根",
        examples=["sqrt(close)", "sqrt(volume)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "root"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.sqrt(x)

# aliases: SQRT



# canonical=sqrt_abs backend=pandas_numpy selected=sqrt_abs source=math/utility_ops.py
@register_operator(name="sqrt_abs", category="math", business_category="elementwise_math", canonical="sqrt_abs", source="factor_dsl_np")
class SqrtAbs(SeriesOperator):
    """返回sqrt(abs(x))"""
    metadata = OperatorMetadata(
        name="sqrt_abs",
        category="math",
        description="返回sqrt(abs(x))",
        examples=["sqrt_abs(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "utility"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.sqrt(x.abs()).replace([np.inf, -np.inf], np.nan)



# canonical=square backend=pandas_numpy selected=square source=math/utility_ops.py
@register_operator(name="square", category="math", business_category="elementwise_math", canonical="square", source="factor_dsl_np")
class Square(SeriesOperator):
    """返回x的平方"""
    metadata = OperatorMetadata(
        name="square",
        category="math",
        description="返回x的平方",
        examples=["square(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "utility"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return (x ** 2).replace([np.inf, -np.inf], np.nan)



# canonical=standardize backend=pandas_numpy selected=standardize source=math/utility_ops.py
@register_operator(name="standardize", category="math", business_category="elementwise_math", canonical="standardize", source="factor_dsl_np")
class Standardize(SeriesOperator):
    """Z-Score标准化"""
    metadata = OperatorMetadata(
        name="standardize",
        category="math",
        description="Z-Score标准化",
        examples=["standardize(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "utility", "standardize", "zscore"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        mean = x.mean(axis=1)
        std = x.std(axis=1).replace(0, np.nan)
        result = x.sub(mean, axis=0).div(std, axis=0)
        return result.replace([np.inf, -np.inf], np.nan)



# canonical=svd backend=pandas_numpy selected=svd source=math/matrix_ops.py
@register_operator(name="svd", category="math", business_category="elementwise_math", canonical="svd", source="factor_dsl_np")
class Svd(SeriesOperator):
    """SVD奇异值分解"""
    metadata = OperatorMetadata(
        name="svd",
        category="math",
        description="SVD奇异值分解",
        examples=["svd(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "matrix", "svd", "decomposition"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        try:
            return expanding_svd_diagonal(x).replace([np.inf, -np.inf], np.nan)
        except np.linalg.LinAlgError:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=tan backend=pandas_numpy selected=tan source=math/trigonometric.py
@register_operator(name="tan", category="math", business_category="elementwise_math", canonical="tan", source="factor_dsl_np")
class Tan(SeriesOperator):
    """正切"""

    metadata = OperatorMetadata(
        name="tan",
        category="math",
        description="计算正切值（弧度）",
        examples=["tan(angle)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "trigonometric"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.tan(x)



# canonical=tanh backend=pandas_numpy selected=tanh source=math/trigonometric.py
@register_operator(name="tanh", category="math", business_category="elementwise_math", canonical="tanh", source="factor_dsl_np")
class Tanh(SeriesOperator):
    """双曲正切"""

    metadata = OperatorMetadata(
        name="tanh",
        category="math",
        description="计算双曲正切值",
        examples=["tanh(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "hyperbolic"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.tanh(x)



# canonical=truncate backend=pandas_numpy selected=truncate source=math/rounding.py
@register_operator(name="truncate", category="math", business_category="elementwise_math", canonical="truncate", source="factor_dsl_np")
class Truncate(SeriesOperator):
    """截断"""

    metadata = OperatorMetadata(
        name="truncate",
        category="math",
        description="截断（保留k位小数）",
        examples=["truncate(price, 2)"],
        param_names=["x", "k"],
        return_type="series",
        tags=["math", "rounding", "truncation"]
    )

    def _calculate_series(self, x: pd.DataFrame, k: int = 0, **kwargs) -> pd.DataFrame:
        return np.trunc(x.round(decimals=k))



# canonical=tukey_transform backend=pandas_numpy selected=tukey_transform source=math/utility_ops.py
@register_operator(name="tukey_transform", category="math", business_category="elementwise_math", canonical="tukey_transform", source="factor_dsl_np")
class TukeyTransform(SeriesOperator):
    """Tukey变换"""
    metadata = OperatorMetadata(
        name="tukey_transform",
        category="math",
        description="Tukey变换",
        examples=["tukey_transform(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "utility", "transform"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ranked = x.rank(axis=1, pct=True)
        result = sp_stats.norm.ppf(ranked)
        return pd.DataFrame(result, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)



# canonical=unitize backend=pandas_numpy selected=unitize source=math/utility_ops.py
@register_operator(name="unitize", category="math", business_category="elementwise_math", canonical="unitize", source="factor_dsl_np")
class Unitize(SeriesOperator):
    """归一化到[-1, 1]"""
    metadata = OperatorMetadata(
        name="unitize",
        category="math",
        description="归一化到[-1, 1]",
        examples=["unitize(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "utility", "normalize"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        x_abs_max = x.abs().max(axis=1).replace(0, np.nan)
        result = x.div(x_abs_max, axis=0)
        return result.clip(-1, 1).replace([np.inf, -np.inf], np.nan)



# canonical=unwrap backend=pandas_numpy selected=unwrap source=math/fourier_ops.py
@register_operator(name="unwrap", category="math", business_category="elementwise_math", canonical="unwrap", source="factor_dsl_np")
class Unwrap(SeriesOperator):
    """相位展开"""
    metadata = OperatorMetadata(
        name="unwrap",
        category="math",
        description="相位展开",
        examples=["unwrap(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "fourier", "phase", "unwrap"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        try:
            return expanding_unwrap_panel(x).replace([np.inf, -np.inf], np.nan)
        except Exception:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=van_der_waerden_transform backend=pandas_numpy selected=van_der_waerden_transform source=math/utility_ops.py
@register_operator(name="van_der_waerden_transform", category="math", business_category="elementwise_math", canonical="van_der_waerden_transform", source="factor_dsl_np")
class VanDerWaerdenTransform(SeriesOperator):
    """Van der Waerden变换"""
    metadata = OperatorMetadata(
        name="van_der_waerden_transform",
        category="math",
        description="Van der Waerden变换",
        examples=["van_der_waerden_transform(x)"],
        param_names=["x"],
        return_type="series",
        tags=["math", "utility", "transform"]
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ranked = x.rank(axis=1, method='average')
        n = x.count(axis=1).to_numpy(dtype=float)[:, None]
        result = sp_stats.norm.ppf(ranked / (n + 1))
        return pd.DataFrame(result, index=x.index, columns=x.columns).replace([np.inf, -np.inf], np.nan)



# canonical=wavelet backend=pandas_numpy selected=wavelet source=math/fourier_ops.py
@register_operator(name="wavelet", category="math", business_category="elementwise_math", canonical="wavelet", source="factor_dsl_np")
class Wavelet(SeriesOperator):
    """小波变换"""
    metadata = OperatorMetadata(
        name="wavelet",
        category="math",
        description="小波变换",
        examples=["wavelet(x, 'db4')"],
        param_names=["x", "wavelet_type"],
        return_type="series",
        tags=["math", "fourier", "wavelet"]
    )

    def _calculate_series(self, x: pd.DataFrame, wavelet_type: str = 'db4', **kwargs) -> pd.DataFrame:
        try:
            return rolling_wavelet_panel(x, wavelet_type=wavelet_type, threshold=0.0).replace(
                [np.inf, -np.inf], np.nan
            )
        except Exception:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=wavelet_denoise backend=pandas_numpy selected=wavelet_denoise source=math/fourier_ops.py
@register_operator(name="wavelet_denoise", category="math", business_category="elementwise_math", canonical="wavelet_denoise", source="factor_dsl_np")
class WaveletDenoise(SeriesOperator):
    """小波去噪"""
    metadata = OperatorMetadata(
        name="wavelet_denoise",
        category="math",
        description="小波去噪",
        examples=["wavelet_denoise(x, 0.1)"],
        param_names=["x", "threshold"],
        return_type="series",
        tags=["math", "fourier", "wavelet", "denoise"]
    )

    def _calculate_series(self, x: pd.DataFrame, threshold: float = 0.1, **kwargs) -> pd.DataFrame:
        try:
            return rolling_wavelet_panel(x, threshold=threshold).replace([np.inf, -np.inf], np.nan)
        except Exception:
            return pd.DataFrame(np.nan, index=x.index, columns=x.columns)



# canonical=weighted_mean backend=pandas_numpy selected=weighted_mean source=math/utility_ops.py
@register_operator(name="weighted_mean", category="math", business_category="elementwise_math", canonical="weighted_mean", source="factor_dsl_np")
class WeightedMean(SeriesOperator):
    """加权平均数"""
    metadata = OperatorMetadata(
        name="weighted_mean",
        category="math",
        description="加权平均数",
        examples=["weighted_mean(x, weights)"],
        param_names=["x", "weights"],
        return_type="series",
        tags=["math", "utility", "mean", "weighted"]
    )

    def _calculate_series(self, x: pd.DataFrame, weights: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
        if weights is None:
            row_mean = x.mean(axis=1)
        else:
            w_sum = weights.sum(axis=1).replace(0, np.nan)
            row_mean = (x * weights).sum(axis=1) / w_sum
        return pd.DataFrame(
            np.tile(row_mean.values[:, None], (1, x.shape[1])),
            index=x.index,
            columns=x.columns,
        ).replace([np.inf, -np.inf], np.nan)



# canonical=winsorize backend=pandas_numpy selected=winsorize source=math/utility_ops.py
@register_operator(name="winsorize", category="math", business_category="elementwise_math", canonical="winsorize", source="factor_dsl_np")
class Winsorize(SeriesOperator):
    """缩尾处理"""
    metadata = OperatorMetadata(
        name="winsorize",
        category="math",
        description="缩尾处理",
        examples=["winsorize(x, 0.05, 0.95)"],
        param_names=["x", "lower", "upper"],
        return_type="series",
        tags=["math", "utility", "winsorize"]
    )

    def _calculate_series(self, x: pd.DataFrame, lower: float = 0.05, upper: float = 0.95, **kwargs) -> pd.DataFrame:
        lower_val = x.quantile(lower, axis=1)
        upper_val = x.quantile(upper, axis=1)
        result = x.clip(lower=lower_val, upper=upper_val, axis=0)
        return result.replace([np.inf, -np.inf], np.nan)

# aliases: WINSORIZE



# canonical=winsorize_mean backend=pandas_numpy selected=winsorize_mean source=math/utility_ops.py
@register_operator(name="winsorize_mean", category="math", business_category="elementwise_math", canonical="winsorize_mean", source="factor_dsl_np")
class WinsorizeMean(SeriesOperator):
    """截尾均值"""
    metadata = OperatorMetadata(
        name="winsorize_mean",
        category="math",
        description="截尾均值",
        examples=["winsorize_mean(x, 0.1)"],
        param_names=["x", "trim_pct"],
        return_type="series",
        tags=["math", "utility", "mean", "trimmed"]
    )

    def _calculate_series(self, x: pd.DataFrame, trim_pct: float = 0.1, **kwargs) -> pd.DataFrame:
        trim_pct = max(0.0, min(trim_pct, 0.49))
        lower = x.quantile(trim_pct, axis=1)
        upper = x.quantile(1 - trim_pct, axis=1)
        clipped = x.clip(lower=lower, upper=upper, axis=0)
        row_mean = clipped.mean(axis=1)
        return pd.DataFrame(
            np.tile(row_mean.values[:, None], (1, x.shape[1])),
            index=x.index,
            columns=x.columns,
        ).replace([np.inf, -np.inf], np.nan)


@register_operator(name="signed_sqrt", category="elementwise_math", business_category="elementwise_math", canonical="signed_sqrt", source="factor_dsl_np")
class SignedSqrt(SeriesOperator):
    """保留符号开方: sign(x)*sqrt(|x|)"""
    metadata = OperatorMetadata(
        name="signed_sqrt",
        category="elementwise_math",
        description="保留符号开方: sign(x)*sqrt(|x|)",
        param_names=["x"],
        return_type="series",
        tags=["math", "signed_sqrt"],
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        from cleaned_operators._numpy_kernels import signed_sqrt_

        return x.apply(lambda s: signed_sqrt_(s.values))


@register_operator(name="sigmoid", category="elementwise_math", business_category="elementwise_math", canonical="sigmoid", source="factor_dsl_np")
class Sigmoid(SeriesOperator):
    """Sigmoid 映射"""
    metadata = OperatorMetadata(
        name="sigmoid",
        category="elementwise_math",
        description="Sigmoid 映射",
        param_names=["x"],
        return_type="series",
        tags=["math", "sigmoid"],
    )

    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        from cleaned_operators._numpy_kernels import sigmoid_

        return x.apply(lambda s: sigmoid_(s.values))


@register_operator(name="coalesce", category="elementwise_math", business_category="elementwise_math", canonical="coalesce", source="factor_dsl_np")
class Coalesce(SeriesOperator):
    """返回第一个非 NaN 值"""
    metadata = OperatorMetadata(
        name="coalesce",
        category="elementwise_math",
        description="返回第一个非 NaN 值",
        param_names=["x", "y"],
        return_type="series",
        tags=["math", "coalesce"],
    )

    def _calculate_series(self, *args, **kwargs) -> pd.DataFrame:
        from cleaned_operators._numpy_kernels import coalesce_

        if len(args) < 2:
            raise ValueError("coalesce 至少需要两个 panel 参数")
        base = args[0]
        result = base.copy()
        for other in args[1:]:
            for idx in result.index:
                for col in result.columns:
                    if col in other.columns and pd.isna(result.at[idx, col]):
                        result.at[idx, col] = other.at[idx, col]
        return result

# canonical=add backend=pandas_numpy selected=add source=basic_runtime
@register_operator(name="add", category="elementwise_math", business_category="elementwise_math", canonical="add", source="basic_runtime")
class AddOp(SeriesOperator):
    """Basic runtime operator"""
    metadata = OperatorMetadata(
        name="add",
        category="elementwise_math",
        description="Basic runtime operator",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        if len(args) < 2:
            raise ValueError("add requires at least 2 inputs")
        out = args[0]
        for a in args[1:]:
            out = out + a
        return out


# canonical=subtract backend=pandas_numpy selected=subtract source=basic_runtime
@register_operator(name="subtract", category="elementwise_math", business_category="elementwise_math", canonical="subtract", source="basic_runtime")
class SubtractOp(SeriesOperator):
    """Basic runtime operator"""
    metadata = OperatorMetadata(
        name="subtract",
        category="elementwise_math",
        description="Basic runtime operator",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        if len(args) < 2:
            raise ValueError("subtract requires at least 2 inputs")
        out = args[0]
        for a in args[1:]:
            out = out - a
        return out


# canonical=multiply backend=pandas_numpy selected=multiply source=basic_runtime
@register_operator(name="multiply", category="elementwise_math", business_category="elementwise_math", canonical="multiply", source="basic_runtime")
class MultiplyOp(SeriesOperator):
    """Basic runtime operator"""
    metadata = OperatorMetadata(
        name="multiply",
        category="elementwise_math",
        description="Basic runtime operator",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, *args, **kwargs):
        if len(args) < 2:
            raise ValueError("multiply requires at least 2 inputs")
        out = args[0]
        for a in args[1:]:
            out = out * a
        return out


# canonical=divide backend=pandas_numpy selected=divide source=basic_runtime
@register_operator(name="divide", category="elementwise_math", business_category="elementwise_math", canonical="divide", source="basic_runtime")
class DivideOp(TwoVarOperator):
    """Basic runtime operator"""
    metadata = OperatorMetadata(
        name="divide",
        category="elementwise_math",
        description="Basic runtime operator",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, x, y, **kwargs):
        return x / y


# canonical=lt backend=pandas_numpy selected=lt source=basic_runtime
@register_operator(name="lt", category="elementwise_math", business_category="elementwise_math", canonical="lt", source="basic_runtime")
class LtOp(TwoVarOperator):
    """Basic runtime operator"""
    metadata = OperatorMetadata(
        name="lt",
        category="elementwise_math",
        description="Basic runtime operator",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, x, y, **kwargs):
        return (x < y).astype(float)


# canonical=le backend=pandas_numpy selected=le source=basic_runtime
@register_operator(name="le", category="elementwise_math", business_category="elementwise_math", canonical="le", source="basic_runtime")
class LeOp(TwoVarOperator):
    """Basic runtime operator"""
    metadata = OperatorMetadata(
        name="le",
        category="elementwise_math",
        description="Basic runtime operator",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, x, y, **kwargs):
        return (x <= y).astype(float)


# canonical=eq backend=pandas_numpy selected=eq source=basic_runtime
@register_operator(name="eq", category="elementwise_math", business_category="elementwise_math", canonical="eq", source="basic_runtime")
class EqOp(TwoVarOperator):
    """Basic runtime operator"""
    metadata = OperatorMetadata(
        name="eq",
        category="elementwise_math",
        description="Basic runtime operator",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, x, y, **kwargs):
        return (x == y).astype(float)


# canonical=gt backend=pandas_numpy selected=gt source=basic_runtime
@register_operator(name="gt", category="elementwise_math", business_category="elementwise_math", canonical="gt", source="basic_runtime")
class GtOp(TwoVarOperator):
    """Basic runtime operator"""
    metadata = OperatorMetadata(
        name="gt",
        category="elementwise_math",
        description="Basic runtime operator",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, x, y, **kwargs):
        return (x > y).astype(float)


# canonical=ge backend=pandas_numpy selected=ge source=basic_runtime
@register_operator(name="ge", category="elementwise_math", business_category="elementwise_math", canonical="ge", source="basic_runtime")
class GeOp(TwoVarOperator):
    """Basic runtime operator"""
    metadata = OperatorMetadata(
        name="ge",
        category="elementwise_math",
        description="Basic runtime operator",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, x, y, **kwargs):
        return (x >= y).astype(float)


# canonical=ne backend=pandas_numpy selected=ne source=basic_runtime
@register_operator(name="ne", category="elementwise_math", business_category="elementwise_math", canonical="ne", source="basic_runtime")
class NeOp(TwoVarOperator):
    """Basic runtime operator"""
    metadata = OperatorMetadata(
        name="ne",
        category="elementwise_math",
        description="Basic runtime operator",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, x, y, **kwargs):
        return (x != y).astype(float)


def _truthy_series(x: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """非空且非零为真；NULL/NaN 视为 false（与 Polars/SQL fast path 一致）。"""
    return x.notna() & (x != 0)


# canonical=and_ backend=pandas_numpy selected=and_ source=basic_runtime
@register_operator(name="and_", category="elementwise_math", business_category="elementwise_math", canonical="and_", source="basic_runtime")
class AndOp(TwoVarOperator):
    """Basic runtime operator"""
    metadata = OperatorMetadata(
        name="and_",
        category="elementwise_math",
        description="Basic runtime operator",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, x, y, **kwargs):
        return (_truthy_series(x) & _truthy_series(y)).astype(float)


# canonical=or_ backend=pandas_numpy selected=or_ source=basic_runtime
@register_operator(name="or_", category="elementwise_math", business_category="elementwise_math", canonical="or_", source="basic_runtime")
class OrOp(TwoVarOperator):
    """Basic runtime operator"""
    metadata = OperatorMetadata(
        name="or_",
        category="elementwise_math",
        description="Basic runtime operator",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, x, y, **kwargs):
        return (_truthy_series(x) | _truthy_series(y)).astype(float)


# canonical=not_ backend=pandas_numpy selected=not_ source=basic_runtime
@register_operator(name="not_", category="elementwise_math", business_category="elementwise_math", canonical="not_", source="basic_runtime")
class NotOp(SeriesOperator):
    """Basic runtime operator"""
    metadata = OperatorMetadata(
        name="not_",
        category="elementwise_math",
        description="Basic runtime operator",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, x, **kwargs):
        return (~_truthy_series(x)).astype(float)


# canonical=inverse backend=pandas_numpy selected=inverse source=basic_runtime
@register_operator(name="inverse", category="elementwise_math", business_category="elementwise_math", canonical="inverse", source="basic_runtime")
class InverseOp(SeriesOperator):
    """Basic runtime operator"""
    metadata = OperatorMetadata(
        name="inverse",
        category="elementwise_math",
        description="Basic runtime operator",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, x, **kwargs):
        return 1.0 / x.replace(0, np.nan)


# canonical=reverse backend=pandas_numpy selected=reverse source=basic_runtime
@register_operator(name="reverse", category="elementwise_math", business_category="elementwise_math", canonical="reverse", source="basic_runtime")
class ReverseOp(SeriesOperator):
    """Basic runtime operator"""
    metadata = OperatorMetadata(
        name="reverse",
        category="elementwise_math",
        description="Basic runtime operator",
        param_names=[],
        return_type="series",
    )

    def _calculate_series(self, x, **kwargs):
        return -x
