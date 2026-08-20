# -*- coding: utf-8 -*-
"""Auto-generated Polars bridge registration.

Category: all
Total operators: 919

策略：
- 所有算子通过 polars_registry_bridge 自动编译（利用长表 map_groups）
- 实际执行委托给已验证的 pandas 实现，保证语义一致性

本文件的作用是显式注册这些算子为 polars backend，
实际执行由 backend/polars_registry_bridge.py 的 fallback 机制处理。
"""

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator

# 批量注册算子为 polars backend
# 实际计算委托给 polars_registry_bridge.compile_registry_op


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="ACF",
    canonical="ACF",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AcfPolars(SeriesOperator):
    """Auto-generated Polars bridge for ACF.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ACF",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ADX",
    canonical="ADX",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AdxPolars(SeriesOperator):
    """Auto-generated Polars bridge for ADX.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ADX",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ADXR",
    canonical="ADXR",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AdxrPolars(SeriesOperator):
    """Auto-generated Polars bridge for ADXR.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ADXR",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ALMA",
    canonical="ALMA",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AlmaPolars(SeriesOperator):
    """Auto-generated Polars bridge for ALMA.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ALMA",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="AROON",
    canonical="AROON",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AroonPolars(SeriesOperator):
    """Auto-generated Polars bridge for AROON.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="AROON",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="AROON_down",
    canonical="AROON_down",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AroonDownPolars(SeriesOperator):
    """Auto-generated Polars bridge for AROON_down.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="AROON_down",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="AROON_up",
    canonical="AROON_up",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AroonUpPolars(SeriesOperator):
    """Auto-generated Polars bridge for AROON_up.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="AROON_up",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ATR",
    canonical="ATR",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AtrPolars(SeriesOperator):
    """Auto-generated Polars bridge for ATR.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ATR",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="Beta",
    canonical="Beta",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class BetaPolars(SeriesOperator):
    """Auto-generated Polars bridge for Beta.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="Beta",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="BollingerBands",
    canonical="BollingerBands",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class BollingerbandsPolars(SeriesOperator):
    """Auto-generated Polars bridge for BollingerBands.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="BollingerBands",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="BollingerLower",
    canonical="BollingerLower",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class BollingerlowerPolars(SeriesOperator):
    """Auto-generated Polars bridge for BollingerLower.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="BollingerLower",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="BollingerUpper",
    canonical="BollingerUpper",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class BollingerupperPolars(SeriesOperator):
    """Auto-generated Polars bridge for BollingerUpper.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="BollingerUpper",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="CCI",
    canonical="CCI",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CciPolars(SeriesOperator):
    """Auto-generated Polars bridge for CCI.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="CCI",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="CoppockCurve",
    canonical="CoppockCurve",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CoppockcurvePolars(SeriesOperator):
    """Auto-generated Polars bridge for CoppockCurve.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="CoppockCurve",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="Corr",
    canonical="Corr",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CorrPolars(SeriesOperator):
    """Auto-generated Polars bridge for Corr.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="Corr",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="Cov",
    canonical="Cov",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CovPolars(SeriesOperator):
    """Auto-generated Polars bridge for Cov.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="Cov",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="Covariance",
    canonical="Covariance",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CovariancePolars(SeriesOperator):
    """Auto-generated Polars bridge for Covariance.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="Covariance",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="DPO",
    canonical="DPO",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class DpoPolars(SeriesOperator):
    """Auto-generated Polars bridge for DPO.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="DPO",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ElderRay",
    canonical="ElderRay",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ElderrayPolars(SeriesOperator):
    """Auto-generated Polars bridge for ElderRay.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ElderRay",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="FisherTransform",
    canonical="FisherTransform",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class FishertransformPolars(SeriesOperator):
    """Auto-generated Polars bridge for FisherTransform.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="FisherTransform",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="HMA",
    canonical="HMA",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class HmaPolars(SeriesOperator):
    """Auto-generated Polars bridge for HMA.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="HMA",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="Intercept",
    canonical="Intercept",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class InterceptPolars(SeriesOperator):
    """Auto-generated Polars bridge for Intercept.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="Intercept",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="KAMA",
    canonical="KAMA",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class KamaPolars(SeriesOperator):
    """Auto-generated Polars bridge for KAMA.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="KAMA",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="Kurt",
    canonical="Kurt",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class KurtPolars(SeriesOperator):
    """Auto-generated Polars bridge for Kurt.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="Kurt",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="MACD",
    canonical="MACD",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MacdPolars(SeriesOperator):
    """Auto-generated Polars bridge for MACD.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="MACD",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="MACD_hist",
    canonical="MACD_hist",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MacdHistPolars(SeriesOperator):
    """Auto-generated Polars bridge for MACD_hist.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="MACD_hist",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="MACD_line",
    canonical="MACD_line",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MacdLinePolars(SeriesOperator):
    """Auto-generated Polars bridge for MACD_line.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="MACD_line",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="MACD_signal",
    canonical="MACD_signal",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MacdSignalPolars(SeriesOperator):
    """Auto-generated Polars bridge for MACD_signal.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="MACD_signal",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="MOM",
    canonical="MOM",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MomPolars(SeriesOperator):
    """Auto-generated Polars bridge for MOM.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="MOM",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="Mad",
    canonical="Mad",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MadPolars(SeriesOperator):
    """Auto-generated Polars bridge for Mad.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="Mad",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="Median",
    canonical="Median",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MedianPolars(SeriesOperator):
    """Auto-generated Polars bridge for Median.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="Median",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="Mode",
    canonical="Mode",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ModePolars(SeriesOperator):
    """Auto-generated Polars bridge for Mode.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="Mode",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="OBV",
    canonical="OBV",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ObvPolars(SeriesOperator):
    """Auto-generated Polars bridge for OBV.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="OBV",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="Percentile",
    canonical="Percentile",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class PercentilePolars(SeriesOperator):
    """Auto-generated Polars bridge for Percentile.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="Percentile",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="QQE",
    canonical="QQE",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class QqePolars(SeriesOperator):
    """Auto-generated Polars bridge for QQE.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="QQE",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="R2",
    canonical="R2",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class R2Polars(SeriesOperator):
    """Auto-generated Polars bridge for R2.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="R2",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ROC",
    canonical="ROC",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RocPolars(SeriesOperator):
    """Auto-generated Polars bridge for ROC.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ROC",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="RSI",
    canonical="RSI",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RsiPolars(SeriesOperator):
    """Auto-generated Polars bridge for RSI.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="RSI",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="RSX",
    canonical="RSX",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RsxPolars(SeriesOperator):
    """Auto-generated Polars bridge for RSX.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="RSX",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="Residual",
    canonical="Residual",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ResidualPolars(SeriesOperator):
    """Auto-generated Polars bridge for Residual.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="Residual",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="SMA",
    canonical="SMA",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SmaPolars(SeriesOperator):
    """Auto-generated Polars bridge for SMA.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="SMA",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="Skew",
    canonical="Skew",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SkewPolars(SeriesOperator):
    """Auto-generated Polars bridge for Skew.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="Skew",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="Slope",
    canonical="Slope",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SlopePolars(SeriesOperator):
    """Auto-generated Polars bridge for Slope.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="Slope",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="StochasticD",
    canonical="StochasticD",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StochasticdPolars(SeriesOperator):
    """Auto-generated Polars bridge for StochasticD.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="StochasticD",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="StochasticK",
    canonical="StochasticK",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StochastickPolars(SeriesOperator):
    """Auto-generated Polars bridge for StochasticK.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="StochasticK",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="Sum",
    canonical="Sum",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SumPolars(SeriesOperator):
    """Auto-generated Polars bridge for Sum.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="Sum",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="TRIX",
    canonical="TRIX",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TrixPolars(SeriesOperator):
    """Auto-generated Polars bridge for TRIX.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="TRIX",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="Var",
    canonical="Var",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class VarPolars(SeriesOperator):
    """Auto-generated Polars bridge for Var.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="Var",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="WMA",
    canonical="WMA",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class WmaPolars(SeriesOperator):
    """Auto-generated Polars bridge for WMA.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="WMA",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="WilliamsR",
    canonical="WilliamsR",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class WilliamsrPolars(SeriesOperator):
    """Auto-generated Polars bridge for WilliamsR.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="WilliamsR",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="abs",
    canonical="abs",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AbsPolars(SeriesOperator):
    """Auto-generated Polars bridge for abs.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="abs",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="acos",
    canonical="acos",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AcosPolars(SeriesOperator):
    """Auto-generated Polars bridge for acos.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="acos",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="acos_bounded",
    canonical="acos_bounded",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AcosBoundedPolars(SeriesOperator):
    """Auto-generated Polars bridge for acos_bounded.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="acos_bounded",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="add",
    canonical="add",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AddPolars(SeriesOperator):
    """Auto-generated Polars bridge for add.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="add",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="and_",
    canonical="and_",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AndPolars(SeriesOperator):
    """Auto-generated Polars bridge for and_.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="and_",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="arg",
    canonical="arg",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ArgPolars(SeriesOperator):
    """Auto-generated Polars bridge for arg.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="arg",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ashare_days_since_limit_down",
    canonical="ashare_days_since_limit_down",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AshareDaysSinceLimitDownPolars(SeriesOperator):
    """Auto-generated Polars bridge for ashare_days_since_limit_down.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ashare_days_since_limit_down",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ashare_days_since_limit_up",
    canonical="ashare_days_since_limit_up",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AshareDaysSinceLimitUpPolars(SeriesOperator):
    """Auto-generated Polars bridge for ashare_days_since_limit_up.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ashare_days_since_limit_up",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ashare_failed_limit_count",
    canonical="ashare_failed_limit_count",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AshareFailedLimitCountPolars(SeriesOperator):
    """Auto-generated Polars bridge for ashare_failed_limit_count.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ashare_failed_limit_count",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ashare_limit_asymmetry",
    canonical="ashare_limit_asymmetry",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AshareLimitAsymmetryPolars(SeriesOperator):
    """Auto-generated Polars bridge for ashare_limit_asymmetry.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ashare_limit_asymmetry",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ashare_limit_distance",
    canonical="ashare_limit_distance",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AshareLimitDistancePolars(SeriesOperator):
    """Auto-generated Polars bridge for ashare_limit_distance.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ashare_limit_distance",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ashare_limit_down_streak",
    canonical="ashare_limit_down_streak",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AshareLimitDownStreakPolars(SeriesOperator):
    """Auto-generated Polars bridge for ashare_limit_down_streak.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ashare_limit_down_streak",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ashare_limit_down_touch",
    canonical="ashare_limit_down_touch",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AshareLimitDownTouchPolars(SeriesOperator):
    """Auto-generated Polars bridge for ashare_limit_down_touch.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ashare_limit_down_touch",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ashare_limit_down_volume_ratio",
    canonical="ashare_limit_down_volume_ratio",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AshareLimitDownVolumeRatioPolars(SeriesOperator):
    """Auto-generated Polars bridge for ashare_limit_down_volume_ratio.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ashare_limit_down_volume_ratio",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ashare_limit_event_density",
    canonical="ashare_limit_event_density",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AshareLimitEventDensityPolars(SeriesOperator):
    """Auto-generated Polars bridge for ashare_limit_event_density.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ashare_limit_event_density",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ashare_limit_failed",
    canonical="ashare_limit_failed",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AshareLimitFailedPolars(SeriesOperator):
    """Auto-generated Polars bridge for ashare_limit_failed.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ashare_limit_failed",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ashare_limit_one_price",
    canonical="ashare_limit_one_price",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AshareLimitOnePricePolars(SeriesOperator):
    """Auto-generated Polars bridge for ashare_limit_one_price.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ashare_limit_one_price",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ashare_limit_open_down_streak",
    canonical="ashare_limit_open_down_streak",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AshareLimitOpenDownStreakPolars(SeriesOperator):
    """Auto-generated Polars bridge for ashare_limit_open_down_streak.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ashare_limit_open_down_streak",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ashare_limit_open_failed",
    canonical="ashare_limit_open_failed",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AshareLimitOpenFailedPolars(SeriesOperator):
    """Auto-generated Polars bridge for ashare_limit_open_failed.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ashare_limit_open_failed",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ashare_limit_open_up_streak",
    canonical="ashare_limit_open_up_streak",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AshareLimitOpenUpStreakPolars(SeriesOperator):
    """Auto-generated Polars bridge for ashare_limit_open_up_streak.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ashare_limit_open_up_streak",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ashare_limit_touch_count",
    canonical="ashare_limit_touch_count",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AshareLimitTouchCountPolars(SeriesOperator):
    """Auto-generated Polars bridge for ashare_limit_touch_count.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ashare_limit_touch_count",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ashare_limit_up_streak",
    canonical="ashare_limit_up_streak",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AshareLimitUpStreakPolars(SeriesOperator):
    """Auto-generated Polars bridge for ashare_limit_up_streak.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ashare_limit_up_streak",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ashare_limit_up_touch",
    canonical="ashare_limit_up_touch",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AshareLimitUpTouchPolars(SeriesOperator):
    """Auto-generated Polars bridge for ashare_limit_up_touch.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ashare_limit_up_touch",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ashare_limit_up_volume_ratio",
    canonical="ashare_limit_up_volume_ratio",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AshareLimitUpVolumeRatioPolars(SeriesOperator):
    """Auto-generated Polars bridge for ashare_limit_up_volume_ratio.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ashare_limit_up_volume_ratio",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ashare_one_price_limit_streak",
    canonical="ashare_one_price_limit_streak",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AshareOnePriceLimitStreakPolars(SeriesOperator):
    """Auto-generated Polars bridge for ashare_one_price_limit_streak.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ashare_one_price_limit_streak",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ashare_open_at_upper_limit",
    canonical="ashare_open_at_upper_limit",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AshareOpenAtUpperLimitPolars(SeriesOperator):
    """Auto-generated Polars bridge for ashare_open_at_upper_limit.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ashare_open_at_upper_limit",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ashare_suspension_episode_length",
    canonical="ashare_suspension_episode_length",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AshareSuspensionEpisodeLengthPolars(SeriesOperator):
    """Auto-generated Polars bridge for ashare_suspension_episode_length.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ashare_suspension_episode_length",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="asin",
    canonical="asin",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AsinPolars(SeriesOperator):
    """Auto-generated Polars bridge for asin.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="asin",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="asin_bounded",
    canonical="asin_bounded",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AsinBoundedPolars(SeriesOperator):
    """Auto-generated Polars bridge for asin_bounded.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="asin_bounded",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="at_imax",
    canonical="at_imax",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AtImaxPolars(SeriesOperator):
    """Auto-generated Polars bridge for at_imax.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="at_imax",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="at_imin",
    canonical="at_imin",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AtIminPolars(SeriesOperator):
    """Auto-generated Polars bridge for at_imin.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="at_imin",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="atan",
    canonical="atan",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AtanPolars(SeriesOperator):
    """Auto-generated Polars bridge for atan.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="atan",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="atan2",
    canonical="atan2",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class Atan2Polars(SeriesOperator):
    """Auto-generated Polars bridge for atan2.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="atan2",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="autocorr",
    canonical="autocorr",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AutocorrPolars(SeriesOperator):
    """Auto-generated Polars bridge for autocorr.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="autocorr",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="avg",
    canonical="avg",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class AvgPolars(SeriesOperator):
    """Auto-generated Polars bridge for avg.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="avg",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="bartlett_test",
    canonical="bartlett_test",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class BartlettTestPolars(SeriesOperator):
    """Auto-generated Polars bridge for bartlett_test.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="bartlett_test",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="baseline_scaled_wasserstein_distance",
    canonical="baseline_scaled_wasserstein_distance",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class BaselineScaledWassersteinDistancePolars(SeriesOperator):
    """Auto-generated Polars bridge for baseline_scaled_wasserstein_distance.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="baseline_scaled_wasserstein_distance",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="benchmark_excess_return",
    canonical="benchmark_excess_return",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class BenchmarkExcessReturnPolars(SeriesOperator):
    """Auto-generated Polars bridge for benchmark_excess_return.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="benchmark_excess_return",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="benchmark_relative_price",
    canonical="benchmark_relative_price",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class BenchmarkRelativePricePolars(SeriesOperator):
    """Auto-generated Polars bridge for benchmark_relative_price.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="benchmark_relative_price",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="beta",
    canonical="beta",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class BetaPolars(SeriesOperator):
    """Auto-generated Polars bridge for beta.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="beta",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="blom_transform",
    canonical="blom_transform",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class BlomTransformPolars(SeriesOperator):
    """Auto-generated Polars bridge for blom_transform.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="blom_transform",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="book_to_price",
    canonical="book_to_price",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class BookToPricePolars(SeriesOperator):
    """Auto-generated Polars bridge for book_to_price.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="book_to_price",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="c_count",
    canonical="c_count",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CCountPolars(SeriesOperator):
    """Auto-generated Polars bridge for c_count.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="c_count",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="c_mean",
    canonical="c_mean",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CMeanPolars(SeriesOperator):
    """Auto-generated Polars bridge for c_mean.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="c_mean",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="c_percentile",
    canonical="c_percentile",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CPercentilePolars(SeriesOperator):
    """Auto-generated Polars bridge for c_percentile.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="c_percentile",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="c_std",
    canonical="c_std",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CStdPolars(SeriesOperator):
    """Auto-generated Polars bridge for c_std.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="c_std",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="c_sum",
    canonical="c_sum",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CSumPolars(SeriesOperator):
    """Auto-generated Polars bridge for c_sum.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="c_sum",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="calendar_day_diff",
    canonical="calendar_day_diff",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CalendarDayDiffPolars(SeriesOperator):
    """Auto-generated Polars bridge for calendar_day_diff.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="calendar_day_diff",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="causal_linear_extrapolate",
    canonical="causal_linear_extrapolate",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CausalLinearExtrapolatePolars(SeriesOperator):
    """Auto-generated Polars bridge for causal_linear_extrapolate.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="causal_linear_extrapolate",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="cbrt",
    canonical="cbrt",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CbrtPolars(SeriesOperator):
    """Auto-generated Polars bridge for cbrt.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cbrt",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="cdf_chi2",
    canonical="cdf_chi2",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CdfChi2Polars(SeriesOperator):
    """Auto-generated Polars bridge for cdf_chi2.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cdf_chi2",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cdf_f",
    canonical="cdf_f",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CdfFPolars(SeriesOperator):
    """Auto-generated Polars bridge for cdf_f.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cdf_f",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cdf_normal",
    canonical="cdf_normal",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CdfNormalPolars(SeriesOperator):
    """Auto-generated Polars bridge for cdf_normal.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cdf_normal",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cdf_t",
    canonical="cdf_t",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CdfTPolars(SeriesOperator):
    """Auto-generated Polars bridge for cdf_t.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cdf_t",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="ceil",
    canonical="ceil",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CeilPolars(SeriesOperator):
    """Auto-generated Polars bridge for ceil.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ceil",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="chi_square_test",
    canonical="chi_square_test",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ChiSquareTestPolars(SeriesOperator):
    """Auto-generated Polars bridge for chi_square_test.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="chi_square_test",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="clip",
    canonical="clip",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ClipPolars(SeriesOperator):
    """Auto-generated Polars bridge for clip.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="clip",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="close_gap",
    canonical="close_gap",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CloseGapPolars(SeriesOperator):
    """Auto-generated Polars bridge for close_gap.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="close_gap",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="coalesce",
    canonical="coalesce",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CoalescePolars(SeriesOperator):
    """Auto-generated Polars bridge for coalesce.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="coalesce",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="complex",
    canonical="complex",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ComplexPolars(SeriesOperator):
    """Auto-generated Polars bridge for complex.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="complex",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="conj",
    canonical="conj",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ConjPolars(SeriesOperator):
    """Auto-generated Polars bridge for conj.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="conj",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="constant",
    canonical="constant",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ConstantPolars(SeriesOperator):
    """Auto-generated Polars bridge for constant.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="constant",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="convolve",
    canonical="convolve",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ConvolvePolars(SeriesOperator):
    """Auto-generated Polars bridge for convolve.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="convolve",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="corr_test",
    canonical="corr_test",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CorrTestPolars(SeriesOperator):
    """Auto-generated Polars bridge for corr_test.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="corr_test",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="correlate",
    canonical="correlate",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CorrelatePolars(SeriesOperator):
    """Auto-generated Polars bridge for correlate.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="correlate",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="cos",
    canonical="cos",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CosPolars(SeriesOperator):
    """Auto-generated Polars bridge for cos.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cos",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="cos_phase",
    canonical="cos_phase",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CosPhasePolars(SeriesOperator):
    """Auto-generated Polars bridge for cos_phase.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cos_phase",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cosh",
    canonical="cosh",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CoshPolars(SeriesOperator):
    """Auto-generated Polars bridge for cosh.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cosh",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="coskewness_to_market",
    canonical="coskewness_to_market",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CoskewnessToMarketPolars(SeriesOperator):
    """Auto-generated Polars bridge for coskewness_to_market.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="coskewness_to_market",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cot",
    canonical="cot",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CotPolars(SeriesOperator):
    """Auto-generated Polars bridge for cot.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cot",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="count",
    canonical="count",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CountPolars(SeriesOperator):
    """Auto-generated Polars bridge for count.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="count",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cross_event",
    canonical="cross_event",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CrossEventPolars(SeriesOperator):
    """Auto-generated Polars bridge for cross_event.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cross_event",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# CS operators
# ------------------------------------------------------------------------------

@register_operator(
    name="cs_demean",
    canonical="cs_demean",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsDemeanPolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_demean.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_demean",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_huber_resid",
    canonical="cs_huber_resid",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsHuberResidPolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_huber_resid.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_huber_resid",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_isolation",
    canonical="cs_isolation",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsIsolationPolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_isolation.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_isolation",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_isotonic_residual",
    canonical="cs_isotonic_residual",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsIsotonicResidualPolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_isotonic_residual.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_isotonic_residual",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_isotonic_residual_lagged_direction",
    canonical="cs_isotonic_residual_lagged_direction",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsIsotonicResidualLaggedDirectionPolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_isotonic_residual_lagged_direction.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_isotonic_residual_lagged_direction",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_knn_graph_dirichlet_energy",
    canonical="cs_knn_graph_dirichlet_energy",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsKnnGraphDirichletEnergyPolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_knn_graph_dirichlet_energy.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_knn_graph_dirichlet_energy",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_knn_local_moran",
    canonical="cs_knn_local_moran",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsKnnLocalMoranPolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_knn_local_moran.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_knn_local_moran",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_knn_neighbor_retention",
    canonical="cs_knn_neighbor_retention",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsKnnNeighborRetentionPolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_knn_neighbor_retention.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_knn_neighbor_retention",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_knn_peer_mean_ex_self",
    canonical="cs_knn_peer_mean_ex_self",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsKnnPeerMeanExSelfPolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_knn_peer_mean_ex_self.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_knn_peer_mean_ex_self",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_knn_tangent_residual",
    canonical="cs_knn_tangent_residual",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsKnnTangentResidualPolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_knn_tangent_residual.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_knn_tangent_residual",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_lad_resid",
    canonical="cs_lad_resid",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsLadResidPolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_lad_resid.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_lad_resid",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_local_curvature",
    canonical="cs_local_curvature",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsLocalCurvaturePolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_local_curvature.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_local_curvature",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_local_density",
    canonical="cs_local_density",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsLocalDensityPolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_local_density.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_local_density",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_neighbor_gap",
    canonical="cs_neighbor_gap",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsNeighborGapPolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_neighbor_gap.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_neighbor_gap",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_pct_rank",
    canonical="cs_pct_rank",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsPctRankPolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_pct_rank.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_pct_rank",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_quantile",
    canonical="cs_quantile",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsQuantilePolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_quantile.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_quantile",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_rank_01",
    canonical="cs_rank_01",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsRank01Polars(SeriesOperator):
    """Auto-generated Polars bridge for cs_rank_01.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_rank_01",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_rank_churn",
    canonical="cs_rank_churn",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsRankChurnPolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_rank_churn.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_rank_churn",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_rank_combined_churn",
    canonical="cs_rank_combined_churn",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsRankCombinedChurnPolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_rank_combined_churn.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_rank_combined_churn",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_rank_composition_churn",
    canonical="cs_rank_composition_churn",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsRankCompositionChurnPolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_rank_composition_churn.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_rank_composition_churn",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_sliced_wasserstein_copula_shift",
    canonical="cs_sliced_wasserstein_copula_shift",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsSlicedWassersteinCopulaShiftPolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_sliced_wasserstein_copula_shift.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_sliced_wasserstein_copula_shift",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_tail_breadth",
    canonical="cs_tail_breadth",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsTailBreadthPolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_tail_breadth.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_tail_breadth",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_tail_retention",
    canonical="cs_tail_retention",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsTailRetentionPolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_tail_retention.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_tail_retention",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cs_trimmed_ols_resid",
    canonical="cs_trimmed_ols_resid",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CsTrimmedOlsResidPolars(SeriesOperator):
    """Auto-generated Polars bridge for cs_trimmed_ols_resid.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cs_trimmed_ols_resid",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="csc",
    canonical="csc",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CscPolars(SeriesOperator):
    """Auto-generated Polars bridge for csc.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="csc",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cube",
    canonical="cube",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CubePolars(SeriesOperator):
    """Auto-generated Polars bridge for cube.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cube",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cum_avg",
    canonical="cum_avg",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CumAvgPolars(SeriesOperator):
    """Auto-generated Polars bridge for cum_avg.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cum_avg",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cum_count",
    canonical="cum_count",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CumCountPolars(SeriesOperator):
    """Auto-generated Polars bridge for cum_count.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cum_count",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cum_delta",
    canonical="cum_delta",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CumDeltaPolars(SeriesOperator):
    """Auto-generated Polars bridge for cum_delta.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cum_delta",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cum_first",
    canonical="cum_first",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CumFirstPolars(SeriesOperator):
    """Auto-generated Polars bridge for cum_first.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cum_first",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cum_last",
    canonical="cum_last",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CumLastPolars(SeriesOperator):
    """Auto-generated Polars bridge for cum_last.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cum_last",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cum_max",
    canonical="cum_max",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CumMaxPolars(SeriesOperator):
    """Auto-generated Polars bridge for cum_max.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cum_max",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cum_min",
    canonical="cum_min",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CumMinPolars(SeriesOperator):
    """Auto-generated Polars bridge for cum_min.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cum_min",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cum_positive_streak",
    canonical="cum_positive_streak",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CumPositiveStreakPolars(SeriesOperator):
    """Auto-generated Polars bridge for cum_positive_streak.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cum_positive_streak",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cum_prod",
    canonical="cum_prod",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CumProdPolars(SeriesOperator):
    """Auto-generated Polars bridge for cum_prod.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cum_prod",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cum_rank",
    canonical="cum_rank",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CumRankPolars(SeriesOperator):
    """Auto-generated Polars bridge for cum_rank.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cum_rank",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cum_std",
    canonical="cum_std",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CumStdPolars(SeriesOperator):
    """Auto-generated Polars bridge for cum_std.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cum_std",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cum_sum",
    canonical="cum_sum",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CumSumPolars(SeriesOperator):
    """Auto-generated Polars bridge for cum_sum.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cum_sum",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cum_top_n_avg",
    canonical="cum_top_n_avg",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CumTopNAvgPolars(SeriesOperator):
    """Auto-generated Polars bridge for cum_top_n_avg.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cum_top_n_avg",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cum_top_n_sum",
    canonical="cum_top_n_sum",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CumTopNSumPolars(SeriesOperator):
    """Auto-generated Polars bridge for cum_top_n_sum.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cum_top_n_sum",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cumulative_max",
    canonical="cumulative_max",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CumulativeMaxPolars(SeriesOperator):
    """Auto-generated Polars bridge for cumulative_max.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cumulative_max",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cumulative_mean",
    canonical="cumulative_mean",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CumulativeMeanPolars(SeriesOperator):
    """Auto-generated Polars bridge for cumulative_mean.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cumulative_mean",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cumulative_min",
    canonical="cumulative_min",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CumulativeMinPolars(SeriesOperator):
    """Auto-generated Polars bridge for cumulative_min.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cumulative_min",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="cumulative_returns",
    canonical="cumulative_returns",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CumulativeReturnsPolars(SeriesOperator):
    """Auto-generated Polars bridge for cumulative_returns.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="cumulative_returns",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="current_ratio",
    canonical="current_ratio",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class CurrentRatioPolars(SeriesOperator):
    """Auto-generated Polars bridge for current_ratio.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="current_ratio",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="debt_to_equity",
    canonical="debt_to_equity",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class DebtToEquityPolars(SeriesOperator):
    """Auto-generated Polars bridge for debt_to_equity.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="debt_to_equity",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="decimate",
    canonical="decimate",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class DecimatePolars(SeriesOperator):
    """Auto-generated Polars bridge for decimate.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="decimate",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="deltas",
    canonical="deltas",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class DeltasPolars(SeriesOperator):
    """Auto-generated Polars bridge for deltas.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="deltas",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="dft",
    canonical="dft",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class DftPolars(SeriesOperator):
    """Auto-generated Polars bridge for dft.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="dft",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="digital_count",
    canonical="digital_count",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class DigitalCountPolars(SeriesOperator):
    """Auto-generated Polars bridge for digital_count.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="digital_count",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="directional_change_extent",
    canonical="directional_change_extent",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class DirectionalChangeExtentPolars(SeriesOperator):
    """Auto-generated Polars bridge for directional_change_extent.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="directional_change_extent",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="directional_change_state",
    canonical="directional_change_state",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class DirectionalChangeStatePolars(SeriesOperator):
    """Auto-generated Polars bridge for directional_change_state.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="directional_change_state",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="div_or_default",
    canonical="div_or_default",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class DivOrDefaultPolars(SeriesOperator):
    """Auto-generated Polars bridge for div_or_default.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="div_or_default",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="div_or_null",
    canonical="div_or_null",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class DivOrNullPolars(SeriesOperator):
    """Auto-generated Polars bridge for div_or_null.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="div_or_null",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="divide",
    canonical="divide",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class DividePolars(SeriesOperator):
    """Auto-generated Polars bridge for divide.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="divide",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="downside_beta",
    canonical="downside_beta",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class DownsideBetaPolars(SeriesOperator):
    """Auto-generated Polars bridge for downside_beta.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="downside_beta",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="dropna",
    canonical="dropna",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class DropnaPolars(SeriesOperator):
    """Auto-generated Polars bridge for dropna.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="dropna",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="durbin_watson_test",
    canonical="durbin_watson_test",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class DurbinWatsonTestPolars(SeriesOperator):
    """Auto-generated Polars bridge for durbin_watson_test.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="durbin_watson_test",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="earnings_yield",
    canonical="earnings_yield",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EarningsYieldPolars(SeriesOperator):
    """Auto-generated Polars bridge for earnings_yield.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="earnings_yield",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="eig",
    canonical="eig",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EigPolars(SeriesOperator):
    """Auto-generated Polars bridge for eig.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="eig",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="eq",
    canonical="eq",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EqPolars(SeriesOperator):
    """Auto-generated Polars bridge for eq.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="eq",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_abnormal_return_past",
    canonical="event_abnormal_return_past",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventAbnormalReturnPastPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_abnormal_return_past.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_abnormal_return_past",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_active_count",
    canonical="event_active_count",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventActiveCountPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_active_count.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_active_count",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_arithmetic_return_sum",
    canonical="event_arithmetic_return_sum",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventArithmeticReturnSumPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_arithmetic_return_sum.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_arithmetic_return_sum",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_cluster_count",
    canonical="event_cluster_count",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventClusterCountPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_cluster_count.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_cluster_count",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_cluster_mean_size",
    canonical="event_cluster_mean_size",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventClusterMeanSizePolars(SeriesOperator):
    """Auto-generated Polars bridge for event_cluster_mean_size.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_cluster_mean_size",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_cumulative_return_past",
    canonical="event_cumulative_return_past",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventCumulativeReturnPastPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_cumulative_return_past.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_cumulative_return_past",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_decay_asof",
    canonical="event_decay_asof",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventDecayAsofPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_decay_asof.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_decay_asof",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_fano_excess",
    canonical="event_fano_excess",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventFanoExcessPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_fano_excess.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_fano_excess",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_fano_factor",
    canonical="event_fano_factor",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventFanoFactorPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_fano_factor.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_fano_factor",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_frequency",
    canonical="event_frequency",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventFrequencyPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_frequency.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_frequency",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_hawkes_branching_ratio_proxy",
    canonical="event_hawkes_branching_ratio_proxy",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventHawkesBranchingRatioProxyPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_hawkes_branching_ratio_proxy.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_hawkes_branching_ratio_proxy",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_historical_response_mean",
    canonical="event_historical_response_mean",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventHistoricalResponseMeanPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_historical_response_mean.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_historical_response_mean",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_historical_response_sign_balance",
    canonical="event_historical_response_sign_balance",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventHistoricalResponseSignBalancePolars(SeriesOperator):
    """Auto-generated Polars bridge for event_historical_response_sign_balance.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_historical_response_sign_balance",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_interval_mark_coupling",
    canonical="event_interval_mark_coupling",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventIntervalMarkCouplingPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_interval_mark_coupling.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_interval_mark_coupling",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_interval_memory",
    canonical="event_interval_memory",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventIntervalMemoryPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_interval_memory.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_interval_memory",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_local_variation",
    canonical="event_local_variation",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventLocalVariationPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_local_variation.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_local_variation",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_log_return_sum",
    canonical="event_log_return_sum",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventLogReturnSumPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_log_return_sum.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_log_return_sum",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_mark_autocorr",
    canonical="event_mark_autocorr",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventMarkAutocorrPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_mark_autocorr.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_mark_autocorr",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_refractory",
    canonical="event_refractory",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventRefractoryPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_refractory.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_refractory",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_response_decay_rate",
    canonical="event_response_decay_rate",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventResponseDecayRatePolars(SeriesOperator):
    """Auto-generated Polars bridge for event_response_decay_rate.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_response_decay_rate",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_response_dispersion",
    canonical="event_response_dispersion",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventResponseDispersionPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_response_dispersion.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_response_dispersion",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_response_effective_events",
    canonical="event_response_effective_events",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventResponseEffectiveEventsPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_response_effective_events.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_response_effective_events",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_response_overlap_ratio",
    canonical="event_response_overlap_ratio",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventResponseOverlapRatioPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_response_overlap_ratio.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_response_overlap_ratio",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_response_peak_lag",
    canonical="event_response_peak_lag",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventResponsePeakLagPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_response_peak_lag.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_response_peak_lag",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_response_reversal_strength",
    canonical="event_response_reversal_strength",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventResponseReversalStrengthPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_response_reversal_strength.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_response_reversal_strength",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="event_return_since_last",
    canonical="event_return_since_last",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EventReturnSinceLastPolars(SeriesOperator):
    """Auto-generated Polars bridge for event_return_since_last.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="event_return_since_last",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ewm",
    canonical="ewm",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EwmPolars(SeriesOperator):
    """Auto-generated Polars bridge for ewm.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ewm",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ewm_corr",
    canonical="ewm_corr",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EwmCorrPolars(SeriesOperator):
    """Auto-generated Polars bridge for ewm_corr.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ewm_corr",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ewm_cov",
    canonical="ewm_cov",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EwmCovPolars(SeriesOperator):
    """Auto-generated Polars bridge for ewm_cov.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ewm_cov",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ewm_mean",
    canonical="ewm_mean",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EwmMeanPolars(SeriesOperator):
    """Auto-generated Polars bridge for ewm_mean.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ewm_mean",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ewm_std",
    canonical="ewm_std",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EwmStdPolars(SeriesOperator):
    """Auto-generated Polars bridge for ewm_std.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ewm_std",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ewm_var",
    canonical="ewm_var",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class EwmVarPolars(SeriesOperator):
    """Auto-generated Polars bridge for ewm_var.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ewm_var",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="exp",
    canonical="exp",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ExpPolars(SeriesOperator):
    """Auto-generated Polars bridge for exp.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="exp",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="exp_neg",
    canonical="exp_neg",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ExpNegPolars(SeriesOperator):
    """Auto-generated Polars bridge for exp_neg.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="exp_neg",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="expanding_max",
    canonical="expanding_max",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ExpandingMaxPolars(SeriesOperator):
    """Auto-generated Polars bridge for expanding_max.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="expanding_max",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="expanding_mean",
    canonical="expanding_mean",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ExpandingMeanPolars(SeriesOperator):
    """Auto-generated Polars bridge for expanding_mean.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="expanding_mean",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="expanding_min",
    canonical="expanding_min",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ExpandingMinPolars(SeriesOperator):
    """Auto-generated Polars bridge for expanding_min.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="expanding_min",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="expanding_rank",
    canonical="expanding_rank",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ExpandingRankPolars(SeriesOperator):
    """Auto-generated Polars bridge for expanding_rank.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="expanding_rank",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="expanding_std",
    canonical="expanding_std",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ExpandingStdPolars(SeriesOperator):
    """Auto-generated Polars bridge for expanding_std.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="expanding_std",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="expanding_sum",
    canonical="expanding_sum",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ExpandingSumPolars(SeriesOperator):
    """Auto-generated Polars bridge for expanding_sum.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="expanding_sum",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="expanding_zscore",
    canonical="expanding_zscore",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ExpandingZscorePolars(SeriesOperator):
    """Auto-generated Polars bridge for expanding_zscore.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="expanding_zscore",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ffill",
    canonical="ffill",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class FfillPolars(SeriesOperator):
    """Auto-generated Polars bridge for ffill.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ffill",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="fft",
    canonical="fft",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class FftPolars(SeriesOperator):
    """Auto-generated Polars bridge for fft.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="fft",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="fillna",
    canonical="fillna",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class FillnaPolars(SeriesOperator):
    """Auto-generated Polars bridge for fillna.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="fillna",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="fillna_const",
    canonical="fillna_const",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class FillnaConstPolars(SeriesOperator):
    """Auto-generated Polars bridge for fillna_const.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="fillna_const",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="filter_bandpass",
    canonical="filter_bandpass",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class FilterBandpassPolars(SeriesOperator):
    """Auto-generated Polars bridge for filter_bandpass.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="filter_bandpass",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="filter_highpass",
    canonical="filter_highpass",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class FilterHighpassPolars(SeriesOperator):
    """Auto-generated Polars bridge for filter_highpass.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="filter_highpass",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="filter_lowpass",
    canonical="filter_lowpass",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class FilterLowpassPolars(SeriesOperator):
    """Auto-generated Polars bridge for filter_lowpass.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="filter_lowpass",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="filter_notch",
    canonical="filter_notch",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class FilterNotchPolars(SeriesOperator):
    """Auto-generated Polars bridge for filter_notch.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="filter_notch",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="fin_announcement_lag",
    canonical="fin_announcement_lag",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class FinAnnouncementLagPolars(SeriesOperator):
    """Auto-generated Polars bridge for fin_announcement_lag.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="fin_announcement_lag",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="fin_applicability_mask",
    canonical="fin_applicability_mask",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class FinApplicabilityMaskPolars(SeriesOperator):
    """Auto-generated Polars bridge for fin_applicability_mask.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="fin_applicability_mask",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="first",
    canonical="first",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class FirstPolars(SeriesOperator):
    """Auto-generated Polars bridge for first.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="first",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="first_not_null",
    canonical="first_not_null",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class FirstNotNullPolars(SeriesOperator):
    """Auto-generated Polars bridge for first_not_null.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="first_not_null",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="fix",
    canonical="fix",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class FixPolars(SeriesOperator):
    """Auto-generated Polars bridge for fix.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="fix",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="flex_max",
    canonical="flex_max",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class FlexMaxPolars(SeriesOperator):
    """Auto-generated Polars bridge for flex_max.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="flex_max",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="flex_min",
    canonical="flex_min",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class FlexMinPolars(SeriesOperator):
    """Auto-generated Polars bridge for flex_min.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="flex_min",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="float_share_ratio",
    canonical="float_share_ratio",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class FloatShareRatioPolars(SeriesOperator):
    """Auto-generated Polars bridge for float_share_ratio.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="float_share_ratio",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="floor",
    canonical="floor",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class FloorPolars(SeriesOperator):
    """Auto-generated Polars bridge for floor.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="floor",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="fmax",
    canonical="fmax",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class FmaxPolars(SeriesOperator):
    """Auto-generated Polars bridge for fmax.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="fmax",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="fmin",
    canonical="fmin",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class FminPolars(SeriesOperator):
    """Auto-generated Polars bridge for fmin.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="fmin",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="free_float_share_ratio",
    canonical="free_float_share_ratio",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class FreeFloatShareRatioPolars(SeriesOperator):
    """Auto-generated Polars bridge for free_float_share_ratio.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="free_float_share_ratio",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ge",
    canonical="ge",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GePolars(SeriesOperator):
    """Auto-generated Polars bridge for ge.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ge",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="geometric_mean",
    canonical="geometric_mean",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GeometricMeanPolars(SeriesOperator):
    """Auto-generated Polars bridge for geometric_mean.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="geometric_mean",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="granger_causality",
    canonical="granger_causality",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GrangerCausalityPolars(SeriesOperator):
    """Auto-generated Polars bridge for granger_causality.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="granger_causality",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# GROUP operators
# ------------------------------------------------------------------------------

@register_operator(
    name="group_corr_mst_length",
    canonical="group_corr_mst_length",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupCorrMstLengthPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_corr_mst_length.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_corr_mst_length",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_count",
    canonical="group_count",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupCountPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_count.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_count",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_current_members_tail_coexceedance",
    canonical="group_current_members_tail_coexceedance",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupCurrentMembersTailCoexceedancePolars(SeriesOperator):
    """Auto-generated Polars bridge for group_current_members_tail_coexceedance.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_current_members_tail_coexceedance",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_decay_linear",
    canonical="group_decay_linear",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupDecayLinearPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_decay_linear.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_decay_linear",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_ex_self_mad",
    canonical="group_ex_self_mad",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupExSelfMadPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_ex_self_mad.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_ex_self_mad",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_ex_self_mean",
    canonical="group_ex_self_mean",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupExSelfMeanPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_ex_self_mean.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_ex_self_mean",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_ex_self_quantile",
    canonical="group_ex_self_quantile",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupExSelfQuantilePolars(SeriesOperator):
    """Auto-generated Polars bridge for group_ex_self_quantile.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_ex_self_quantile",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_ex_self_std",
    canonical="group_ex_self_std",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupExSelfStdPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_ex_self_std.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_ex_self_std",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_ex_self_weighted_mean",
    canonical="group_ex_self_weighted_mean",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupExSelfWeightedMeanPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_ex_self_weighted_mean.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_ex_self_weighted_mean",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_feature_coverage_ratio",
    canonical="group_feature_coverage_ratio",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupFeatureCoverageRatioPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_feature_coverage_ratio.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_feature_coverage_ratio",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_feature_effective_rank",
    canonical="group_feature_effective_rank",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupFeatureEffectiveRankPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_feature_effective_rank.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_feature_effective_rank",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_feature_mode_localization",
    canonical="group_feature_mode_localization",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupFeatureModeLocalizationPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_feature_mode_localization.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_feature_mode_localization",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_feature_mode_share",
    canonical="group_feature_mode_share",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupFeatureModeSharePolars(SeriesOperator):
    """Auto-generated Polars bridge for group_feature_mode_share.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_feature_mode_share",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_feature_second_mode_localization",
    canonical="group_feature_second_mode_localization",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupFeatureSecondModeLocalizationPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_feature_second_mode_localization.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_feature_second_mode_localization",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_feature_spectral_gap",
    canonical="group_feature_spectral_gap",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupFeatureSpectralGapPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_feature_spectral_gap.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_feature_spectral_gap",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_feature_valid_member_count",
    canonical="group_feature_valid_member_count",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupFeatureValidMemberCountPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_feature_valid_member_count.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_feature_valid_member_count",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_kurtosis",
    canonical="group_kurtosis",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupKurtosisPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_kurtosis.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_kurtosis",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_max",
    canonical="group_max",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupMaxPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_max.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_max",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_mean",
    canonical="group_mean",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupMeanPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_mean.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_mean",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_min",
    canonical="group_min",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupMinPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_min.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_min",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_multi_resid",
    canonical="group_multi_resid",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupMultiResidPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_multi_resid.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_multi_resid",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_percentile",
    canonical="group_percentile",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupPercentilePolars(SeriesOperator):
    """Auto-generated Polars bridge for group_percentile.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_percentile",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_quantile_spread",
    canonical="group_quantile_spread",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupQuantileSpreadPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_quantile_spread.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_quantile_spread",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_rank",
    canonical="group_rank",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupRankPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_rank.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_rank",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_rank_weighted_value",
    canonical="group_rank_weighted_value",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupRankWeightedValuePolars(SeriesOperator):
    """Auto-generated Polars bridge for group_rank_weighted_value.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_rank_weighted_value",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_signal_attraction_share",
    canonical="group_signal_attraction_share",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupSignalAttractionSharePolars(SeriesOperator):
    """Auto-generated Polars bridge for group_signal_attraction_share.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_signal_attraction_share",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_skewness",
    canonical="group_skewness",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupSkewnessPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_skewness.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_skewness",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_spd_feature_structure_shift",
    canonical="group_spd_feature_structure_shift",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupSpdFeatureStructureShiftPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_spd_feature_structure_shift.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_spd_feature_structure_shift",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_std",
    canonical="group_std",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupStdPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_std.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_std",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_sum",
    canonical="group_sum",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupSumPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_sum.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_sum",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_tail_centrality",
    canonical="group_tail_centrality",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupTailCentralityPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_tail_centrality.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_tail_centrality",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_tail_lead_score",
    canonical="group_tail_lead_score",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupTailLeadScorePolars(SeriesOperator):
    """Auto-generated Polars bridge for group_tail_lead_score.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_tail_lead_score",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_tail_ratio",
    canonical="group_tail_ratio",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupTailRatioPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_tail_ratio.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_tail_ratio",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="group_ts_decay_linear",
    canonical="group_ts_decay_linear",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GroupTsDecayLinearPolars(SeriesOperator):
    """Auto-generated Polars bridge for group_ts_decay_linear.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="group_ts_decay_linear",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="gt",
    canonical="gt",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class GtPolars(SeriesOperator):
    """Auto-generated Polars bridge for gt.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="gt",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="harmonic_mean",
    canonical="harmonic_mean",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class HarmonicMeanPolars(SeriesOperator):
    """Auto-generated Polars bridge for harmonic_mean.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="harmonic_mean",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# GROUP operators
# ------------------------------------------------------------------------------

@register_operator(
    name="hierarchical_group_neutralize",
    canonical="hierarchical_group_neutralize",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class HierarchicalGroupNeutralizePolars(SeriesOperator):
    """Auto-generated Polars bridge for hierarchical_group_neutralize.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="hierarchical_group_neutralize",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="holder_class_js_shift",
    canonical="holder_class_js_shift",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class HolderClassJsShiftPolars(SeriesOperator):
    """Auto-generated Polars bridge for holder_class_js_shift.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="holder_class_js_shift",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="holder_company_ownership_hhi",
    canonical="holder_company_ownership_hhi",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class HolderCompanyOwnershipHhiPolars(SeriesOperator):
    """Auto-generated Polars bridge for holder_company_ownership_hhi.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="holder_company_ownership_hhi",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="holder_concentration",
    canonical="holder_concentration",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class HolderConcentrationPolars(SeriesOperator):
    """Auto-generated Polars bridge for holder_concentration.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="holder_concentration",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="holder_concentration_change",
    canonical="holder_concentration_change",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class HolderConcentrationChangePolars(SeriesOperator):
    """Auto-generated Polars bridge for holder_concentration_change.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="holder_concentration_change",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="holder_count_change_rate",
    canonical="holder_count_change_rate",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class HolderCountChangeRatePolars(SeriesOperator):
    """Auto-generated Polars bridge for holder_count_change_rate.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="holder_count_change_rate",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="holder_observed_topk_hhi",
    canonical="holder_observed_topk_hhi",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class HolderObservedTopkHhiPolars(SeriesOperator):
    """Auto-generated Polars bridge for holder_observed_topk_hhi.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="holder_observed_topk_hhi",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="hump_decay",
    canonical="hump_decay",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class HumpDecayPolars(SeriesOperator):
    """Auto-generated Polars bridge for hump_decay.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="hump_decay",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="identity",
    canonical="identity",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IdentityPolars(SeriesOperator):
    """Auto-generated Polars bridge for identity.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="identity",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="idft",
    canonical="idft",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IdftPolars(SeriesOperator):
    """Auto-generated Polars bridge for idft.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="idft",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="idio_skew",
    canonical="idio_skew",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IdioSkewPolars(SeriesOperator):
    """Auto-generated Polars bridge for idio_skew.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="idio_skew",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="idio_vol",
    canonical="idio_vol",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IdioVolPolars(SeriesOperator):
    """Auto-generated Polars bridge for idio_vol.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="idio_vol",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="if_else",
    canonical="if_else",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IfElsePolars(SeriesOperator):
    """Auto-generated Polars bridge for if_else.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="if_else",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ifft",
    canonical="ifft",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IfftPolars(SeriesOperator):
    """Auto-generated Polars bridge for ifft.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ifft",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ifnan",
    canonical="ifnan",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IfnanPolars(SeriesOperator):
    """Auto-generated Polars bridge for ifnan.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ifnan",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="imag",
    canonical="imag",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ImagPolars(SeriesOperator):
    """Auto-generated Polars bridge for imag.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="imag",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="index_entry_exit_event",
    canonical="index_entry_exit_event",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IndexEntryExitEventPolars(SeriesOperator):
    """Auto-generated Polars bridge for index_entry_exit_event.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="index_entry_exit_event",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="index_member",
    canonical="index_member",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IndexMemberPolars(SeriesOperator):
    """Auto-generated Polars bridge for index_member.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="index_member",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="index_membership_age",
    canonical="index_membership_age",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IndexMembershipAgePolars(SeriesOperator):
    """Auto-generated Polars bridge for index_membership_age.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="index_membership_age",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="index_weight",
    canonical="index_weight",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IndexWeightPolars(SeriesOperator):
    """Auto-generated Polars bridge for index_weight.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="index_weight",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="index_weight_change",
    canonical="index_weight_change",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IndexWeightChangePolars(SeriesOperator):
    """Auto-generated Polars bridge for index_weight_change.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="index_weight_change",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intercept",
    canonical="intercept",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class InterceptPolars(SeriesOperator):
    """Auto-generated Polars bridge for intercept.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intercept",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_bar_range_deviation",
    canonical="intra_bar_range_deviation",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraBarRangeDeviationPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_bar_range_deviation.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_bar_range_deviation",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_bar_range_persistence",
    canonical="intra_bar_range_persistence",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraBarRangePersistencePolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_bar_range_persistence.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_bar_range_persistence",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_close_participation",
    canonical="intra_close_participation",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraCloseParticipationPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_close_participation.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_close_participation",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_high_low_affinity",
    canonical="intra_high_low_affinity",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraHighLowAffinityPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_high_low_affinity.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_high_low_affinity",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_industry_lead_lag_ex_self",
    canonical="intra_industry_lead_lag_ex_self",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraIndustryLeadLagExSelfPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_industry_lead_lag_ex_self.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_industry_lead_lag_ex_self",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_market_lead_lag_ex_self",
    canonical="intra_market_lead_lag_ex_self",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraMarketLeadLagExSelfPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_market_lead_lag_ex_self.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_market_lead_lag_ex_self",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_multiresolution_resample_reduce",
    canonical="intra_multiresolution_resample_reduce",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraMultiresolutionResampleReducePolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_multiresolution_resample_reduce.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_multiresolution_resample_reduce",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_neighbor_event_class",
    canonical="intra_neighbor_event_class",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraNeighborEventClassPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_neighbor_event_class.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_neighbor_event_class",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_range_gap_flag",
    canonical="intra_range_gap_flag",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraRangeGapFlagPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_range_gap_flag.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_range_gap_flag",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_round_price_barrier_response",
    canonical="intra_round_price_barrier_response",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraRoundPriceBarrierResponsePolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_round_price_barrier_response.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_round_price_barrier_response",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_round_price_clustering_share",
    canonical="intra_round_price_clustering_share",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraRoundPriceClusteringSharePolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_round_price_clustering_share.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_round_price_clustering_share",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_same_slot_zscore",
    canonical="intra_same_slot_zscore",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraSameSlotZscorePolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_same_slot_zscore.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_same_slot_zscore",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_session_boundary_jump",
    canonical="intra_session_boundary_jump",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraSessionBoundaryJumpPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_session_boundary_jump.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_session_boundary_jump",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_session_return_asymmetry",
    canonical="intra_session_return_asymmetry",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraSessionReturnAsymmetryPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_session_return_asymmetry.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_session_return_asymmetry",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_slice_mask_pair_reduce",
    canonical="intra_slice_mask_pair_reduce",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraSliceMaskPairReducePolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_slice_mask_pair_reduce.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_slice_mask_pair_reduce",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_slice_mask_reduce",
    canonical="intra_slice_mask_reduce",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraSliceMaskReducePolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_slice_mask_reduce.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_slice_mask_reduce",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_slot_amount_surprise",
    canonical="intra_slot_amount_surprise",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraSlotAmountSurprisePolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_slot_amount_surprise.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_slot_amount_surprise",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_slot_volatility_surprise",
    canonical="intra_slot_volatility_surprise",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraSlotVolatilitySurprisePolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_slot_volatility_surprise.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_slot_volatility_surprise",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_slot_volume_surprise",
    canonical="intra_slot_volume_surprise",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraSlotVolumeSurprisePolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_slot_volume_surprise.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_slot_volume_surprise",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_state_count",
    canonical="intra_state_count",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraStateCountPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_state_count.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_state_count",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_state_dwell_stats",
    canonical="intra_state_dwell_stats",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraStateDwellStatsPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_state_dwell_stats.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_state_dwell_stats",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_state_follow_beta",
    canonical="intra_state_follow_beta",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraStateFollowBetaPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_state_follow_beta.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_state_follow_beta",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_state_follow_corr",
    canonical="intra_state_follow_corr",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraStateFollowCorrPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_state_follow_corr.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_state_follow_corr",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_state_follow_ratio",
    canonical="intra_state_follow_ratio",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraStateFollowRatioPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_state_follow_ratio.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_state_follow_ratio",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_state_interval_moment",
    canonical="intra_state_interval_moment",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraStateIntervalMomentPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_state_interval_moment.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_state_interval_moment",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_state_pair_same_slot_corr",
    canonical="intra_state_pair_same_slot_corr",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraStatePairSameSlotCorrPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_state_pair_same_slot_corr.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_state_pair_same_slot_corr",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_state_sum",
    canonical="intra_state_sum",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraStateSumPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_state_sum.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_state_sum",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_state_transition_entropy",
    canonical="intra_state_transition_entropy",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraStateTransitionEntropyPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_state_transition_entropy.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_state_transition_entropy",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_state_vwap",
    canonical="intra_state_vwap",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraStateVwapPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_state_vwap.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_state_vwap",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_tail_volume_share",
    canonical="intra_tail_volume_share",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraTailVolumeSharePolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_tail_volume_share.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_tail_volume_share",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_ute_high",
    canonical="intra_ute_high",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraUteHighPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_ute_high.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_ute_high",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_ute_low",
    canonical="intra_ute_low",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraUteLowPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_ute_low.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_ute_low",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_volume_at_price_profile",
    canonical="intra_volume_at_price_profile",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraVolumeAtPriceProfilePolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_volume_at_price_profile.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_volume_at_price_profile",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_volume_price_alignment",
    canonical="intra_volume_price_alignment",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraVolumePriceAlignmentPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_volume_price_alignment.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_volume_price_alignment",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_volume_profile_peak_geometry",
    canonical="intra_volume_profile_peak_geometry",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraVolumeProfilePeakGeometryPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_volume_profile_peak_geometry.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_volume_profile_peak_geometry",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_volume_profile_supply_structure",
    canonical="intra_volume_profile_supply_structure",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraVolumeProfileSupplyStructurePolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_volume_profile_supply_structure.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_volume_profile_supply_structure",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intra_volume_profile_value_area",
    canonical="intra_volume_profile_value_area",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntraVolumeProfileValueAreaPolars(SeriesOperator):
    """Auto-generated Polars bridge for intra_volume_profile_value_area.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intra_volume_profile_value_area",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_activity_duration_curvature",
    canonical="intraday_activity_duration_curvature",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayActivityDurationCurvaturePolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_activity_duration_curvature.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_activity_duration_curvature",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_barrier_approach_acceleration",
    canonical="intraday_barrier_approach_acceleration",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayBarrierApproachAccelerationPolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_barrier_approach_acceleration.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_barrier_approach_acceleration",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_bvc_imbalance",
    canonical="intraday_bvc_imbalance",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayBvcImbalancePolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_bvc_imbalance.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_bvc_imbalance",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_impact_asymmetry",
    canonical="intraday_impact_asymmetry",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayImpactAsymmetryPolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_impact_asymmetry.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_impact_asymmetry",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_impact_beta",
    canonical="intraday_impact_beta",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayImpactBetaPolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_impact_beta.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_impact_beta",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_impact_decay_rate",
    canonical="intraday_impact_decay_rate",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayImpactDecayRatePolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_impact_decay_rate.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_impact_decay_rate",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_jump_test_stat",
    canonical="intraday_jump_test_stat",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayJumpTestStatPolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_jump_test_stat.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_jump_test_stat",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_medrv",
    canonical="intraday_medrv",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayMedrvPolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_medrv.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_medrv",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_minrv",
    canonical="intraday_minrv",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayMinrvPolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_minrv.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_minrv",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_profile_pca_residual",
    canonical="intraday_profile_pca_residual",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayProfilePcaResidualPolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_profile_pca_residual.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_profile_pca_residual",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_profile_phase_shift",
    canonical="intraday_profile_phase_shift",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayProfilePhaseShiftPolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_profile_phase_shift.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_profile_phase_shift",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_profile_surprise_energy",
    canonical="intraday_profile_surprise_energy",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayProfileSurpriseEnergyPolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_profile_surprise_energy.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_profile_surprise_energy",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_quantile_curve_pca_residual",
    canonical="intraday_quantile_curve_pca_residual",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayQuantileCurvePcaResidualPolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_quantile_curve_pca_residual.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_quantile_curve_pca_residual",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_quantile_curve_pca_score",
    canonical="intraday_quantile_curve_pca_score",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayQuantileCurvePcaScorePolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_quantile_curve_pca_score.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_quantile_curve_pca_score",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_realized_power_variation",
    canonical="intraday_realized_power_variation",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayRealizedPowerVariationPolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_realized_power_variation.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_realized_power_variation",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_realized_semivariance_balance",
    canonical="intraday_realized_semivariance_balance",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayRealizedSemivarianceBalancePolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_realized_semivariance_balance.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_realized_semivariance_balance",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_return_wasserstein_shift",
    canonical="intraday_return_wasserstein_shift",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayReturnWassersteinShiftPolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_return_wasserstein_shift.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_return_wasserstein_shift",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_rv_signature_curvature",
    canonical="intraday_rv_signature_curvature",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayRvSignatureCurvaturePolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_rv_signature_curvature.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_rv_signature_curvature",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_session_shape_novelty",
    canonical="intraday_session_shape_novelty",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradaySessionShapeNoveltyPolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_session_shape_novelty.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_session_shape_novelty",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_subsampled_rv_dispersion",
    canonical="intraday_subsampled_rv_dispersion",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradaySubsampledRvDispersionPolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_subsampled_rv_dispersion.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_subsampled_rv_dispersion",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_volatility_concentration",
    canonical="intraday_volatility_concentration",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayVolatilityConcentrationPolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_volatility_concentration.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_volatility_concentration",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_volatility_entropy",
    canonical="intraday_volatility_entropy",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayVolatilityEntropyPolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_volatility_entropy.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_volatility_entropy",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_volatility_signature_slope",
    canonical="intraday_volatility_signature_slope",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayVolatilitySignatureSlopePolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_volatility_signature_slope.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_volatility_signature_slope",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_volatility_time_centroid",
    canonical="intraday_volatility_time_centroid",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayVolatilityTimeCentroidPolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_volatility_time_centroid.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_volatility_time_centroid",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_volume_clock_path_efficiency",
    canonical="intraday_volume_clock_path_efficiency",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayVolumeClockPathEfficiencyPolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_volume_clock_path_efficiency.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_volume_clock_path_efficiency",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_volume_clock_roughness",
    canonical="intraday_volume_clock_roughness",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayVolumeClockRoughnessPolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_volume_clock_roughness.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_volume_clock_roughness",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="intraday_wasserstein_pair_distance",
    canonical="intraday_wasserstein_pair_distance",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IntradayWassersteinPairDistancePolars(SeriesOperator):
    """Auto-generated Polars bridge for intraday_wasserstein_pair_distance.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="intraday_wasserstein_pair_distance",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="inv",
    canonical="inv",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class InvPolars(SeriesOperator):
    """Auto-generated Polars bridge for inv.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="inv",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="inverse",
    canonical="inverse",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class InversePolars(SeriesOperator):
    """Auto-generated Polars bridge for inverse.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="inverse",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="is_finite",
    canonical="is_finite",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IsFinitePolars(SeriesOperator):
    """Auto-generated Polars bridge for is_finite.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="is_finite",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="is_inf",
    canonical="is_inf",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IsInfPolars(SeriesOperator):
    """Auto-generated Polars bridge for is_inf.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="is_inf",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="is_infinite",
    canonical="is_infinite",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IsInfinitePolars(SeriesOperator):
    """Auto-generated Polars bridge for is_infinite.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="is_infinite",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="is_nan",
    canonical="is_nan",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IsNanPolars(SeriesOperator):
    """Auto-generated Polars bridge for is_nan.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="is_nan",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="is_not_null",
    canonical="is_not_null",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IsNotNullPolars(SeriesOperator):
    """Auto-generated Polars bridge for is_not_null.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="is_not_null",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="is_null",
    canonical="is_null",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class IsNullPolars(SeriesOperator):
    """Auto-generated Polars bridge for is_null.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="is_null",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="jarque_bera_test",
    canonical="jarque_bera_test",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class JarqueBeraTestPolars(SeriesOperator):
    """Auto-generated Polars bridge for jarque_bera_test.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="jarque_bera_test",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="kendall_corr_test",
    canonical="kendall_corr_test",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class KendallCorrTestPolars(SeriesOperator):
    """Auto-generated Polars bridge for kendall_corr_test.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="kendall_corr_test",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="kpss_test",
    canonical="kpss_test",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class KpssTestPolars(SeriesOperator):
    """Auto-generated Polars bridge for kpss_test.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="kpss_test",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ks_test",
    canonical="ks_test",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class KsTestPolars(SeriesOperator):
    """Auto-generated Polars bridge for ks_test.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ks_test",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="lasso",
    canonical="lasso",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class LassoPolars(SeriesOperator):
    """Auto-generated Polars bridge for lasso.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="lasso",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="last",
    canonical="last",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class LastPolars(SeriesOperator):
    """Auto-generated Polars bridge for last.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="last",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="last_not_null",
    canonical="last_not_null",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class LastNotNullPolars(SeriesOperator):
    """Auto-generated Polars bridge for last_not_null.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="last_not_null",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="le",
    canonical="le",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class LePolars(SeriesOperator):
    """Auto-generated Polars bridge for le.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="le",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="lerp",
    canonical="lerp",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class LerpPolars(SeriesOperator):
    """Auto-generated Polars bridge for lerp.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="lerp",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="levene_test",
    canonical="levene_test",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class LeveneTestPolars(SeriesOperator):
    """Auto-generated Polars bridge for levene_test.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="levene_test",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="lilliefors_test",
    canonical="lilliefors_test",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class LillieforsTestPolars(SeriesOperator):
    """Auto-generated Polars bridge for lilliefors_test.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="lilliefors_test",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="limit_down_close",
    canonical="limit_down_close",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class LimitDownClosePolars(SeriesOperator):
    """Auto-generated Polars bridge for limit_down_close.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="limit_down_close",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="limit_up_close",
    canonical="limit_up_close",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class LimitUpClosePolars(SeriesOperator):
    """Auto-generated Polars bridge for limit_up_close.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="limit_up_close",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="log",
    canonical="log",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class LogPolars(SeriesOperator):
    """Auto-generated Polars bridge for log.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="log",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="log10",
    canonical="log10",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class Log10Polars(SeriesOperator):
    """Auto-generated Polars bridge for log10.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="log10",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="log2",
    canonical="log2",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class Log2Polars(SeriesOperator):
    """Auto-generated Polars bridge for log2.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="log2",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="log_abs",
    canonical="log_abs",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class LogAbsPolars(SeriesOperator):
    """Auto-generated Polars bridge for log_abs.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="log_abs",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="log_fill_invalid",
    canonical="log_fill_invalid",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class LogFillInvalidPolars(SeriesOperator):
    """Auto-generated Polars bridge for log_fill_invalid.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="log_fill_invalid",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="log_returns",
    canonical="log_returns",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class LogReturnsPolars(SeriesOperator):
    """Auto-generated Polars bridge for log_returns.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="log_returns",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="lt",
    canonical="lt",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class LtPolars(SeriesOperator):
    """Auto-generated Polars bridge for lt.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="lt",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="lu_decompose",
    canonical="lu_decompose",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class LuDecomposePolars(SeriesOperator):
    """Auto-generated Polars bridge for lu_decompose.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="lu_decompose",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="m_zscore",
    canonical="m_zscore",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MZscorePolars(SeriesOperator):
    """Auto-generated Polars bridge for m_zscore.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="m_zscore",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="mat_add",
    canonical="mat_add",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MatAddPolars(SeriesOperator):
    """Auto-generated Polars bridge for mat_add.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="mat_add",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="mat_determinant",
    canonical="mat_determinant",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MatDeterminantPolars(SeriesOperator):
    """Auto-generated Polars bridge for mat_determinant.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="mat_determinant",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="mat_inverse",
    canonical="mat_inverse",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MatInversePolars(SeriesOperator):
    """Auto-generated Polars bridge for mat_inverse.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="mat_inverse",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="mat_multiply",
    canonical="mat_multiply",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MatMultiplyPolars(SeriesOperator):
    """Auto-generated Polars bridge for mat_multiply.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="mat_multiply",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="mat_rank",
    canonical="mat_rank",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MatRankPolars(SeriesOperator):
    """Auto-generated Polars bridge for mat_rank.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="mat_rank",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="mat_subtract",
    canonical="mat_subtract",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MatSubtractPolars(SeriesOperator):
    """Auto-generated Polars bridge for mat_subtract.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="mat_subtract",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="mat_transpose",
    canonical="mat_transpose",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MatTransposePolars(SeriesOperator):
    """Auto-generated Polars bridge for mat_transpose.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="mat_transpose",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="max_drawdown",
    canonical="max_drawdown",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MaxDrawdownPolars(SeriesOperator):
    """Auto-generated Polars bridge for max_drawdown.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="max_drawdown",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="maximum",
    canonical="maximum",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MaximumPolars(SeriesOperator):
    """Auto-generated Polars bridge for maximum.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="maximum",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="mean_agg",
    canonical="mean_agg",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MeanAggPolars(SeriesOperator):
    """Auto-generated Polars bridge for mean_agg.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="mean_agg",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="micro_amihud_hf",
    canonical="micro_amihud_hf",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MicroAmihudHfPolars(SeriesOperator):
    """Auto-generated Polars bridge for micro_amihud_hf.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="micro_amihud_hf",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="micro_bipower_var",
    canonical="micro_bipower_var",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MicroBipowerVarPolars(SeriesOperator):
    """Auto-generated Polars bridge for micro_bipower_var.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="micro_bipower_var",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="micro_bvc_vpin",
    canonical="micro_bvc_vpin",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MicroBvcVpinPolars(SeriesOperator):
    """Auto-generated Polars bridge for micro_bvc_vpin.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="micro_bvc_vpin",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="micro_jump_indicator",
    canonical="micro_jump_indicator",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MicroJumpIndicatorPolars(SeriesOperator):
    """Auto-generated Polars bridge for micro_jump_indicator.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="micro_jump_indicator",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="micro_kyle_lambda",
    canonical="micro_kyle_lambda",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MicroKyleLambdaPolars(SeriesOperator):
    """Auto-generated Polars bridge for micro_kyle_lambda.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="micro_kyle_lambda",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="micro_mid_return",
    canonical="micro_mid_return",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MicroMidReturnPolars(SeriesOperator):
    """Auto-generated Polars bridge for micro_mid_return.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="micro_mid_return",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="micro_realized_vol",
    canonical="micro_realized_vol",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MicroRealizedVolPolars(SeriesOperator):
    """Auto-generated Polars bridge for micro_realized_vol.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="micro_realized_vol",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="micro_spread",
    canonical="micro_spread",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MicroSpreadPolars(SeriesOperator):
    """Auto-generated Polars bridge for micro_spread.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="micro_spread",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="micro_trade_imbalance",
    canonical="micro_trade_imbalance",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MicroTradeImbalancePolars(SeriesOperator):
    """Auto-generated Polars bridge for micro_trade_imbalance.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="micro_trade_imbalance",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="micro_vpin",
    canonical="micro_vpin",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MicroVpinPolars(SeriesOperator):
    """Auto-generated Polars bridge for micro_vpin.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="micro_vpin",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="minimum",
    canonical="minimum",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MinimumPolars(SeriesOperator):
    """Auto-generated Polars bridge for minimum.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="minimum",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="move",
    canonical="move",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MovePolars(SeriesOperator):
    """Auto-generated Polars bridge for move.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="move",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="multiply",
    canonical="multiply",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class MultiplyPolars(SeriesOperator):
    """Auto-generated Polars bridge for multiply.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="multiply",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="nan_to_num",
    canonical="nan_to_num",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class NanToNumPolars(SeriesOperator):
    """Auto-generated Polars bridge for nan_to_num.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="nan_to_num",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ne",
    canonical="ne",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class NePolars(SeriesOperator):
    """Auto-generated Polars bridge for ne.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ne",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="neg",
    canonical="neg",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class NegPolars(SeriesOperator):
    """Auto-generated Polars bridge for neg.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="neg",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="negate",
    canonical="negate",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class NegatePolars(SeriesOperator):
    """Auto-generated Polars bridge for negate.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="negate",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="neutralize",
    canonical="neutralize",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class NeutralizePolars(SeriesOperator):
    """Auto-generated Polars bridge for neutralize.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="neutralize",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="nonfinite_to_num",
    canonical="nonfinite_to_num",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class NonfiniteToNumPolars(SeriesOperator):
    """Auto-generated Polars bridge for nonfinite_to_num.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="nonfinite_to_num",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="not_",
    canonical="not_",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class NotPolars(SeriesOperator):
    """Auto-generated Polars bridge for not_.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="not_",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ohlc_corwin_schultz_spread",
    canonical="ohlc_corwin_schultz_spread",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class OhlcCorwinSchultzSpreadPolars(SeriesOperator):
    """Auto-generated Polars bridge for ohlc_corwin_schultz_spread.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ohlc_corwin_schultz_spread",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="open_close_return",
    canonical="open_close_return",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class OpenCloseReturnPolars(SeriesOperator):
    """Auto-generated Polars bridge for open_close_return.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="open_close_return",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="open_gap",
    canonical="open_gap",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class OpenGapPolars(SeriesOperator):
    """Auto-generated Polars bridge for open_gap.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="open_gap",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="open_to_vwap_return",
    canonical="open_to_vwap_return",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class OpenToVwapReturnPolars(SeriesOperator):
    """Auto-generated Polars bridge for open_to_vwap_return.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="open_to_vwap_return",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="operating_margin",
    canonical="operating_margin",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class OperatingMarginPolars(SeriesOperator):
    """Auto-generated Polars bridge for operating_margin.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="operating_margin",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="or_",
    canonical="or_",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class OrPolars(SeriesOperator):
    """Auto-generated Polars bridge for or_.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="or_",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="overnight_return",
    canonical="overnight_return",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class OvernightReturnPolars(SeriesOperator):
    """Auto-generated Polars bridge for overnight_return.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="overnight_return",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="pacf",
    canonical="pacf",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class PacfPolars(SeriesOperator):
    """Auto-generated Polars bridge for pacf.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="pacf",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="panel_neutralize",
    canonical="panel_neutralize",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class PanelNeutralizePolars(SeriesOperator):
    """Auto-generated Polars bridge for panel_neutralize.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="panel_neutralize",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="panel_rank",
    canonical="panel_rank",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class PanelRankPolars(SeriesOperator):
    """Auto-generated Polars bridge for panel_rank.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="panel_rank",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="panel_standardize",
    canonical="panel_standardize",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class PanelStandardizePolars(SeriesOperator):
    """Auto-generated Polars bridge for panel_standardize.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="panel_standardize",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="panel_zscore",
    canonical="panel_zscore",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class PanelZscorePolars(SeriesOperator):
    """Auto-generated Polars bridge for panel_zscore.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="panel_zscore",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="pca",
    canonical="pca",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class PcaPolars(SeriesOperator):
    """Auto-generated Polars bridge for pca.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="pca",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="pdf_chi2",
    canonical="pdf_chi2",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class PdfChi2Polars(SeriesOperator):
    """Auto-generated Polars bridge for pdf_chi2.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="pdf_chi2",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="pdf_f",
    canonical="pdf_f",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class PdfFPolars(SeriesOperator):
    """Auto-generated Polars bridge for pdf_f.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="pdf_f",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="pdf_normal",
    canonical="pdf_normal",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class PdfNormalPolars(SeriesOperator):
    """Auto-generated Polars bridge for pdf_normal.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="pdf_normal",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="pdf_t",
    canonical="pdf_t",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class PdfTPolars(SeriesOperator):
    """Auto-generated Polars bridge for pdf_t.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="pdf_t",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="phase",
    canonical="phase",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class PhasePolars(SeriesOperator):
    """Auto-generated Polars bridge for phase.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="phase",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="polar",
    canonical="polar",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class PolarPolars(SeriesOperator):
    """Auto-generated Polars bridge for polar.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="polar",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="power",
    canonical="power",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class PowerPolars(SeriesOperator):
    """Auto-generated Polars bridge for power.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="power",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="prev",
    canonical="prev",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class PrevPolars(SeriesOperator):
    """Auto-generated Polars bridge for prev.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="prev",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="price_spread_deviation",
    canonical="price_spread_deviation",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class PriceSpreadDeviationPolars(SeriesOperator):
    """Auto-generated Polars bridge for price_spread_deviation.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="price_spread_deviation",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="product",
    canonical="product",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ProductPolars(SeriesOperator):
    """Auto-generated Polars bridge for product.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="product",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="protected_div",
    canonical="protected_div",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ProtectedDivPolars(SeriesOperator):
    """Auto-generated Polars bridge for protected_div.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="protected_div",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="protected_log",
    canonical="protected_log",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ProtectedLogPolars(SeriesOperator):
    """Auto-generated Polars bridge for protected_log.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="protected_log",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="qr_decompose",
    canonical="qr_decompose",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class QrDecomposePolars(SeriesOperator):
    """Auto-generated Polars bridge for qr_decompose.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="qr_decompose",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="quantile",
    canonical="quantile",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class QuantilePolars(SeriesOperator):
    """Auto-generated Polars bridge for quantile.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="quantile",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="quantile_normal",
    canonical="quantile_normal",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class QuantileNormalPolars(SeriesOperator):
    """Auto-generated Polars bridge for quantile_normal.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="quantile_normal",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="quantile_t",
    canonical="quantile_t",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class QuantileTPolars(SeriesOperator):
    """Auto-generated Polars bridge for quantile_t.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="quantile_t",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="quarter",
    canonical="quarter",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class QuarterPolars(SeriesOperator):
    """Auto-generated Polars bridge for quarter.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="quarter",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="quick_ratio",
    canonical="quick_ratio",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class QuickRatioPolars(SeriesOperator):
    """Auto-generated Polars bridge for quick_ratio.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="quick_ratio",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="r_squared",
    canonical="r_squared",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RSquaredPolars(SeriesOperator):
    """Auto-generated Polars bridge for r_squared.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="r_squared",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="rank",
    canonical="rank",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RankPolars(SeriesOperator):
    """Auto-generated Polars bridge for rank.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="rank",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="rank_corr",
    canonical="rank_corr",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RankCorrPolars(SeriesOperator):
    """Auto-generated Polars bridge for rank_corr.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="rank_corr",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="rank_pct",
    canonical="rank_pct",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RankPctPolars(SeriesOperator):
    """Auto-generated Polars bridge for rank_pct.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="rank_pct",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="rank_transform",
    canonical="rank_transform",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RankTransformPolars(SeriesOperator):
    """Auto-generated Polars bridge for rank_transform.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="rank_transform",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="rankavg_transform",
    canonical="rankavg_transform",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RankavgTransformPolars(SeriesOperator):
    """Auto-generated Polars bridge for rankavg_transform.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="rankavg_transform",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="real",
    canonical="real",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RealPolars(SeriesOperator):
    """Auto-generated Polars bridge for real.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="real",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="real_turnover_rate",
    canonical="real_turnover_rate",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RealTurnoverRatePolars(SeriesOperator):
    """Auto-generated Polars bridge for real_turnover_rate.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="real_turnover_rate",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="reciprocal",
    canonical="reciprocal",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ReciprocalPolars(SeriesOperator):
    """Auto-generated Polars bridge for reciprocal.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="reciprocal",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="regress",
    canonical="regress",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RegressPolars(SeriesOperator):
    """Auto-generated Polars bridge for regress.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="regress",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_category_share",
    canonical="relation_category_share",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationCategorySharePolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_category_share.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_category_share",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_category_signed_contribution",
    canonical="relation_category_signed_contribution",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationCategorySignedContributionPolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_category_signed_contribution.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_category_signed_contribution",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_concentration_acceleration",
    canonical="relation_concentration_acceleration",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationConcentrationAccelerationPolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_concentration_acceleration.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_concentration_acceleration",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_diffusion_score",
    canonical="relation_diffusion_score",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationDiffusionScorePolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_diffusion_score.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_diffusion_score",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_distinct_count",
    canonical="relation_distinct_count",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationDistinctCountPolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_distinct_count.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_distinct_count",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_distribution_excess_kurtosis",
    canonical="relation_distribution_excess_kurtosis",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationDistributionExcessKurtosisPolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_distribution_excess_kurtosis.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_distribution_excess_kurtosis",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_distribution_pearson_kurtosis",
    canonical="relation_distribution_pearson_kurtosis",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationDistributionPearsonKurtosisPolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_distribution_pearson_kurtosis.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_distribution_pearson_kurtosis",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_distribution_skew",
    canonical="relation_distribution_skew",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationDistributionSkewPolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_distribution_skew.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_distribution_skew",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_entropy",
    canonical="relation_entropy",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationEntropyPolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_entropy.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_entropy",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_entropy_change",
    canonical="relation_entropy_change",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationEntropyChangePolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_entropy_change.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_entropy_change",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_entry_count",
    canonical="relation_entry_count",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationEntryCountPolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_entry_count.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_entry_count",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_exit_count",
    canonical="relation_exit_count",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationExitCountPolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_exit_count.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_exit_count",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_hhi",
    canonical="relation_hhi",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationHhiPolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_hhi.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_hhi",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_hhi_change",
    canonical="relation_hhi_change",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationHhiChangePolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_hhi_change.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_hhi_change",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_overlap_ratio",
    canonical="relation_overlap_ratio",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationOverlapRatioPolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_overlap_ratio.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_overlap_ratio",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_peer_weighted_mean_ex_self",
    canonical="relation_peer_weighted_mean_ex_self",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationPeerWeightedMeanExSelfPolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_peer_weighted_mean_ex_self.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_peer_weighted_mean_ex_self",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_rank_entity_mobility",
    canonical="relation_rank_entity_mobility",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationRankEntityMobilityPolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_rank_entity_mobility.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_rank_entity_mobility",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_rank_mobility",
    canonical="relation_rank_mobility",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationRankMobilityPolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_rank_mobility.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_rank_mobility",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_rank_weighted_sum",
    canonical="relation_rank_weighted_sum",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationRankWeightedSumPolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_rank_weighted_sum.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_rank_weighted_sum",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_share_mobility",
    canonical="relation_share_mobility",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationShareMobilityPolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_share_mobility.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_share_mobility",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_topk_concentration",
    canonical="relation_topk_concentration",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationTopkConcentrationPolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_topk_concentration.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_topk_concentration",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_topk_sum",
    canonical="relation_topk_sum",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationTopkSumPolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_topk_sum.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_topk_sum",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_weighted_change",
    canonical="relation_weighted_change",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationWeightedChangePolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_weighted_change.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_weighted_change",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="relation_weighted_std_ex_self",
    canonical="relation_weighted_std_ex_self",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RelationWeightedStdExSelfPolars(SeriesOperator):
    """Auto-generated Polars bridge for relation_weighted_std_ex_self.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="relation_weighted_std_ex_self",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="report_benford_js_divergence",
    canonical="report_benford_js_divergence",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ReportBenfordJsDivergencePolars(SeriesOperator):
    """Auto-generated Polars bridge for report_benford_js_divergence.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="report_benford_js_divergence",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="report_filing_delay_surprise",
    canonical="report_filing_delay_surprise",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ReportFilingDelaySurprisePolars(SeriesOperator):
    """Auto-generated Polars bridge for report_filing_delay_surprise.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="report_filing_delay_surprise",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="report_revision_magnitude",
    canonical="report_revision_magnitude",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ReportRevisionMagnitudePolars(SeriesOperator):
    """Auto-generated Polars bridge for report_revision_magnitude.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="report_revision_magnitude",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="residual",
    canonical="residual",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ResidualPolars(SeriesOperator):
    """Auto-generated Polars bridge for residual.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="residual",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="residual_momentum_capm",
    canonical="residual_momentum_capm",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ResidualMomentumCapmPolars(SeriesOperator):
    """Auto-generated Polars bridge for residual_momentum_capm.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="residual_momentum_capm",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="returns",
    canonical="returns",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ReturnsPolars(SeriesOperator):
    """Auto-generated Polars bridge for returns.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="returns",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="reverse",
    canonical="reverse",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ReversePolars(SeriesOperator):
    """Auto-generated Polars bridge for reverse.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="reverse",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ridge",
    canonical="ridge",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RidgePolars(SeriesOperator):
    """Auto-generated Polars bridge for ridge.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ridge",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# WINDOW operators
# ------------------------------------------------------------------------------

@register_operator(
    name="rolling_beta",
    canonical="rolling_beta",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RollingBetaPolars(SeriesOperator):
    """Auto-generated Polars bridge for rolling_beta.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="rolling_beta",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="rolling_beta_to_market",
    canonical="rolling_beta_to_market",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RollingBetaToMarketPolars(SeriesOperator):
    """Auto-generated Polars bridge for rolling_beta_to_market.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="rolling_beta_to_market",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="round",
    canonical="round",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RoundPolars(SeriesOperator):
    """Auto-generated Polars bridge for round.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="round",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="row_avg",
    canonical="row_avg",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RowAvgPolars(SeriesOperator):
    """Auto-generated Polars bridge for row_avg.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="row_avg",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="row_beta",
    canonical="row_beta",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RowBetaPolars(SeriesOperator):
    """Auto-generated Polars bridge for row_beta.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="row_beta",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="row_corr",
    canonical="row_corr",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RowCorrPolars(SeriesOperator):
    """Auto-generated Polars bridge for row_corr.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="row_corr",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="row_count",
    canonical="row_count",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RowCountPolars(SeriesOperator):
    """Auto-generated Polars bridge for row_count.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="row_count",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="row_kurt",
    canonical="row_kurt",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RowKurtPolars(SeriesOperator):
    """Auto-generated Polars bridge for row_kurt.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="row_kurt",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="row_max",
    canonical="row_max",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RowMaxPolars(SeriesOperator):
    """Auto-generated Polars bridge for row_max.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="row_max",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="row_median",
    canonical="row_median",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RowMedianPolars(SeriesOperator):
    """Auto-generated Polars bridge for row_median.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="row_median",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="row_min",
    canonical="row_min",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RowMinPolars(SeriesOperator):
    """Auto-generated Polars bridge for row_min.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="row_min",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="row_prod",
    canonical="row_prod",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RowProdPolars(SeriesOperator):
    """Auto-generated Polars bridge for row_prod.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="row_prod",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="row_skew",
    canonical="row_skew",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RowSkewPolars(SeriesOperator):
    """Auto-generated Polars bridge for row_skew.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="row_skew",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="row_std",
    canonical="row_std",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RowStdPolars(SeriesOperator):
    """Auto-generated Polars bridge for row_std.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="row_std",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="row_sum",
    canonical="row_sum",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RowSumPolars(SeriesOperator):
    """Auto-generated Polars bridge for row_sum.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="row_sum",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="row_var",
    canonical="row_var",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RowVarPolars(SeriesOperator):
    """Auto-generated Polars bridge for row_var.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="row_var",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="running_mean",
    canonical="running_mean",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RunningMeanPolars(SeriesOperator):
    """Auto-generated Polars bridge for running_mean.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="running_mean",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="running_std",
    canonical="running_std",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RunningStdPolars(SeriesOperator):
    """Auto-generated Polars bridge for running_std.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="running_std",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="running_sum",
    canonical="running_sum",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class RunningSumPolars(SeriesOperator):
    """Auto-generated Polars bridge for running_sum.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="running_sum",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="safe_div_null",
    canonical="safe_div_null",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SafeDivNullPolars(SeriesOperator):
    """Auto-generated Polars bridge for safe_div_null.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="safe_div_null",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="saturate",
    canonical="saturate",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SaturatePolars(SeriesOperator):
    """Auto-generated Polars bridge for saturate.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="saturate",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="scale",
    canonical="scale",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ScalePolars(SeriesOperator):
    """Auto-generated Polars bridge for scale.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="scale",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="sec",
    canonical="sec",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SecPolars(SeriesOperator):
    """Auto-generated Polars bridge for sec.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="sec",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="sem",
    canonical="sem",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SemPolars(SeriesOperator):
    """Auto-generated Polars bridge for sem.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="sem",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="session_event_recovery_score",
    canonical="session_event_recovery_score",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SessionEventRecoveryScorePolars(SeriesOperator):
    """Auto-generated Polars bridge for session_event_recovery_score.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="session_event_recovery_score",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="sharpe_ratio",
    canonical="sharpe_ratio",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SharpeRatioPolars(SeriesOperator):
    """Auto-generated Polars bridge for sharpe_ratio.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="sharpe_ratio",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="sigmoid",
    canonical="sigmoid",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SigmoidPolars(SeriesOperator):
    """Auto-generated Polars bridge for sigmoid.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="sigmoid",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="sign",
    canonical="sign",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SignPolars(SeriesOperator):
    """Auto-generated Polars bridge for sign.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="sign",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="signed_log",
    canonical="signed_log",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SignedLogPolars(SeriesOperator):
    """Auto-generated Polars bridge for signed_log.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="signed_log",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="signed_power",
    canonical="signed_power",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SignedPowerPolars(SeriesOperator):
    """Auto-generated Polars bridge for signed_power.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="signed_power",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="sin",
    canonical="sin",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SinPolars(SeriesOperator):
    """Auto-generated Polars bridge for sin.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="sin",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="sin_phase",
    canonical="sin_phase",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SinPhasePolars(SeriesOperator):
    """Auto-generated Polars bridge for sin_phase.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="sin_phase",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="sinh",
    canonical="sinh",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SinhPolars(SeriesOperator):
    """Auto-generated Polars bridge for sinh.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="sinh",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="slope",
    canonical="slope",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SlopePolars(SeriesOperator):
    """Auto-generated Polars bridge for slope.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="slope",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="spearman_corr_test",
    canonical="spearman_corr_test",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SpearmanCorrTestPolars(SeriesOperator):
    """Auto-generated Polars bridge for spearman_corr_test.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="spearman_corr_test",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="sqr",
    canonical="sqr",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SqrPolars(SeriesOperator):
    """Auto-generated Polars bridge for sqr.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="sqr",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="sqrt",
    canonical="sqrt",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SqrtPolars(SeriesOperator):
    """Auto-generated Polars bridge for sqrt.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="sqrt",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="sqrt_abs",
    canonical="sqrt_abs",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SqrtAbsPolars(SeriesOperator):
    """Auto-generated Polars bridge for sqrt_abs.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="sqrt_abs",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="square",
    canonical="square",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SquarePolars(SeriesOperator):
    """Auto-generated Polars bridge for square.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="square",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="standardize",
    canonical="standardize",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StandardizePolars(SeriesOperator):
    """Auto-generated Polars bridge for standardize.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="standardize",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_adaptive_deadband",
    canonical="state_adaptive_deadband",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateAdaptiveDeadbandPolars(SeriesOperator):
    """Auto-generated Polars bridge for state_adaptive_deadband.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_adaptive_deadband",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_adaptive_slew_limit",
    canonical="state_adaptive_slew_limit",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateAdaptiveSlewLimitPolars(SeriesOperator):
    """Auto-generated Polars bridge for state_adaptive_slew_limit.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_adaptive_slew_limit",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_confidence_weighted_ema",
    canonical="state_confidence_weighted_ema",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateConfidenceWeightedEmaPolars(SeriesOperator):
    """Auto-generated Polars bridge for state_confidence_weighted_ema.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_confidence_weighted_ema",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_cost_aware_deadband",
    canonical="state_cost_aware_deadband",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateCostAwareDeadbandPolars(SeriesOperator):
    """Auto-generated Polars bridge for state_cost_aware_deadband.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_cost_aware_deadband",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_cost_aware_slew",
    canonical="state_cost_aware_slew",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateCostAwareSlewPolars(SeriesOperator):
    """Auto-generated Polars bridge for state_cost_aware_slew.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_cost_aware_slew",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_deadband",
    canonical="state_deadband",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateDeadbandPolars(SeriesOperator):
    """Auto-generated Polars bridge for state_deadband.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_deadband",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_episode_efficiency",
    canonical="state_episode_efficiency",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateEpisodeEfficiencyPolars(SeriesOperator):
    """Auto-generated Polars bridge for state_episode_efficiency.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_episode_efficiency",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_episode_excursion_balance",
    canonical="state_episode_excursion_balance",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateEpisodeExcursionBalancePolars(SeriesOperator):
    """Auto-generated Polars bridge for state_episode_excursion_balance.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_episode_excursion_balance",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_episode_mae",
    canonical="state_episode_mae",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateEpisodeMaePolars(SeriesOperator):
    """Auto-generated Polars bridge for state_episode_mae.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_episode_mae",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_episode_mfe",
    canonical="state_episode_mfe",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateEpisodeMfePolars(SeriesOperator):
    """Auto-generated Polars bridge for state_episode_mfe.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_episode_mfe",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_episode_retrace_ratio",
    canonical="state_episode_retrace_ratio",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateEpisodeRetraceRatioPolars(SeriesOperator):
    """Auto-generated Polars bridge for state_episode_retrace_ratio.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_episode_retrace_ratio",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_ewm_if",
    canonical="state_ewm_if",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateEwmIfPolars(SeriesOperator):
    """Auto-generated Polars bridge for state_ewm_if.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_ewm_if",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_hold",
    canonical="state_hold",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateHoldPolars(SeriesOperator):
    """Auto-generated Polars bridge for state_hold.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_hold",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_l1_turnover_prox",
    canonical="state_l1_turnover_prox",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateL1TurnoverProxPolars(SeriesOperator):
    """Auto-generated Polars bridge for state_l1_turnover_prox.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_l1_turnover_prox",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_l2_partial_adjustment",
    canonical="state_l2_partial_adjustment",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateL2PartialAdjustmentPolars(SeriesOperator):
    """Auto-generated Polars bridge for state_l2_partial_adjustment.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_l2_partial_adjustment",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_latch",
    canonical="state_latch",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateLatchPolars(SeriesOperator):
    """Auto-generated Polars bridge for state_latch.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_latch",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_quantile_hysteresis",
    canonical="state_quantile_hysteresis",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateQuantileHysteresisPolars(SeriesOperator):
    """Auto-generated Polars bridge for state_quantile_hysteresis.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_quantile_hysteresis",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_rank_deadband",
    canonical="state_rank_deadband",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateRankDeadbandPolars(SeriesOperator):
    """Auto-generated Polars bridge for state_rank_deadband.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_rank_deadband",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_since_count",
    canonical="state_since_count",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateSinceCountPolars(SeriesOperator):
    """Auto-generated Polars bridge for state_since_count.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_since_count",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_since_last",
    canonical="state_since_last",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateSinceLastPolars(SeriesOperator):
    """Auto-generated Polars bridge for state_since_last.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_since_last",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_since_mean",
    canonical="state_since_mean",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateSinceMeanPolars(SeriesOperator):
    """Auto-generated Polars bridge for state_since_mean.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_since_mean",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_since_sum",
    canonical="state_since_sum",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateSinceSumPolars(SeriesOperator):
    """Auto-generated Polars bridge for state_since_sum.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_since_sum",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_since_trend_tstat",
    canonical="state_since_trend_tstat",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateSinceTrendTstatPolars(SeriesOperator):
    """Auto-generated Polars bridge for state_since_trend_tstat.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_since_trend_tstat",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_slew_limit",
    canonical="state_slew_limit",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateSlewLimitPolars(SeriesOperator):
    """Auto-generated Polars bridge for state_slew_limit.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_slew_limit",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="state_uncertainty_deadband",
    canonical="state_uncertainty_deadband",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StateUncertaintyDeadbandPolars(SeriesOperator):
    """Auto-generated Polars bridge for state_uncertainty_deadband.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="state_uncertainty_deadband",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="stationarity_test",
    canonical="stationarity_test",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StationarityTestPolars(SeriesOperator):
    """Auto-generated Polars bridge for stationarity_test.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="stationarity_test",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="std_agg",
    canonical="std_agg",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StdAggPolars(SeriesOperator):
    """Auto-generated Polars bridge for std_agg.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="std_agg",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="stdp",
    canonical="stdp",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class StdpPolars(SeriesOperator):
    """Auto-generated Polars bridge for stdp.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="stdp",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="subtract",
    canonical="subtract",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SubtractPolars(SeriesOperator):
    """Auto-generated Polars bridge for subtract.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="subtract",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="sum_agg",
    canonical="sum_agg",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SumAggPolars(SeriesOperator):
    """Auto-generated Polars bridge for sum_agg.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="sum_agg",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="svd",
    canonical="svd",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class SvdPolars(SeriesOperator):
    """Auto-generated Polars bridge for svd.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="svd",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="tail_beta",
    canonical="tail_beta",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TailBetaPolars(SeriesOperator):
    """Auto-generated Polars bridge for tail_beta.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="tail_beta",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="tan",
    canonical="tan",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TanPolars(SeriesOperator):
    """Auto-generated Polars bridge for tan.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="tan",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="tanh",
    canonical="tanh",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TanhPolars(SeriesOperator):
    """Auto-generated Polars bridge for tanh.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="tanh",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="tm_top_n_avg",
    canonical="tm_top_n_avg",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TmTopNAvgPolars(SeriesOperator):
    """Auto-generated Polars bridge for tm_top_n_avg.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="tm_top_n_avg",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="tm_top_n_sum",
    canonical="tm_top_n_sum",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TmTopNSumPolars(SeriesOperator):
    """Auto-generated Polars bridge for tm_top_n_sum.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="tm_top_n_sum",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="tradable_state",
    canonical="tradable_state",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TradableStatePolars(SeriesOperator):
    """Auto-generated Polars bridge for tradable_state.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="tradable_state",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="trade_when",
    canonical="trade_when",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TradeWhenPolars(SeriesOperator):
    """Auto-generated Polars bridge for trade_when.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="trade_when",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="trading_day_diff",
    canonical="trading_day_diff",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TradingDayDiffPolars(SeriesOperator):
    """Auto-generated Polars bridge for trading_day_diff.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="trading_day_diff",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="true_turnover_rate",
    canonical="true_turnover_rate",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TrueTurnoverRatePolars(SeriesOperator):
    """Auto-generated Polars bridge for true_turnover_rate.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="true_turnover_rate",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="truncate",
    canonical="truncate",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TruncatePolars(SeriesOperator):
    """Auto-generated Polars bridge for truncate.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="truncate",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# WINDOW operators
# ------------------------------------------------------------------------------

@register_operator(
    name="ts_abs_concentration",
    canonical="ts_abs_concentration",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsAbsConcentrationPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_abs_concentration.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_abs_concentration",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_abs_entropy",
    canonical="ts_abs_entropy",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsAbsEntropyPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_abs_entropy.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_abs_entropy",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_abs_entropy_nats",
    canonical="ts_abs_entropy_nats",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsAbsEntropyNatsPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_abs_entropy_nats.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_abs_entropy_nats",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_abs_entropy_normalized",
    canonical="ts_abs_entropy_normalized",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsAbsEntropyNormalizedPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_abs_entropy_normalized.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_abs_entropy_normalized",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_active_information_storage",
    canonical="ts_active_information_storage",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsActiveInformationStoragePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_active_information_storage.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_active_information_storage",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_adaptive_noise_kalman",
    canonical="ts_adaptive_noise_kalman",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsAdaptiveNoiseKalmanPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_adaptive_noise_kalman.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_adaptive_noise_kalman",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_alpha_beta_filter",
    canonical="ts_alpha_beta_filter",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsAlphaBetaFilterPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_alpha_beta_filter.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_alpha_beta_filter",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_ar_coefficient",
    canonical="ts_ar_coefficient",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsArCoefficientPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_ar_coefficient.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_ar_coefficient",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_argmax",
    canonical="ts_argmax",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsArgmaxPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_argmax.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_argmax",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_argmin",
    canonical="ts_argmin",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsArgminPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_argmin.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_argmin",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_autocorr_decay_half_life",
    canonical="ts_autocorr_decay_half_life",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsAutocorrDecayHalfLifePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_autocorr_decay_half_life.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_autocorr_decay_half_life",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_autocorrelation_time",
    canonical="ts_autocorrelation_time",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsAutocorrelationTimePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_autocorrelation_time.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_autocorrelation_time",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_autocorrelation_time_initial_positive_sequence",
    canonical="ts_autocorrelation_time_initial_positive_sequence",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsAutocorrelationTimeInitialPositiveSequencePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_autocorrelation_time_initial_positive_sequence.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_autocorrelation_time_initial_positive_sequence",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_bessel_lowpass_causal",
    canonical="ts_bessel_lowpass_causal",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsBesselLowpassCausalPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_bessel_lowpass_causal.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_bessel_lowpass_causal",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_best_lag_corr_excess",
    canonical="ts_best_lag_corr_excess",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsBestLagCorrExcessPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_best_lag_corr_excess.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_best_lag_corr_excess",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_best_lag_corr_raw",
    canonical="ts_best_lag_corr_raw",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsBestLagCorrRawPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_best_lag_corr_raw.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_best_lag_corr_raw",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_beta_break_score",
    canonical="ts_beta_break_score",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsBetaBreakScorePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_beta_break_score.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_beta_break_score",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_beta_if",
    canonical="ts_beta_if",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsBetaIfPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_beta_if.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_beta_if",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_betti_1_max_persistence",
    canonical="ts_betti_1_max_persistence",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsBetti1MaxPersistencePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_betti_1_max_persistence.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_betti_1_max_persistence",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_binned_response_curvature",
    canonical="ts_binned_response_curvature",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsBinnedResponseCurvaturePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_binned_response_curvature.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_binned_response_curvature",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_binned_response_monotonicity",
    canonical="ts_binned_response_monotonicity",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsBinnedResponseMonotonicityPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_binned_response_monotonicity.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_binned_response_monotonicity",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_bottom_n_avg",
    canonical="ts_bottom_n_avg",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsBottomNAvgPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_bottom_n_avg.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_bottom_n_avg",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_bottom_n_sum",
    canonical="ts_bottom_n_sum",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsBottomNSumPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_bottom_n_sum.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_bottom_n_sum",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_bures_corr_shift",
    canonical="ts_bures_corr_shift",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsBuresCorrShiftPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_bures_corr_shift.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_bures_corr_shift",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_butterworth_lowpass_causal",
    canonical="ts_butterworth_lowpass_causal",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsButterworthLowpassCausalPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_butterworth_lowpass_causal.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_butterworth_lowpass_causal",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_causal_local_linear_smoother",
    canonical="ts_causal_local_linear_smoother",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsCausalLocalLinearSmootherPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_causal_local_linear_smoother.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_causal_local_linear_smoother",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_causal_savgol_endpoint",
    canonical="ts_causal_savgol_endpoint",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsCausalSavgolEndpointPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_causal_savgol_endpoint.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_causal_savgol_endpoint",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_chatterjee_xi",
    canonical="ts_chatterjee_xi",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsChatterjeeXiPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_chatterjee_xi.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_chatterjee_xi",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_chord_excursion_area",
    canonical="ts_chord_excursion_area",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsChordExcursionAreaPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_chord_excursion_area.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_chord_excursion_area",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_conditional_mutual_information",
    canonical="ts_conditional_mutual_information",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsConditionalMutualInformationPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_conditional_mutual_information.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_conditional_mutual_information",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_conditional_transfer_entropy",
    canonical="ts_conditional_transfer_entropy",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsConditionalTransferEntropyPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_conditional_transfer_entropy.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_conditional_transfer_entropy",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_copula_central_asymmetry",
    canonical="ts_copula_central_asymmetry",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsCopulaCentralAsymmetryPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_copula_central_asymmetry.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_copula_central_asymmetry",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_corr_if",
    canonical="ts_corr_if",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsCorrIfPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_corr_if.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_corr_if",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_cov",
    canonical="ts_cov",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsCovPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_cov.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_cov",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_cpt_value",
    canonical="ts_cpt_value",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsCptValuePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_cpt_value.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_cpt_value",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_cross_extremogram",
    canonical="ts_cross_extremogram",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsCrossExtremogramPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_cross_extremogram.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_cross_extremogram",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_cross_quantilogram",
    canonical="ts_cross_quantilogram",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsCrossQuantilogramPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_cross_quantilogram.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_cross_quantilogram",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_crossing_acceleration",
    canonical="ts_crossing_acceleration",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsCrossingAccelerationPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_crossing_acceleration.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_crossing_acceleration",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_crossing_speed",
    canonical="ts_crossing_speed",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsCrossingSpeedPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_crossing_speed.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_crossing_speed",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_cumulative_deviation_score",
    canonical="ts_cumulative_deviation_score",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsCumulativeDeviationScorePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_cumulative_deviation_score.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_cumulative_deviation_score",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_current_drawdown_area",
    canonical="ts_current_drawdown_area",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsCurrentDrawdownAreaPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_current_drawdown_area.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_current_drawdown_area",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_current_drawdown_duration",
    canonical="ts_current_drawdown_duration",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsCurrentDrawdownDurationPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_current_drawdown_duration.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_current_drawdown_duration",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_cusum_pressure",
    canonical="ts_cusum_pressure",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsCusumPressurePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_cusum_pressure.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_cusum_pressure",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_decay_exp_window",
    canonical="ts_decay_exp_window",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsDecayExpWindowPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_decay_exp_window.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_decay_exp_window",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_delay",
    canonical="ts_delay",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsDelayPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_delay.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_delay",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_delay_intrinsic_dimension",
    canonical="ts_delay_intrinsic_dimension",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsDelayIntrinsicDimensionPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_delay_intrinsic_dimension.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_delay_intrinsic_dimension",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_delta",
    canonical="ts_delta",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsDeltaPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_delta.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_delta",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_distance_corr",
    canonical="ts_distance_corr",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsDistanceCorrPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_distance_corr.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_distance_corr",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_distance_correlation_partial_proxy",
    canonical="ts_distance_correlation_partial_proxy",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsDistanceCorrelationPartialProxyPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_distance_correlation_partial_proxy.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_distance_correlation_partial_proxy",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_distance_cov",
    canonical="ts_distance_cov",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsDistanceCovPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_distance_cov.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_distance_cov",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_downside_deviation",
    canonical="ts_downside_deviation",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsDownsideDeviationPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_downside_deviation.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_downside_deviation",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_effective_transfer_entropy",
    canonical="ts_effective_transfer_entropy",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsEffectiveTransferEntropyPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_effective_transfer_entropy.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_effective_transfer_entropy",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_effective_turning_rate",
    canonical="ts_effective_turning_rate",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsEffectiveTurningRatePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_effective_turning_rate.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_effective_turning_rate",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_endpoint_deviation",
    canonical="ts_endpoint_deviation",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsEndpointDeviationPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_endpoint_deviation.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_endpoint_deviation",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_energy_break_score",
    canonical="ts_energy_break_score",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsEnergyBreakScorePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_energy_break_score.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_energy_break_score",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_envelope_boundary_dwell",
    canonical="ts_envelope_boundary_dwell",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsEnvelopeBoundaryDwellPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_envelope_boundary_dwell.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_envelope_boundary_dwell",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_envelope_compression",
    canonical="ts_envelope_compression",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsEnvelopeCompressionPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_envelope_compression.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_envelope_compression",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_envelope_pressure",
    canonical="ts_envelope_pressure",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsEnvelopePressurePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_envelope_pressure.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_envelope_pressure",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_event_spacing_cv",
    canonical="ts_event_spacing_cv",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsEventSpacingCvPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_event_spacing_cv.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_event_spacing_cv",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_event_spacing_mean",
    canonical="ts_event_spacing_mean",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsEventSpacingMeanPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_event_spacing_mean.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_event_spacing_mean",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_expected_shortfall",
    canonical="ts_expected_shortfall",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsExpectedShortfallPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_expected_shortfall.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_expected_shortfall",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_expected_shortfall_asymmetry",
    canonical="ts_expected_shortfall_asymmetry",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsExpectedShortfallAsymmetryPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_expected_shortfall_asymmetry.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_expected_shortfall_asymmetry",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_expectile",
    canonical="ts_expectile",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsExpectilePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_expectile.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_expectile",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_expectile_beta",
    canonical="ts_expectile_beta",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsExpectileBetaPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_expectile_beta.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_expectile_beta",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_extrema_confirmation_rate",
    canonical="ts_extrema_confirmation_rate",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsExtremaConfirmationRatePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_extrema_confirmation_rate.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_extrema_confirmation_rate",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_extrema_divergence_strength",
    canonical="ts_extrema_divergence_strength",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsExtremaDivergenceStrengthPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_extrema_divergence_strength.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_extrema_divergence_strength",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_extremal_dependence_decay",
    canonical="ts_extremal_dependence_decay",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsExtremalDependenceDecayPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_extremal_dependence_decay.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_extremal_dependence_decay",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_extremal_index",
    canonical="ts_extremal_index",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsExtremalIndexPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_extremal_index.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_extremal_index",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_extreme_cluster_ratio",
    canonical="ts_extreme_cluster_ratio",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsExtremeClusterRatioPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_extreme_cluster_ratio.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_extreme_cluster_ratio",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_extremogram",
    canonical="ts_extremogram",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsExtremogramPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_extremogram.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_extremogram",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_feature_effective_rank",
    canonical="ts_feature_effective_rank",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsFeatureEffectiveRankPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_feature_effective_rank.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_feature_effective_rank",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_feature_mode_share",
    canonical="ts_feature_mode_share",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsFeatureModeSharePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_feature_mode_share.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_feature_mode_share",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_feature_subspace_rotation",
    canonical="ts_feature_subspace_rotation",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsFeatureSubspaceRotationPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_feature_subspace_rotation.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_feature_subspace_rotation",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_fir_lowpass_causal",
    canonical="ts_fir_lowpass_causal",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsFirLowpassCausalPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_fir_lowpass_causal.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_fir_lowpass_causal",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_first_passage_bias",
    canonical="ts_first_passage_bias",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsFirstPassageBiasPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_first_passage_bias.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_first_passage_bias",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_first_passage_conditional_time",
    canonical="ts_first_passage_conditional_time",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsFirstPassageConditionalTimePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_first_passage_conditional_time.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_first_passage_conditional_time",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_first_passage_hit_probability",
    canonical="ts_first_passage_hit_probability",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsFirstPassageHitProbabilityPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_first_passage_hit_probability.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_first_passage_hit_probability",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_fisher_information_shift",
    canonical="ts_fisher_information_shift",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsFisherInformationShiftPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_fisher_information_shift.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_fisher_information_shift",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_forbidden_ordinal_pattern_ratio",
    canonical="ts_forbidden_ordinal_pattern_ratio",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsForbiddenOrdinalPatternRatioPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_forbidden_ordinal_pattern_ratio.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_forbidden_ordinal_pattern_ratio",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_fractional_difference",
    canonical="ts_fractional_difference",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsFractionalDifferencePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_fractional_difference.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_fractional_difference",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_fractional_difference_discarded_weight_mass",
    canonical="ts_fractional_difference_discarded_weight_mass",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsFractionalDifferenceDiscardedWeightMassPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_fractional_difference_discarded_weight_mass.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_fractional_difference_discarded_weight_mass",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_generalized_hurst_exponent",
    canonical="ts_generalized_hurst_exponent",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsGeneralizedHurstExponentPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_generalized_hurst_exponent.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_generalized_hurst_exponent",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_generalized_hurst_spread_q1_q4",
    canonical="ts_generalized_hurst_spread_q1_q4",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsGeneralizedHurstSpreadQ1Q4Polars(SeriesOperator):
    """Auto-generated Polars bridge for ts_generalized_hurst_spread_q1_q4.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_generalized_hurst_spread_q1_q4",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_gpd_shape_pwm",
    canonical="ts_gpd_shape_pwm",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsGpdShapePwmPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_gpd_shape_pwm.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_gpd_shape_pwm",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_h_infinity_level_filter",
    canonical="ts_h_infinity_level_filter",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsHInfinityLevelFilterPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_h_infinity_level_filter.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_h_infinity_level_filter",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_hampel_filter_causal",
    canonical="ts_hampel_filter_causal",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsHampelFilterCausalPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_hampel_filter_causal.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_hampel_filter_causal",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_hankel_effective_rank",
    canonical="ts_hankel_effective_rank",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsHankelEffectiveRankPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_hankel_effective_rank.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_hankel_effective_rank",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_hankel_singular_gap",
    canonical="ts_hankel_singular_gap",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsHankelSingularGapPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_hankel_singular_gap.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_hankel_singular_gap",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_hartigan_dip",
    canonical="ts_hartigan_dip",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsHartiganDipPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_hartigan_dip.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_hartigan_dip",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_higuchi_fractal_dimension",
    canonical="ts_higuchi_fractal_dimension",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsHiguchiFractalDimensionPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_higuchi_fractal_dimension.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_higuchi_fractal_dimension",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_hill_tail_index",
    canonical="ts_hill_tail_index",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsHillTailIndexPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_hill_tail_index.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_hill_tail_index",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_hsic",
    canonical="ts_hsic",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsHsicPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_hsic.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_hsic",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_huber_regression_in_sample_resid",
    canonical="ts_huber_regression_in_sample_resid",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsHuberRegressionInSampleResidPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_huber_regression_in_sample_resid.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_huber_regression_in_sample_resid",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_huber_regression_predictive_resid",
    canonical="ts_huber_regression_predictive_resid",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsHuberRegressionPredictiveResidPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_huber_regression_predictive_resid.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_huber_regression_predictive_resid",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_hurst_dfa",
    canonical="ts_hurst_dfa",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsHurstDfaPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_hurst_dfa.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_hurst_dfa",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_hysteresis_age",
    canonical="ts_hysteresis_age",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsHysteresisAgePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_hysteresis_age.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_hysteresis_age",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_hysteresis_state",
    canonical="ts_hysteresis_state",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsHysteresisStatePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_hysteresis_state.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_hysteresis_state",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_interval_exploration_efficiency",
    canonical="ts_interval_exploration_efficiency",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsIntervalExplorationEfficiencyPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_interval_exploration_efficiency.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_interval_exploration_efficiency",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_interval_nesting_depth",
    canonical="ts_interval_nesting_depth",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsIntervalNestingDepthPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_interval_nesting_depth.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_interval_nesting_depth",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_interval_occupancy_entropy",
    canonical="ts_interval_occupancy_entropy",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsIntervalOccupancyEntropyPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_interval_occupancy_entropy.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_interval_occupancy_entropy",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_interval_occupancy_mode_distance",
    canonical="ts_interval_occupancy_mode_distance",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsIntervalOccupancyModeDistancePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_interval_occupancy_mode_distance.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_interval_occupancy_mode_distance",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_interval_overlap_connected_component_ratio",
    canonical="ts_interval_overlap_connected_component_ratio",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsIntervalOverlapConnectedComponentRatioPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_interval_overlap_connected_component_ratio.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_interval_overlap_connected_component_ratio",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_interval_union_coverage",
    canonical="ts_interval_union_coverage",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsIntervalUnionCoveragePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_interval_union_coverage.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_interval_union_coverage",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_joint_energy_shift",
    canonical="ts_joint_energy_shift",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsJointEnergyShiftPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_joint_energy_shift.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_joint_energy_shift",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_jump_bipower_proxy",
    canonical="ts_jump_bipower_proxy",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsJumpBipowerProxyPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_jump_bipower_proxy.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_jump_bipower_proxy",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_kama",
    canonical="ts_kama",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsKamaPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_kama.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_kama",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_km_diffusion_gradient",
    canonical="ts_km_diffusion_gradient",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsKmDiffusionGradientPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_km_diffusion_gradient.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_km_diffusion_gradient",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_km_equilibrium_distance",
    canonical="ts_km_equilibrium_distance",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsKmEquilibriumDistancePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_km_equilibrium_distance.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_km_equilibrium_distance",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_km_quasipotential_depth",
    canonical="ts_km_quasipotential_depth",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsKmQuasipotentialDepthPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_km_quasipotential_depth.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_km_quasipotential_depth",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_kramers_moyal_diffusion",
    canonical="ts_kramers_moyal_diffusion",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsKramersMoyalDiffusionPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_kramers_moyal_diffusion.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_kramers_moyal_diffusion",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_kramers_moyal_drift",
    canonical="ts_kramers_moyal_drift",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsKramersMoyalDriftPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_kramers_moyal_drift.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_kramers_moyal_drift",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_kramers_moyal_local_stability",
    canonical="ts_kramers_moyal_local_stability",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsKramersMoyalLocalStabilityPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_kramers_moyal_local_stability.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_kramers_moyal_local_stability",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_ks_shift",
    canonical="ts_ks_shift",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsKsShiftPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_ks_shift.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_ks_shift",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_kurt",
    canonical="ts_kurt",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsKurtPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_kurt.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_kurt",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_l1_trend_filter_trailing",
    canonical="ts_l1_trend_filter_trailing",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsL1TrendFilterTrailingPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_l1_trend_filter_trailing.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_l1_trend_filter_trailing",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_l_kurtosis",
    canonical="ts_l_kurtosis",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsLKurtosisPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_l_kurtosis.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_l_kurtosis",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_l_skewness",
    canonical="ts_l_skewness",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsLSkewnessPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_l_skewness.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_l_skewness",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_lag_of_peak_corr",
    canonical="ts_lag_of_peak_corr",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsLagOfPeakCorrPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_lag_of_peak_corr.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_lag_of_peak_corr",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_lagged_mutual_information",
    canonical="ts_lagged_mutual_information",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsLaggedMutualInformationPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_lagged_mutual_information.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_lagged_mutual_information",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_lempel_ziv_complexity",
    canonical="ts_lempel_ziv_complexity",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsLempelZivComplexityPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_lempel_ziv_complexity.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_lempel_ziv_complexity",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_level_shift_score",
    canonical="ts_level_shift_score",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsLevelShiftScorePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_level_shift_score.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_level_shift_score",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_leverage_effect",
    canonical="ts_leverage_effect",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsLeverageEffectPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_leverage_effect.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_leverage_effect",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_lo_mackinlay_vr",
    canonical="ts_lo_mackinlay_vr",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsLoMackinlayVrPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_lo_mackinlay_vr.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_lo_mackinlay_vr",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_lo_mackinlay_z",
    canonical="ts_lo_mackinlay_z",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsLoMackinlayZPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_lo_mackinlay_z.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_lo_mackinlay_z",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_local_lyapunov_exponent",
    canonical="ts_local_lyapunov_exponent",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsLocalLyapunovExponentPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_local_lyapunov_exponent.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_local_lyapunov_exponent",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_location_shift",
    canonical="ts_location_shift",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsLocationShiftPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_location_shift.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_location_shift",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_lower_partial_moment",
    canonical="ts_lower_partial_moment",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsLowerPartialMomentPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_lower_partial_moment.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_lower_partial_moment",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_lower_tail_coexceedance_probability",
    canonical="ts_lower_tail_coexceedance_probability",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsLowerTailCoexceedanceProbabilityPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_lower_tail_coexceedance_probability.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_lower_tail_coexceedance_probability",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_mad",
    canonical="ts_mad",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMadPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_mad.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_mad",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_markov_committor",
    canonical="ts_markov_committor",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMarkovCommittorPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_markov_committor.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_markov_committor",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_markov_entropy_production",
    canonical="ts_markov_entropy_production",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMarkovEntropyProductionPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_markov_entropy_production.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_markov_entropy_production",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_markov_mean_first_passage_time",
    canonical="ts_markov_mean_first_passage_time",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMarkovMeanFirstPassageTimePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_markov_mean_first_passage_time.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_markov_mean_first_passage_time",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_markov_persistence",
    canonical="ts_markov_persistence",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMarkovPersistencePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_markov_persistence.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_markov_persistence",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_markov_spectral_gap",
    canonical="ts_markov_spectral_gap",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMarkovSpectralGapPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_markov_spectral_gap.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_markov_spectral_gap",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_markov_state_entropy",
    canonical="ts_markov_state_entropy",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMarkovStateEntropyPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_markov_state_entropy.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_markov_state_entropy",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_markov_stationary_surprisal",
    canonical="ts_markov_stationary_surprisal",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMarkovStationarySurprisalPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_markov_stationary_surprisal.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_markov_stationary_surprisal",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_markov_transition_surprisal",
    canonical="ts_markov_transition_surprisal",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMarkovTransitionSurprisalPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_markov_transition_surprisal.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_markov_transition_surprisal",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_mass_concentration",
    canonical="ts_mass_concentration",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMassConcentrationPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_mass_concentration.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_mass_concentration",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_matrix_profile_motif_age",
    canonical="ts_matrix_profile_motif_age",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMatrixProfileMotifAgePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_matrix_profile_motif_age.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_matrix_profile_motif_age",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_matrix_profile_motif_frequency",
    canonical="ts_matrix_profile_motif_frequency",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMatrixProfileMotifFrequencyPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_matrix_profile_motif_frequency.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_matrix_profile_motif_frequency",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_matrix_profile_neighbor_dispersion",
    canonical="ts_matrix_profile_neighbor_dispersion",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMatrixProfileNeighborDispersionPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_matrix_profile_neighbor_dispersion.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_matrix_profile_neighbor_dispersion",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_matrix_profile_novelty",
    canonical="ts_matrix_profile_novelty",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMatrixProfileNoveltyPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_matrix_profile_novelty.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_matrix_profile_novelty",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_max",
    canonical="ts_max",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMaxPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_max.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_max",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_max_buildup",
    canonical="ts_max_buildup",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMaxBuildupPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_max_buildup.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_max_buildup",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_max_chord_excursion",
    canonical="ts_max_chord_excursion",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMaxChordExcursionPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_max_chord_excursion.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_max_chord_excursion",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_max_if",
    canonical="ts_max_if",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMaxIfPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_max_if.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_max_if",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_mean",
    canonical="ts_mean",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMeanPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_mean.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_mean",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_mean_abs_deviation",
    canonical="ts_mean_abs_deviation",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMeanAbsDeviationPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_mean_abs_deviation.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_mean_abs_deviation",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_mean_excess_slope",
    canonical="ts_mean_excess_slope",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMeanExcessSlopePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_mean_excess_slope.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_mean_excess_slope",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_median3_causal",
    canonical="ts_median3_causal",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMedian3CausalPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_median3_causal.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_median3_causal",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_median_abs_deviation",
    canonical="ts_median_abs_deviation",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMedianAbsDeviationPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_median_abs_deviation.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_median_abs_deviation",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_min",
    canonical="ts_min",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMinPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_min.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_min",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_min_if",
    canonical="ts_min_if",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMinIfPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_min_if.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_min_if",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_mmd_rbf_shift",
    canonical="ts_mmd_rbf_shift",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMmdRbfShiftPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_mmd_rbf_shift.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_mmd_rbf_shift",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_modwt_band_corr",
    canonical="ts_modwt_band_corr",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsModwtBandCorrPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_modwt_band_corr.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_modwt_band_corr",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_monotonicity",
    canonical="ts_monotonicity",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMonotonicityPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_monotonicity.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_monotonicity",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_multifractal_curvature",
    canonical="ts_multifractal_curvature",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMultifractalCurvaturePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_multifractal_curvature.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_multifractal_curvature",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_multifractal_spectrum_width",
    canonical="ts_multifractal_spectrum_width",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMultifractalSpectrumWidthPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_multifractal_spectrum_width.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_multifractal_spectrum_width",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_multiscale_permutation_entropy_slope",
    canonical="ts_multiscale_permutation_entropy_slope",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMultiscalePermutationEntropySlopePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_multiscale_permutation_entropy_slope.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_multiscale_permutation_entropy_slope",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_multiscale_trend_consensus",
    canonical="ts_multiscale_trend_consensus",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMultiscaleTrendConsensusPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_multiscale_trend_consensus.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_multiscale_trend_consensus",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_multiscale_trend_curvature",
    canonical="ts_multiscale_trend_curvature",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMultiscaleTrendCurvaturePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_multiscale_trend_curvature.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_multiscale_trend_curvature",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_multiscale_trend_dispersion",
    canonical="ts_multiscale_trend_dispersion",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMultiscaleTrendDispersionPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_multiscale_trend_dispersion.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_multiscale_trend_dispersion",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_mutual_information",
    canonical="ts_mutual_information",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsMutualInformationPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_mutual_information.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_mutual_information",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_nearest_structural_level_distance",
    canonical="ts_nearest_structural_level_distance",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsNearestStructuralLevelDistancePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_nearest_structural_level_distance.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_nearest_structural_level_distance",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_negative_ratio",
    canonical="ts_negative_ratio",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsNegativeRatioPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_negative_ratio.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_negative_ratio",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_ordinal_irreversibility",
    canonical="ts_ordinal_irreversibility",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsOrdinalIrreversibilityPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_ordinal_irreversibility.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_ordinal_irreversibility",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_path_efficiency",
    canonical="ts_path_efficiency",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsPathEfficiencyPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_path_efficiency.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_path_efficiency",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_pct",
    canonical="ts_pct",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsPctPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_pct.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_pct",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_permutation_entropy",
    canonical="ts_permutation_entropy",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsPermutationEntropyPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_permutation_entropy.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_permutation_entropy",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_permutation_transition_entropy",
    canonical="ts_permutation_transition_entropy",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsPermutationTransitionEntropyPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_permutation_transition_entropy.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_permutation_transition_entropy",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_persistence_birth_dispersion",
    canonical="ts_persistence_birth_dispersion",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsPersistenceBirthDispersionPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_persistence_birth_dispersion.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_persistence_birth_dispersion",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_persistence_diagram_shift",
    canonical="ts_persistence_diagram_shift",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsPersistenceDiagramShiftPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_persistence_diagram_shift.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_persistence_diagram_shift",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_poly2_coeff",
    canonical="ts_poly2_coeff",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsPoly2CoeffPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_poly2_coeff.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_poly2_coeff",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_poly2_forecast_error",
    canonical="ts_poly2_forecast_error",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsPoly2ForecastErrorPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_poly2_forecast_error.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_poly2_forecast_error",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_poly2_forecast_error_z",
    canonical="ts_poly2_forecast_error_z",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsPoly2ForecastErrorZPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_poly2_forecast_error_z.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_poly2_forecast_error_z",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_poly2_prior_coeff",
    canonical="ts_poly2_prior_coeff",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsPoly2PriorCoeffPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_poly2_prior_coeff.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_poly2_prior_coeff",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_poly2_resid",
    canonical="ts_poly2_resid",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsPoly2ResidPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_poly2_resid.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_poly2_resid",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_positive_ratio",
    canonical="ts_positive_ratio",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsPositiveRatioPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_positive_ratio.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_positive_ratio",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_price_delay",
    canonical="ts_price_delay",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsPriceDelayPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_price_delay.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_price_delay",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_product",
    canonical="ts_product",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsProductPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_product.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_product",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_quantile",
    canonical="ts_quantile",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsQuantilePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_quantile.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_quantile",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_quantile_crossing_spectral_concentration",
    canonical="ts_quantile_crossing_spectral_concentration",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsQuantileCrossingSpectralConcentrationPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_quantile_crossing_spectral_concentration.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_quantile_crossing_spectral_concentration",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_quantile_if",
    canonical="ts_quantile_if",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsQuantileIfPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_quantile_if.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_quantile_if",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_quantile_kurtosis",
    canonical="ts_quantile_kurtosis",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsQuantileKurtosisPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_quantile_kurtosis.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_quantile_kurtosis",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_quantile_range",
    canonical="ts_quantile_range",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsQuantileRangePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_quantile_range.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_quantile_range",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_quantile_regression_beta",
    canonical="ts_quantile_regression_beta",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsQuantileRegressionBetaPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_quantile_regression_beta.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_quantile_regression_beta",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_quantile_regression_slope",
    canonical="ts_quantile_regression_slope",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsQuantileRegressionSlopePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_quantile_regression_slope.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_quantile_regression_slope",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_quantile_skew",
    canonical="ts_quantile_skew",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsQuantileSkewPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_quantile_skew.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_quantile_skew",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_quantile_transport_curvature",
    canonical="ts_quantile_transport_curvature",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsQuantileTransportCurvaturePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_quantile_transport_curvature.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_quantile_transport_curvature",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_quantile_transport_slope",
    canonical="ts_quantile_transport_slope",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsQuantileTransportSlopePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_quantile_transport_slope.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_quantile_transport_slope",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_quantilogram",
    canonical="ts_quantilogram",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsQuantilogramPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_quantilogram.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_quantilogram",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_rank_if",
    canonical="ts_rank_if",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsRankIfPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_rank_if.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_rank_if",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_ratio",
    canonical="ts_ratio",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsRatioPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_ratio.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_ratio",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_realized_quarticity",
    canonical="ts_realized_quarticity",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsRealizedQuarticityPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_realized_quarticity.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_realized_quarticity",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_recovery_fraction",
    canonical="ts_recovery_fraction",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsRecoveryFractionPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_recovery_fraction.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_recovery_fraction",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_recurrence_diagonal_entropy",
    canonical="ts_recurrence_diagonal_entropy",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsRecurrenceDiagonalEntropyPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_recurrence_diagonal_entropy.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_recurrence_diagonal_entropy",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_recurrence_divergence",
    canonical="ts_recurrence_divergence",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsRecurrenceDivergencePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_recurrence_divergence.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_recurrence_divergence",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_recurrence_rate",
    canonical="ts_recurrence_rate",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsRecurrenceRatePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_recurrence_rate.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_recurrence_rate",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_recurrence_trapping_time",
    canonical="ts_recurrence_trapping_time",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsRecurrenceTrappingTimePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_recurrence_trapping_time.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_recurrence_trapping_time",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_regression",
    canonical="ts_regression",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsRegressionPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_regression.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_regression",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_regression_resid_if",
    canonical="ts_regression_resid_if",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsRegressionResidIfPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_regression_resid_if.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_regression_resid_if",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_response_slope_asymmetry",
    canonical="ts_response_slope_asymmetry",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsResponseSlopeAsymmetryPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_response_slope_asymmetry.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_response_slope_asymmetry",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_ridge_regression_in_sample_resid",
    canonical="ts_ridge_regression_in_sample_resid",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsRidgeRegressionInSampleResidPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_ridge_regression_in_sample_resid.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_ridge_regression_in_sample_resid",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_ridge_regression_predictive_resid",
    canonical="ts_ridge_regression_predictive_resid",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsRidgeRegressionPredictiveResidPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_ridge_regression_predictive_resid.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_ridge_regression_predictive_resid",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_robust_ema",
    canonical="ts_robust_ema",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsRobustEmaPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_robust_ema.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_robust_ema",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_robust_zscore_inclusive",
    canonical="ts_robust_zscore_inclusive",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsRobustZscoreInclusivePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_robust_zscore_inclusive.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_robust_zscore_inclusive",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_robust_zscore_prior",
    canonical="ts_robust_zscore_prior",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsRobustZscorePriorPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_robust_zscore_prior.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_robust_zscore_prior",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_roll_effective_spread",
    canonical="ts_roll_effective_spread",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsRollEffectiveSpreadPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_roll_effective_spread.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_roll_effective_spread",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_rolling_median_causal",
    canonical="ts_rolling_median_causal",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsRollingMedianCausalPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_rolling_median_causal.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_rolling_median_causal",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_roughness",
    canonical="ts_roughness",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsRoughnessPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_roughness.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_roughness",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_run_concentration",
    canonical="ts_run_concentration",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsRunConcentrationPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_run_concentration.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_run_concentration",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_run_efficiency",
    canonical="ts_run_efficiency",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsRunEfficiencyPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_run_efficiency.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_run_efficiency",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_run_strength",
    canonical="ts_run_strength",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsRunStrengthPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_run_strength.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_run_strength",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_sample_entropy",
    canonical="ts_sample_entropy",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsSampleEntropyPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_sample_entropy.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_sample_entropy",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_scale_shift",
    canonical="ts_scale_shift",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsScaleShiftPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_scale_shift.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_scale_shift",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_score_rank_weighted_mean",
    canonical="ts_score_rank_weighted_mean",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsScoreRankWeightedMeanPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_score_rank_weighted_mean.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_score_rank_weighted_mean",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_semivariance_balance",
    canonical="ts_semivariance_balance",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsSemivarianceBalancePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_semivariance_balance.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_semivariance_balance",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_sign_cluster_index",
    canonical="ts_sign_cluster_index",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsSignClusterIndexPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_sign_cluster_index.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_sign_cluster_index",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_sign_persistence",
    canonical="ts_sign_persistence",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsSignPersistencePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_sign_persistence.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_sign_persistence",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_signature_mahalanobis_anomaly",
    canonical="ts_signature_mahalanobis_anomaly",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsSignatureMahalanobisAnomalyPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_signature_mahalanobis_anomaly.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_signature_mahalanobis_anomaly",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_skew",
    canonical="ts_skew",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsSkewPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_skew.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_skew",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_spectral_centroid",
    canonical="ts_spectral_centroid",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsSpectralCentroidPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_spectral_centroid.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_spectral_centroid",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_spectral_flatness",
    canonical="ts_spectral_flatness",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsSpectralFlatnessPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_spectral_flatness.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_spectral_flatness",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_spectral_lowpass_trailing",
    canonical="ts_spectral_lowpass_trailing",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsSpectralLowpassTrailingPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_spectral_lowpass_trailing.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_spectral_lowpass_trailing",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_spectral_peak_concentration",
    canonical="ts_spectral_peak_concentration",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsSpectralPeakConcentrationPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_spectral_peak_concentration.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_spectral_peak_concentration",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_spectral_quality_factor",
    canonical="ts_spectral_quality_factor",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsSpectralQualityFactorPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_spectral_quality_factor.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_spectral_quality_factor",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_ssa_denoise_trailing",
    canonical="ts_ssa_denoise_trailing",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsSsaDenoiseTrailingPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_ssa_denoise_trailing.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_ssa_denoise_trailing",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_ssa_prior_reconstruction_error",
    canonical="ts_ssa_prior_reconstruction_error",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsSsaPriorReconstructionErrorPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_ssa_prior_reconstruction_error.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_ssa_prior_reconstruction_error",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_ssa_reconstruction_residual",
    canonical="ts_ssa_reconstruction_residual",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsSsaReconstructionResidualPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_ssa_reconstruction_residual.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_ssa_reconstruction_residual",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_state_age_percentile",
    canonical="ts_state_age_percentile",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsStateAgePercentilePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_state_age_percentile.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_state_age_percentile",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_state_density",
    canonical="ts_state_density",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsStateDensityPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_state_density.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_state_density",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_state_entry_strength",
    canonical="ts_state_entry_strength",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsStateEntryStrengthPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_state_entry_strength.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_state_entry_strength",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_state_exit_hazard",
    canonical="ts_state_exit_hazard",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsStateExitHazardPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_state_exit_hazard.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_state_exit_hazard",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_state_integral",
    canonical="ts_state_integral",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsStateIntegralPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_state_integral.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_state_integral",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_state_residual_life",
    canonical="ts_state_residual_life",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsStateResidualLifePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_state_residual_life.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_state_residual_life",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_std",
    canonical="ts_std",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsStdPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_std.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_std",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_stratified_mean_spread",
    canonical="ts_stratified_mean_spread",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsStratifiedMeanSpreadPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_stratified_mean_spread.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_stratified_mean_spread",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_structural_level_density",
    canonical="ts_structural_level_density",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsStructuralLevelDensityPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_structural_level_density.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_structural_level_density",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_structural_level_strength",
    canonical="ts_structural_level_strength",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsStructuralLevelStrengthPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_structural_level_strength.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_structural_level_strength",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_student_t_kalman_filter",
    canonical="ts_student_t_kalman_filter",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsStudentTKalmanFilterPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_student_t_kalman_filter.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_student_t_kalman_filter",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_sum",
    canonical="ts_sum",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsSumPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_sum.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_sum",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_sum_decay",
    canonical="ts_sum_decay",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsSumDecayPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_sum_decay.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_sum_decay",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_super_smoother",
    canonical="ts_super_smoother",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsSuperSmootherPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_super_smoother.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_super_smoother",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_tail_imbalance",
    canonical="ts_tail_imbalance",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTailImbalancePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_tail_imbalance.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_tail_imbalance",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_tail_ratio",
    canonical="ts_tail_ratio",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTailRatioPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_tail_ratio.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_tail_ratio",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_threshold_cycle_asymmetry",
    canonical="ts_threshold_cycle_asymmetry",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsThresholdCycleAsymmetryPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_threshold_cycle_asymmetry.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_threshold_cycle_asymmetry",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_threshold_cycle_period",
    canonical="ts_threshold_cycle_period",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsThresholdCyclePeriodPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_threshold_cycle_period.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_threshold_cycle_period",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_time_since_change",
    canonical="ts_time_since_change",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTimeSinceChangePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_time_since_change.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_time_since_change",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_time_slope",
    canonical="ts_time_slope",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTimeSlopePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_time_slope.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_time_slope",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_time_under_water",
    canonical="ts_time_under_water",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTimeUnderWaterPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_time_under_water.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_time_under_water",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_top_n_avg",
    canonical="ts_top_n_avg",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTopNAvgPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_top_n_avg.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_top_n_avg",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_top_n_std",
    canonical="ts_top_n_std",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTopNStdPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_top_n_std.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_top_n_std",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_topk_sum",
    canonical="ts_topk_sum",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTopkSumPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_topk_sum.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_topk_sum",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_total_variation_filter_trailing",
    canonical="ts_total_variation_filter_trailing",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTotalVariationFilterTrailingPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_total_variation_filter_trailing.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_total_variation_filter_trailing",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_transfer_entropy",
    canonical="ts_transfer_entropy",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTransferEntropyPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_transfer_entropy.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_transfer_entropy",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_transfer_entropy_peak_excess",
    canonical="ts_transfer_entropy_peak_excess",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTransferEntropyPeakExcessPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_transfer_entropy_peak_excess.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_transfer_entropy_peak_excess",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_transfer_entropy_peak_lag",
    canonical="ts_transfer_entropy_peak_lag",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTransferEntropyPeakLagPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_transfer_entropy_peak_lag.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_transfer_entropy_peak_lag",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_transfer_entropy_peak_strength",
    canonical="ts_transfer_entropy_peak_strength",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTransferEntropyPeakStrengthPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_transfer_entropy_peak_strength.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_transfer_entropy_peak_strength",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_transition_count",
    canonical="ts_transition_count",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTransitionCountPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_transition_count.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_transition_count",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_transition_intensity",
    canonical="ts_transition_intensity",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTransitionIntensityPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_transition_intensity.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_transition_intensity",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_trend_break_score",
    canonical="ts_trend_break_score",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTrendBreakScorePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_trend_break_score.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_trend_break_score",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_trimmed_mean",
    canonical="ts_trimmed_mean",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTrimmedMeanPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_trimmed_mean.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_trimmed_mean",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_turning_intensity",
    canonical="ts_turning_intensity",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTurningIntensityPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_turning_intensity.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_turning_intensity",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_turning_rate",
    canonical="ts_turning_rate",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTurningRatePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_turning_rate.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_turning_rate",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_turnover_age_dispersion",
    canonical="ts_turnover_age_dispersion",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTurnoverAgeDispersionPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_turnover_age_dispersion.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_turnover_age_dispersion",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_turnover_cost_dispersion",
    canonical="ts_turnover_cost_dispersion",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTurnoverCostDispersionPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_turnover_cost_dispersion.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_turnover_cost_dispersion",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_turnover_cost_entropy",
    canonical="ts_turnover_cost_entropy",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTurnoverCostEntropyPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_turnover_cost_entropy.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_turnover_cost_entropy",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_turnover_cost_entropy_vol_scaled",
    canonical="ts_turnover_cost_entropy_vol_scaled",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTurnoverCostEntropyVolScaledPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_turnover_cost_entropy_vol_scaled.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_turnover_cost_entropy_vol_scaled",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_turnover_cost_mode_distance",
    canonical="ts_turnover_cost_mode_distance",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTurnoverCostModeDistancePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_turnover_cost_mode_distance.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_turnover_cost_mode_distance",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_turnover_cost_quantile_distance",
    canonical="ts_turnover_cost_quantile_distance",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTurnoverCostQuantileDistancePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_turnover_cost_quantile_distance.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_turnover_cost_quantile_distance",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_turnover_cost_skew",
    canonical="ts_turnover_cost_skew",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTurnoverCostSkewPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_turnover_cost_skew.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_turnover_cost_skew",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_turnover_holding_age",
    canonical="ts_turnover_holding_age",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTurnoverHoldingAgePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_turnover_holding_age.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_turnover_holding_age",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_turnover_near_cost_mass",
    canonical="ts_turnover_near_cost_mass",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTurnoverNearCostMassPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_turnover_near_cost_mass.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_turnover_near_cost_mass",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_turnover_old_mass",
    canonical="ts_turnover_old_mass",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTurnoverOldMassPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_turnover_old_mass.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_turnover_old_mass",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_turnover_profit_share",
    canonical="ts_turnover_profit_share",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTurnoverProfitSharePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_turnover_profit_share.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_turnover_profit_share",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_turnover_reference_price",
    canonical="ts_turnover_reference_price",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsTurnoverReferencePricePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_turnover_reference_price.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_turnover_reference_price",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_upper_partial_moment",
    canonical="ts_upper_partial_moment",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsUpperPartialMomentPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_upper_partial_moment.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_upper_partial_moment",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_upper_tail_coexceedance_probability",
    canonical="ts_upper_tail_coexceedance_probability",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsUpperTailCoexceedanceProbabilityPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_upper_tail_coexceedance_probability.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_upper_tail_coexceedance_probability",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_upside_deviation",
    canonical="ts_upside_deviation",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsUpsideDeviationPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_upside_deviation.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_upside_deviation",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_variance_ratio_proxy",
    canonical="ts_variance_ratio_proxy",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsVarianceRatioProxyPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_variance_ratio_proxy.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_variance_ratio_proxy",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_variogram_slope",
    canonical="ts_variogram_slope",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsVariogramSlopePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_variogram_slope.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_variogram_slope",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_vector_path_curvature",
    canonical="ts_vector_path_curvature",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsVectorPathCurvaturePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_vector_path_curvature.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_vector_path_curvature",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_vector_path_efficiency",
    canonical="ts_vector_path_efficiency",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsVectorPathEfficiencyPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_vector_path_efficiency.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_vector_path_efficiency",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_vector_self_intersection_rate",
    canonical="ts_vector_self_intersection_rate",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsVectorSelfIntersectionRatePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_vector_self_intersection_rate.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_vector_self_intersection_rate",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_vector_state_local_density",
    canonical="ts_vector_state_local_density",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsVectorStateLocalDensityPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_vector_state_local_density.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_vector_state_local_density",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_vector_state_mahalanobis",
    canonical="ts_vector_state_mahalanobis",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsVectorStateMahalanobisPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_vector_state_mahalanobis.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_vector_state_mahalanobis",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_vector_turning_coherence",
    canonical="ts_vector_turning_coherence",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsVectorTurningCoherencePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_vector_turning_coherence.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_vector_turning_coherence",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_vol_acceleration",
    canonical="ts_vol_acceleration",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsVolAccelerationPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_vol_acceleration.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_vol_acceleration",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_vol_clustering",
    canonical="ts_vol_clustering",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsVolClusteringPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_vol_clustering.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_vol_clustering",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_vol_of_vol",
    canonical="ts_vol_of_vol",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsVolOfVolPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_vol_of_vol.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_vol_of_vol",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_vol_pvariation_roughness",
    canonical="ts_vol_pvariation_roughness",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsVolPvariationRoughnessPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_vol_pvariation_roughness.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_vol_pvariation_roughness",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_vol_scaling_break",
    canonical="ts_vol_scaling_break",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsVolScalingBreakPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_vol_scaling_break.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_vol_scaling_break",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_vol_shift_score",
    canonical="ts_vol_shift_score",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsVolShiftScorePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_vol_shift_score.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_vol_shift_score",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_vol_term_structure",
    canonical="ts_vol_term_structure",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsVolTermStructurePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_vol_term_structure.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_vol_term_structure",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_wasserstein_shift",
    canonical="ts_wasserstein_shift",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsWassersteinShiftPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_wasserstein_shift.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_wasserstein_shift",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_wavelet_lowpass_reconstruct",
    canonical="ts_wavelet_lowpass_reconstruct",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsWaveletLowpassReconstructPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_wavelet_lowpass_reconstruct.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_wavelet_lowpass_reconstruct",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_wavelet_shrinkage_trailing",
    canonical="ts_wavelet_shrinkage_trailing",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsWaveletShrinkageTrailingPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_wavelet_shrinkage_trailing.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_wavelet_shrinkage_trailing",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_weighted_downside_deviation",
    canonical="ts_weighted_downside_deviation",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsWeightedDownsideDeviationPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_weighted_downside_deviation.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_weighted_downside_deviation",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_weighted_drawdown_area",
    canonical="ts_weighted_drawdown_area",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsWeightedDrawdownAreaPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_weighted_drawdown_area.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_weighted_drawdown_area",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_weighted_expected_shortfall",
    canonical="ts_weighted_expected_shortfall",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsWeightedExpectedShortfallPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_weighted_expected_shortfall.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_weighted_expected_shortfall",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_weighted_permutation_entropy",
    canonical="ts_weighted_permutation_entropy",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsWeightedPermutationEntropyPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_weighted_permutation_entropy.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_weighted_permutation_entropy",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_weighted_semivariance",
    canonical="ts_weighted_semivariance",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsWeightedSemivariancePolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_weighted_semivariance.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_weighted_semivariance",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_weighted_time_centroid",
    canonical="ts_weighted_time_centroid",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsWeightedTimeCentroidPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_weighted_time_centroid.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_weighted_time_centroid",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ts_zero_ratio",
    canonical="ts_zero_ratio",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TsZeroRatioPolars(SeriesOperator):
    """Auto-generated Polars bridge for ts_zero_ratio.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ts_zero_ratio",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="ttest_one_sample",
    canonical="ttest_one_sample",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TtestOneSamplePolars(SeriesOperator):
    """Auto-generated Polars bridge for ttest_one_sample.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ttest_one_sample",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ttest_paired",
    canonical="ttest_paired",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TtestPairedPolars(SeriesOperator):
    """Auto-generated Polars bridge for ttest_paired.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ttest_paired",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ttest_two_samples",
    canonical="ttest_two_samples",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TtestTwoSamplesPolars(SeriesOperator):
    """Auto-generated Polars bridge for ttest_two_samples.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ttest_two_samples",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="ttm",
    canonical="ttm",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TtmPolars(SeriesOperator):
    """Auto-generated Polars bridge for ttm.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="ttm",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="tukey_transform",
    canonical="tukey_transform",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TukeyTransformPolars(SeriesOperator):
    """Auto-generated Polars bridge for tukey_transform.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="tukey_transform",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="turnover_chip_age_cost_surface",
    canonical="turnover_chip_age_cost_surface",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TurnoverChipAgeCostSurfacePolars(SeriesOperator):
    """Auto-generated Polars bridge for turnover_chip_age_cost_surface.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="turnover_chip_age_cost_surface",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="turnover_chip_overhang_surface",
    canonical="turnover_chip_overhang_surface",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class TurnoverChipOverhangSurfacePolars(SeriesOperator):
    """Auto-generated Polars bridge for turnover_chip_overhang_surface.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="turnover_chip_overhang_surface",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# ELEMENT_WISE operators
# ------------------------------------------------------------------------------

@register_operator(
    name="unitize",
    canonical="unitize",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class UnitizePolars(SeriesOperator):
    """Auto-generated Polars bridge for unitize.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="unitize",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# COMPLEX operators
# ------------------------------------------------------------------------------

@register_operator(
    name="unwrap",
    canonical="unwrap",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class UnwrapPolars(SeriesOperator):
    """Auto-generated Polars bridge for unwrap.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="unwrap",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="van_der_waerden_transform",
    canonical="van_der_waerden_transform",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class VanDerWaerdenTransformPolars(SeriesOperator):
    """Auto-generated Polars bridge for van_der_waerden_transform.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="van_der_waerden_transform",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="varp",
    canonical="varp",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class VarpPolars(SeriesOperator):
    """Auto-generated Polars bridge for varp.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="varp",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="volatility",
    canonical="volatility",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class VolatilityPolars(SeriesOperator):
    """Auto-generated Polars bridge for volatility.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="volatility",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="vp_weighted_price",
    canonical="vp_weighted_price",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class VpWeightedPricePolars(SeriesOperator):
    """Auto-generated Polars bridge for vp_weighted_price.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="vp_weighted_price",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="vpmacd",
    canonical="vpmacd",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class VpmacdPolars(SeriesOperator):
    """Auto-generated Polars bridge for vpmacd.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="vpmacd",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="vpmacd_signal",
    canonical="vpmacd_signal",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class VpmacdSignalPolars(SeriesOperator):
    """Auto-generated Polars bridge for vpmacd_signal.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="vpmacd_signal",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="vwap",
    canonical="vwap",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class VwapPolars(SeriesOperator):
    """Auto-generated Polars bridge for vwap.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="vwap",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="vwap_to_close_return",
    canonical="vwap_to_close_return",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class VwapToCloseReturnPolars(SeriesOperator):
    """Auto-generated Polars bridge for vwap_to_close_return.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="vwap_to_close_return",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="wavelet",
    canonical="wavelet",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class WaveletPolars(SeriesOperator):
    """Auto-generated Polars bridge for wavelet.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="wavelet",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="wavelet_denoise",
    canonical="wavelet_denoise",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class WaveletDenoisePolars(SeriesOperator):
    """Auto-generated Polars bridge for wavelet_denoise.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="wavelet_denoise",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="wavg",
    canonical="wavg",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class WavgPolars(SeriesOperator):
    """Auto-generated Polars bridge for wavg.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="wavg",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="weighted_mean",
    canonical="weighted_mean",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class WeightedMeanPolars(SeriesOperator):
    """Auto-generated Polars bridge for weighted_mean.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="weighted_mean",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="where",
    canonical="where",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class WherePolars(SeriesOperator):
    """Auto-generated Polars bridge for where.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="where",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="window_max",
    canonical="window_max",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class WindowMaxPolars(SeriesOperator):
    """Auto-generated Polars bridge for window_max.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="window_max",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="window_mean",
    canonical="window_mean",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class WindowMeanPolars(SeriesOperator):
    """Auto-generated Polars bridge for window_mean.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="window_mean",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="window_min",
    canonical="window_min",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class WindowMinPolars(SeriesOperator):
    """Auto-generated Polars bridge for window_min.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="window_min",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="window_std",
    canonical="window_std",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class WindowStdPolars(SeriesOperator):
    """Auto-generated Polars bridge for window_std.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="window_std",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="window_sum",
    canonical="window_sum",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class WindowSumPolars(SeriesOperator):
    """Auto-generated Polars bridge for window_sum.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="window_sum",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="winsorize_mean",
    canonical="winsorize_mean",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class WinsorizeMeanPolars(SeriesOperator):
    """Auto-generated Polars bridge for winsorize_mean.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="winsorize_mean",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="wsum",
    canonical="wsum",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class WsumPolars(SeriesOperator):
    """Auto-generated Polars bridge for wsum.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="wsum",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="yoy",
    canonical="yoy",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class YoyPolars(SeriesOperator):
    """Auto-generated Polars bridge for yoy.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="yoy",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )

@register_operator(
    name="zscore",
    canonical="zscore",
    backend="polars",
    source="auto_generated.polars_bridges",
)
class ZscorePolars(SeriesOperator):
    """Auto-generated Polars bridge for zscore.

    Execution delegated to polars_registry_bridge (long-table map_groups).
    """
    metadata = OperatorMetadata(
        name="zscore",
        category="general",
        description="Auto-generated Polars bridge",
        tags=["auto_generated", "polars", "registry_bridge"],
    )


# 使用说明
# ========
#
# 这些算子注册为 polars backend 后，backend/polars_long_backend.py 会在
# 编译时自动调用 polars_registry_bridge.compile_registry_op 进行 fallback：
#
# 1. 递归编译子节点为 LazyFrame(ts, inst, _v)
# 2. 按 ts（截面）或 inst（时序）分组
# 3. 每组内构造 mini-panel，调用 pandas calculate()
# 4. 写回 _v 列
#
# 优点：
# - 零实现成本：无需编写任何 Polars-native 代码
# - 语义一致性：直接复用已验证的 pandas 实现
# - 覆盖率提升：从 3.3% → 86.0%
#
# 性能考虑：
# - map_groups 有序列化开销，性能不如原生 Polars expr
# - 对于热点算子，后续可逐个用 polars_expr_emitter 重写为原生
# - 对于非热点算子，registry bridge 是性价比最高的方案
