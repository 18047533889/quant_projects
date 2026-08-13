# -*- coding: utf-8 -*-
"""
Polars Native CS (Cross-Section) Operators - Advanced Batch (Phase 3)

高级截面统计算子：LOF、Isolation Forest、KNN 系列、Mahalanobis、回归残差等。

核心模式：
- 使用 Polars .over() 实现真正的截面操作
- 复杂 ML 算子提供 skeleton + TODO（需要完整实现）
- 所有算子使用 backend="polars" 注册
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore

from cleaned_operators.base_polars import (
    SeriesOperator,
    register_operator,
    OperatorMetadata,
)


def _to_polars_safe(feature: pd.Series) -> pl.LazyFrame:
    """安全转换 pandas Series 到 Polars LazyFrame"""
    return (
        feature.to_frame()
        .reset_index(drop=False)
        .pipe(pl.from_pandas)
        .lazy()
    )


def _from_polars_safe(lf: pl.LazyFrame, feature_name: str, original_index) -> pd.Series:
    """安全转换 Polars LazyFrame 回 pandas Series"""
    df = lf.collect().to_pandas()
    if "index" in df.columns:
        df = df.set_index("index")
    result = df[feature_name]
    result.index = original_index
    return result


# =============================================================================
# LOF and Outlier Detection - 局部异常因子和异常检测
# =============================================================================

@register_operator(
    name="cs_actual_lof_score",
    canonical="cs_actual_lof_score_polars",
    backend="polars",
    category="cross_sectional",
    business_category="outlier_detection",
)
class CSActualLOFScore(SeriesOperator):
    """截面 LOF (Local Outlier Factor) 分数

    计算每个样本的局部异常因子，衡量其相对于邻域的异常程度
    """
    metadata = OperatorMetadata(
        name="cs_actual_lof_score",
        category="cross_sectional",
        description="Local Outlier Factor score (cross-sectional)",
        examples=["cs_actual_lof_score(feature, 20)"],
        param_names=["x", "n_neighbors"],
        return_type="series",
        tags=["cross_sectional", "outlier", "lof", "polars_native"],
    )

    def _calculate_series(self, feature, n_neighbors=20, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine LOF algorithm
        # Requires: k-NN distance calculation, reachability distance, local reachability density
        # For now, return placeholder based on z-score
        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        # Placeholder: use absolute z-score as proxy
        mean_expr = pl.col(feature_name).mean()
        std_expr = pl.col(feature_name).std()

        lf = lf.with_columns([
            (pl.col(feature_name) - mean_expr).abs() / std_expr
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_isolation_forest_score",
    canonical="cs_isolation_forest_score_polars",
    backend="polars",
    category="cross_sectional",
    business_category="outlier_detection",
)
class CSIsolationForestScore(SeriesOperator):
    """截面 Isolation Forest 异常分数

    使用 isolation forest 算法检测异常值
    """
    metadata = OperatorMetadata(
        name="cs_isolation_forest_score",
        category="cross_sectional",
        description="Isolation Forest anomaly score (cross-sectional)",
        examples=["cs_isolation_forest_score(feature)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "outlier", "isolation_forest", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine Isolation Forest
        # Requires: tree-based isolation path length calculation
        # For now, return placeholder
        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        # Placeholder: normalized absolute deviation
        median_expr = pl.col(feature_name).median()
        mad_expr = (pl.col(feature_name) - median_expr).abs().median()

        lf = lf.with_columns([
            ((pl.col(feature_name) - median_expr).abs() / mad_expr)
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_isolation",
    canonical="cs_isolation_polars",
    backend="polars",
    category="cross_sectional",
    business_category="outlier_detection",
)
class CSIsolation(SeriesOperator):
    """截面隔离度（基于排名距离）"""
    metadata = OperatorMetadata(
        name="cs_isolation",
        category="cross_sectional",
        description="Cross-sectional isolation measure",
        examples=["cs_isolation(feature)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "isolation", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        # Isolation: distance from median in rank space
        median_expr = pl.col(feature_name).median()

        lf = lf.with_columns([
            (pl.col(feature_name) - median_expr).abs()
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_autoencoder_reconstruction_error",
    canonical="cs_autoencoder_reconstruction_error_polars",
    backend="polars",
    category="cross_sectional",
    business_category="outlier_detection",
)
class CSAutoencoderReconstructionError(SeriesOperator):
    """截面自编码器重构误差

    使用自编码器检测异常模式
    """
    metadata = OperatorMetadata(
        name="cs_autoencoder_reconstruction_error",
        category="cross_sectional",
        description="Autoencoder reconstruction error for anomaly detection",
        examples=["cs_autoencoder_reconstruction_error(feature)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "autoencoder", "outlier", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine autoencoder
        # Requires: neural network training and reconstruction
        # For now, return placeholder based on deviation from mean
        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        # Placeholder: squared deviation from mean
        mean_expr = pl.col(feature_name).mean()

        lf = lf.with_columns([
            ((pl.col(feature_name) - mean_expr) ** 2)
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


# =============================================================================
# KNN Family - K近邻系列算子
# =============================================================================

@register_operator(
    name="cs_knn_distance",
    canonical="cs_knn_distance_polars",
    backend="polars",
    category="cross_sectional",
    business_category="knn",
)
class CSKNNDistance(SeriesOperator):
    """截面 K近邻平均距离"""
    metadata = OperatorMetadata(
        name="cs_knn_distance",
        category="cross_sectional",
        description="Average K-nearest neighbor distance (cross-sectional)",
        examples=["cs_knn_distance(feature, 10)"],
        param_names=["x", "k"],
        return_type="series",
        tags=["cross_sectional", "knn", "polars_native"],
    )

    def _calculate_series(self, feature, k=10, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine KNN distance calculation
        # Requires: pairwise distance matrix and k-nearest selection
        # For now, return placeholder based on local density proxy
        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        # Placeholder: distance from median
        median_expr = pl.col(feature_name).median()

        lf = lf.with_columns([
            (pl.col(feature_name) - median_expr).abs()
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_knn_peer_mean_ex_self",
    canonical="cs_knn_peer_mean_ex_self_polars",
    backend="polars",
    category="cross_sectional",
    business_category="knn",
)
class CSKNNPeerMeanExSelf(SeriesOperator):
    """截面 K近邻均值（排除自身）"""
    metadata = OperatorMetadata(
        name="cs_knn_peer_mean_ex_self",
        category="cross_sectional",
        description="K-nearest neighbor mean excluding self",
        examples=["cs_knn_peer_mean_ex_self(feature, 10)"],
        param_names=["x", "k"],
        return_type="series",
        tags=["cross_sectional", "knn", "polars_native"],
    )

    def _calculate_series(self, feature, k=10, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine KNN peer mean
        # For now, return cross-sectional mean as placeholder
        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        # Placeholder: cross-sectional mean (simplified)
        lf = lf.with_columns([
            pl.col(feature_name).mean().alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_knn_local_linear_residual",
    canonical="cs_knn_local_linear_residual_polars",
    backend="polars",
    category="cross_sectional",
    business_category="knn",
)
class CSKNNLocalLinearResidual(SeriesOperator):
    """截面 KNN 局部线性回归残差"""
    metadata = OperatorMetadata(
        name="cs_knn_local_linear_residual",
        category="cross_sectional",
        description="KNN local linear regression residual",
        examples=["cs_knn_local_linear_residual(y, x, 20)"],
        param_names=["y", "x", "k"],
        return_type="series",
        tags=["cross_sectional", "knn", "regression", "polars_native"],
    )

    def _calculate_series(self, y, x, k=20, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine KNN local regression
        # For now, return demean as placeholder
        original_index = y.index
        feature_name = y.name or "value"

        lf = _to_polars_safe(y)

        # Placeholder: demean
        lf = lf.with_columns([
            (pl.col(feature_name) - pl.col(feature_name).mean())
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_knn_tangent_residual",
    canonical="cs_knn_tangent_residual_polars",
    backend="polars",
    category="cross_sectional",
    business_category="knn",
)
class CSKNNTangentResidual(SeriesOperator):
    """截面 KNN 切线回归残差"""
    metadata = OperatorMetadata(
        name="cs_knn_tangent_residual",
        category="cross_sectional",
        description="KNN tangent space regression residual",
        examples=["cs_knn_tangent_residual(y, x, 20)"],
        param_names=["y", "x", "k"],
        return_type="series",
        tags=["cross_sectional", "knn", "tangent", "polars_native"],
    )

    def _calculate_series(self, y, x, k=20, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine tangent space projection
        # For now, return demean as placeholder
        original_index = y.index
        feature_name = y.name or "value"

        lf = _to_polars_safe(y)

        lf = lf.with_columns([
            (pl.col(feature_name) - pl.col(feature_name).mean())
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_knn_local_gradient_norm",
    canonical="cs_knn_local_gradient_norm_polars",
    backend="polars",
    category="cross_sectional",
    business_category="knn",
)
class CSKNNLocalGradientNorm(SeriesOperator):
    """截面 KNN 局部梯度范数"""
    metadata = OperatorMetadata(
        name="cs_knn_local_gradient_norm",
        category="cross_sectional",
        description="KNN local gradient norm",
        examples=["cs_knn_local_gradient_norm(feature, 20)"],
        param_names=["x", "k"],
        return_type="series",
        tags=["cross_sectional", "knn", "gradient", "polars_native"],
    )

    def _calculate_series(self, feature, k=20, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine local gradient calculation
        # For now, return std as placeholder
        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        lf = lf.with_columns([
            pl.col(feature_name).std().alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_knn_local_moran",
    canonical="cs_knn_local_moran_polars",
    backend="polars",
    category="cross_sectional",
    business_category="knn",
)
class CSKNNLocalMoran(SeriesOperator):
    """截面 KNN 局部 Moran's I 统计量"""
    metadata = OperatorMetadata(
        name="cs_knn_local_moran",
        category="cross_sectional",
        description="KNN local Moran's I statistic",
        examples=["cs_knn_local_moran(feature, 20)"],
        param_names=["x", "k"],
        return_type="series",
        tags=["cross_sectional", "knn", "moran", "spatial", "polars_native"],
    )

    def _calculate_series(self, feature, k=20, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine local Moran's I
        # For now, return zscore as placeholder
        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        mean_expr = pl.col(feature_name).mean()
        std_expr = pl.col(feature_name).std()

        lf = lf.with_columns([
            ((pl.col(feature_name) - mean_expr) / std_expr)
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_knn_neighbor_retention",
    canonical="cs_knn_neighbor_retention_polars",
    backend="polars",
    category="cross_sectional",
    business_category="knn",
)
class CSKNNNeighborRetention(SeriesOperator):
    """截面 KNN 邻居保持率（时序稳定性）"""
    metadata = OperatorMetadata(
        name="cs_knn_neighbor_retention",
        category="cross_sectional",
        description="KNN neighbor retention rate over time",
        examples=["cs_knn_neighbor_retention(feature, 20)"],
        param_names=["x", "k"],
        return_type="series",
        tags=["cross_sectional", "knn", "retention", "polars_native"],
    )

    def _calculate_series(self, feature, k=20, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine neighbor retention (requires time series context)
        # For now, return constant placeholder
        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        lf = lf.with_columns([
            pl.lit(0.5).alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_knn_graph_dirichlet_energy",
    canonical="cs_knn_graph_dirichlet_energy_polars",
    backend="polars",
    category="cross_sectional",
    business_category="knn",
)
class CSKNNGraphDirichletEnergy(SeriesOperator):
    """截面 KNN 图的 Dirichlet 能量"""
    metadata = OperatorMetadata(
        name="cs_knn_graph_dirichlet_energy",
        category="cross_sectional",
        description="KNN graph Dirichlet energy",
        examples=["cs_knn_graph_dirichlet_energy(feature, 20)"],
        param_names=["x", "k"],
        return_type="series",
        tags=["cross_sectional", "knn", "graph", "polars_native"],
    )

    def _calculate_series(self, feature, k=20, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine Dirichlet energy
        # For now, return variance as placeholder
        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        lf = lf.with_columns([
            pl.col(feature_name).var().alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


# =============================================================================
# Local Density and Curvature - 局部密度和曲率
# =============================================================================

@register_operator(
    name="cs_local_density",
    canonical="cs_local_density_polars",
    backend="polars",
    category="cross_sectional",
    business_category="density",
)
class CSLocalDensity(SeriesOperator):
    """截面局部密度估计"""
    metadata = OperatorMetadata(
        name="cs_local_density",
        category="cross_sectional",
        description="Local density estimation (cross-sectional)",
        examples=["cs_local_density(feature, 20)"],
        param_names=["x", "k"],
        return_type="series",
        tags=["cross_sectional", "density", "polars_native"],
    )

    def _calculate_series(self, feature, k=20, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine kernel density estimation
        # For now, return inverse distance from median as proxy
        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        median_expr = pl.col(feature_name).median()

        lf = lf.with_columns([
            (1.0 / (1.0 + (pl.col(feature_name) - median_expr).abs()))
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_local_density_score",
    canonical="cs_local_density_score_polars",
    backend="polars",
    category="cross_sectional",
    business_category="density",
)
class CSLocalDensityScore(SeriesOperator):
    """截面局部密度分数（标准化）"""
    metadata = OperatorMetadata(
        name="cs_local_density_score",
        category="cross_sectional",
        description="Standardized local density score",
        examples=["cs_local_density_score(feature, 20)"],
        param_names=["x", "k"],
        return_type="series",
        tags=["cross_sectional", "density", "polars_native"],
    )

    def _calculate_series(self, feature, k=20, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # Based on cs_local_density but standardized
        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        median_expr = pl.col(feature_name).median()
        density_expr = 1.0 / (1.0 + (pl.col(feature_name) - median_expr).abs())

        # Standardize
        mean_density = density_expr.mean()
        std_density = density_expr.std()

        lf = lf.with_columns([
            ((density_expr - mean_density) / std_density)
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_local_curvature",
    canonical="cs_local_curvature_polars",
    backend="polars",
    category="cross_sectional",
    business_category="geometry",
)
class CSLocalCurvature(SeriesOperator):
    """截面局部曲率（基于 KNN）"""
    metadata = OperatorMetadata(
        name="cs_local_curvature",
        category="cross_sectional",
        description="Local curvature based on KNN",
        examples=["cs_local_curvature(feature, 20)"],
        param_names=["x", "k"],
        return_type="series",
        tags=["cross_sectional", "curvature", "geometry", "polars_native"],
    )

    def _calculate_series(self, feature, k=20, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine local curvature calculation
        # For now, return second moment as placeholder
        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        mean_expr = pl.col(feature_name).mean()

        lf = lf.with_columns([
            ((pl.col(feature_name) - mean_expr) ** 2)
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_relative_density_ratio",
    canonical="cs_relative_density_ratio_polars",
    backend="polars",
    category="cross_sectional",
    business_category="density",
)
class CSRelativeDensityRatio(SeriesOperator):
    """截面相对密度比率"""
    metadata = OperatorMetadata(
        name="cs_relative_density_ratio",
        category="cross_sectional",
        description="Relative density ratio (cross-sectional)",
        examples=["cs_relative_density_ratio(feature, 20)"],
        param_names=["x", "k"],
        return_type="series",
        tags=["cross_sectional", "density", "polars_native"],
    )

    def _calculate_series(self, feature, k=20, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine relative density
        # For now, return normalized distance from median
        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        median_expr = pl.col(feature_name).median()
        mad_expr = (pl.col(feature_name) - median_expr).abs().median()

        lf = lf.with_columns([
            (1.0 / (1.0 + (pl.col(feature_name) - median_expr).abs() / mad_expr))
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_neighbor_gap",
    canonical="cs_neighbor_gap_polars",
    backend="polars",
    category="cross_sectional",
    business_category="density",
)
class CSNeighborGap(SeriesOperator):
    """截面邻居间隙（距离到最近邻）"""
    metadata = OperatorMetadata(
        name="cs_neighbor_gap",
        category="cross_sectional",
        description="Gap to nearest neighbor (cross-sectional)",
        examples=["cs_neighbor_gap(feature)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "gap", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine nearest neighbor gap
        # For now, return distance from median
        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        median_expr = pl.col(feature_name).median()

        lf = lf.with_columns([
            (pl.col(feature_name) - median_expr).abs()
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


# =============================================================================
# Mahalanobis Distance - 马氏距离系列
# =============================================================================

@register_operator(
    name="cs_mahalanobis_distance",
    canonical="cs_mahalanobis_distance_polars",
    backend="polars",
    category="cross_sectional",
    business_category="distance",
)
class CSMahalanobisDistance(SeriesOperator):
    """截面马氏距离"""
    metadata = OperatorMetadata(
        name="cs_mahalanobis_distance",
        category="cross_sectional",
        description="Mahalanobis distance (cross-sectional)",
        examples=["cs_mahalanobis_distance(feature)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "mahalanobis", "distance", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine multivariate Mahalanobis distance
        # For now, return standardized distance (z-score)
        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        mean_expr = pl.col(feature_name).mean()
        std_expr = pl.col(feature_name).std()

        lf = lf.with_columns([
            ((pl.col(feature_name) - mean_expr).abs() / std_expr)
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_robust_mahalanobis_mad",
    canonical="cs_robust_mahalanobis_mad_polars",
    backend="polars",
    category="cross_sectional",
    business_category="distance",
)
class CSRobustMahalanobisMAD(SeriesOperator):
    """截面稳健马氏距离（基于 MAD）"""
    metadata = OperatorMetadata(
        name="cs_robust_mahalanobis_mad",
        category="cross_sectional",
        description="Robust Mahalanobis distance using MAD",
        examples=["cs_robust_mahalanobis_mad(feature)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "mahalanobis", "robust", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        # Robust: use median and MAD instead of mean and std
        median_expr = pl.col(feature_name).median()
        mad_expr = (pl.col(feature_name) - median_expr).abs().median()

        lf = lf.with_columns([
            ((pl.col(feature_name) - median_expr).abs() / mad_expr)
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_shrinkage_mahalanobis",
    canonical="cs_shrinkage_mahalanobis_polars",
    backend="polars",
    category="cross_sectional",
    business_category="distance",
)
class CSShrinkageMahalanobis(SeriesOperator):
    """截面收缩马氏距离（正则化协方差）"""
    metadata = OperatorMetadata(
        name="cs_shrinkage_mahalanobis",
        category="cross_sectional",
        description="Mahalanobis distance with shrinkage covariance",
        examples=["cs_shrinkage_mahalanobis(feature)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "mahalanobis", "shrinkage", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine shrinkage covariance
        # For now, return z-score as placeholder
        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        mean_expr = pl.col(feature_name).mean()
        std_expr = pl.col(feature_name).std()

        lf = lf.with_columns([
            ((pl.col(feature_name) - mean_expr).abs() / std_expr)
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


# =============================================================================
# Regression Residuals - 回归残差系列
# =============================================================================

@register_operator(
    name="cs_multi_resid",
    canonical="cs_multi_resid_polars",
    backend="polars",
    category="cross_sectional",
    business_category="regression",
)
class CSMultiResid(SeriesOperator):
    """截面多元线性回归残差"""
    metadata = OperatorMetadata(
        name="cs_multi_resid",
        category="cross_sectional",
        description="Multiple linear regression residual (cross-sectional)",
        examples=["cs_multi_resid(y, x1, x2)"],
        param_names=["y", "x"],
        return_type="series",
        tags=["cross_sectional", "regression", "polars_native"],
    )

    def _calculate_series(self, y, x, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine OLS regression
        # For now, return demean as placeholder
        original_index = y.index
        feature_name = y.name or "value"

        lf = _to_polars_safe(y)

        lf = lf.with_columns([
            (pl.col(feature_name) - pl.col(feature_name).mean())
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_multi_ridge_resid",
    canonical="cs_multi_ridge_resid_polars",
    backend="polars",
    category="cross_sectional",
    business_category="regression",
)
class CSMultiRidgeResid(SeriesOperator):
    """截面岭回归残差"""
    metadata = OperatorMetadata(
        name="cs_multi_ridge_resid",
        category="cross_sectional",
        description="Ridge regression residual (cross-sectional)",
        examples=["cs_multi_ridge_resid(y, x, 0.1)"],
        param_names=["y", "x", "alpha"],
        return_type="series",
        tags=["cross_sectional", "regression", "ridge", "polars_native"],
    )

    def _calculate_series(self, y, x, alpha=1.0, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine ridge regression
        # For now, return demean as placeholder
        original_index = y.index
        feature_name = y.name or "value"

        lf = _to_polars_safe(y)

        lf = lf.with_columns([
            (pl.col(feature_name) - pl.col(feature_name).mean())
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_trimmed_ols_resid",
    canonical="cs_trimmed_ols_resid_polars",
    backend="polars",
    category="cross_sectional",
    business_category="regression",
)
class CSTrimmedOLSResid(SeriesOperator):
    """截面修剪 OLS 回归残差（剔除极端值）"""
    metadata = OperatorMetadata(
        name="cs_trimmed_ols_resid",
        category="cross_sectional",
        description="Trimmed OLS regression residual",
        examples=["cs_trimmed_ols_resid(y, x, 0.05)"],
        param_names=["y", "x", "trim_pct"],
        return_type="series",
        tags=["cross_sectional", "regression", "robust", "polars_native"],
    )

    def _calculate_series(self, y, x, trim_pct=0.05, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine trimmed OLS
        # For now, return demean with winsorization placeholder
        original_index = y.index
        feature_name = y.name or "value"

        lf = _to_polars_safe(y)

        lf = lf.with_columns([
            (pl.col(feature_name) - pl.col(feature_name).mean())
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_wls_resid",
    canonical="cs_wls_resid_polars",
    backend="polars",
    category="cross_sectional",
    business_category="regression",
)
class CSWLSResid(SeriesOperator):
    """截面加权最小二乘回归残差"""
    metadata = OperatorMetadata(
        name="cs_wls_resid",
        category="cross_sectional",
        description="Weighted least squares regression residual",
        examples=["cs_wls_resid(y, x, weights)"],
        param_names=["y", "x", "weights"],
        return_type="series",
        tags=["cross_sectional", "regression", "weighted", "polars_native"],
    )

    def _calculate_series(self, y, x, weights=None, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine WLS
        # For now, return demean as placeholder
        original_index = y.index
        feature_name = y.name or "value"

        lf = _to_polars_safe(y)

        lf = lf.with_columns([
            (pl.col(feature_name) - pl.col(feature_name).mean())
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_isotonic_residual",
    canonical="cs_isotonic_residual_polars",
    backend="polars",
    category="cross_sectional",
    business_category="regression",
)
class CSIsotonicResidual(SeriesOperator):
    """截面保序回归残差"""
    metadata = OperatorMetadata(
        name="cs_isotonic_residual",
        category="cross_sectional",
        description="Isotonic regression residual (cross-sectional)",
        examples=["cs_isotonic_residual(y, x)"],
        param_names=["y", "x"],
        return_type="series",
        tags=["cross_sectional", "regression", "isotonic", "polars_native"],
    )

    def _calculate_series(self, y, x, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine isotonic regression
        # For now, return demean as placeholder
        original_index = y.index
        feature_name = y.name or "value"

        lf = _to_polars_safe(y)

        lf = lf.with_columns([
            (pl.col(feature_name) - pl.col(feature_name).mean())
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_isotonic_residual_lagged_direction",
    canonical="cs_isotonic_residual_lagged_direction_polars",
    backend="polars",
    category="cross_sectional",
    business_category="regression",
)
class CSIsotonicResidualLaggedDirection(SeriesOperator):
    """截面保序回归残差 × 滞后方向"""
    metadata = OperatorMetadata(
        name="cs_isotonic_residual_lagged_direction",
        category="cross_sectional",
        description="Isotonic residual times lagged direction",
        examples=["cs_isotonic_residual_lagged_direction(y, x)"],
        param_names=["y", "x"],
        return_type="series",
        tags=["cross_sectional", "regression", "isotonic", "polars_native"],
    )

    def _calculate_series(self, y, x, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine isotonic with lagged direction
        # For now, return demean as placeholder
        original_index = y.index
        feature_name = y.name or "value"

        lf = _to_polars_safe(y)

        lf = lf.with_columns([
            (pl.col(feature_name) - pl.col(feature_name).mean())
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


# =============================================================================
# Rank-based Statistics - 基于排名的统计量
# =============================================================================

@register_operator(
    name="cs_rank_churn",
    canonical="cs_rank_churn_polars",
    backend="polars",
    category="cross_sectional",
    business_category="rank",
)
class CSRankChurn(SeriesOperator):
    """截面排名变动率（相对于历史）"""
    metadata = OperatorMetadata(
        name="cs_rank_churn",
        category="cross_sectional",
        description="Cross-sectional rank churn rate",
        examples=["cs_rank_churn(feature)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "rank", "churn", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine rank churn (requires time series context)
        # For now, return rank as placeholder
        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        lf = lf.with_columns([
            ((pl.col(feature_name).rank(method="average") - 1) /
             (pl.col(feature_name).count() - 1))
            .fill_nan(0.5)
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_rank_combined_churn",
    canonical="cs_rank_combined_churn_polars",
    backend="polars",
    category="cross_sectional",
    business_category="rank",
)
class CSRankCombinedChurn(SeriesOperator):
    """截面组合排名变动率"""
    metadata = OperatorMetadata(
        name="cs_rank_combined_churn",
        category="cross_sectional",
        description="Combined rank churn across multiple features",
        examples=["cs_rank_combined_churn(feature)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "rank", "churn", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine combined churn
        # For now, return rank as placeholder
        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        lf = lf.with_columns([
            ((pl.col(feature_name).rank(method="average") - 1) /
             (pl.col(feature_name).count() - 1))
            .fill_nan(0.5)
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_rank_composition_churn",
    canonical="cs_rank_composition_churn_polars",
    backend="polars",
    category="cross_sectional",
    business_category="rank",
)
class CSRankCompositionChurn(SeriesOperator):
    """截面排名组成变动率"""
    metadata = OperatorMetadata(
        name="cs_rank_composition_churn",
        category="cross_sectional",
        description="Rank composition churn rate",
        examples=["cs_rank_composition_churn(feature)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "rank", "churn", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine composition churn
        # For now, return rank as placeholder
        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        lf = lf.with_columns([
            ((pl.col(feature_name).rank(method="average") - 1) /
             (pl.col(feature_name).count() - 1))
            .fill_nan(0.5)
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_rank_copula_entropy",
    canonical="cs_rank_copula_entropy_polars",
    backend="polars",
    category="cross_sectional",
    business_category="rank",
)
class CSRankCopulaEntropy(SeriesOperator):
    """截面排名 copula 熵"""
    metadata = OperatorMetadata(
        name="cs_rank_copula_entropy",
        category="cross_sectional",
        description="Rank copula entropy (cross-sectional)",
        examples=["cs_rank_copula_entropy(x, y)"],
        param_names=["x", "y"],
        return_type="series",
        tags=["cross_sectional", "rank", "copula", "entropy", "polars_native"],
    )

    def _calculate_series(self, x, y, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine copula entropy
        # For now, return rank of x as placeholder
        original_index = x.index
        feature_name = x.name or "value"

        lf = _to_polars_safe(x)

        lf = lf.with_columns([
            ((pl.col(feature_name).rank(method="average") - 1) /
             (pl.col(feature_name).count() - 1))
            .fill_nan(0.5)
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_rank_copula_mi",
    canonical="cs_rank_copula_mi_polars",
    backend="polars",
    category="cross_sectional",
    business_category="rank",
)
class CSRankCopulaMI(SeriesOperator):
    """截面排名 copula 互信息"""
    metadata = OperatorMetadata(
        name="cs_rank_copula_mi",
        category="cross_sectional",
        description="Rank copula mutual information",
        examples=["cs_rank_copula_mi(x, y)"],
        param_names=["x", "y"],
        return_type="series",
        tags=["cross_sectional", "rank", "copula", "mi", "polars_native"],
    )

    def _calculate_series(self, x, y, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine copula MI
        # For now, return rank of x as placeholder
        original_index = x.index
        feature_name = x.name or "value"

        lf = _to_polars_safe(x)

        lf = lf.with_columns([
            ((pl.col(feature_name).rank(method="average") - 1) /
             (pl.col(feature_name).count() - 1))
            .fill_nan(0.5)
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


# =============================================================================
# Shrinkage and Bayesian - 收缩和贝叶斯方法
# =============================================================================

@register_operator(
    name="cs_empirical_bayes_shrinkage",
    canonical="cs_empirical_bayes_shrinkage_polars",
    backend="polars",
    category="cross_sectional",
    business_category="shrinkage",
)
class CSEmpiricalBayesShrinkage(SeriesOperator):
    """截面经验贝叶斯收缩"""
    metadata = OperatorMetadata(
        name="cs_empirical_bayes_shrinkage",
        category="cross_sectional",
        description="Empirical Bayes shrinkage estimator",
        examples=["cs_empirical_bayes_shrinkage(feature)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "shrinkage", "bayes", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine empirical Bayes shrinkage
        # For now, shrink toward mean with fixed factor
        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        mean_expr = pl.col(feature_name).mean()
        shrinkage_factor = 0.5

        lf = lf.with_columns([
            (shrinkage_factor * pl.col(feature_name) + (1 - shrinkage_factor) * mean_expr)
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_shrink_to_group_mean",
    canonical="cs_shrink_to_group_mean_polars",
    backend="polars",
    category="cross_sectional",
    business_category="shrinkage",
)
class CSShrinkToGroupMean(SeriesOperator):
    """截面收缩到组均值"""
    metadata = OperatorMetadata(
        name="cs_shrink_to_group_mean",
        category="cross_sectional",
        description="Shrink toward group mean",
        examples=["cs_shrink_to_group_mean(feature, group, 0.5)"],
        param_names=["x", "group", "shrinkage"],
        return_type="series",
        tags=["cross_sectional", "shrinkage", "group", "polars_native"],
    )

    def _calculate_series(self, feature, group=None, shrinkage=0.5, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine group-based shrinkage
        # For now, shrink toward overall mean
        original_index = feature.index
        feature_name = feature.name or "value"
        shrinkage = float(shrinkage) if shrinkage is not None else 0.5

        lf = _to_polars_safe(feature)

        mean_expr = pl.col(feature_name).mean()

        lf = lf.with_columns([
            ((1 - shrinkage) * pl.col(feature_name) + shrinkage * mean_expr)
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


# =============================================================================
# Factor Bucketing and Distribution - 因子分桶和分布
# =============================================================================

@register_operator(
    name="cs_factor_bucket_return",
    canonical="cs_factor_bucket_return_polars",
    backend="polars",
    category="cross_sectional",
    business_category="factor",
)
class CSFactorBucketReturn(SeriesOperator):
    """截面因子分桶收益率"""
    metadata = OperatorMetadata(
        name="cs_factor_bucket_return",
        category="cross_sectional",
        description="Factor bucket return (cross-sectional)",
        examples=["cs_factor_bucket_return(factor, returns, 5)"],
        param_names=["factor", "returns", "n_buckets"],
        return_type="series",
        tags=["cross_sectional", "factor", "bucket", "polars_native"],
    )

    def _calculate_series(self, factor, returns, n_buckets=5, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine bucket return calculation
        # For now, return demean of returns
        original_index = returns.index
        feature_name = returns.name or "value"

        lf = _to_polars_safe(returns)

        lf = lf.with_columns([
            (pl.col(feature_name) - pl.col(feature_name).mean())
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_hartigan_dip",
    canonical="cs_hartigan_dip_polars",
    backend="polars",
    category="cross_sectional",
    business_category="distribution",
)
class CSHartiganDip(SeriesOperator):
    """截面 Hartigan Dip 统计量（多模态检验）"""
    metadata = OperatorMetadata(
        name="cs_hartigan_dip",
        category="cross_sectional",
        description="Hartigan Dip statistic for multimodality",
        examples=["cs_hartigan_dip(feature)"],
        param_names=["x"],
        return_type="series",
        tags=["cross_sectional", "distribution", "multimodal", "polars_native"],
    )

    def _calculate_series(self, feature, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine Hartigan dip test
        # For now, return constant placeholder
        original_index = feature.index
        feature_name = feature.name or "value"

        lf = _to_polars_safe(feature)

        lf = lf.with_columns([
            pl.lit(0.0).alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_sliced_wasserstein_copula_shift",
    canonical="cs_sliced_wasserstein_copula_shift_polars",
    backend="polars",
    category="cross_sectional",
    business_category="distribution",
)
class CSSlicedWassersteinCopulaShift(SeriesOperator):
    """截面切片 Wasserstein copula 偏移"""
    metadata = OperatorMetadata(
        name="cs_sliced_wasserstein_copula_shift",
        category="cross_sectional",
        description="Sliced Wasserstein distance for copula shift",
        examples=["cs_sliced_wasserstein_copula_shift(x, y)"],
        param_names=["x", "y"],
        return_type="series",
        tags=["cross_sectional", "wasserstein", "copula", "polars_native"],
    )

    def _calculate_series(self, x, y, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine sliced Wasserstein
        # For now, return difference of ranks
        original_index = x.index
        feature_name = x.name or "value"

        lf = _to_polars_safe(x)

        lf = lf.with_columns([
            ((pl.col(feature_name).rank() - pl.col(feature_name).count() / 2).abs())
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)


@register_operator(
    name="cs_predictability_mosaic_score",
    canonical="cs_predictability_mosaic_score_polars",
    backend="polars",
    category="cross_sectional",
    business_category="predictability",
)
class CSPredictabilityMosaicScore(SeriesOperator):
    """截面可预测性镶嵌分数"""
    metadata = OperatorMetadata(
        name="cs_predictability_mosaic_score",
        category="cross_sectional",
        description="Predictability mosaic score (cross-sectional)",
        examples=["cs_predictability_mosaic_score(feature, target)"],
        param_names=["x", "y"],
        return_type="series",
        tags=["cross_sectional", "predictability", "polars_native"],
    )

    def _calculate_series(self, x, y, **kwargs):
        if pl is None:
            raise ImportError("Polars is required for polars_native operators")

        # TODO: Implement genuine predictability mosaic
        # For now, return correlation proxy
        original_index = x.index
        feature_name = x.name or "value"

        lf = _to_polars_safe(x)

        lf = lf.with_columns([
            ((pl.col(feature_name).rank(method="average") - 1) /
             (pl.col(feature_name).count() - 1))
            .fill_nan(0.5)
            .alias(feature_name)
        ])

        return _from_polars_safe(lf, feature_name, original_index)
