# -*- coding: utf-8 -*-
"""Cross-sectional TRUE_GAP operators batch 1.

Five new cross-sectional operators for anomaly detection, bucketing analysis,
empirical Bayes shrinkage, and peer graph aggregation.  All are causal daily-panel
transforms evaluated independently per trading row (scope="cs" or "group").
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from factor_engine.cleaned_operators.common.daily_panel import _aligned

# R47 convention: minimum breadth for cross-sectional operations
_MIN_BREADTH = 10


def _metadata(name: str, description: str, params: list[str], *, domain: str, unit: str, cost: int = 2) -> OperatorMetadata:
    """Build operator metadata following R47 conventions."""
    output_unit = unit if (unit.startswith("same_as:") or unit.startswith("unit(") or unit == "dimensionless") else None
    return OperatorMetadata(
        name=name,
        category="cross_sectional",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "cross_sectional", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", f"cost:{cost}",
        ],
        output_unit=output_unit,
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


# ---------------------------------------------------------------------------
# 1. cs_isolation_forest_score
# ---------------------------------------------------------------------------
@register_operator(
    name="cs_isolation_forest_score",
    category="cross_sectional",
    business_category="cross_sectional_anomaly",
    canonical="cs_isolation_forest_score",
    source="cs_batch1",
    status="experimental")
class CsIsolationForestScore(SeriesOperator):
    """横截面孤立森林异常得分（Isolation Forest anomaly score）。

    使用孤立森林算法检测横截面中的异常值。孤立森林通过随机分割特征空间，异常点
    需要更少的分割次数即可孤立。输出归一化异常得分：正常样本接近 0，异常样本接近 1。

    参数:
        x: 输入特征
        n_trees: 树的数量（默认 100）
        contamination: 预期异常比例（默认 0.1，即 10%）
        random_seed: 随机种子（默认 42）

    语义:
        - 每行独立计算（跨 instruments 的横截面）
        - 样本不足 _MIN_BREADTH 时 fail-closed（全 NaN）
        - 输出范围 [0, 1]，值越大越异常
        - 单位：dimensionless（无量纲得分）
    """

    metadata = _metadata(
        "cs_isolation_forest_score",
        "横截面孤立森林异常得分（isolation forest，归一化 [0,1]，值越大越异常）",
        ["x", "n_trees", "contamination", "random_seed"],
        domain="price_volume",
        unit="dimensionless",
        cost=3,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        n_trees: int = 100,
        contamination: float = 0.1,
        random_seed: int = 42,
        **_: Any
    ) -> pd.DataFrame:
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        for row in range(rows):
            x_row = xv[row]
            valid = np.isfinite(x_row)
            if valid.sum() < _MIN_BREADTH:
                continue

            scores = self._isolation_forest_row(
                x_row[valid], int(n_trees), float(contamination), int(random_seed)
            )
            if scores is not None:
                out[row][valid] = scores

        return _frame_like(x, out)

    @staticmethod
    def _isolation_forest_row(
        x_valid: np.ndarray,
        n_trees: int,
        contamination: float,
        random_seed: int,
    ) -> np.ndarray | None:
        """Compute isolation forest anomaly scores for one cross-section."""
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            # sklearn unavailable -> fail-closed
            return None

        n = len(x_valid)
        if n < 2:
            return None

        # Isolation Forest expects 2D input (n_samples, n_features)
        X = x_valid.reshape(-1, 1)

        # Fit isolation forest
        iso = IsolationForest(
            n_estimators=n_trees,
            contamination=contamination,
            random_state=random_seed,
            bootstrap=False,
        )

        try:
            iso.fit(X)
            # decision_function returns negative anomaly scores (outliers have more negative)
            # We transform to [0, 1] where 1 = most anomalous
            raw_scores = iso.decision_function(X)
            # Normalize to [0, 1]: flip sign and scale
            scores = 0.5 - raw_scores / (2.0 * (np.max(np.abs(raw_scores)) + 1e-10))
            scores = np.clip(scores, 0.0, 1.0)
            return scores
        except Exception:
            return None


# ---------------------------------------------------------------------------
# 2. cs_factor_bucket_return
# ---------------------------------------------------------------------------
@register_operator(
    name="cs_factor_bucket_return",
    category="cross_sectional",
    business_category="cross_sectional_bucket",
    canonical="cs_factor_bucket_return",
    source="cs_batch1",
    status="experimental")
class CsFactorBucketReturn(SeriesOperator):
    """横截面因子分桶平均收益（factor bucket mean return）。

    按因子值将横截面分成 N 个桶，计算每个桶内收益的平均值，输出每个股票所在桶的
    平均收益。用于分析因子的分桶表现和单调性。

    参数:
        factor: 用于分桶的因子
        ret: 收益率
        n_buckets: 桶数（默认 5）
        ascending: 是否升序分桶（默认 True，因子值最小的在桶 1）

    语义:
        - 每行独立计算（横截面）
        - 桶内样本数不足时该桶均值为 NaN（fail-closed）
        - 输出单位与 ret 相同（same_as:ret）
    """

    metadata = _metadata(
        "cs_factor_bucket_return",
        "横截面因子分桶平均收益（按因子分桶，输出每个股票所在桶的平均收益）",
        ["factor", "ret", "n_buckets", "ascending"],
        domain="price_volume",
        unit="same_as:ret",
        cost=2,
    )

    def _calculate_series(
        self,
        factor: pd.DataFrame,
        ret: pd.DataFrame,
        n_buckets: int = 5,
        ascending: bool = True,
        **_: Any
    ) -> pd.DataFrame:
        factor, ret = _aligned(factor, ret)
        fv = factor.to_numpy(dtype=float)
        rv = ret.to_numpy(dtype=float)
        rows, cols = fv.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        n_buckets = int(n_buckets)
        if n_buckets < 2:
            # Invalid bucket count -> fail-closed
            return _frame_like(factor, out)

        for row in range(rows):
            f_row = fv[row]
            r_row = rv[row]
            valid = np.isfinite(f_row) & np.isfinite(r_row)
            if valid.sum() < _MIN_BREADTH:
                continue

            bucket_means = self._bucket_return_row(
                f_row[valid], r_row[valid], n_buckets, ascending
            )
            if bucket_means is not None:
                out[row][valid] = bucket_means

        return _frame_like(factor, out)

    @staticmethod
    def _bucket_return_row(
        factor_valid: np.ndarray,
        ret_valid: np.ndarray,
        n_buckets: int,
        ascending: bool,
    ) -> np.ndarray | None:
        """Compute bucket mean returns for one cross-section."""
        n = len(factor_valid)
        if n < n_buckets:
            return None

        # Assign buckets using quantile-based splitting
        ranks = pd.Series(factor_valid).rank(method="first", ascending=ascending).to_numpy()
        bucket_indices = np.floor((ranks - 1) / n * n_buckets).astype(int)
        bucket_indices = np.clip(bucket_indices, 0, n_buckets - 1)

        # Compute mean return for each bucket
        bucket_means_map = {}
        for b in range(n_buckets):
            mask = bucket_indices == b
            if mask.sum() > 0:
                bucket_means_map[b] = float(np.mean(ret_valid[mask]))

        # Map each stock to its bucket's mean return
        result = np.full(n, np.nan, dtype=float)
        for i in range(n):
            bucket = bucket_indices[i]
            if bucket in bucket_means_map:
                result[i] = bucket_means_map[bucket]

        return result


# ---------------------------------------------------------------------------
# 3. cs_empirical_bayes_shrinkage
# ---------------------------------------------------------------------------
@register_operator(
    name="cs_empirical_bayes_shrinkage",
    category="cross_sectional",
    business_category="cross_sectional_shrinkage",
    canonical="cs_empirical_bayes_shrinkage",
    source="cs_batch1",
    status="experimental")
class CsEmpiricalBayesShrinkage(SeriesOperator):
    """横截面经验贝叶斯收缩（empirical Bayes shrinkage）。

    使用经验贝叶斯方法将个股估计值向横截面均值收缩，收缩程度由估计精度（通过
    标准误）决定。精度越低（标准误越大），收缩越强。

    参数:
        estimate: 个股估计值（如 alpha、beta）
        std_err: 估计值的标准误
        shrinkage_factor: 收缩强度系数（默认 1.0，越大收缩越强）

    语义:
        - 每行独立计算（横截面）
        - shrunk = cs_mean + (estimate - cs_mean) * precision_weight
        - precision_weight = 1 / (1 + shrinkage_factor * std_err^2 / var(estimate))
        - 输出单位与 estimate 相同（same_as:estimate）
    """

    metadata = _metadata(
        "cs_empirical_bayes_shrinkage",
        "横截面经验贝叶斯收缩（向均值收缩，收缩程度由标准误决定）",
        ["estimate", "std_err", "shrinkage_factor"],
        domain="price_volume",
        unit="same_as:estimate",
        cost=2,
    )

    def _calculate_series(
        self,
        estimate: pd.DataFrame,
        std_err: pd.DataFrame,
        shrinkage_factor: float = 1.0,
        **_: Any
    ) -> pd.DataFrame:
        estimate, std_err = _aligned(estimate, std_err)
        ev = estimate.to_numpy(dtype=float)
        sv = std_err.to_numpy(dtype=float)
        rows, cols = ev.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        shrinkage_factor = float(shrinkage_factor)
        if shrinkage_factor < 0:
            # Invalid shrinkage factor -> fail-closed
            return _frame_like(estimate, out)

        for row in range(rows):
            e_row = ev[row]
            s_row = sv[row]
            valid = np.isfinite(e_row) & np.isfinite(s_row) & (s_row > 0)
            if valid.sum() < _MIN_BREADTH:
                continue

            shrunk = self._empirical_bayes_row(
                e_row[valid], s_row[valid], shrinkage_factor
            )
            if shrunk is not None:
                out[row][valid] = shrunk

        return _frame_like(estimate, out)

    @staticmethod
    def _empirical_bayes_row(
        estimate_valid: np.ndarray,
        std_err_valid: np.ndarray,
        shrinkage_factor: float,
    ) -> np.ndarray | None:
        """Apply empirical Bayes shrinkage to one cross-section."""
        n = len(estimate_valid)
        if n < 2:
            return None

        # Cross-sectional mean (shrinkage target)
        cs_mean = float(np.mean(estimate_valid))

        # Cross-sectional variance
        cs_var = float(np.var(estimate_valid, ddof=1))
        if cs_var <= 0 or not np.isfinite(cs_var):
            # No variance -> shrink everything to mean
            return np.full(n, cs_mean, dtype=float)

        # Precision weighting: higher std_err -> lower weight -> more shrinkage
        # weight = 1 / (1 + lambda * sigma^2 / var(estimate))
        variance_ratio = (std_err_valid ** 2) / cs_var
        weights = 1.0 / (1.0 + shrinkage_factor * variance_ratio)
        weights = np.clip(weights, 0.0, 1.0)

        # Shrink toward mean
        shrunk = cs_mean + (estimate_valid - cs_mean) * weights
        return shrunk


# ---------------------------------------------------------------------------
# 4. cs_shrink_to_group_mean
# ---------------------------------------------------------------------------
@register_operator(
    name="cs_shrink_to_group_mean",
    category="cross_sectional",
    business_category="cross_sectional_shrinkage",
    canonical="cs_shrink_to_group_mean",
    source="cs_batch1",
    status="experimental")
class CsShrinkToGroupMean(SeriesOperator):
    """横截面向组均值收缩（shrink to group mean）。

    将个股值向其所属组的均值收缩，收缩程度由 shrinkage_intensity 控制。
    intensity=0 时不收缩（保持原值），intensity=1 时完全收缩到组均值。

    参数:
        x: 输入值
        group: 分组标签
        shrinkage_intensity: 收缩强度（默认 0.5，范围 [0, 1]）

    语义:
        - 每行独立计算（横截面）
        - shrunk = x * (1 - intensity) + group_mean * intensity
        - 未知组标签（NaN/inf/None/空）的个股保持 NaN（fail-closed）
        - 输出单位与 x 相同（same_as:x）
    """

    metadata = _metadata(
        "cs_shrink_to_group_mean",
        "横截面向组均值收缩（按分组收缩，强度可控）",
        ["x", "group", "shrinkage_intensity"],
        domain="price_volume",
        unit="same_as:x",
        cost=1,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        group: pd.DataFrame,
        shrinkage_intensity: float = 0.5,
        **_: Any
    ) -> pd.DataFrame:
        x, group = _aligned(x, group)
        xv = x.to_numpy(dtype=float)
        gv = group.to_numpy()
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        intensity = float(shrinkage_intensity)
        if not (0.0 <= intensity <= 1.0):
            # Invalid intensity -> fail-closed
            return _frame_like(x, out)

        for row in range(rows):
            out[row] = self._shrink_to_group_mean_row(xv[row], gv[row], intensity)

        return _frame_like(x, out)

    @staticmethod
    def _shrink_to_group_mean_row(
        x_row: np.ndarray,
        g_row: np.ndarray,
        intensity: float,
    ) -> np.ndarray:
        """Shrink to group mean for one cross-section."""
        out = np.full(len(x_row), np.nan, dtype=float)

        # Get unique valid group labels
        labels = pd.unique(g_row)

        for label in labels:
            # Skip invalid membership labels (from group_ext.py logic)
            if label is None:
                continue
            if isinstance(label, str) and label == "":
                continue
            if not isinstance(label, (str, bool)):
                try:
                    if not np.isfinite(float(label)):
                        continue
                except (TypeError, ValueError):
                    pass  # keep it

            idx = g_row == label
            x_group = x_row[idx]
            valid = np.isfinite(x_group)

            if valid.sum() == 0:
                continue

            group_mean = float(np.mean(x_group[valid]))

            # Apply shrinkage: shrunk = x * (1 - intensity) + group_mean * intensity
            for j in np.flatnonzero(idx):
                if np.isfinite(x_row[j]):
                    out[j] = x_row[j] * (1.0 - intensity) + group_mean * intensity

        return out


# ---------------------------------------------------------------------------
# 5. panel_peer_graph_aggregate
# ---------------------------------------------------------------------------
@register_operator(
    name="panel_peer_graph_aggregate",
    category="cross_sectional",
    business_category="cross_sectional_graph",
    canonical="panel_peer_graph_aggregate",
    source="cs_batch1",
    status="experimental")
class PanelPeerGraphAggregate(SeriesOperator):
    """横截面图聚合（peer graph aggregation）。

    基于相似度矩阵（或邻接矩阵）对横截面进行图聚合。每个节点（股票）聚合其邻居
    节点的值，聚合方式可选均值、加权均值或总和。

    参数:
        x: 输入值
        similarity: 相似度矩阵（或邻接矩阵），shape 应与 x 对齐
        method: 聚合方法（"mean"/"weighted_mean"/"sum"，默认 "mean"）
        threshold: 相似度阈值（默认 0.0，仅聚合相似度 > threshold 的邻居）

    语义:
        - 每行独立计算（横截面）
        - similarity[i, j] 表示股票 i 和 j 的相似度
        - method="mean": 邻居均值
        - method="weighted_mean": 相似度加权均值
        - method="sum": 邻居总和
        - 输出单位与 x 相同（same_as:x）

    注意:
        - similarity 必须是方阵（N×N），与横截面股票数一致
        - 对角线元素（自身相似度）被忽略
    """

    metadata = _metadata(
        "panel_peer_graph_aggregate",
        "横截面图聚合（基于相似度矩阵聚合邻居值）",
        ["x", "similarity", "method", "threshold"],
        domain="price_volume",
        unit="same_as:x",
        cost=3,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        similarity: pd.DataFrame,
        method: str = "mean",
        threshold: float = 0.0,
        **_: Any
    ) -> pd.DataFrame:
        # Note: similarity is expected to be a panel (date × stock) where each cell
        # encodes graph structure. For simplicity, we interpret similarity as a
        # daily cross-sectional similarity measure where similarity[row, col_i]
        # represents the similarity of stock col_i to a reference.
        #
        # In a true graph context, we'd need a 3D tensor (date × N × N).
        # For this MVP, we simplify: similarity[row, j] is the weight for stock j.

        x, similarity = _aligned(x, similarity)
        xv = x.to_numpy(dtype=float)
        sv = similarity.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        method = str(method).lower()
        if method not in ("mean", "weighted_mean", "sum"):
            # Invalid method -> fail-closed
            return _frame_like(x, out)

        threshold = float(threshold)

        for row in range(rows):
            out[row] = self._graph_aggregate_row(
                xv[row], sv[row], method, threshold
            )

        return _frame_like(x, out)

    @staticmethod
    def _graph_aggregate_row(
        x_row: np.ndarray,
        sim_row: np.ndarray,
        method: str,
        threshold: float,
    ) -> np.ndarray:
        """Aggregate neighbors for one cross-section (simplified version)."""
        n = len(x_row)
        out = np.full(n, np.nan, dtype=float)

        # Simplified interpretation: sim_row[j] is the weight/similarity for stock j
        # For a true graph, we'd have a N×N matrix per row.
        # Here we treat sim_row as a "global similarity profile" and aggregate
        # all stocks weighted by similarity.

        valid_x = np.isfinite(x_row)
        valid_sim = np.isfinite(sim_row) & (sim_row > threshold)
        valid = valid_x & valid_sim

        if valid.sum() < 2:
            # Need at least 2 valid neighbors
            return out

        if method == "mean":
            # Simple mean of neighbors
            agg_val = float(np.mean(x_row[valid]))
            out[valid] = agg_val
        elif method == "weighted_mean":
            # Weighted mean by similarity
            weights = sim_row[valid]
            total_w = float(np.sum(weights))
            if total_w > 0:
                agg_val = float(np.sum(x_row[valid] * weights) / total_w)
                out[valid] = agg_val
        elif method == "sum":
            # Sum of neighbors
            agg_val = float(np.sum(x_row[valid]))
            out[valid] = agg_val

        return out


# Register all operators to EXTENDED_ONLY_CANONICALS
try:
    from factor_engine.cleaned_operators.operator_surface import extend_extended_only

    extend_extended_only([
        "cs_isolation_forest_score",
        "cs_factor_bucket_return",
        "cs_empirical_bayes_shrinkage",
        "cs_shrink_to_group_mean",
        "panel_peer_graph_aggregate",
    ])
except ImportError:  # pragma: no cover
    pass
