# -*- coding: utf-8 -*-
"""
测试 Polars Native CS 算子 - Batch 1

测试真正的 Polars native 实现，验证：
1. 基础功能正确性
2. 边界情况处理
3. 性能特性
"""
import pytest
import pandas as pd
import numpy as np

from cleaned_operators.polars_native.cs_batch1 import (
    CSRankPolarsNative,
    CSDemeanPolarsNative,
    CSZscorePolarsNative,
    CSBucketPolarsNative,
    CSQuantilePolarsNative,
    CSFillMeanPolarsNative,
    CSFillMedianPolarsNative,
    CSImputeMeanPolarsNative,
    CSImputeMedianPolarsNative,
    CSValidCountPolarsNative,
    CSCoverageRatioPolarsNative,
    CSWeightedMeanPolarsNative,
    CSWeightedDemeanPolarsNative,
    CSWeightedZscorePolarsNative,
    CSNeutralizePolarsNative,
    CSLadResidPolarsNative,
    CSQuantileResidPolarsNative,
)


@pytest.fixture
def sample_series():
    """生成测试 Series"""
    dates = pd.date_range('2024-01-01', periods=10)
    np.random.seed(42)
    data = np.random.randn(10) * 10 + 100
    return pd.Series(data, index=dates, name='test_feature')


@pytest.fixture
def sample_series_with_nan():
    """生成包含 NaN 的测试 Series"""
    dates = pd.date_range('2024-01-01', periods=10)
    np.random.seed(42)
    data = np.random.randn(10) * 10 + 100
    data[2] = np.nan
    data[5] = np.nan
    return pd.Series(data, index=dates, name='test_feature')


class TestCSRankPolarsNative:
    """测试截面排名"""

    def test_basic_rank(self, sample_series):
        op = CSRankPolarsNative()
        result = op._calculate_series(sample_series)

        # 验证结果范围 [0, 1]
        assert result.min() >= 0
        assert result.max() <= 1

        # 验证索引保持不变
        assert result.index.equals(sample_series.index)

    def test_rank_with_nan(self, sample_series_with_nan):
        op = CSRankPolarsNative()
        result = op._calculate_series(sample_series_with_nan)

        # NaN 位置应该保持 NaN
        assert pd.isna(result.iloc[2])
        assert pd.isna(result.iloc[5])

    def test_rank_single_value(self):
        series = pd.Series([100.0], index=[pd.Timestamp('2024-01-01')])
        op = CSRankPolarsNative()
        result = op._calculate_series(series)

        # 单个值应该返回 0.5
        assert result.iloc[0] == 0.5


class TestCSDemeanPolarsNative:
    """测试截面去均值"""

    def test_basic_demean(self, sample_series):
        op = CSDemeanPolarsNative()
        result = op._calculate_series(sample_series)

        # 验证均值接近 0
        assert abs(result.mean()) < 1e-10

        # 验证索引保持不变
        assert result.index.equals(sample_series.index)

    def test_demean_with_nan(self, sample_series_with_nan):
        op = CSDemeanPolarsNative()
        result = op._calculate_series(sample_series_with_nan)

        # NaN 位置应该保持 NaN
        assert pd.isna(result.iloc[2])
        assert pd.isna(result.iloc[5])


class TestCSZscorePolarsNative:
    """测试截面 Z-score"""

    def test_basic_zscore(self, sample_series):
        op = CSZscorePolarsNative()
        result = op._calculate_series(sample_series)

        # 验证均值接近 0，标准差接近 1
        assert abs(result.mean()) < 1e-10
        assert abs(result.std() - 1.0) < 1e-10

        # 验证索引保持不变
        assert result.index.equals(sample_series.index)

    def test_zscore_with_nan(self, sample_series_with_nan):
        op = CSZscorePolarsNative()
        result = op._calculate_series(sample_series_with_nan)

        # NaN 位置应该保持 NaN
        assert pd.isna(result.iloc[2])
        assert pd.isna(result.iloc[5])


class TestCSBucketPolarsNative:
    """测试截面分桶"""

    def test_basic_bucket(self, sample_series):
        op = CSBucketPolarsNative()
        result = op._calculate_series(sample_series, n_buckets=5)

        # 验证分桶结果在 [0, 4] 范围内
        valid_result = result.dropna()
        assert valid_result.min() >= 0
        assert valid_result.max() <= 4


class TestCSQuantilePolarsNative:
    """测试截面分位数"""

    def test_basic_quantile(self, sample_series):
        op = CSQuantilePolarsNative()
        result = op._calculate_series(sample_series, q=0.5)

        # 所有值应该是中位数
        median_value = sample_series.median()
        assert (result == median_value).all()

    def test_quantile_different_q(self, sample_series):
        op = CSQuantilePolarsNative()

        q25 = op._calculate_series(sample_series, q=0.25)
        q75 = op._calculate_series(sample_series, q=0.75)

        # q75 应该大于 q25
        assert q75.iloc[0] > q25.iloc[0]


class TestCSFillMeanPolarsNative:
    """测试截面均值填充"""

    def test_fill_mean(self, sample_series_with_nan):
        op = CSFillMeanPolarsNative()
        result = op._calculate_series(sample_series_with_nan)

        # 验证没有 NaN
        assert not result.isna().any()

        # 验证填充值是均值
        mean_value = sample_series_with_nan.mean()
        assert abs(result.iloc[2] - mean_value) < 1e-10
        assert abs(result.iloc[5] - mean_value) < 1e-10


class TestCSFillMedianPolarsNative:
    """测试截面中位数填充"""

    def test_fill_median(self, sample_series_with_nan):
        op = CSFillMedianPolarsNative()
        result = op._calculate_series(sample_series_with_nan)

        # 验证没有 NaN
        assert not result.isna().any()

        # 验证填充值是中位数
        median_value = sample_series_with_nan.median()
        assert abs(result.iloc[2] - median_value) < 1e-10
        assert abs(result.iloc[5] - median_value) < 1e-10


class TestCSValidCountPolarsNative:
    """测试截面有效计数"""

    def test_valid_count(self, sample_series_with_nan):
        op = CSValidCountPolarsNative()
        result = op._calculate_series(sample_series_with_nan)

        # 所有位置应该返回相同的有效计数
        expected_count = sample_series_with_nan.notna().sum()
        assert (result == expected_count).all()


class TestCSCoverageRatioPolarsNative:
    """测试截面覆盖率"""

    def test_coverage_ratio(self, sample_series_with_nan):
        op = CSCoverageRatioPolarsNative()
        result = op._calculate_series(sample_series_with_nan)

        # 验证覆盖率在 [0, 1] 范围内
        assert result.min() >= 0
        assert result.max() <= 1

        # 验证覆盖率计算正确
        expected_ratio = sample_series_with_nan.notna().sum() / len(sample_series_with_nan)
        assert abs(result.iloc[0] - expected_ratio) < 1e-10


class TestCSWeightedOperations:
    """测试加权操作"""

    def test_weighted_mean(self, sample_series):
        weights = pd.Series(np.ones(len(sample_series)), index=sample_series.index)

        op = CSWeightedMeanPolarsNative()
        result = op._calculate_series(sample_series, weight=weights)

        # 等权重应该等于简单均值
        expected = sample_series.mean()
        assert abs(result.iloc[0] - expected) < 1e-10

    def test_weighted_demean(self, sample_series):
        weights = pd.Series(np.ones(len(sample_series)), index=sample_series.index)

        op = CSWeightedDemeanPolarsNative()
        result = op._calculate_series(sample_series, weight=weights)

        # 等权重应该等于简单 demean
        expected_mean = sample_series.mean()
        expected = sample_series - expected_mean
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_weighted_zscore(self, sample_series):
        weights = pd.Series(np.ones(len(sample_series)), index=sample_series.index)

        op = CSWeightedZscorePolarsNative()
        result = op._calculate_series(sample_series, weight=weights)

        # 验证加权 Z-score 均值接近 0
        assert abs(result.mean()) < 1e-10


class TestCSResidualOperations:
    """测试残差操作"""

    def test_lad_resid(self, sample_series):
        op = CSLadResidPolarsNative()
        result = op._calculate_series(sample_series)

        # LAD 残差：x - median(x)
        expected = sample_series - sample_series.median()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_quantile_resid(self, sample_series):
        op = CSQuantileResidPolarsNative()
        result = op._calculate_series(sample_series, q=0.5)

        # 分位数残差：x - quantile(x, q)
        expected = sample_series - sample_series.quantile(0.5)
        pd.testing.assert_series_equal(result, expected, check_names=False)


class TestCSNeutralizePolarsNative:
    """测试中性化"""

    def test_neutralize(self, sample_series):
        op = CSNeutralizePolarsNative()
        result = op._calculate_series(sample_series)

        # 中性化就是 demean
        assert abs(result.mean()) < 1e-10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
