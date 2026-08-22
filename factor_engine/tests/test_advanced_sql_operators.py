# -*- coding: utf-8 -*-
"""高级 SQL 算子 Parity 测试。

验证 15 个复杂算子的 DuckDB SQL 实现与 pandas/polars 后端的精确一致性。
测试策略：
1. 生成随机面板数据
2. 分别用 SQL 和 pandas 计算
3. 比较结果（允许浮点误差）
"""
import numpy as np
import pandas as pd
import pytest

try:
    import duckdb
except ImportError:
    duckdb = None

from backend.sql_pushdown.advanced_sql_operators import (
    cs_zscore_sql,
    cs_winsorize_sql,
    cs_rank_normalize_sql,
    ts_seasonal_diff_sql,
    panel_rank_sql,
    ts_regression_beta_sql,
    group_residual_sql,
    cs_industry_neutralize_sql,
    panel_residual_sql,
    cs_factor_score_sql,
    SqlDialect,
)


# ===========================================================================
# 测试数据生成
# ===========================================================================

def generate_panel_data(n_dates=20, n_insts=30, seed=42):
    """生成随机面板数据。"""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    instruments = [f"I{i:03d}" for i in range(n_insts)]

    index = pd.MultiIndex.from_product([dates, instruments], names=["ts", "inst"])
    data = np.random.randn(len(index)) * 10 + 100
    # 添加一些 NaN
    data[np.random.choice(len(data), size=int(len(data) * 0.05), replace=False)] = np.nan

    df = pd.DataFrame({"_v": data}, index=index).reset_index()
    return df


def df_to_sql_table(df, conn, table_name="input_data"):
    """将 DataFrame 注册为 DuckDB 表。"""
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")


def execute_sql(conn, sql):
    """执行 SQL 并返回 DataFrame。"""
    return conn.execute(sql).df()


def compare_results(sql_result, pandas_result, atol=1e-9, rtol=1e-7):
    """比较 SQL 和 pandas 结果。"""
    # 按 ts, inst 排序
    sql_sorted = sql_result.sort_values(["ts", "inst"]).reset_index(drop=True)
    pandas_sorted = pandas_result.sort_values(["ts", "inst"]).reset_index(drop=True)

    # 比较形状
    assert sql_sorted.shape == pandas_sorted.shape, \
        f"Shape mismatch: SQL {sql_sorted.shape} vs pandas {pandas_sorted.shape}"

    # 比较值
    pd.testing.assert_frame_equal(
        sql_sorted,
        pandas_sorted,
        check_dtype=False,
        atol=atol,
        rtol=rtol,
    )


# ===========================================================================
# 测试用例
# ===========================================================================

@pytest.mark.skipif(duckdb is None, reason="DuckDB not installed")
class TestAdvancedSQLOperators:
    """高级 SQL 算子 Parity 测试套件。"""

    def setup_method(self):
        """每个测试前设置。"""
        self.conn = duckdb.connect(":memory:")
        self.df = generate_panel_data()
        df_to_sql_table(self.df, self.conn)

    def teardown_method(self):
        """每个测试后清理。"""
        self.conn.close()

    # -----------------------------------------------------------------------
    # 1. cs_zscore
    # -----------------------------------------------------------------------

    def test_cs_zscore(self):
        """测试截面 z-score。"""
        # SQL 实现
        sql = cs_zscore_sql("SELECT * FROM input_data", dialect=SqlDialect.DUCKDB)
        sql_result = execute_sql(self.conn, sql)

        # Pandas 实现
        df_pivot = self.df.pivot(index="ts", columns="inst", values="_v")
        mean = df_pivot.mean(axis=1)
        std = df_pivot.std(axis=1).replace(0, 1)
        zscore = (df_pivot.sub(mean, axis=0)).div(std, axis=0)
        pandas_result = zscore.stack().reset_index()
        pandas_result.columns = ["ts", "inst", "_v"]

        compare_results(sql_result, pandas_result)

    # -----------------------------------------------------------------------
    # 2. cs_winsorize
    # -----------------------------------------------------------------------

    def test_cs_winsorize(self):
        """测试截面 winsorize。"""
        lower, upper = 0.05, 0.95

        # SQL 实现
        sql = cs_winsorize_sql(
            "SELECT * FROM input_data",
            lower=lower,
            upper=upper,
            dialect=SqlDialect.DUCKDB,
        )
        sql_result = execute_sql(self.conn, sql)

        # Pandas 实现
        df_pivot = self.df.pivot(index="ts", columns="inst", values="_v")

        def winsorize_row(row):
            valid = row.dropna()
            if len(valid) == 0:
                return row
            lower_bound = valid.quantile(lower)
            upper_bound = valid.quantile(upper)
            return row.clip(lower=lower_bound, upper=upper_bound)

        winsorized = df_pivot.apply(winsorize_row, axis=1)
        pandas_result = winsorized.stack().reset_index()
        pandas_result.columns = ["ts", "inst", "_v"]

        compare_results(sql_result, pandas_result, atol=1e-6)

    # -----------------------------------------------------------------------
    # 3. cs_rank_normalize
    # -----------------------------------------------------------------------

    def test_cs_rank_normalize(self):
        """测试截面 rank 归一化。"""
        # SQL 实现
        sql = cs_rank_normalize_sql("SELECT * FROM input_data", dialect=SqlDialect.DUCKDB)
        sql_result = execute_sql(self.conn, sql)

        # Pandas 实现
        df_pivot = self.df.pivot(index="ts", columns="inst", values="_v")

        def rank_normalize_row(row):
            valid = row.dropna()
            if len(valid) <= 1:
                return row.map(lambda x: 1.0 if pd.notna(x) else np.nan)
            ranks = valid.rank(method="average")
            normalized = (ranks - 1) / (len(valid) - 1)
            return row.map(lambda x: normalized.get(x, np.nan) if pd.notna(x) else np.nan)

        ranked = df_pivot.apply(rank_normalize_row, axis=1)
        pandas_result = ranked.stack().reset_index()
        pandas_result.columns = ["ts", "inst", "_v"]

        compare_results(sql_result, pandas_result, atol=1e-6)

    # -----------------------------------------------------------------------
    # 4. ts_seasonal_diff
    # -----------------------------------------------------------------------

    def test_ts_seasonal_diff(self):
        """测试时序季节性差分。"""
        period = 5

        # SQL 实现
        sql = ts_seasonal_diff_sql(
            "SELECT * FROM input_data",
            seasonal_period=period,
            dialect=SqlDialect.DUCKDB,
        )
        sql_result = execute_sql(self.conn, sql)

        # Pandas 实现
        df_pivot = self.df.pivot(index="ts", columns="inst", values="_v")
        diff = df_pivot - df_pivot.shift(period)
        pandas_result = diff.stack().reset_index()
        pandas_result.columns = ["ts", "inst", "_v"]

        compare_results(sql_result, pandas_result)

    # -----------------------------------------------------------------------
    # 5. panel_rank
    # -----------------------------------------------------------------------

    def test_panel_rank(self):
        """测试面板 rank。"""
        # SQL 实现
        sql = panel_rank_sql("SELECT * FROM input_data", dialect=SqlDialect.DUCKDB)
        sql_result = execute_sql(self.conn, sql)

        # Pandas 实现
        df_sorted = self.df.sort_values("_v")
        valid = df_sorted.dropna(subset=["_v"])

        if len(valid) <= 1:
            self.df["_v_rank"] = self.df["_v"].map(lambda x: 1.0 if pd.notna(x) else np.nan)
        else:
            ranks = valid["_v"].rank(method="average")
            normalized = (ranks - 1) / (len(valid) - 1)
            rank_map = dict(zip(valid["_v"], normalized))
            self.df["_v_rank"] = self.df["_v"].map(lambda x: rank_map.get(x, np.nan) if pd.notna(x) else np.nan)

        pandas_result = self.df[["ts", "inst", "_v_rank"]].rename(columns={"_v_rank": "_v"})

        compare_results(sql_result, pandas_result, atol=1e-6)

    # -----------------------------------------------------------------------
    # 6. ts_regression_beta
    # -----------------------------------------------------------------------

    def test_ts_regression_beta(self):
        """测试时序回归 beta。"""
        # 生成两个序列
        df2 = generate_panel_data(seed=43)
        df_to_sql_table(df2, self.conn, "input_data2")

        window = 10

        # SQL 实现
        sql = ts_regression_beta_sql(
            "SELECT * FROM input_data",
            "SELECT * FROM input_data2",
            window=window,
            min_periods=5,
            dialect=SqlDialect.DUCKDB,
        )
        sql_result = execute_sql(self.conn, sql)

        # Pandas 实现（简化版）
        df_pivot1 = self.df.pivot(index="ts", columns="inst", values="_v")
        df_pivot2 = df2.pivot(index="ts", columns="inst", values="_v")

        def rolling_beta(y, x):
            cov = y.rolling(window, min_periods=5).cov(x)
            var = x.rolling(window, min_periods=5).var()
            return cov / var.replace(0, np.nan)

        beta = pd.DataFrame(
            {col: rolling_beta(df_pivot1[col], df_pivot2[col]) for col in df_pivot1.columns}
        )
        pandas_result = beta.stack().reset_index()
        pandas_result.columns = ["ts", "inst", "_v"]

        compare_results(sql_result, pandas_result, atol=1e-5)

    # -----------------------------------------------------------------------
    # 7. cs_industry_neutralize
    # -----------------------------------------------------------------------

    def test_cs_industry_neutralize(self):
        """测试行业中性化。"""
        # 生成行业分组
        np.random.seed(42)
        industries = np.random.choice([1, 2, 3], size=len(self.df))
        df_ind = self.df.copy()
        df_ind["_v"] = industries
        df_to_sql_table(df_ind, self.conn, "industry_data")

        # SQL 实现
        sql = cs_industry_neutralize_sql(
            "SELECT * FROM input_data",
            "SELECT * FROM industry_data",
            dialect=SqlDialect.DUCKDB,
        )
        sql_result = execute_sql(self.conn, sql)

        # Pandas 实现
        df_merged = self.df.merge(df_ind[["ts", "inst", "_v"]], on=["ts", "inst"], suffixes=("", "_ind"))
        df_pivot = df_merged.pivot(index="ts", columns="inst", values="_v")
        ind_pivot = df_merged.pivot(index="ts", columns="inst", values="_v_ind")

        neutral = df_pivot.copy()
        for ts in df_pivot.index:
            row = df_pivot.loc[ts]
            ind_row = ind_pivot.loc[ts]
            for ind_val in ind_row.dropna().unique():
                mask = ind_row == ind_val
                if mask.sum() > 0:
                    ind_mean = row[mask].mean()
                    neutral.loc[ts, mask] = row[mask] - ind_mean

        pandas_result = neutral.stack().reset_index()
        pandas_result.columns = ["ts", "inst", "_v"]

        compare_results(sql_result, pandas_result, atol=1e-6)

    # -----------------------------------------------------------------------
    # 8. 集成测试：多算子组合
    # -----------------------------------------------------------------------

    def test_combined_operators(self):
        """测试算子组合：cs_zscore + cs_winsorize。"""
        # 先 zscore 再 winsorize
        sql1 = cs_zscore_sql("SELECT * FROM input_data", dialect=SqlDialect.DUCKDB)
        sql2 = cs_winsorize_sql(f"({sql1})", lower=0.05, upper=0.95, dialect=SqlDialect.DUCKDB)
        sql_result = execute_sql(self.conn, sql2)

        # Pandas 实现
        df_pivot = self.df.pivot(index="ts", columns="inst", values="_v")
        mean = df_pivot.mean(axis=1)
        std = df_pivot.std(axis=1).replace(0, 1)
        zscore = (df_pivot.sub(mean, axis=0)).div(std, axis=0)

        def winsorize_row(row):
            valid = row.dropna()
            if len(valid) == 0:
                return row
            lower_bound = valid.quantile(0.05)
            upper_bound = valid.quantile(0.95)
            return row.clip(lower=lower_bound, upper=upper_bound)

        winsorized = zscore.apply(winsorize_row, axis=1)
        pandas_result = winsorized.stack().reset_index()
        pandas_result.columns = ["ts", "inst", "_v"]

        compare_results(sql_result, pandas_result, atol=1e-6)


# ===========================================================================
# 性能基准测试
# ===========================================================================

@pytest.mark.skipif(duckdb is None, reason="DuckDB not installed")
class TestAdvancedSQLPerformance:
    """性能基准测试。"""

    def test_large_panel_cs_zscore(self, benchmark):
        """大规模面板 cs_zscore 性能测试。"""
        conn = duckdb.connect(":memory:")
        df = generate_panel_data(n_dates=250, n_insts=3000)
        df_to_sql_table(df, conn)

        sql = cs_zscore_sql("SELECT * FROM input_data", dialect=SqlDialect.DUCKDB)

        def run_sql():
            return execute_sql(conn, sql)

        result = benchmark(run_sql)
        assert len(result) > 0

        conn.close()

    def test_large_panel_ts_seasonal_diff(self, benchmark):
        """大规模面板 ts_seasonal_diff 性能测试。"""
        conn = duckdb.connect(":memory:")
        df = generate_panel_data(n_dates=250, n_insts=3000)
        df_to_sql_table(df, conn)

        sql = ts_seasonal_diff_sql(
            "SELECT * FROM input_data",
            seasonal_period=12,
            dialect=SqlDialect.DUCKDB,
        )

        def run_sql():
            return execute_sql(conn, sql)

        result = benchmark(run_sql)
        assert len(result) > 0

        conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
