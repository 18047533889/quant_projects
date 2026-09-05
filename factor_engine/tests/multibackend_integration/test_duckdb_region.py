# -*- coding: utf-8 -*-
"""DuckDB backend-specific integration tests.

Tests cover:
- DuckDB SQL execution
- DuckDB relation API
- Window functions
- Aggregations and joins
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd


class TestDuckDBBackend:
    """Test DuckDB SQL backend."""

    def test_duckdb_available(self):
        """Test DuckDB is available for import."""
        try:
            import duckdb
            assert duckdb is not None
        except ImportError:
            pytest.skip("DuckDB not installed")

    def test_duckdb_basic_query(self, sample_panel_data):
        """Test basic SQL query in DuckDB."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")
        conn.register("data", sample_panel_data)

        result = conn.execute("SELECT COUNT(*) as cnt FROM data").fetchone()
        assert result[0] == len(sample_panel_data)

        conn.close()

    def test_duckdb_relation_api(self, sample_panel_data):
        """Test DuckDB Relation API."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")

        # Create relation
        rel = conn.from_df(sample_panel_data)

        # Query via relation API
        result = rel.filter("close > 100.0").aggregate("COUNT(*) as cnt").fetchone()

        assert result[0] >= 0

        conn.close()


class TestDuckDBWindowFunctions:
    """Test DuckDB window functions."""

    def test_duckdb_rolling_mean(self, sample_panel_data):
        """Test rolling mean using window function."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")
        conn.register("data", sample_panel_data)

        result = conn.execute("""
            SELECT
                date,
                instrument,
                close,
                AVG(close) OVER (
                    PARTITION BY instrument
                    ORDER BY date
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) as rolling_mean
            FROM data
            ORDER BY instrument, date
        """).df()

        assert "rolling_mean" in result.columns
        assert len(result) == len(sample_panel_data)

        conn.close()

    def test_duckdb_rank_function(self, sample_panel_data):
        """Test rank window function."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")
        conn.register("data", sample_panel_data)

        result = conn.execute("""
            SELECT
                date,
                instrument,
                close,
                RANK() OVER (PARTITION BY date ORDER BY close DESC) as rank
            FROM data
        """).df()

        assert "rank" in result.columns
        # Ranks should start from 1
        assert result["rank"].min() == 1

        conn.close()

    def test_duckdb_lag_lead(self, sample_panel_data):
        """Test LAG and LEAD functions."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")
        conn.register("data", sample_panel_data)

        result = conn.execute("""
            SELECT
                date,
                instrument,
                close,
                LAG(close, 1) OVER (PARTITION BY instrument ORDER BY date) as close_lag1,
                LEAD(close, 1) OVER (PARTITION BY instrument ORDER BY date) as close_lead1
            FROM data
            ORDER BY instrument, date
        """).df()

        assert "close_lag1" in result.columns
        assert "close_lead1" in result.columns

        conn.close()


class TestDuckDBAggregations:
    """Test DuckDB aggregation operations."""

    def test_duckdb_group_by(self, sample_panel_data):
        """Test GROUP BY aggregation."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")
        conn.register("data", sample_panel_data)

        result = conn.execute("""
            SELECT
                instrument,
                AVG(close) as avg_close,
                STDDEV(close) as std_close,
                SUM(volume) as total_volume,
                COUNT(*) as count
            FROM data
            GROUP BY instrument
        """).df()

        assert len(result) == sample_panel_data["instrument"].nunique()
        assert all(col in result.columns for col in [
            "instrument", "avg_close", "std_close", "total_volume", "count"
        ])

        conn.close()

    def test_duckdb_multiple_grouping_keys(self, sample_panel_data):
        """Test GROUP BY with multiple keys."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")
        conn.register("data", sample_panel_data)

        # Add year column for grouping
        result = conn.execute("""
            SELECT
                YEAR(date) as year,
                instrument,
                AVG(close) as avg_close
            FROM data
            GROUP BY year, instrument
        """).df()

        assert result["year"].notna().all()
        assert len(result) > 0

        conn.close()

    def test_duckdb_having_clause(self, sample_panel_data):
        """Test HAVING clause filtering."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")
        conn.register("data", sample_panel_data)

        result = conn.execute("""
            SELECT
                instrument,
                AVG(close) as avg_close
            FROM data
            GROUP BY instrument
            HAVING AVG(close) > 100.0
        """).df()

        # All results should have avg_close > 100
        assert (result["avg_close"] > 100.0).all()

        conn.close()


class TestDuckDBJoins:
    """Test DuckDB join operations."""

    def test_duckdb_inner_join(self, sample_panel_data):
        """Test inner join."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")

        df1 = sample_panel_data[["date", "instrument", "close"]].copy()
        df2 = sample_panel_data[["date", "instrument", "volume"]].copy()

        conn.register("df1", df1)
        conn.register("df2", df2)

        result = conn.execute("""
            SELECT
                df1.date,
                df1.instrument,
                df1.close,
                df2.volume
            FROM df1
            INNER JOIN df2
                ON df1.date = df2.date
                AND df1.instrument = df2.instrument
        """).df()

        assert len(result) == len(sample_panel_data)
        assert "close" in result.columns
        assert "volume" in result.columns

        conn.close()

    def test_duckdb_left_join(self, sample_panel_data):
        """Test left outer join."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")

        df1 = sample_panel_data.copy()
        df2 = sample_panel_data.sample(frac=0.8).copy()  # 80% of data

        conn.register("df1", df1)
        conn.register("df2", df2)

        result = conn.execute("""
            SELECT
                df1.date,
                df1.instrument,
                df1.close,
                df2.volume
            FROM df1
            LEFT JOIN df2
                ON df1.date = df2.date
                AND df1.instrument = df2.instrument
        """).df()

        # Result should have all rows from df1
        assert len(result) == len(df1)

        conn.close()


class TestDuckDBPerformance:
    """Test DuckDB performance characteristics."""

    def test_duckdb_large_aggregation(self, large_panel_data):
        """Test DuckDB on large aggregation."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")
        conn.register("data", large_panel_data)

        result = conn.execute("""
            SELECT
                instrument,
                AVG(close) as avg_close,
                STDDEV(close) as std_close
            FROM data
            GROUP BY instrument
        """).df()

        assert len(result) == large_panel_data["instrument"].nunique()

        conn.close()

    def test_duckdb_complex_window(self, large_panel_data):
        """Test DuckDB complex window function."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")
        conn.register("data", large_panel_data)

        result = conn.execute("""
            SELECT
                date,
                instrument,
                close,
                AVG(close) OVER (
                    PARTITION BY instrument
                    ORDER BY date
                    ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
                ) as rolling_mean,
                RANK() OVER (
                    PARTITION BY date
                    ORDER BY close DESC
                ) as rank
            FROM data
            LIMIT 1000
        """).df()

        assert len(result) == 1000
        assert "rolling_mean" in result.columns
        assert "rank" in result.columns

        conn.close()


class TestDuckDBDataTypes:
    """Test DuckDB data type handling."""

    def test_duckdb_datetime_operations(self, sample_panel_data):
        """Test datetime operations."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")
        conn.register("data", sample_panel_data)

        result = conn.execute("""
            SELECT
                date,
                instrument,
                YEAR(date) as year,
                MONTH(date) as month,
                DAY(date) as day
            FROM data
            LIMIT 10
        """).df()

        assert "year" in result.columns
        assert "month" in result.columns
        assert "day" in result.columns

        conn.close()

    def test_duckdb_numeric_operations(self, sample_panel_data):
        """Test numeric operations."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")
        conn.register("data", sample_panel_data)

        result = conn.execute("""
            SELECT
                close,
                ROUND(close, 2) as close_rounded,
                FLOOR(close) as close_floor,
                CEIL(close) as close_ceil,
                ABS(returns) as abs_returns
            FROM data
            LIMIT 10
        """).df()

        assert all(col in result.columns for col in [
            "close_rounded", "close_floor", "close_ceil", "abs_returns"
        ])

        conn.close()


class TestDuckDBStreaming:
    """Test DuckDB streaming capabilities."""

    def test_duckdb_streaming_aggregation(self, large_panel_data):
        """Test streaming aggregation (memory efficient)."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")
        conn.register("data", large_panel_data)

        # DuckDB uses streaming internally for aggregations
        result = conn.execute("""
            SELECT
                instrument,
                COUNT(*) as count,
                SUM(volume) as total_volume
            FROM data
            GROUP BY instrument
        """).df()

        assert len(result) > 0

        conn.close()

    def test_duckdb_limited_memory_query(self, large_panel_data):
        """Test query that works with limited memory."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")
        conn.register("data", large_panel_data)

        # Query with LIMIT to avoid loading everything
        result = conn.execute("""
            SELECT *
            FROM data
            WHERE close > 100.0
            ORDER BY date
            LIMIT 1000
        """).df()

        assert len(result) <= 1000

        conn.close()


class TestDuckDBRelationConversion:
    """Test DuckDB relation conversion to other formats."""

    def test_duckdb_to_pandas(self, sample_panel_data):
        """Test DuckDB Relation → Pandas conversion."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")
        rel = conn.from_df(sample_panel_data)

        # Convert to Pandas
        result_df = rel.df()

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(sample_panel_data)

        conn.close()

    def test_duckdb_to_arrow(self, sample_panel_data):
        """Test DuckDB Relation → Arrow conversion."""
        try:
            import duckdb
            import pyarrow as pa
        except ImportError:
            pytest.skip("DuckDB or PyArrow not installed")

        conn = duckdb.connect(":memory:")
        rel = conn.from_df(sample_panel_data)

        # Convert to Arrow — newer duckdb versions return a pyarrow
        # RecordBatchReader instead of a Table; normalize so the row-count
        # contract is checked on an actual Table either way.
        arrow_result = rel.arrow()
        if isinstance(arrow_result, pa.RecordBatchReader):
            arrow_result = arrow_result.read_all()
        assert isinstance(arrow_result, pa.Table)
        assert arrow_result.num_rows == len(sample_panel_data)

        conn.close()


class TestDuckDBCTE:
    """Test DuckDB Common Table Expressions (CTEs)."""

    def test_duckdb_with_clause(self, sample_panel_data):
        """Test WITH clause (CTE)."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")
        conn.register("data", sample_panel_data)

        result = conn.execute("""
            WITH daily_stats AS (
                SELECT
                    date,
                    AVG(close) as avg_close,
                    STDDEV(close) as std_close
                FROM data
                GROUP BY date
            )
            SELECT
                date,
                avg_close,
                std_close
            FROM daily_stats
            WHERE avg_close > 95.0
        """).df()

        assert "avg_close" in result.columns
        assert (result["avg_close"] > 95.0).all()

        conn.close()

    def test_duckdb_nested_cte(self, sample_panel_data):
        """Test nested CTEs."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")
        conn.register("data", sample_panel_data)

        result = conn.execute("""
            WITH base AS (
                SELECT
                    instrument,
                    AVG(close) as avg_close
                FROM data
                GROUP BY instrument
            ),
            filtered AS (
                SELECT *
                FROM base
                WHERE avg_close > 95.0
            )
            SELECT COUNT(*) as cnt
            FROM filtered
        """).fetchone()

        assert result[0] >= 0

        conn.close()


class TestDuckDBOrderingGuarantees:
    """Test DuckDB ordering behavior (Section 22)."""

    def test_duckdb_no_order_without_order_by(self, sample_panel_data):
        """Test that DuckDB doesn't guarantee order without ORDER BY."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")
        conn.register("data", sample_panel_data)

        # Query without ORDER BY - order is not guaranteed
        result = conn.execute("""
            SELECT date, instrument, close
            FROM data
        """).df()

        # We can't assert anything about the order here
        # Just verify we got the data
        assert len(result) == len(sample_panel_data)

        conn.close()

    def test_duckdb_explicit_order_by(self, sample_panel_data):
        """Test explicit ORDER BY produces ordered results."""
        try:
            import duckdb
        except ImportError:
            pytest.skip("DuckDB not installed")

        conn = duckdb.connect(":memory:")
        conn.register("data", sample_panel_data)

        # Query with explicit ORDER BY
        result = conn.execute("""
            SELECT date, instrument, close
            FROM data
            ORDER BY date, instrument
        """).df()

        # Verify ordering
        for i in range(len(result) - 1):
            current = (result.iloc[i]["date"], result.iloc[i]["instrument"])
            next_row = (result.iloc[i+1]["date"], result.iloc[i+1]["instrument"])
            assert current <= next_row

        conn.close()
