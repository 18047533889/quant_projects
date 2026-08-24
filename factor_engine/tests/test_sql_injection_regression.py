# -*- coding: utf-8 -*-
"""SQL injection regression tests for backend/sql_pushdown/emitter.py.

Validates that dataset/table name interpolation in FROM clauses is fail-closed
and rejects identifiers with special characters that could enable injection.
"""
import pytest

from factor_engine.backend.sql_pushdown.emitter import (
    _validate_sql_identifier,
    _duckdb_dataset_ref,
    compile_plan_to_sql,
    SqlDialect,
)
from factor_engine.planner.logical_plan import PlanNode


class TestSqlIdentifierValidation:
    """Test _validate_sql_identifier fail-closed validation."""

    def test_valid_identifiers(self):
        """Valid identifiers: alphanumeric, underscore, non-leading digit."""
        valid_names = [
            "valid_table",
            "table123",
            "_private",
            "table_name",
            "CamelCase",
            "UPPERCASE",
            "a",
            "a1",
            "_",
            "__init__",
        ]
        for name in valid_names:
            # Should not raise
            _validate_sql_identifier(name, context="table")

    def test_reject_injection_attempts(self):
        """Reject identifiers with injection-enabling special chars."""
        injection_attempts = [
            "table; DROP TABLE users--",
            "table'; DROP TABLE users--",
            'table"; DROP TABLE users--',
            "table /* comment */",
            "table -- comment",
            "table\n--comment",
            "table OR 1=1",
            "table UNION SELECT",
        ]
        for name in injection_attempts:
            with pytest.raises(ValueError, match="unsafe characters"):
                _validate_sql_identifier(name, context="table")

    def test_reject_special_chars(self):
        """Reject identifiers with spaces, dots, braces, slashes."""
        invalid_names = [
            "table name",
            "table.schema",
            "table-name",
            "table+name",
            "table*name",
            "table/name",
            "table\\name",
            "table(name)",
            "table[name]",
            "{{placeholder}}",
            "table@name",
            "table#name",
            "table$name",
            "table%name",
            "table&name",
            "table|name",
            "table<name",
            "table>name",
            "table=name",
            "table!name",
            "table?name",
            "table:name",
            "table,name",
        ]
        for name in invalid_names:
            with pytest.raises(ValueError, match="unsafe characters"):
                _validate_sql_identifier(name, context="table")

    def test_reject_empty_identifier(self):
        """Empty identifier should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_sql_identifier("", context="dataset")

    def test_reject_leading_digit(self):
        """Identifier starting with digit should be rejected."""
        with pytest.raises(ValueError, match="unsafe characters"):
            _validate_sql_identifier("123table", context="table")

    def test_reject_too_long(self):
        """Identifier exceeding 128 chars should raise ValueError."""
        long_name = "a" * 129
        with pytest.raises(ValueError, match="too long"):
            _validate_sql_identifier(long_name, context="dataset")

    def test_context_in_error_message(self):
        """Error message should include context parameter."""
        with pytest.raises(ValueError, match="dataset.*unsafe"):
            _validate_sql_identifier("bad;name", context="dataset")
        with pytest.raises(ValueError, match="table.*unsafe"):
            _validate_sql_identifier("bad;name", context="table")


class TestDuckdbDatasetRef:
    """Test _duckdb_dataset_ref validation + placeholder generation."""

    def test_valid_dataset_placeholder(self):
        """Valid dataset name should generate {{dataset}} placeholder."""
        result = _duckdb_dataset_ref("my_dataset")
        assert result == "{{my_dataset}}"

    def test_reject_invalid_dataset(self):
        """Invalid dataset name should raise before placeholder generation."""
        with pytest.raises(ValueError, match="dataset.*unsafe"):
            _duckdb_dataset_ref("dataset; DROP TABLE")


class TestCompilePlanInjectionProtection:
    """Integration tests: compile_plan_to_sql rejects malicious dataset/table."""

    def test_duckdb_reject_malicious_dataset(self):
        """DuckDB: malicious dataset name should raise ValueError."""
        plan = PlanNode(op="column", attrs={"column": "close"})

        with pytest.raises(ValueError, match="dataset.*unsafe"):
            compile_plan_to_sql(
                plan,
                dataset="dataset'; DROP TABLE users--",
                time_column="trade_date",
                instrument_column="instrument",
                dialect=SqlDialect.DUCKDB,
            )

    def test_clickhouse_reject_malicious_table(self):
        """ClickHouse: malicious table name should raise ValueError."""
        plan = PlanNode(op="column", attrs={"column": "close"})

        with pytest.raises(ValueError, match="table.*unsafe"):
            compile_plan_to_sql(
                plan,
                table="table; DROP TABLE users",
                time_column="trade_date",
                instrument_column="instrument",
                dialect=SqlDialect.CLICKHOUSE,
            )

    def test_duckdb_valid_dataset_compiles(self):
        """DuckDB: valid dataset name should compile successfully."""
        plan = PlanNode(op="column", attrs={"column": "close"})

        result = compile_plan_to_sql(
            plan,
            dataset="valid_dataset",
            time_column="trade_date",
            instrument_column="instrument",
            dialect=SqlDialect.DUCKDB,
        )

        assert result is not None
        assert "{{valid_dataset}}" in result.query
        assert result.read_datasets == ("valid_dataset",)

    def test_clickhouse_valid_table_compiles(self):
        """ClickHouse: valid table name should compile successfully."""
        plan = PlanNode(op="column", attrs={"column": "close"})

        result = compile_plan_to_sql(
            plan,
            table="valid_table",
            time_column="trade_date",
            instrument_column="instrument",
            dialect=SqlDialect.CLICKHOUSE,
        )

        assert result is not None
        assert "FROM valid_table" in result.query
        assert result.table == "valid_table"


class TestDateFilterPitSafety:
    """Verify date filters remain inclusive (<=, >=) and PIT-safe after changes."""

    def test_date_filter_inclusive_bounds(self):
        """Date filters should use >= for start and <= for end (inclusive)."""
        from factor_engine.backend.sql_pushdown.emitter import _build_filter_clause, SqlPushdownFilter

        filt = SqlPushdownFilter(
            time_column="trade_date",
            start="2024-01-01",
            end="2024-12-31",
        )

        clause = _build_filter_clause(filt, dialect=SqlDialect.DUCKDB)

        # Should be inclusive bounds for PIT safety
        assert ">=" in clause
        assert "<=" in clause
        assert "trade_date" in clause or '"trade_date"' in clause
        # Should NOT use exclusive bounds
        assert ">" not in clause.replace(">=", "")
        assert "<" not in clause.replace("<=", "")

    def test_date_filter_quoted_identifiers(self):
        """Date filter column names should be properly quoted."""
        from factor_engine.backend.sql_pushdown.emitter import _build_filter_clause, SqlPushdownFilter

        filt = SqlPushdownFilter(
            time_column="trade_date",
            start="2024-01-01",
            end="2024-12-31",
        )

        clause = _build_filter_clause(filt, dialect=SqlDialect.DUCKDB)

        # Column name should be quoted to prevent injection
        assert '"trade_date"' in clause

    def test_date_filter_no_sql_literal_injection(self):
        """Date literals should be properly escaped (no raw interpolation)."""
        from factor_engine.backend.sql_pushdown.emitter import _build_filter_clause, SqlPushdownFilter

        # Attempt injection via date literal
        filt = SqlPushdownFilter(
            time_column="trade_date",
            start="2024-01-01' OR '1'='1",
            end="2024-12-31",
        )

        clause = _build_filter_clause(filt, dialect=SqlDialect.DUCKDB)

        # _sql_literal should escape single quotes
        assert "''" in clause or "OR" not in clause
