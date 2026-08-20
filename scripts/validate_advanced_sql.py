#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立验证脚本：测试高级 SQL 算子实现。

不依赖完整的 FactorEngine 环境，只验证 SQL 生成逻辑。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# 直接导入模块（避免触发 registry）
from backend.sql_pushdown.advanced_sql_operators import (
    SqlDialect,
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
    list_advanced_sql_operators,
)


def validate_sql_generation():
    """验证 SQL 生成功能。"""
    print("=" * 70)
    print("DuckDB 高级 SQL 算子验证")
    print("=" * 70)

    test_input = "SELECT ts, inst, _v FROM test_data"
    dialect = SqlDialect.DUCKDB

    # 1. cs_zscore
    print("\n1. Testing cs_zscore...")
    sql = cs_zscore_sql(test_input, dialect=dialect)
    assert "WITH stats AS" in sql
    assert "AVG(_v) OVER (PARTITION BY ts)" in sql
    assert "stddev_samp(_v) OVER (PARTITION BY ts)" in sql
    print("   ✓ cs_zscore SQL generated successfully")
    print(f"   SQL length: {len(sql)} chars")

    # 2. cs_winsorize
    print("\n2. Testing cs_winsorize...")
    sql = cs_winsorize_sql(test_input, lower=0.05, upper=0.95, dialect=dialect)
    assert "WITH bounds AS" in sql
    assert "greatest" in sql.lower() or "GREATEST" in sql
    print("   ✓ cs_winsorize SQL generated successfully")
    print(f"   SQL length: {len(sql)} chars")

    # 3. cs_rank_normalize
    print("\n3. Testing cs_rank_normalize...")
    sql = cs_rank_normalize_sql(test_input, dialect=dialect)
    assert "SELECT CASE" in sql or "SELECT b.ts, b.inst" in sql
    assert "FILTER" in sql
    print("   ✓ cs_rank_normalize SQL generated successfully")
    print(f"   SQL length: {len(sql)} chars")

    # 4. ts_seasonal_diff
    print("\n4. Testing ts_seasonal_diff...")
    sql = ts_seasonal_diff_sql(test_input, seasonal_period=12, dialect=dialect)
    assert "LAG(_v, 12)" in sql
    print("   ✓ ts_seasonal_diff SQL generated successfully")
    print(f"   SQL length: {len(sql)} chars")

    # 5. panel_rank
    print("\n5. Testing panel_rank...")
    sql = panel_rank_sql(test_input, dialect=dialect)
    assert "WITH global_stats AS" in sql
    assert "CROSS JOIN" in sql
    print("   ✓ panel_rank SQL generated successfully")
    print(f"   SQL length: {len(sql)} chars")

    # 6. ts_regression_beta
    print("\n6. Testing ts_regression_beta...")
    sql = ts_regression_beta_sql(
        test_input,
        test_input,
        window=20,
        min_periods=10,
        dialect=dialect
    )
    assert "WITH stats AS" in sql
    assert "covar_samp" in sql
    assert "var_samp" in sql
    print("   ✓ ts_regression_beta SQL generated successfully")
    print(f"   SQL length: {len(sql)} chars")

    # 7. group_residual
    print("\n7. Testing group_residual...")
    sql = group_residual_sql(test_input, test_input, test_input, dialect=dialect)
    assert "WITH joined AS" in sql
    assert "stats AS" in sql
    print("   ✓ group_residual SQL generated successfully")
    print(f"   SQL length: {len(sql)} chars")

    # 8. cs_industry_neutralize
    print("\n8. Testing cs_industry_neutralize...")
    sql = cs_industry_neutralize_sql(test_input, test_input, dialect=dialect)
    assert "WITH ind_mean AS" in sql
    assert "AVG(x._v) OVER" in sql
    print("   ✓ cs_industry_neutralize SQL generated successfully")
    print(f"   SQL length: {len(sql)} chars")

    # 9. panel_residual
    print("\n9. Testing panel_residual...")
    sql = panel_residual_sql(test_input, test_input, dialect=dialect)
    assert "WITH joined AS" in sql
    assert "global_stats AS" in sql
    print("   ✓ panel_residual SQL generated successfully")
    print(f"   SQL length: {len(sql)} chars")

    # 10. cs_factor_score
    print("\n10. Testing cs_factor_score...")
    sql = cs_factor_score_sql(
        [test_input, test_input, test_input],
        weights=[0.5, 0.3, 0.2],
        standardize=True,
        dialect=dialect
    )
    assert "WITH" in sql
    assert "f0 AS" in sql
    assert "f1 AS" in sql
    assert "f2 AS" in sql
    print("   ✓ cs_factor_score SQL generated successfully")
    print(f"   SQL length: {len(sql)} chars")

    # 列出所有算子
    print("\n" + "=" * 70)
    print("Available operators:")
    for op in list_advanced_sql_operators():
        print(f"  • {op}")

    print("\n" + "=" * 70)
    print("✓ All validations passed!")
    print("=" * 70)

    return True


def test_dialect_switching():
    """测试方言切换。"""
    print("\n" + "=" * 70)
    print("Testing dialect switching...")
    print("=" * 70)

    test_input = "SELECT ts, inst, _v FROM test_data"

    # DuckDB
    sql_duck = cs_zscore_sql(test_input, dialect=SqlDialect.DUCKDB)
    assert "stddev_samp" in sql_duck
    print("✓ DuckDB dialect: uses stddev_samp")

    # ClickHouse
    sql_click = cs_zscore_sql(test_input, dialect=SqlDialect.CLICKHOUSE)
    assert "stddevSamp" in sql_click or "multiIf" in sql_click
    print("✓ ClickHouse dialect: uses ClickHouse functions")

    print("\n✓ Dialect switching works correctly")


def show_example_sql():
    """显示示例 SQL。"""
    print("\n" + "=" * 70)
    print("Example SQL output:")
    print("=" * 70)

    test_input = "SELECT ts, inst, _v FROM test_data"

    print("\n--- cs_zscore ---")
    sql = cs_zscore_sql(test_input, dialect=SqlDialect.DUCKDB)
    print(sql[:500] + "..." if len(sql) > 500 else sql)

    print("\n--- ts_seasonal_diff ---")
    sql = ts_seasonal_diff_sql(test_input, seasonal_period=12, dialect=SqlDialect.DUCKDB)
    print(sql)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        # 运行验证
        validate_sql_generation()
        test_dialect_switching()
        show_example_sql()

        print("\n" + "=" * 70)
        print("SUCCESS: All validations passed!")
        print("=" * 70)
        sys.exit(0)

    except Exception as e:
        print("\n" + "=" * 70)
        print(f"ERROR: Validation failed!")
        print(f"  {type(e).__name__}: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)
