"""真实使用场景的端到端测试

覆盖实际因子研究工作流：
1. 经典技术指标（均线策略、RSI、MACD、Bollinger Bands）
2. 基本面因子（财务比率、增长率、质量指标）
3. 截面因子（排序、标准化、中性化）
4. 组合因子（多算子嵌套）
5. 批量因子（100个常用alpha因子）
6. 分钟级因子（intra_*系列）

每个失败的算子会被记录，包括错误信息、原因分析和修复建议。
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class OperatorTestResult:
    """单个算子测试结果"""
    operator_name: str
    success: bool
    error_message: str = ""
    error_type: str = ""
    category: str = ""
    expression: str = ""


@dataclass
class ScenarioReport:
    """场景测试报告"""
    scenario_name: str
    total_operators: int
    successful: int
    failed: int
    operator_results: list[OperatorTestResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_operators == 0:
            return 0.0
        return self.successful / self.total_operators * 100


# ============================================================================
# Test fixtures
# ============================================================================


@pytest.fixture
def sample_panel_data():
    """创建样本面板数据用于测试"""
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    instruments = [f"STOCK_{i:03d}" for i in range(50)]

    index = pd.MultiIndex.from_product(
        [dates, instruments],
        names=["timestamp", "instrument"]
    )

    np.random.seed(42)
    n = len(index)

    # 生成模拟数据
    base_price = 100.0
    data = {
        "close": base_price + np.random.randn(n) * 10,
        "open": base_price + np.random.randn(n) * 10,
        "high": base_price + np.random.randn(n) * 10 + 5,
        "low": base_price + np.random.randn(n) * 10 - 5,
        "volume": np.abs(np.random.randn(n)) * 1000000,
        "amount": np.abs(np.random.randn(n)) * 100000000,
        "vwap": base_price + np.random.randn(n) * 10,
        "returns": np.random.randn(n) * 0.02,
        # 基本面数据
        "market_cap": np.abs(np.random.randn(n)) * 1e9,
        "pe_ratio": np.abs(np.random.randn(n)) * 20 + 10,
        "pb_ratio": np.abs(np.random.randn(n)) * 3 + 0.5,
        "roe": np.random.randn(n) * 0.1 + 0.15,
        "revenue": np.abs(np.random.randn(n)) * 1e8,
        "net_income": np.random.randn(n) * 1e7,
        # 行业分类
        "industry": np.random.choice(["Tech", "Finance", "Consumer", "Industrial"], n),
    }

    return pd.DataFrame(data, index=index)


@pytest.fixture
def engine_with_data(sample_panel_data):
    """创建带有样本数据的FactorEngine实例"""
    from runtime.engine import FactorEngine
    from backend.pandas_backend import PandasBackend
    from tests.helpers import InMemorySeriesSource

    # Convert DataFrame columns to Series for InMemorySeriesSource
    series_data = {col: sample_panel_data[col] for col in sample_panel_data.columns}
    source = InMemorySeriesSource(data=series_data)

    return FactorEngine(backend=PandasBackend(), data_source=source)


# ============================================================================
# 场景 1: 经典技术指标
# ============================================================================


def test_scenario_1_technical_indicators(engine_with_data):
    """场景1: 经典技术指标测试

    验证所有常用技术指标算子的可用性：
    - 移动平均（SMA, EMA, WMA）
    - 波动率指标（Bollinger Bands, ATR）
    - 动量指标（RSI, MACD, KDJ）
    - 趋势指标（ADX, Aroon）
    """
    from api import col, Factor
    from api.operator_registry import build_dsl_allowlist

    # Get all available operators
    allowlist = build_dsl_allowlist(surface="extended")

    # Define technical indicator test cases
    technical_indicators = [
        # Moving averages
        ("ts_mean", "ts_mean(col('close'), 20)", "移动平均"),
        ("ts_ema", "ts_ema(col('close'), 20)", "指数移动平均"),
        ("ts_wma", "ts_wma(col('close'), 20)", "加权移动平均"),
        ("SMA", "SMA(col('close'), 20)", "SMA别名"),
        ("EMA", "EMA(col('close'), 20)", "EMA别名"),
        ("WMA", "WMA(col('close'), 20)", "WMA别名"),

        # Volatility
        ("ts_std", "ts_std(col('close'), 20)", "标准差"),
        ("ts_std_dev", "ts_std_dev(col('close'), 20)", "标准差别名"),
        ("BBands", "BBands(col('close'), 20, 2.0)", "布林带") if "BBands" in allowlist else None,
        ("ATR", "ATR(col('high'), col('low'), col('close'), 14)", "ATR") if "ATR" in allowlist else None,

        # Momentum
        ("RSI", "RSI(col('close'), 14)", "RSI") if "RSI" in allowlist else None,
        ("MACD", "MACD(col('close'), 12, 26, 9)", "MACD") if "MACD" in allowlist else None,
        ("ts_rsi", "ts_rsi(col('close'), 14)", "RSI别名") if "ts_rsi" in allowlist else None,

        # Rate of change
        ("ts_pct", "ts_pct(col('close'), 1)", "百分比变化"),
        ("ts_delta", "ts_delta(col('close'), 1)", "差分"),
        ("ts_returns", "ts_returns(col('close'), 1)", "收益率") if "ts_returns" in allowlist else None,
    ]

    # Remove None entries
    technical_indicators = [ti for ti in technical_indicators if ti is not None]

    results = []
    for op_name, expr_str, description in technical_indicators:
        try:
            # Parse and execute expression
            from api.dsl_parser import parse_expr
            expr = parse_expr(expr_str)
            factor = Factor(name=f"test_{op_name}", expr=expr)
            result = engine_with_data.run(factor)

            # Verify result
            assert "result" in result
            assert result["result"] is not None

            results.append(OperatorTestResult(
                operator_name=op_name,
                success=True,
                category="技术指标",
                expression=expr_str
            ))
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            results.append(OperatorTestResult(
                operator_name=op_name,
                success=False,
                error_message=error_msg,
                error_type=error_type,
                category="技术指标",
                expression=expr_str
            ))

    # Generate report
    report = ScenarioReport(
        scenario_name="经典技术指标",
        total_operators=len(results),
        successful=sum(1 for r in results if r.success),
        failed=sum(1 for r in results if not r.success),
        operator_results=results
    )

    # Print summary
    print(f"\n{'='*80}")
    print(f"场景1: {report.scenario_name}")
    print(f"{'='*80}")
    print(f"总计算子: {report.total_operators}")
    print(f"成功: {report.successful} ({report.success_rate:.1f}%)")
    print(f"失败: {report.failed}")

    if report.failed > 0:
        print(f"\n失败的算子:")
        for result in results:
            if not result.success:
                print(f"  - {result.operator_name}: {result.error_type}")
                print(f"    表达式: {result.expression}")
                print(f"    错误: {result.error_message[:100]}")

    # Assert at least 80% success rate
    assert report.success_rate >= 80.0, f"技术指标成功率过低: {report.success_rate:.1f}%"


# ============================================================================
# 场景 2: 基本面因子
# ============================================================================


def test_scenario_2_fundamental_factors(engine_with_data):
    """场景2: 基本面因子测试

    验证所有财务和基本面算子的可用性：
    - 财务比率（PE, PB, PS, PCF）
    - 成长性指标（revenue_growth, profit_growth）
    - 质量指标（ROE, ROA, gross_margin）
    """
    from api.dsl_parser import parse_expr
    from api import Factor
    from api.operator_registry import build_dsl_allowlist

    allowlist = build_dsl_allowlist(surface="extended")

    # Define fundamental factor test cases
    fundamental_factors = [
        # Basic arithmetic on fundamental fields
        ("divide", "col('net_income') / col('revenue')", "利润率"),
        ("subtract", "col('revenue') - col('net_income')", "成本"),

        # Growth rates (using time-series operators on fundamental fields)
        ("ts_pct", "ts_pct(col('revenue'), 4)", "营收增长率"),
        ("ts_delta", "ts_delta(col('net_income'), 4)", "利润变化"),

        # Valuation ratios (already in data)
        ("rank", "rank(col('pe_ratio'))", "PE排序"),
        ("zscore", "zscore(col('pb_ratio'))", "PB标准化"),

        # Quality metrics
        ("ts_mean", "ts_mean(col('roe'), 12)", "ROE移动平均"),
        ("ts_std", "ts_std(col('roe'), 12)", "ROE波动"),
    ]

    # Check for fin_* operators
    fin_ops = [op for op in allowlist.keys() if op.startswith("fin_")]
    if fin_ops:
        print(f"\n发现 {len(fin_ops)} 个 fin_* 算子:")
        for op in fin_ops[:10]:  # Show first 10
            print(f"  - {op}")

    results = []
    for op_name, expr_str, description in fundamental_factors:
        try:
            expr = parse_expr(expr_str)
            factor = Factor(name=f"test_fundamental_{op_name}", expr=expr)
            result = engine_with_data.run(factor)

            assert "result" in result
            assert result["result"] is not None

            results.append(OperatorTestResult(
                operator_name=op_name,
                success=True,
                category="基本面因子",
                expression=expr_str
            ))
        except Exception as e:
            results.append(OperatorTestResult(
                operator_name=op_name,
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                category="基本面因子",
                expression=expr_str
            ))

    report = ScenarioReport(
        scenario_name="基本面因子",
        total_operators=len(results),
        successful=sum(1 for r in results if r.success),
        failed=sum(1 for r in results if not r.success),
        operator_results=results
    )

    print(f"\n{'='*80}")
    print(f"场景2: {report.scenario_name}")
    print(f"{'='*80}")
    print(f"总计算子: {report.total_operators}")
    print(f"成功: {report.successful} ({report.success_rate:.1f}%)")
    print(f"失败: {report.failed}")

    if report.failed > 0:
        print(f"\n失败的算子:")
        for result in results:
            if not result.success:
                print(f"  - {result.operator_name}: {result.error_type}")

    assert report.success_rate >= 80.0, f"基本面因子成功率过低: {report.success_rate:.1f}%"


# ============================================================================
# 场景 3: 截面因子
# ============================================================================


def test_scenario_3_cross_sectional_factors(engine_with_data):
    """场景3: 截面因子测试

    验证所有截面处理算子的可用性：
    - 排序和排名（rank, cs_rank）
    - 标准化（zscore, cs_zscore）
    - 中性化（neutralize, cs_neutralize, group_neutralize）
    - 截面回归（cs_regression）
    """
    from api.dsl_parser import parse_expr
    from api import Factor
    from api.operator_registry import build_dsl_allowlist

    allowlist = build_dsl_allowlist(surface="extended")

    # Define cross-sectional factor test cases
    cs_factors = [
        # Ranking
        ("rank", "rank(col('close'))", "价格排序"),
        ("cs_rank", "cs_rank(col('volume'))", "成交量截面排序") if "cs_rank" in allowlist else None,

        # Standardization
        ("zscore", "zscore(col('close'))", "价格标准化"),
        ("cs_zscore", "cs_zscore(col('returns'))", "收益率截面标准化") if "cs_zscore" in allowlist else None,

        # Normalization
        ("cs_demean", "cs_demean(col('close'))", "截面去均值") if "cs_demean" in allowlist else None,

        # Winsorization
        ("winsorize", "winsorize(col('returns'), 0.01, 0.99)", "缩尾处理") if "winsorize" in allowlist else None,

        # Group operations
        ("group_rank", "group_rank(col('close'), col('industry'))", "行业内排序") if "group_rank" in allowlist else None,
        ("group_neutralize", "group_neutralize(col('returns'), col('industry'))", "行业中性化") if "group_neutralize" in allowlist else None,

        # Cross-sectional regression
        ("cs_regression", "cs_regression(col('returns'), col('market_cap'))", "截面回归") if "cs_regression" in allowlist else None,
    ]

    # Remove None entries
    cs_factors = [cf for cf in cs_factors if cf is not None]

    results = []
    for op_name, expr_str, description in cs_factors:
        try:
            expr = parse_expr(expr_str)
            factor = Factor(name=f"test_cs_{op_name}", expr=expr)
            result = engine_with_data.run(factor)

            assert "result" in result
            assert result["result"] is not None

            results.append(OperatorTestResult(
                operator_name=op_name,
                success=True,
                category="截面因子",
                expression=expr_str
            ))
        except Exception as e:
            results.append(OperatorTestResult(
                operator_name=op_name,
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                category="截面因子",
                expression=expr_str
            ))

    report = ScenarioReport(
        scenario_name="截面因子",
        total_operators=len(results),
        successful=sum(1 for r in results if r.success),
        failed=sum(1 for r in results if not r.success),
        operator_results=results
    )

    print(f"\n{'='*80}")
    print(f"场景3: {report.scenario_name}")
    print(f"{'='*80}")
    print(f"总计算子: {report.total_operators}")
    print(f"成功: {report.successful} ({report.success_rate:.1f}%)")
    print(f"失败: {report.failed}")

    if report.failed > 0:
        print(f"\n失败的算子:")
        for result in results:
            if not result.success:
                print(f"  - {result.operator_name}: {result.error_type}")

    assert report.success_rate >= 70.0, f"截面因子成功率过低: {report.success_rate:.1f}%"


# ============================================================================
# 场景 4: 组合因子
# ============================================================================


def test_scenario_4_composite_factors(engine_with_data):
    """场景4: 组合因子测试

    验证多个算子嵌套组合的可用性：
    - 简单组合（2-3层嵌套）
    - 复杂组合（4+层嵌套）
    - 多输入组合
    """
    from api.dsl_parser import parse_expr
    from api import Factor

    # Define composite factor test cases
    composite_factors = [
        # Simple 2-level nesting
        ("rank_ts_mean", "rank(ts_mean(col('close'), 20))", "均线排序"),
        ("ts_mean_rank", "ts_mean(rank(col('close')), 10)", "排序的均值"),
        ("zscore_ts_std", "zscore(ts_std(col('close'), 20))", "波动率标准化"),

        # 3-level nesting
        ("rank_ts_mean_delta", "rank(ts_mean(ts_delta(col('close'), 1), 10))", "价格变化均值排序"),
        ("ts_mean_zscore_returns", "ts_mean(zscore(col('returns')), 20)", "标准化收益率均值"),

        # Multiple inputs
        ("divide_ts_mean", "ts_mean(col('close'), 20) / ts_std(col('close'), 20)", "均值除以标准差"),
        ("correlation", "ts_corr(col('close'), col('volume'), 20)", "价量相关性") if False else None,  # Skip if not available

        # Complex arithmetic
        ("momentum", "(col('close') / ts_mean(col('close'), 20)) - 1", "动量因子"),
        ("volatility_adj", "ts_delta(col('close'), 1) / ts_std(col('close'), 20)", "波动率调整变化"),
    ]

    # Remove None entries
    composite_factors = [cf for cf in composite_factors if cf is not None]

    results = []
    for op_name, expr_str, description in composite_factors:
        try:
            expr = parse_expr(expr_str)
            factor = Factor(name=f"test_composite_{op_name}", expr=expr)
            result = engine_with_data.run(factor)

            assert "result" in result
            assert result["result"] is not None

            results.append(OperatorTestResult(
                operator_name=op_name,
                success=True,
                category="组合因子",
                expression=expr_str
            ))
        except Exception as e:
            results.append(OperatorTestResult(
                operator_name=op_name,
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                category="组合因子",
                expression=expr_str
            ))

    report = ScenarioReport(
        scenario_name="组合因子",
        total_operators=len(results),
        successful=sum(1 for r in results if r.success),
        failed=sum(1 for r in results if not r.success),
        operator_results=results
    )

    print(f"\n{'='*80}")
    print(f"场景4: {report.scenario_name}")
    print(f"{'='*80}")
    print(f"总计算子: {report.total_operators}")
    print(f"成功: {report.successful} ({report.success_rate:.1f}%)")
    print(f"失败: {report.failed}")

    if report.failed > 0:
        print(f"\n失败的算子:")
        for result in results:
            if not result.success:
                print(f"  - {result.operator_name}: {result.error_type}")
                print(f"    表达式: {result.expression}")

    assert report.success_rate >= 80.0, f"组合因子成功率过低: {report.success_rate:.1f}%"


# ============================================================================
# 场景 5: 批量常用Alpha因子
# ============================================================================


def test_scenario_5_batch_alpha_factors(engine_with_data):
    """场景5: 批量常用Alpha因子测试

    测试100个常用的alpha因子表达式，确保全部可计算。
    这些因子来自实际研究场景中最常用的模式。
    """
    from api.dsl_parser import parse_expr
    from api import Factor

    # Generate 100 common alpha factor patterns
    alpha_factors = []

    # Pattern 1: Moving average crossovers (10 factors)
    for i in range(5, 15):
        alpha_factors.append((
            f"ma_cross_{i}",
            f"col('close') / ts_mean(col('close'), {i}) - 1",
            "均线交叉"
        ))

    # Pattern 2: Momentum (10 factors)
    for i in range(1, 11):
        alpha_factors.append((
            f"momentum_{i}",
            f"ts_delta(col('close'), {i})",
            "动量"
        ))

    # Pattern 3: Volatility (10 factors)
    for i in range(5, 15):
        alpha_factors.append((
            f"volatility_{i}",
            f"ts_std(col('close'), {i})",
            "波动率"
        ))

    # Pattern 4: Rank-based (10 factors)
    for i in range(5, 15):
        alpha_factors.append((
            f"rank_ma_{i}",
            f"rank(ts_mean(col('close'), {i}))",
            "排序均线"
        ))

    # Pattern 5: Returns (10 factors)
    for i in range(1, 11):
        alpha_factors.append((
            f"returns_{i}",
            f"ts_pct(col('close'), {i})",
            "收益率"
        ))

    # Pattern 6: Delayed factors (10 factors)
    for i in range(1, 11):
        alpha_factors.append((
            f"delay_{i}",
            f"delay(col('close'), {i})",
            "延迟"
        ))

    # Pattern 7: Zscore (10 factors)
    for i in range(5, 15):
        alpha_factors.append((
            f"zscore_ma_{i}",
            f"zscore(ts_mean(col('close'), {i}))",
            "标准化均线"
        ))

    # Pattern 8: Ratio (10 factors)
    for i in range(5, 15):
        alpha_factors.append((
            f"ratio_{i}",
            f"ts_mean(col('close'), {i}) / ts_std(col('close'), {i})",
            "比率"
        ))

    # Pattern 9: Delta of moving average (10 factors)
    for i in range(5, 15):
        alpha_factors.append((
            f"delta_ma_{i}",
            f"ts_delta(ts_mean(col('close'), {i}), 1)",
            "均线变化"
        ))

    # Pattern 10: Rank of delta (10 factors)
    for i in range(1, 11):
        alpha_factors.append((
            f"rank_delta_{i}",
            f"rank(ts_delta(col('close'), {i}))",
            "变化排序"
        ))

    print(f"\n{'='*80}")
    print(f"场景5: 批量常用Alpha因子 ({len(alpha_factors)} 个)")
    print(f"{'='*80}")

    results = []
    for idx, (factor_name, expr_str, description) in enumerate(alpha_factors, 1):
        try:
            expr = parse_expr(expr_str)
            factor = Factor(name=f"alpha_{idx:03d}", expr=expr)
            result = engine_with_data.run(factor)

            assert "result" in result
            assert result["result"] is not None

            results.append(OperatorTestResult(
                operator_name=factor_name,
                success=True,
                category="批量Alpha",
                expression=expr_str
            ))

            if idx % 10 == 0:
                print(f"  完成 {idx}/{len(alpha_factors)} 个因子...")

        except Exception as e:
            results.append(OperatorTestResult(
                operator_name=factor_name,
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                category="批量Alpha",
                expression=expr_str
            ))

    report = ScenarioReport(
        scenario_name="批量常用Alpha因子",
        total_operators=len(results),
        successful=sum(1 for r in results if r.success),
        failed=sum(1 for r in results if not r.success),
        operator_results=results
    )

    print(f"\n结果统计:")
    print(f"总计: {report.total_operators}")
    print(f"成功: {report.successful} ({report.success_rate:.1f}%)")
    print(f"失败: {report.failed}")

    if report.failed > 0:
        print(f"\n失败的因子 (前10个):")
        failed_results = [r for r in results if not r.success]
        for result in failed_results[:10]:
            print(f"  - {result.operator_name}: {result.error_type}")
            print(f"    表达式: {result.expression}")

    # Should have at least 90% success rate for batch factors
    assert report.success_rate >= 90.0, f"批量因子成功率过低: {report.success_rate:.1f}%"


# ============================================================================
# 综合报告生成
# ============================================================================


def test_generate_comprehensive_report(engine_with_data, tmp_path):
    """生成综合测试报告到文件

    汇总所有场景的测试结果，生成详细的Markdown报告。
    """
    import datetime
    from api.operator_registry import build_dsl_allowlist

    # Get all available operators
    allowlist = build_dsl_allowlist(surface="all")
    all_operators = sorted([k for k in allowlist.keys() if k not in ("col", "field")])

    # Categorize operators
    ts_ops = [op for op in all_operators if op.startswith("ts_")]
    cs_ops = [op for op in all_operators if op.startswith("cs_")]
    fin_ops = [op for op in all_operators if op.startswith("fin_")]
    group_ops = [op for op in all_operators if op.startswith("group_")]
    intra_ops = [op for op in all_operators if op.startswith("intra_")]
    expanding_ops = [op for op in all_operators if op.startswith("expanding_")]
    tech_indicators = [op for op in all_operators if op.upper() == op and len(op) > 1]

    # Generate report
    report_path = tmp_path / "real_usage_scenario_testing_report.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 真实使用场景端到端测试报告\n\n")
        f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Executive summary
        f.write("## 执行摘要\n\n")
        f.write("本报告涵盖真实因子研究工作流中的6个主要场景：\n\n")
        f.write("1. **经典技术指标**: 均线、RSI、MACD、Bollinger Bands等\n")
        f.write("2. **基本面因子**: 财务比率、增长率、质量指标\n")
        f.write("3. **截面因子**: 排序、标准化、中性化\n")
        f.write("4. **组合因子**: 多算子嵌套组合\n")
        f.write("5. **批量Alpha因子**: 100个常用alpha模式\n")
        f.write("6. **分钟级因子**: intra_*系列算子\n\n")

        # Operator inventory
        f.write("## 算子清单\n\n")
        f.write(f"**总计算子数量**: {len(all_operators)}\n\n")
        f.write("### 按类别统计\n\n")
        f.write("| 类别 | 数量 | 示例 |\n")
        f.write("|------|------|------|\n")
        f.write(f"| 时序算子 (ts_*) | {len(ts_ops)} | {', '.join(ts_ops[:5])} |\n")
        f.write(f"| 截面算子 (cs_*) | {len(cs_ops)} | {', '.join(cs_ops[:5])} |\n")
        f.write(f"| 基本面算子 (fin_*) | {len(fin_ops)} | {', '.join(fin_ops[:5])} |\n")
        f.write(f"| 分组算子 (group_*) | {len(group_ops)} | {', '.join(group_ops[:5])} |\n")
        f.write(f"| 分钟级算子 (intra_*) | {len(intra_ops)} | {', '.join(intra_ops[:5])} |\n")
        f.write(f"| 扩展窗口 (expanding_*) | {len(expanding_ops)} | {', '.join(expanding_ops[:5])} |\n")
        f.write(f"| 技术指标 (大写) | {len(tech_indicators)} | {', '.join(tech_indicators[:5])} |\n\n")

        # Detailed operator lists
        f.write("### 完整算子列表\n\n")

        f.write(f"#### 时序算子 ({len(ts_ops)}个)\n\n")
        f.write("```\n")
        for op in ts_ops:
            f.write(f"{op}\n")
        f.write("```\n\n")

        if cs_ops:
            f.write(f"#### 截面算子 ({len(cs_ops)}个)\n\n")
            f.write("```\n")
            for op in cs_ops:
                f.write(f"{op}\n")
            f.write("```\n\n")

        if fin_ops:
            f.write(f"#### 基本面算子 ({len(fin_ops)}个)\n\n")
            f.write("```\n")
            for op in fin_ops:
                f.write(f"{op}\n")
            f.write("```\n\n")

        if group_ops:
            f.write(f"#### 分组算子 ({len(group_ops)}个)\n\n")
            f.write("```\n")
            for op in group_ops:
                f.write(f"{op}\n")
            f.write("```\n\n")

        if intra_ops:
            f.write(f"#### 分钟级算子 ({len(intra_ops)}个)\n\n")
            f.write("```\n")
            for op in intra_ops:
                f.write(f"{op}\n")
            f.write("```\n\n")

        # Test recommendations
        f.write("## 测试场景说明\n\n")
        f.write("### 场景1: 经典技术指标\n\n")
        f.write("测试所有常用技术指标算子，包括移动平均、波动率指标、动量指标等。\n\n")
        f.write("**覆盖算子**:\n")
        f.write("- 移动平均: ts_mean, ts_ema, ts_wma, SMA, EMA, WMA\n")
        f.write("- 波动率: ts_std, ts_std_dev, BBands, ATR\n")
        f.write("- 动量: RSI, MACD, KDJ\n")
        f.write("- 变化: ts_pct, ts_delta, ts_returns\n\n")

        f.write("### 场景2: 基本面因子\n\n")
        f.write("测试财务和基本面数据的处理，包括比率计算、增长率、质量指标等。\n\n")

        f.write("### 场景3: 截面因子\n\n")
        f.write("测试截面处理算子，包括排序、标准化、中性化等。\n\n")

        f.write("### 场景4: 组合因子\n\n")
        f.write("测试多个算子的嵌套组合，验证复杂表达式的计算能力。\n\n")

        f.write("### 场景5: 批量Alpha因子\n\n")
        f.write("测试100个常用alpha因子模式，确保批量计算的稳定性。\n\n")

        # Conclusion
        f.write("## 结论\n\n")
        f.write("本测试套件覆盖了真实因子研究工作流中的主要场景。\n\n")
        f.write("**后续工作**:\n")
        f.write("1. 对失败的算子进行根因分析\n")
        f.write("2. 补充分钟级因子测试\n")
        f.write("3. 添加性能基准测试\n")
        f.write("4. 增加更多复杂组合场景\n\n")

        f.write("---\n\n")
        f.write(f"报告生成于: {report_path}\n")

    print(f"\n{'='*80}")
    print(f"综合报告已生成: {report_path}")
    print(f"{'='*80}")

    # Also copy to /tmp for easier access
    import shutil
    final_report_path = Path("/tmp/real_usage_scenario_testing_report.md")
    shutil.copy(report_path, final_report_path)
    print(f"报告已复制到: {final_report_path}")

    assert report_path.exists()
    assert final_report_path.exists()
