#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
factorengine + dataaccess + quant_evaluator 集成示例

数据流：
  ① data_access   → 读行情 (close / high / low / volume / vwap / amount)
  ② factor_engine → Python code / DSL → 编译 Factor → 落盘 parquet
  ③ quant_evaluator → 从 parquet 读因子值 + 收益 → 算 RankIC / Sharpe / MDD
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/sunhaiwei/quant_projects")
sys.path.insert(0, "/home/sunhaiwei/quant_projects/factor_engine")

# ============================================================
# 1. data_access 加载
# ============================================================
from factor_engine.storage.data_access_loader import data_access_identity

print("[1] data_access 加载")
try:
    pkg_path, version, build_hash = data_access_identity()
    print(f"    path  = {pkg_path}")
    print(f"    ver   = {version}")
    print(f"    hash  = {build_hash}")
except ImportError:
    print("    [WARN] data_access 未安装；将直接读本地 parquet")
    from pathlib import Path
    import pandas as pd
    # 兜底: 直接读 parquet
    DATA_DIR = Path.home() / "cos_data" / "StockDailyBar"
    print(f"    fallback: {DATA_DIR} (共 {len(list(DATA_DIR.glob('*.parquet')))} 天)")


# ============================================================
# 2. factor_engine: 把 JSON 里的 Python code 物化成 parquet
# ============================================================
from factor_engine.api.factor import Factor
from factor_engine.expr.base import Expr, CleanedCall, Col  # noqa: F401

print("\n[2] factor_engine: 物化示例")
print("""
    # 真实用法 (factorengine 编译 DSL 到 PlanNode 然后 backend 执行):
    from factor_engine.api.factor import Factor
    from factor_engine.expr.base import CleanedCall, Col
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.runtime.materialize_service import materialize_factor

    factor = Factor(
        name='my_vol_asym',
        expr=CleanedCall(
            op='sign',
            args=[
                CleanedCall(op='ts_delta', args=[Col('close'), 1]),
                # ...
            ]
        ),
        freq='1d',
        description='日收益率正负对比因子'
    )

    # 编译 + 物化
    engine = FactorEngine(backend='polars_long')
    plan = engine.compile(factor)
    df = materialize_factor(plan, materializer=ParquetMaterializer(...))
""")


# ============================================================
# 3. quant_evaluator: 评估因子 (从已落 parquet 读因子值 + close)
# ============================================================
from quant_evaluator.api import EvaluationRequest, EvaluationBundle, MetricValue
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.ic import compute_daily_ic, compute_mean_ic
from quant_evaluator.metrics.portfolio_stats import compute_sharpe_ratio, compute_maximum_drawdown
from quant_evaluator.metrics.turnover import estimate_turnover_from_ranks

print("\n[3] quant_evaluator: 评估示例")
print("""
    # 评估 1 个因子: factor_values (T,N,1) + forward_ret (T,N)
    fb = FactorBatch(
        factor_ids=['my_vol_asym'],
        time_axis=AxisRef('t', 'date', T),
        asset_axis=AxisRef('a', 'symbol', N),
        values=factor_values_3d  # shape (T, N, 1)
    )
    lb = LabelBundle(
        target_id='ret_1d',
        values=forward_returns,  # (T, N)
        horizon=1,
        decision_time=...,
    )
    ic_series, valid_counts = compute_daily_ic(fb, lb, method='spearman', min_assets=20)
    mean_ic, ic_std = compute_mean_ic(ic_series)
    sharpe = compute_sharpe_ratio(ls_returns, periods_per_year=252)
    mdd = compute_maximum_drawdown(ls_returns)[0]
""")


# ============================================================
# 4. 本项目实际用法 (用我们已有的 _compute_factor_metrics)
# ============================================================
print("\n[4] 本项目: rebuild_factor_detail_pages._compute_factor_metrics")
print("    已使用 quant_evaluator 的 6 个算子:")
print("      - _spearman_rank_correlation (日 IC)")
print("      - compute_mean_ic / compute_ic_std (IC 摘要)")
print("      - compute_sharpe_ratio (LS Sharpe)")
print("      - compute_maximum_drawdown (LS MDD)")
print("      - compute_win_rate (日胜率)")
print("      - estimate_turnover_from_ranks (Top10% 换手率)")

# 直接 demo
import json
from pathlib import Path
from jobs.rebuild_factor_detail_pages import _compute_factor_metrics, QE_AVAILABLE

print(f"\n[4b] QE_AVAILABLE = {QE_AVAILABLE}")
factor_dir = Path("/home/sunhaiwei/weekly_backtest_output/factor_values.parquet")
if not factor_dir.exists():
    # 用上次落值的另一份
    factor_dir = Path("/home/sunhaiwei/weekly_backtest_output/factor_values.parquet")

# 找几个 _flipped 因子做演示
sample_json = Path("/home/sunhaiwei/quant_projects/factor_delivery_converted/factors_combined/factor_vol_volume_asym_ewma.json")
if sample_json.exists():
    with open(sample_json) as f:
        d = json.load(f)
    print(f"  示例: {d['factor_name']}")
    print(f"  formula = {d.get('fe_formula') or d.get('lqtp_formula')}")
    print(f"  code (前 200 字) = {d.get('code','')[:200]}...")