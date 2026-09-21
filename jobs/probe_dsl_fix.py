#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探测 FE 算子参数绑定：为 3 个 PlanParamError 因子设计等价改写。"""
import os, sys, warnings
from pathlib import Path
ROOT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ASHARE_PARQUET_ROOT", str(Path.home() / "cos_data"))
os.environ.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")
warnings.filterwarnings("ignore")

from factor_engine.api.dsl_parser import parse_expr
from factor_engine.ir.analyzer import Analyzer

TESTS = {
    # 基线：能跑通的同类写法
    "base_slope": "ts_regression_slope(open, high, 60)",
    # 因子 5cf157f2 原式（第 2 参是布尔比较）
    "orig_bool": "ts_regression_slope(pe_ratio, (amount > 10), 60)",
    "fix_where": "ts_regression_slope(pe_ratio, where(amount > 10, 1, 0), 60)",
    "fix_sig_sqrt": "ts_regression_slope(pe_ratio, signed_sqrt((amount > 10)), 60)",
    "fix_isnan_bool": "ts_regression_slope(pe_ratio, where(amount > 10, 1.0, 0.0), 60)",
    # 因子 f70ac242 原式（第 2 参是 rank(...)）
    "orig_rank": "ts_regression_slope(circulating_market_cap, rank(is_nan(eps)), 10)",
    "fix_rank_where": "ts_regression_slope(circulating_market_cap, rank(where(is_nan(eps), 1, 0)), 10)",
    "fix_rank_1_0": "ts_regression_slope(circulating_market_cap, rank(where(is_nan(eps), 1.0, 0.0)), 10)",
    "fix_rank_iff": "ts_regression_slope(circulating_market_cap, rank(iff(is_nan(eps), 1, 0)), 10)",
    # 因子 3c69f233：winsorize 两参形式
    "orig_win": "(high + winsorize(ts_regression_slope(circulating_market_cap, turnover_ratio, 30), 1))",
    "fix_win_3arg": "(high + winsorize(ts_regression_slope(circulating_market_cap, turnover_ratio, 30), 0.01, 0.99))",
    "fix_win_005_095": "(high + winsorize(ts_regression_slope(circulating_market_cap, turnover_ratio, 30), 0.05, 0.95))",
    "fix_win_p": "(high + winsorize(ts_regression_slope(circulating_market_cap, turnover_ratio, 30), 0.01))",
    # 额外：退化因子的 cs_resid 行为
    "cs_resid_self": "rank(scale(cs_resid(turnover_ratio, turnover_ratio), 100))",
    "cs_resid_other": "rank(scale(cs_resid(turnover_ratio, amount), 100))",
    # 全 NaN 因子
    "orig_allnan": "(eps - ts_delta(ts_pct(is_nan(net_profit), 60), 1))",
    "fix_allnan": "(eps - ts_delta(ts_pct(where(is_nan(net_profit), 1, 0), 60), 1))",
}

analyzer = Analyzer(production=False)
for name, expr in TESTS.items():
    try:
        f = parse_expr(expr, surface="lqtp", dialect="lqtp")
        analyzer.lower(f)
        print(f"OK    {name:18s} {expr[:90]}")
    except Exception as e:
        print(f"FAIL  {name:18s} {type(e).__name__}: {str(e)[:130]}")
        print(f"      expr={expr[:110]}")

# 打印算子签名
print("\n=== ts_regression_slope / winsorize 参数规格 ===")
try:
    from factor_engine.ir.registry import get_operator  # 可能路径不同
except Exception:
    pass
for mod in ["factor_engine.ir.registry", "factor_engine.operators.registry",
            "factor_engine.registry", "factor_engine.api.registry"]:
    try:
        m = __import__(mod, fromlist=["*"])
        fn = getattr(m, "OPERATOR_REGISTRY", None) or getattr(m, "REGISTRY", None) or getattr(m, "OPERATORS", None)
        if fn:
            for op in ["ts_regression_slope", "winsorize", "cs_resid"]:
                spec = None
                try:
                    spec = fn[op]
                except Exception:
                    try:
                        spec = fn.get(op)
                    except Exception:
                        pass
                print(f"[{mod}] {op}: {str(spec)[:400]}")
            break
    except Exception as e:
        print(f"[{mod}] 不可用 {e}")
