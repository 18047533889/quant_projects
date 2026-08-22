"""Verify an accelerated accurate backtest against a frozen Python baseline.

Example
-------
python scripts/verify_accurate_parity.py \
    examples/configs/B002_meanvar_1D.yaml \
    tests/baselines/B002_meanvar_1D_python
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent))
sys.path.insert(0, str(PROJECT_ROOT / "vectorbt"))

from vectorbt_qs.mvp.engine.runner import run_backtest


def _assert_exact_frame(
    name: str,
    actual: pd.DataFrame,
    expected: pd.DataFrame,
) -> None:
    """Compare semantic values exactly while ignoring parquet dtype upgrades."""
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_index_type=False,
            check_column_type=False,
            check_freq=False,
            check_exact=True,
        )
    except AssertionError as exc:
        raise AssertionError(f"{name} 与 Python 基线不一致\n{exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="逐日、逐订单验证 accurate 加速引擎与冻结基线完全一致"
    )
    parser.add_argument("config", type=Path, help="回测 YAML 配置")
    parser.add_argument("baseline_dir", type=Path, help="冻结基线目录")
    parser.add_argument(
        "--planner-engine",
        choices=("numba", "python"),
        default="numba",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="验证报告路径；默认写入基线目录 verification_<engine>.json",
    )
    parser.add_argument(
        "--allow-legacy-contract",
        action="store_true",
        help="仅审计历史代码时允许使用旧复权价格基线",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    baseline_dir = args.baseline_dir.resolve()
    metadata_path = baseline_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("contract") == "legacy_adjusted_price_v1"
            and not args.allow_legacy_contract
        ):
            raise RuntimeError(
                "该目录是 legacy_adjusted_price_v1 历史基线，"
                "不能验证 standard_accurate_v2；"
                "仅在审计对应历史代码时使用 --allow-legacy-contract"
            )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    positions_path = Path(config["input"]["positions"])
    if not positions_path.is_absolute():
        positions_path = (config_path.parent / positions_path).resolve()
    targets = pd.read_parquet(positions_path)

    backtest_config = dict(config.get("backtest", {}))
    backtest_config["planner_engine"] = args.planner_engine
    started = time.perf_counter()
    portfolio = run_backtest(
        config.get("market", "ashare"),
        targets,
        config=backtest_config,
    )
    elapsed = time.perf_counter() - started

    actual_nav = pd.DataFrame(
        {
            "nav": portfolio.value(),
            "cash": portfolio.cash(),
            "return": portfolio.returns(),
        }
    )
    actual_orders = portfolio.orders.records_readable.reset_index(drop=True)
    actual_assets = portfolio.assets().iloc[-1].rename("shares").to_frame()
    actual_log = portfolio._qs_execution_log.reset_index(drop=True)

    expected_nav = pd.read_parquet(baseline_dir / "nav.parquet")
    expected_orders = pd.read_parquet(baseline_dir / "orders.parquet")
    expected_assets = pd.read_parquet(baseline_dir / "final_assets.parquet")
    expected_log = pd.read_parquet(baseline_dir / "execution_log.parquet")

    _assert_exact_frame("NAV", actual_nav, expected_nav)
    _assert_exact_frame("订单", actual_orders, expected_orders)
    _assert_exact_frame("期末持仓", actual_assets, expected_assets)
    _assert_exact_frame("执行日志", actual_log, expected_log)

    report = {
        "status": "passed",
        "config": str(config_path),
        "baseline_dir": str(baseline_dir),
        "planner_engine": args.planner_engine,
        "elapsed_seconds": elapsed,
        "nav_rows": len(actual_nav),
        "order_count": len(actual_orders),
        "execution_log_count": len(actual_log),
        "start_nav": float(actual_nav["nav"].iloc[0]),
        "end_nav": float(actual_nav["nav"].iloc[-1]),
        "max_nav_abs_diff": float(
            np.max(np.abs(actual_nav["nav"] - expected_nav["nav"]))
        ),
        "max_cash_abs_diff": float(
            np.max(np.abs(actual_nav["cash"] - expected_nav["cash"]))
        ),
        "max_return_abs_diff": float(
            np.max(np.abs(actual_nav["return"] - expected_nav["return"]))
        ),
        "orders_exact": True,
        "final_assets_exact": True,
        "execution_log_exact": True,
    }
    report_path = args.report
    if report_path is None:
        report_path = baseline_dir / f"verification_{args.planner_engine}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
