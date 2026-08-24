"""仅导出 candidate_pool（不跑训练）。可从 pool JSON 日志或当前内存池导出。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from shared.alphagen.data.expression import Feature, Ref
from shared.alphagen_qlib.stock_data import FeatureType
from alphaprobe.delivery.config import ExperimentConfig
from alphaprobe.delivery.exporter import DiskV1DeliveryExporter
from alphaprobe.fe_bridge.bootstrap import enable_factor_engine_evaluation
from alphaprobe.fe_bridge.stock_data import FactorEngineStockData


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 disk.v1 到 candidate_pool")
    parser.add_argument(
        "--experiment_config",
        default=str(ROOT / "configs" / "experiment_ashare_pv.yaml"),
    )
    parser.add_argument(
        "--pool_json",
        help="AlphaKnowledgeLogger 产出的 pool_*.json（含 exprs 字符串）",
    )
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()

    if not args.pool_json:
        raise SystemExit("请指定 --pool_json（训练日志里的 pool_*.json）")

    experiment = ExperimentConfig.from_yaml(args.experiment_config)
    enable_factor_engine_evaluation()

    pool_data = json.loads(Path(args.pool_json).read_text(encoding="utf-8"))
    expr_strings = pool_data.get("exprs") or []

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    label_days = int((experiment.raw.get("mining") or {}).get("label_days", 20))
    # 与 runner 一致：vwap→vwap 远期收益
    vwap = Feature(FeatureType.VWAP)
    target = Ref(vwap, -label_days) / vwap - 1

    exporter = DiskV1DeliveryExporter(experiment)
    exporter.write_config(finalize=False)

    from shared.alphagen.data.tree import ExpressionParser
    from alphaprobe.fe_bridge.expr_parse import parse_mining_expression

    parser_expr = ExpressionParser()
    for item in expr_strings:
        if not item:
            continue
        expr = parse_mining_expression(str(item), parser_expr)
        from alphaprobe.fe_bridge.metrics import evaluate_all_period_metrics

        metrics = evaluate_all_period_metrics(expr, target, experiment, device)
        exporter.export_candidate(expr, description="", metrics=metrics)

    out = exporter.write_config(finalize=True)
    print("config:", out)
    print("campaign_dir:", exporter.campaign_dir)


if __name__ == "__main__":
    main()
