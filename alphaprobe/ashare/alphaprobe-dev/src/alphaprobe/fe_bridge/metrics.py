"""按 mining_config 各段计算因子 metrics。"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from shared.alphagen.data.calculator import AlphaCalculator
from shared.alphagen.data.expression import Expression
from shared.alphagen_qlib.calculator import QLibStockDataCalculator
from alphaprobe.delivery.config import ExperimentConfig
from alphaprobe.fe_bridge.stock_data import FactorEngineStockData


def _calc_ic_ir(calc: AlphaCalculator, expr: Expression) -> Tuple[float, float, float]:
    ic, rank_ic = calc.calc_single_all_ret(expr)
    value = calc._calc_alpha(expr)  # type: ignore[attr-defined]
    target = calc.target_value  # type: ignore[attr-defined]
    raw_ic = calc._calc_raw_IC(value, target)  # type: ignore[attr-defined]
    icir = float(raw_ic.mean().item() / (raw_ic.std().item() + 1e-6))
    return float(ic), icir, float(rank_ic)


def evaluate_period_metrics(
    expr: Expression,
    target: Expression,
    experiment: ExperimentConfig,
    period_key: str,
    device: torch.device,
) -> dict[str, float]:
    periods = experiment.delivery.mining_config
    start, end = periods[period_key][0], periods[period_key][1]
    data = FactorEngineStockData(
        instrument=experiment.data.instruments,
        start_time=start,
        end_time=end,
        device=device,
        max_files=experiment.data.max_files,
        data_source_config=experiment.factor_engine_data_source_config(
            start_date=start,
            end_date=end,
        ),
    )
    calc = QLibStockDataCalculator(data, target)
    ic, icir, rank_ic = _calc_ic_ir(calc, expr)
    return {"ic": ic, "icir": icir, "rank_ic": rank_ic}


def evaluate_all_period_metrics(
    expr: Expression,
    target: Expression,
    experiment: ExperimentConfig,
    device: torch.device,
) -> Dict[str, dict[str, float]]:
    out: Dict[str, dict[str, float]] = {}
    for key in ("train_period", "valid_period", "test_period"):
        segment = key.replace("_period", "")
        out[segment] = evaluate_period_metrics(expr, target, experiment, key, device)
    return out
