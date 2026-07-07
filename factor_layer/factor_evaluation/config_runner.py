from __future__ import annotations

from pathlib import Path

from factor_layer.factor_evaluation.config import FactorEvaluationConfig, load_config
from factor_layer.factor_evaluation.pipeline import evaluate_factor, save_results


def run_from_config(config_path: str | Path) -> dict[str, object]:
    config: FactorEvaluationConfig = load_config(config_path)
    result = evaluate_factor(config)
    if config.output.save_artifacts:
        save_results(result, config)
    return result