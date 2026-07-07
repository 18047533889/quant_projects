from __future__ import annotations

from pathlib import Path
from typing import Any

from factor_layer.alphapurify import pipeline as _base_pipeline
from factor_layer.alphapurify.Exposures_new import PortfolioExposures, PureExposures
from factor_layer.alphapurify.config import AlphaPurifyAdapterConfig, load_config


def run_pipeline(
    config: AlphaPurifyAdapterConfig,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    original_portfolio = _base_pipeline.PortfolioExposures
    original_pure = _base_pipeline.PureExposures
    try:
        _base_pipeline.PortfolioExposures = PortfolioExposures
        _base_pipeline.PureExposures = PureExposures
        return _base_pipeline.run_pipeline(config, config_path=config_path)
    finally:
        _base_pipeline.PortfolioExposures = original_portfolio
        _base_pipeline.PureExposures = original_pure


def run_from_config(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    return run_pipeline(config, config_path=config_path)
