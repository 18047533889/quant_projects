"""Result containers for portfolio exposure analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd


@dataclass(slots=True)
class ExposureAnalysisResult:
    """All tabular outputs produced by one exposure-analysis run."""

    realized_weights: pd.DataFrame
    portfolio_exposure: pd.DataFrame
    portfolio_exposure_invested: pd.DataFrame
    portfolio_industry: pd.DataFrame
    coverage: pd.DataFrame
    benchmark_exposure: Optional[pd.DataFrame] = None
    benchmark_industry: Optional[pd.DataFrame] = None
    active_exposure: Optional[pd.DataFrame] = None
    active_industry: Optional[pd.DataFrame] = None
    factor_returns: Optional[pd.DataFrame] = None
    factor_contribution: Optional[pd.DataFrame] = None
    active_factor_contribution: Optional[pd.DataFrame] = None
    unexplained_return: Optional[pd.Series] = None
    active_unexplained_return: Optional[pd.Series] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def export(self, output: str | Path) -> Path:
        """Write stable parquet/CSV/JSON artifacts under ``output``."""
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        frames = {
            "realized_weights": self.realized_weights,
            "portfolio_style_exposure": self.portfolio_exposure,
            "portfolio_style_exposure_invested": self.portfolio_exposure_invested,
            "portfolio_industry_exposure": self.portfolio_industry,
            "benchmark_style_exposure": self.benchmark_exposure,
            "benchmark_industry_exposure": self.benchmark_industry,
            "active_style_exposure": self.active_exposure,
            "active_industry_exposure": self.active_industry,
            "factor_returns": self.factor_returns,
            "factor_contribution": self.factor_contribution,
            "active_factor_contribution": self.active_factor_contribution,
        }
        for name, frame in frames.items():
            if frame is not None:
                frame.to_parquet(output_path / f"{name}.parquet")
        for name, series in {
            "unexplained_return": self.unexplained_return,
            "active_unexplained_return": self.active_unexplained_return,
        }.items():
            if series is not None:
                series.rename(name).to_frame().to_parquet(output_path / f"{name}.parquet")
        self.coverage.to_csv(output_path / "coverage.csv")
        (output_path / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return output_path
