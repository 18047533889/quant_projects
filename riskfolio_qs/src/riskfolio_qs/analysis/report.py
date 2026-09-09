"""Persist structured analysis artifacts and a dependency-free HTML report."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .. import __version__
from .contracts import PositionAnalysisResult


ANALYSIS_ARTIFACTS = {
    "position_summary.parquet",
    "holdings_detail.parquet",
    "turnover_detail.parquet",
    "exposure_summary.parquet",
    "risk_contribution.parquet",
    "constraint_summary.parquet",
    "quality_checks.parquet",
    "analysis_manifest.yaml",
    "resolved_analysis_config.yaml",
    "position_report.html",
}


def _serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _table(frame: pd.DataFrame, rows: int = 20) -> str:
    if frame.empty:
        return "<p class=\"empty\">无可用数据</p>"
    return frame.head(rows).to_html(
        border=0,
        classes=["dataframe"],
        na_rep="—",
        float_format=lambda value: f"{value:.6g}",
    )


def build_html_report(
    result: PositionAnalysisResult,
    *,
    top_holdings: int = 20,
    top_trades: int = 20,
) -> str:
    """Render only from structured result tables."""

    metadata = result.metadata
    latest_date = result.position_summary.index.max()
    holdings = result.holdings_detail.xs(latest_date).sort_values(
        "abs_weight", ascending=False
    )
    trades = result.turnover_detail.xs(latest_date).assign(
        abs_delta=lambda frame: frame["delta_weight"].abs()
    ).sort_values("abs_delta", ascending=False)
    constraints = (
        result.constraint_summary.reset_index()
        if not result.constraint_summary.empty
        else result.constraint_summary
    )
    quality = result.quality_checks.loc[
        ~result.quality_checks["status"].eq("passed")
    ] if not result.quality_checks.empty else result.quality_checks
    style = """
    body { font-family: system-ui, sans-serif; margin: 28px; color: #202124; }
    h1, h2 { color: #17324d; }
    .meta { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
    .card { background: #f5f7fa; border-radius: 8px; padding: 12px 16px; }
    table { border-collapse: collapse; width: 100%; margin: 8px 0 28px; }
    th, td { border-bottom: 1px solid #ddd; padding: 6px 8px; text-align: right; }
    th:first-child, td:first-child { text-align: left; }
    .empty { color: #777; }
    """
    return f"""<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>riskfolio_qs 仓位分析</title>
<style>{style}</style></head>
<body>
<h1>riskfolio_qs 仓位分析</h1>
<div class="meta">
  <div class="card">状态：{escape(str(metadata.get("status")))}</div>
  <div class="card">日期：{escape(str(latest_date.date()))}</div>
  <div class="card">期间数：{int(metadata.get("date_count", 0))}</div>
  <div class="card">资产数：{int(metadata.get("asset_count", 0))}</div>
</div>
<h2>组合时间序列</h2>
{_table(result.position_summary.reset_index(), rows=len(result.position_summary))}
<h2>最新目标仓位 Top {int(top_holdings)}</h2>
{_table(holdings.reset_index(), rows=top_holdings)}
<h2>最新调仓 Top {int(top_trades)}</h2>
{_table(trades.reset_index(), rows=top_trades)}
<h2>约束</h2>
{_table(constraints, rows=100)}
<h2>数据质量与告警</h2>
{_table(quality, rows=100)}
<p>本报告分析目标仓位，不代表实际成交、实际持仓或投资绩效。</p>
</body></html>
"""


def write_analysis_outputs(
    result: PositionAnalysisResult,
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    config_path: Path | None,
    overwrite: bool,
) -> Path:
    """Write known analysis artifacts without modifying optimizer artifacts."""

    root = Path(output_dir).expanduser().resolve()
    existing = sorted(name for name in ANALYSIS_ARTIFACTS if (root / name).exists())
    if existing and not overwrite:
        raise FileExistsError(
            f"analysis output already contains artifacts: {existing}; "
            "pass --overwrite or set output.overwrite=true"
        )
    root.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for name in existing:
            path = (root / name).resolve()
            if path.parent != root:
                raise ValueError(f"refusing to remove artifact outside {root}: {path}")
            path.unlink()
    written: list[Path] = []
    if bool(config["output"]["parquet"]):
        frames = {
            "position_summary.parquet": result.position_summary,
            "holdings_detail.parquet": result.holdings_detail,
            "turnover_detail.parquet": result.turnover_detail,
            "exposure_summary.parquet": result.exposure_summary,
            "risk_contribution.parquet": result.risk_contribution,
            "constraint_summary.parquet": result.constraint_summary,
            "quality_checks.parquet": result.quality_checks,
        }
        for name, frame in frames.items():
            path = root / name
            frame.to_parquet(path)
            written.append(path)

    resolved_path = root / "resolved_analysis_config.yaml"
    with resolved_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            _serializable(config), handle, sort_keys=False, allow_unicode=True
        )
    written.append(resolved_path)

    if bool(config["report"]["html"]):
        html_path = root / "position_report.html"
        html_path.write_text(
            build_html_report(
                result,
                top_holdings=int(config["report"]["top_holdings"]),
                top_trades=int(config["report"]["top_trades"]),
            ),
            encoding="utf-8",
        )
        written.append(html_path)

    manifest = {
        **result.metadata,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "riskfolio_qs_version": __version__,
        "config_path": str(config_path) if config_path is not None else None,
        "artifacts": {
            path.name: {
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in written
        },
    }
    source_dir = result.metadata.get("source_input_dir")
    if source_dir:
        source_root = Path(str(source_dir))
        source_files = {}
        for name in (
            "target_positions.parquet",
            "trades.parquet",
            "summary.parquet",
            "metadata.json",
            "run_manifest.yaml",
            "resolved_cli_config.yaml",
            "resolved_params.yaml",
        ):
            path = source_root / name
            if path.exists():
                source_files[name] = {
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
        manifest["source_artifacts"] = source_files
    manifest_path = root / "analysis_manifest.yaml"
    with manifest_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            _serializable(manifest), handle, sort_keys=False, allow_unicode=True
        )
    result.output_dir = root
    return root


__all__ = [
    "ANALYSIS_ARTIFACTS",
    "build_html_report",
    "write_analysis_outputs",
]
