from __future__ import annotations

import importlib
import json
import math
import re
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import polars as pl

from factor_layer.factor_evaluation_alphapurify.Database import DataBase
from factor_layer.factor_evaluation_alphapurify.Exposures import PortfolioExposures, PureExposures
from factor_layer.factor_evaluation_alphapurify.FactorAnalyzer import FactorAnalyzer
from factor_layer.factor_evaluation_alphapurify.config import AlphaPurifyAdapterConfig, load_config


FA_MODULE = importlib.import_module("factor_layer.factor_evaluation_alphapurify.FactorAnalyzer")


def _safe_name(name: str, max_len: int = 100) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff\-]+", "_", name).strip("_")
    return cleaned[:max_len] if len(cleaned) > max_len else cleaned


def _prepare_dirs(output_root: Path) -> tuple[Path, Path, Path, Path]:
    json_root = output_root / "json"
    images_root = output_root / "images"
    tmp_root = output_root / "_tmp_runtime"
    run_summary_path = output_root / "run_summary.json"

    if output_root.exists():
        shutil.rmtree(output_root)
    json_root.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    return json_root, images_root, tmp_root, run_summary_path


def _detect_test_year(market_root: Path, user_year: int | None = None) -> int:
    if user_year is not None:
        return user_year

    market_files = sorted(market_root.glob("daily_market_summary_*.parquet"))
    years: list[int] = []
    for item in market_files:
        token = item.stem.rsplit("_", 1)[-1]
        if token.isdigit():
            years.append(int(token))
    if not years:
        raise FileNotFoundError(f"未找到可解析年份的行情文件: {market_root}")
    years = sorted(set(years))
    if 2025 in years:
        return 2025
    return max(years)


def _has_year_parquet(factor_dir: Path, year: int) -> bool:
    year_dir = factor_dir / f"year={year}"
    return year_dir.exists() and any(year_dir.rglob("*.parquet"))


def _select_specified_factor_dirs(
    *,
    factors_root: Path,
    year: int,
    target_names: tuple[str, ...],
    exposure_names: tuple[str, ...],
) -> tuple[list[Path], list[Path]]:
    def _resolve_one(name: str) -> Path:
        path = factors_root / name
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"指定因子目录不存在: {path}")
        if not _has_year_parquet(path, year):
            raise RuntimeError(f"指定因子在 year={year} 没有可用 parquet: {path}")
        return path

    target_dirs = [_resolve_one(item) for item in target_names]
    exposure_dirs = [_resolve_one(item) for item in exposure_names]
    return target_dirs, exposure_dirs


def _dedup_panel(panel_df: pl.DataFrame, factor_name: str, exposure_cols: list[str]) -> pl.DataFrame:
    agg_exprs = [
        pl.col("close").mean().alias("close"),
        pl.col(factor_name).mean().alias(factor_name),
    ] + [pl.col(column).mean().alias(column) for column in exposure_cols]
    return panel_df.group_by(["datetime", "symbol"]).agg(agg_exprs).sort(["symbol", "datetime"])


def _summarize_pl(df: pl.DataFrame, head_n: int = 5) -> dict[str, Any]:
    return {
        "shape": [df.height, df.width],
        "columns": df.columns,
        "head": df.head(head_n).to_dicts(),
    }


def _summarize_pd(df: pd.DataFrame, head_n: int = 5) -> dict[str, Any]:
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": [str(column) for column in df.columns],
        "head": df.head(head_n).to_dict(orient="records"),
    }


def _normalize_numeric_sequence(values: Any) -> list[Any]:
    if values is None:
        return []
    out: list[Any] = []
    for value in values:
        if value is None:
            out.append(None)
            continue
        if isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                out.append(None)
            else:
                out.append(value)
            continue
        out.append(value)
    return out


def _compress_trace_name(name: Any, *, max_len: int = 42) -> str:
    text = str(name) if name is not None else ""
    text = text.replace("day_aggs_v1_fundamental_", "")
    text = text.replace("_2016_2025_v1", "")
    text = text.replace("_rank_", "_")
    text = text.replace("operating_", "op_")
    text = text.replace("portfolio_", "pf_")
    if len(text) <= max_len:
        return text
    head = text[:20].rstrip("_")
    tail = text[-16:].lstrip("_")
    return f"{head}...{tail}"


def _axis_layout_key(axis_ref: str | None) -> str:
    if axis_ref is None or axis_ref == "x":
        return "xaxis"
    if axis_ref == "y":
        return "yaxis"
    if axis_ref.startswith("x"):
        suffix = axis_ref[1:]
        return f"xaxis{suffix}"
    if axis_ref.startswith("y"):
        suffix = axis_ref[1:]
        return f"yaxis{suffix}"
    return axis_ref


def _is_datetime_like(values: list[Any]) -> bool:
    if not values:
        return False
    sample = values[: min(30, len(values))]
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return float(parsed.notna().mean()) >= 0.8


def _build_export_ready_figure(fig: go.Figure) -> go.Figure:
    ready_fig = go.Figure(fig)

    # 统一静态图尺寸与字体，避免图例/轴标签把绘图区压缩到角落
    ready_fig.update_layout(
        autosize=False,
        width=int(ready_fig.layout.width or 2200),
        height=int(ready_fig.layout.height or 2600),
        font=dict(size=12),
        title=dict(font=dict(size=20)),
        legend=dict(
            font=dict(size=10),
            x=1.01,
            y=1.0,
            xanchor="left",
            yanchor="top",
        ),
        margin=dict(
            l=90,
            r=380,
            t=120,
            b=120,
        ),
    )

    x_values_by_axis: dict[str, list[Any]] = {}
    for trace in ready_fig.data:
        if hasattr(trace, "name"):
            trace.name = _compress_trace_name(getattr(trace, "name"))
        if hasattr(trace, "y"):
            y = getattr(trace, "y")
            if y is not None:
                trace.y = _normalize_numeric_sequence(y)
        if hasattr(trace, "x"):
            x = getattr(trace, "x")
            if x is not None and not isinstance(x, str):
                normalized_x = [None if pd.isna(item) else item for item in x]
                trace.x = normalized_x
                axis_key = _axis_layout_key(getattr(trace, "xaxis", None))
                x_values_by_axis.setdefault(axis_key, []).extend(normalized_x)

    layout_dict = ready_fig.to_dict().get("layout", {})
    for axis_name, axis_payload in layout_dict.items():
        if not axis_name.startswith("xaxis"):
            continue
        axis_payload["automargin"] = True
        axis_payload["tickfont"] = {"size": 9}
        axis_payload["tickangle"] = -35

        axis_values = [item for item in x_values_by_axis.get(axis_name, []) if item is not None]
        if _is_datetime_like(axis_values):
            axis_payload["type"] = "date"
            axis_payload["tickformat"] = "%Y-%m-%d"
            axis_payload["nticks"] = 8
            axis_payload.pop("categoryarray", None)
            axis_payload.pop("categoryorder", None)
        elif axis_payload.get("type") == "category":
            axis_payload.pop("categoryarray", None)
            axis_payload["categoryorder"] = "trace"
            axis_payload["nticks"] = 12
    ready_fig.update_layout(**layout_dict)
    return ready_fig


def _build_export_safe_figure(fig: go.Figure) -> go.Figure:
    safe_fig = go.Figure(fig)
    safe_fig.update_layout(
        autosize=False,
        width=int(safe_fig.layout.width or 2200),
        height=int(safe_fig.layout.height or 2600),
    )

    # Kaleido 在复杂 category axis + categoryarray 组合下偶发 axis scaling 错误；
    # fallback 时统一把 categoryarray 清空，交给 trace 顺序驱动。
    layout_dict = safe_fig.to_dict().get("layout", {})
    for axis_name, axis_payload in layout_dict.items():
        if not axis_name.startswith(("xaxis", "yaxis")):
            continue
        axis_type = axis_payload.get("type")
        if axis_type == "category":
            axis_payload["categoryorder"] = "trace"
            axis_payload.pop("categoryarray", None)
        axis_payload["automargin"] = True

    safe_fig.update_layout(**layout_dict)

    # 对 trace 中的 inf/nan 做兜底清理，避免导出端计算坐标范围失败
    for trace in safe_fig.data:
        if hasattr(trace, "y"):
            y = getattr(trace, "y")
            if y is not None:
                trace.y = _normalize_numeric_sequence(y)
        if hasattr(trace, "x"):
            x = getattr(trace, "x")
            if x is not None and not isinstance(x, str):
                trace.x = [None if pd.isna(item) else item for item in x]
    return safe_fig


def _save_fig(fig: go.Figure, path: Path, output_root: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    ready_fig = _build_export_ready_figure(fig)
    try:
        ready_fig.write_image(
            path,
            format="png",
            scale=2,
            width=int(ready_fig.layout.width or 2200),
            height=int(ready_fig.layout.height or 2600),
        )
    except Exception as exc:
        first_error = str(exc)
        try:
            ready_fig.write_image(
                path,
                format="png",
                scale=1,
                width=int(ready_fig.layout.width or 2200),
                height=int(ready_fig.layout.height or 2600),
            )
        except Exception as exc_retry:
            second_error = str(exc_retry)
            try:
                safe_fig = _build_export_safe_figure(ready_fig)
                safe_fig.write_image(
                    path,
                    format="png",
                    scale=1,
                    width=int(safe_fig.layout.width or 2200),
                    height=int(safe_fig.layout.height or 2600),
                )
            except Exception as exc_fallback:
                raise RuntimeError(
                    "保存 PNG 图片失败。"
                    f" 初次错误: {first_error};"
                    f" 重试错误: {second_error};"
                    f" fallback错误: {exc_fallback}"
                ) from exc_fallback
    return str(path.relative_to(output_root))


def _write_config_snapshot(config_path: Path, output_root: Path) -> str:
    snapshot_path = output_root / "config_snapshot.yaml"
    snapshot_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    return str(snapshot_path)


def run_factor_case(
    *,
    factor_dir: Path,
    exposure_dirs: list[Path],
    year: int,
    data_root: Path,
    min_symbols_per_day: int,
    json_root: Path,
    images_root: Path,
    tmp_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    factor_name = factor_dir.name
    file_key = _safe_name(factor_name)
    factor_result: dict[str, Any] = {
        "factor_name": factor_name,
        "factor_dir": str(factor_dir),
        "exposure_dirs": [str(path) for path in exposure_dirs],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "status": "success",
        "database": {},
        "exposures": {"portfolio": {}, "pure": {}},
        "factor_analyzer": {},
        "images": [],
        "errors": [],
    }

    case_tmp = tmp_root / file_key
    case_tmp.mkdir(parents=True, exist_ok=True)
    image_dir = images_root / file_key
    image_dir.mkdir(parents=True, exist_ok=True)

    adapter_output_root = case_tmp / "adapter_symbol_output"
    adapter = DataBase.build_alphapurify_symbol_parquet_input(
        data_root=str(data_root),
        target_factor_dir=str(factor_dir),
        exposure_factor_dirs=[str(path) for path in exposure_dirs],
        output_root=str(adapter_output_root),
        factor_name=factor_name,
        years=[year],
        min_symbols_per_day=min_symbols_per_day,
        base_dir_name="base_data",
        continuous_dir_name="factors_data",
        trade_date_col="datetime",
        symbol_col="symbol",
    )

    exposure_cols = [Path(path).name for path in exposure_dirs]
    panel_df = _dedup_panel(adapter["panel_df"], factor_name=factor_name, exposure_cols=exposure_cols)
    begin_dt = panel_df.select(pl.col("datetime").min()).item()
    end_dt = panel_df.select(pl.col("datetime").max()).item()

    db = DataBase(
        PathConfig=adapter["PathConfig"],
        stocks_list=adapter["stocks_list"],
        begin_date=begin_dt.strftime("%Y-%m-%d %H:%M:%S"),
        end_date=end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        trade_date_col="datetime",
        symbol_col="symbol",
        dropNaN=False,
        max_workers=1,
    )
    db_df = db.get()
    factor_result["database"] = {
        "stocks_count": len(adapter["stocks_list"]),
        "panel_df_summary": _summarize_pl(panel_df),
        "db_get_summary": _summarize_pd(db_df),
    }

    analysis_df = panel_df.to_pandas()
    original_show = go.Figure.show
    go.Figure.show = lambda self, *args, **kwargs: None
    try:
        try:
            pe = PortfolioExposures(
                base_df=analysis_df,
                trade_date_col="datetime",
                symbol_col="symbol",
                price_col="close",
                factor_name=factor_name,
                exposure_cols=exposure_cols,
                rebalance_period=1,
                bins=5,
                position="l",
                overnight="on",
            )
            pe.run()
            factor_result["exposures"]["portfolio"]["result_df"] = _summarize_pl(pe.result_df)

            fig = pe.plot_portfolio_exposures(staticPlot=True, return_fig=True)
            factor_result["images"].append(
                _save_fig(
                    fig,
                    image_dir / "portfolio_exposures" / "portfolio_exposures.png",
                    output_root,
                )
            )
            fig = pe.plot_portfolio_returns(staticPlot=True, return_fig=True)
            factor_result["images"].append(
                _save_fig(
                    fig,
                    image_dir / "portfolio_exposures" / "portfolio_returns.png",
                    output_root,
                )
            )
            fig = pe.plot_portfolio_exposures_and_returns(staticPlot=True, return_fig=True)
            factor_result["images"].append(
                _save_fig(
                    fig,
                    image_dir / "portfolio_exposures" / "portfolio_exposures_and_returns.png",
                    output_root,
                )
            )
            factor_result["exposures"]["portfolio"]["success"] = True
        except Exception as exc:
            factor_result["exposures"]["portfolio"]["success"] = False
            factor_result["exposures"]["portfolio"]["error"] = str(exc)
            factor_result["errors"].append(f"PortfolioExposures 失败: {exc}")

        try:
            pure = PureExposures(
                base_df=analysis_df,
                trade_date_col="datetime",
                symbol_col="symbol",
                price_col="close",
                factor_name=factor_name,
                exposure_cols=exposure_cols,
                overnight="on",
            )
            pure.run()
            factor_result["exposures"]["pure"]["result_df"] = _summarize_pl(pure.result_df)
            factor_result["exposures"]["pure"]["corr_df"] = _summarize_pd(pure.corr_df)
            factor_result["exposures"]["pure"]["corr_matrix"] = _summarize_pd(pure.corr_matrix.reset_index())

            fig = pure.plot_pure_exposures(staticPlot=True, return_fig=True)
            factor_result["images"].append(
                _save_fig(
                    fig,
                    image_dir / "pure_exposures" / "pure_exposures.png",
                    output_root,
                )
            )
            fig = pure.plot_pure_returns(staticPlot=True, return_fig=True)
            factor_result["images"].append(
                _save_fig(
                    fig,
                    image_dir / "pure_exposures" / "pure_returns.png",
                    output_root,
                )
            )
            fig = pure.plot_pure_exposures_and_returns(staticPlot=True, return_fig=True)
            factor_result["images"].append(
                _save_fig(
                    fig,
                    image_dir / "pure_exposures" / "pure_exposures_and_returns.png",
                    output_root,
                )
            )
            fig = pure.plot_correlations(staticPlot=True, return_fig=True)
            factor_result["images"].append(
                _save_fig(
                    fig,
                    image_dir / "pure_exposures" / "pure_correlations.png",
                    output_root,
                )
            )
            factor_result["exposures"]["pure"]["success"] = True
        except Exception as exc:
            factor_result["exposures"]["pure"]["success"] = False
            factor_result["exposures"]["pure"]["error"] = str(exc)
            factor_result["errors"].append(f"PureExposures 失败: {exc}")

        try:
            FA_MODULE._worker_df = None
            fa = FactorAnalyzer.simple(
                analysis_df,
                factor_name=factor_name,
                trade_date_col="datetime",
                symbol_col="symbol",
                price_col="close",
                research_cfg={
                    "rebalance_periods": [1],
                    "return_horizons": [1],
                    "return_rolling_period": 3,
                    "horizon_rolling_period": 3,
                    "base_rate": 0.0,
                    "overnight": "on",
                    "bins": 5,
                    "fac_shift": None,
                },
                analysis_cfg={
                    "rank_ic": True,
                    "log_scale": False,
                    "agg_freq": None,
                    "group_by": None,
                    "max_workers": 1,
                },
            )
            fa.run()
            factor_result["factor_analyzer"]["ls_stats_panel"] = _summarize_pd(fa.ls_stats_panel)
            factor_result["factor_analyzer"]["l_stats_panel"] = _summarize_pd(fa.l_stats_panel)
            factor_result["factor_analyzer"]["s_stats_panel"] = _summarize_pd(fa.s_stats_panel)
            factor_result["factor_analyzer"]["ic_stats_panel"] = _summarize_pd(fa.ic_stats_panel)

            fig = fa.create_single_fac_ic_sheet(staticPlot=True, return_fig=True)
            factor_result["images"].append(
                _save_fig(
                    fig,
                    image_dir / "factor_analyzer" / "fa_ic_sheet.png",
                    output_root,
                )
            )
            fig = fa.create_long_short_return_sheet(staticPlot=True, return_fig=True)
            factor_result["images"].append(
                _save_fig(
                    fig,
                    image_dir / "factor_analyzer" / "fa_long_short_sheet.png",
                    output_root,
                )
            )
            fig = fa.create_long_return_sheet(staticPlot=True, return_fig=True)
            factor_result["images"].append(
                _save_fig(
                    fig,
                    image_dir / "factor_analyzer" / "fa_long_sheet.png",
                    output_root,
                )
            )
            fig = fa.create_short_return_sheet(staticPlot=True, return_fig=True)
            factor_result["images"].append(
                _save_fig(
                    fig,
                    image_dir / "factor_analyzer" / "fa_short_sheet.png",
                    output_root,
                )
            )
            factor_result["factor_analyzer"]["success"] = True
        except Exception as exc:
            factor_result["factor_analyzer"]["success"] = False
            factor_result["factor_analyzer"]["error"] = str(exc)
            factor_result["errors"].append(f"FactorAnalyzer 失败: {exc}")
    finally:
        go.Figure.show = original_show

    if factor_result["errors"]:
        factor_result["status"] = "partial_failed"

    json_path = json_root / f"{file_key}.json"
    json_path.write_text(json.dumps(factor_result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    factor_result["json_path"] = str(json_path.relative_to(output_root))
    return factor_result


def run_pipeline(
    config: AlphaPurifyAdapterConfig,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    data_root = Path(config.data_root)
    output_root = Path(config.output_root)
    market_root = data_root / "daily_market_summary"
    factors_root = data_root / "factors"
    year = _detect_test_year(market_root=market_root, user_year=config.year)

    target_dirs, exposure_dirs = _select_specified_factor_dirs(
        factors_root=factors_root,
        year=year,
        target_names=config.target_factors,
        exposure_names=config.exposure_factors,
    )

    json_root, images_root, tmp_root, run_summary_path = _prepare_dirs(output_root=output_root)
    results: list[dict[str, Any]] = []
    for factor_dir in target_dirs:
        try:
            result = run_factor_case(
                factor_dir=factor_dir,
                exposure_dirs=exposure_dirs,
                year=year,
                data_root=data_root,
                min_symbols_per_day=config.min_symbols_per_day,
                json_root=json_root,
                images_root=images_root,
                tmp_root=tmp_root,
                output_root=output_root,
            )
        except Exception as exc:
            key = _safe_name(factor_dir.name)
            result = {
                "factor_name": factor_dir.name,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "status": "failed",
                "errors": [str(exc)],
                "traceback": traceback.format_exc(),
                "images": [],
                "json_path": f"json/{key}.json",
            }
            (json_root / f"{key}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        results.append(result)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "year": year,
        "selected_target_factors": [item.name for item in target_dirs],
        "selected_exposure_factors": [item.name for item in exposure_dirs],
        "factors_total": len(target_dirs),
        "factors_success": sum(1 for item in results if item.get("status") == "success"),
        "factors_partial_failed": sum(1 for item in results if item.get("status") == "partial_failed"),
        "factors_failed": sum(1 for item in results if item.get("status") == "failed"),
        "json_files": [item.get("json_path") for item in results],
    }
    run_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    config_snapshot = None
    if config_path is not None and config.save_config_snapshot:
        config_snapshot = _write_config_snapshot(Path(config_path), output_root)

    return {
        "summary": summary,
        "results": results,
        "output_root": str(output_root),
        "run_summary_path": str(run_summary_path),
        "config_snapshot": config_snapshot,
    }


def run_from_config(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    return run_pipeline(config, config_path=config_path)
