"""Plotly dashboard for exposure-analysis results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .result import ExposureAnalysisResult


def _industry_label(result: ExposureAnalysisResult, factor: str) -> str:
    names = result.metadata.get("industry_names") or {}
    name = names.get(str(factor))
    code = str(factor).removeprefix("industry_")
    return f"{name} ({code})" if name else code


def _configure_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _export_one_style_exposure_png(
    frame: Any,
    output: Path,
    *,
    title: str,
    dpi: int,
) -> Path:
    import numpy as np

    plt = _configure_matplotlib()
    if frame is None or frame.empty or len(frame.columns) == 0:
        raise ValueError(f"{title} 没有可绘制的风格暴露")
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    factors = list(frame.columns)
    fig, axes = plt.subplots(
        len(factors),
        1,
        figsize=(14, max(2.4 * len(factors), 4.0)),
        sharex=True,
        squeeze=False,
    )
    axes = np.asarray(axes).reshape(-1)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for position, (axis, factor) in enumerate(zip(axes, factors)):
        values = frame[factor].astype(float)
        color = colors[position % len(colors)]
        axis.plot(
            values.index,
            values.values,
            color=color,
            linewidth=1.15,
            label=str(factor),
        )
        axis.fill_between(
            values.index,
            values.values,
            0.0,
            color=color,
            alpha=0.10,
        )
        axis.axhline(0.0, color="#666666", linewidth=0.7, linestyle="--")
        axis.set_ylabel("Exposure")
        axis.legend(loc="upper left", frameon=False)
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Date")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output


def export_style_exposure_pngs(
    result: ExposureAnalysisResult,
    output: str | Path,
    *,
    dpi: int = 150,
) -> dict[str, Path]:
    """Export portfolio/benchmark/active style-exposure time-series PNGs."""
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    definitions = {
        "portfolio": (
            result.portfolio_exposure,
            "Portfolio Style Exposure",
            "portfolio_style_exposure.png",
        ),
        "benchmark": (
            result.benchmark_exposure,
            "Benchmark Style Exposure",
            "benchmark_style_exposure.png",
        ),
        "active": (
            result.active_exposure,
            "Active Style Exposure",
            "active_style_exposure.png",
        ),
    }
    exported: dict[str, Path] = {}
    for name, (frame, title, filename) in definitions.items():
        if frame is None or frame.empty:
            continue
        exported[name] = _export_one_style_exposure_png(
            frame,
            output_path / filename,
            title=title,
            dpi=dpi,
        )
    return exported


def export_industry_exposure_pngs(
    result: ExposureAnalysisResult,
    output: str | Path,
    *,
    top_n: int = 10,
    dpi: int = 150,
) -> dict[str, Path]:
    """Export the default static industry-exposure report."""
    import numpy as np
    import pandas as pd

    if int(top_n) <= 0:
        raise ValueError("industry_top_n 必须为正整数")
    plt = _configure_matplotlib()
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}

    # 1) Active exposure heatmap across all model industries.
    if result.active_industry is not None and not result.active_industry.empty:
        frame = result.active_industry.astype(float)
        matrix = frame.to_numpy().T
        limit = float(np.nanmax(np.abs(matrix))) if matrix.size else 0.0
        limit = max(limit, 1e-6)
        fig, axis = plt.subplots(
            figsize=(15, max(8.0, 0.30 * len(frame.columns))),
        )
        image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        tick_count = min(10, len(frame.index))
        tick_positions = np.unique(
            np.linspace(0, len(frame.index) - 1, tick_count, dtype=int)
        )
        axis.set_xticks(tick_positions)
        axis.set_xticklabels(
            [frame.index[position].strftime("%Y-%m-%d") for position in tick_positions],
            rotation=35,
            ha="right",
        )
        axis.set_yticks(np.arange(len(frame.columns)))
        axis.set_yticklabels(
            [_industry_label(result, factor) for factor in frame.columns]
        )
        axis.set_xlabel("Date")
        axis.set_ylabel("SW Level-1 Industry")
        axis.set_title("Active Industry Exposure Heatmap", fontweight="bold")
        colorbar = fig.colorbar(image, ax=axis, pad=0.015)
        colorbar.set_label("Active weight")
        fig.tight_layout()
        path = output_path / "active_industry_heatmap.png"
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        exported["active_heatmap"] = path

        # 2) Latest active over/underweight snapshot.
        latest = frame.iloc[-1].sort_values()
        labels = [_industry_label(result, factor) for factor in latest.index]
        colors = ["#2b6cb0" if value < 0 else "#c53030" for value in latest.values]
        fig, axis = plt.subplots(
            figsize=(12, max(8.0, 0.31 * len(latest))),
        )
        axis.barh(labels, latest.values, color=colors, alpha=0.88)
        axis.axvline(0.0, color="#555555", linewidth=0.8)
        axis.set_xlabel("Active weight")
        axis.set_title(
            f"Latest Active Industry Exposure — {latest.name:%Y-%m-%d}",
            fontweight="bold",
        )
        axis.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        path = output_path / "latest_active_industry_exposure.png"
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        exported["latest_active"] = path

    # 3) Portfolio and benchmark latest-date comparison for largest active bets.
    if (
        result.benchmark_industry is not None
        and not result.portfolio_industry.empty
        and not result.benchmark_industry.empty
    ):
        portfolio_latest = result.portfolio_industry.iloc[-1].astype(float)
        benchmark_latest = result.benchmark_industry.iloc[-1].astype(float)
        active_latest = portfolio_latest - benchmark_latest
        selected = active_latest.abs().nlargest(min(int(top_n), len(active_latest))).index
        selected = active_latest.loc[selected].sort_values().index
        labels = [_industry_label(result, factor) for factor in selected]
        positions = np.arange(len(selected))
        fig, axis = plt.subplots(figsize=(12, max(6.0, 0.55 * len(selected))))
        height = 0.36
        axis.barh(
            positions - height / 2,
            portfolio_latest.loc[selected],
            height=height,
            color="#1f77b4",
            label="Portfolio",
        )
        axis.barh(
            positions + height / 2,
            benchmark_latest.loc[selected],
            height=height,
            color="#9ca3af",
            label="Benchmark",
        )
        axis.set_yticks(positions)
        axis.set_yticklabels(labels)
        axis.set_xlabel("Weight")
        axis.set_title(
            f"Portfolio vs Benchmark — Top {len(selected)} Active Industries",
            fontweight="bold",
        )
        axis.legend(frameon=False)
        axis.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        path = output_path / "portfolio_vs_benchmark_industry.png"
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        exported["portfolio_vs_benchmark"] = path

    # 4) Top-N absolute industry allocation, with Other/Unknown/Cash buckets.
    if not result.portfolio_industry.empty:
        portfolio = result.portfolio_industry.astype(float).clip(lower=0.0)
        selected = (
            portfolio.mean(axis=0)
            .nlargest(min(int(top_n), len(portfolio.columns)))
            .index
        )
        allocation = portfolio.loc[:, selected].copy()
        remaining = portfolio.drop(columns=selected).sum(axis=1)
        if (remaining > 1e-12).any():
            allocation["Other"] = remaining
        unknown_column = "portfolio_unknown_weight"
        if unknown_column in result.coverage:
            unknown = result.coverage[unknown_column].clip(lower=0.0)
            if (unknown > 1e-12).any():
                allocation["Unknown"] = unknown
        invested_column = "portfolio_invested_weight"
        if invested_column in result.coverage:
            cash = (1.0 - result.coverage[invested_column]).clip(lower=0.0)
            if (cash > 1e-12).any():
                allocation["Cash"] = cash
        allocation = allocation.rename(
            columns={
                factor: _industry_label(result, factor)
                for factor in selected
            }
        )
        fig, axis = plt.subplots(figsize=(15, 7))
        palette = list(plt.get_cmap("tab20").colors)
        axis.stackplot(
            allocation.index,
            *[allocation[column].values for column in allocation.columns],
            labels=list(allocation.columns),
            colors=palette[: len(allocation.columns)],
            alpha=0.88,
        )
        axis.set_ylabel("Portfolio weight")
        axis.set_xlabel("Date")
        axis.set_title(
            f"Portfolio Industry Allocation — Top {len(selected)} + Other",
            fontweight="bold",
        )
        axis.set_ylim(bottom=0.0)
        axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=False,
        )
        axis.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        path = output_path / "portfolio_industry_allocation.png"
        fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        exported["allocation"] = path

    return exported


def build_exposure_dashboard(
    result: ExposureAnalysisResult,
    *,
    title: str = "Portfolio Exposure Analysis",
) -> Any:
    from plotly import graph_objects as go
    from plotly.subplots import make_subplots

    has_active = result.active_exposure is not None
    contribution = (
        result.active_factor_contribution
        if result.active_factor_contribution is not None
        else result.factor_contribution
    )
    industry_primary = (
        result.active_industry
        if result.active_industry is not None
        else result.portfolio_industry
    )
    rows = 6 if contribution is not None else 5
    fig = make_subplots(
        rows=rows,
        cols=1,
        vertical_spacing=0.08,
        subplot_titles=[
            "风格暴露",
            "最新风格暴露快照",
            "行业暴露热力图",
            "最新行业暴露快照",
            "风险模型覆盖率",
            *(["累计因子贡献（实验性事后归因）"] if contribution is not None else []),
        ],
    )
    primary = result.active_exposure if has_active else result.portfolio_exposure
    prefix = "主动" if has_active else "组合"
    for factor in primary.columns:
        fig.add_trace(
            go.Scatter(
                x=primary.index,
                y=primary[factor],
                mode="lines",
                name=f"{prefix} {factor}",
            ),
            row=1,
            col=1,
        )
    if not primary.empty:
        latest = primary.iloc[-1]
        fig.add_trace(
            go.Bar(x=latest.index, y=latest.values, name=f"最新{prefix}暴露"),
            row=2,
            col=1,
        )
    industry_labels = [
        _industry_label(result, factor) for factor in industry_primary.columns
    ]
    heatmap_kwargs = {
        "colorscale": "RdBu",
        "zmid": 0.0,
    } if result.active_industry is not None else {"colorscale": "Blues"}
    fig.add_trace(
        go.Heatmap(
            x=industry_primary.index,
            y=industry_labels,
            z=industry_primary.to_numpy().T,
            name=f"{prefix}行业暴露",
            colorbar={"title": "Weight"},
            **heatmap_kwargs,
        ),
        row=3,
        col=1,
    )
    latest_industry = industry_primary.iloc[-1].sort_values()
    latest_labels = [
        _industry_label(result, factor) for factor in latest_industry.index
    ]
    latest_colors = [
        "#2b6cb0" if value < 0 else "#c53030"
        for value in latest_industry.values
    ]
    fig.add_trace(
        go.Bar(
            x=latest_industry.values,
            y=latest_labels,
            orientation="h",
            marker_color=latest_colors,
            name=f"最新{prefix}行业暴露",
        ),
        row=4,
        col=1,
    )
    for column in [
        col for col in result.coverage.columns if col.endswith("coverage_weight")
    ]:
        fig.add_trace(
            go.Scatter(
                x=result.coverage.index,
                y=result.coverage[column],
                mode="lines",
                name=column,
            ),
            row=5,
            col=1,
        )
    if contribution is not None:
        cumulative = contribution.fillna(0.0).cumsum()
        for factor in cumulative.columns:
            fig.add_trace(
                go.Scatter(
                    x=cumulative.index,
                    y=cumulative[factor],
                    mode="lines",
                    name=f"贡献 {factor}",
                ),
                row=6,
                col=1,
            )
    fig.update_layout(
        title=title,
        height=max(320 * rows, 900),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Exposure", row=1, col=1)
    fig.update_yaxes(title_text="Exposure", row=2, col=1)
    fig.update_yaxes(title_text="Industry", row=3, col=1)
    fig.update_xaxes(title_text="Exposure", row=4, col=1)
    fig.update_yaxes(title_text="Industry", row=4, col=1)
    fig.update_yaxes(title_text="Coverage", range=[0, 1.01], row=5, col=1)
    if contribution is not None:
        fig.update_yaxes(title_text="Return contribution", row=6, col=1)
    return fig


def export_exposure_dashboard(
    result: ExposureAnalysisResult,
    output: str | Path,
    *,
    title: str = "Portfolio Exposure Analysis",
    include_plotlyjs: bool | str = "cdn",
) -> Path:
    output_path = Path(output)
    if output_path.suffix.lower() != ".html":
        output_path = output_path / "exposure_dashboard.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_exposure_dashboard(result, title=title)
    fig.write_html(
        str(output_path),
        include_plotlyjs=include_plotlyjs,
        full_html=True,
    )
    return output_path
