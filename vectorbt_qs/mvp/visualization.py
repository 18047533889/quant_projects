"""Unified Plotly visualization helpers for vectorbt portfolios.

The plotting primitives remain vectorbt's own Plotly implementation.  This
module provides a stable chart registry, sensible multi-asset defaults, clear
single-asset handling, and HTML export for vectorbt_qs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import pandas as pd


ChartMetadata = Dict[str, Any]


PLOTLY_CHARTS: Dict[str, ChartMetadata] = {
    "nav_comparison": {
        "title": "策略、基准与超额 NAV",
        "scope": "portfolio",
        "method": None,
        "subplot": True,
    },
    "orders": {
        "title": "订单",
        "scope": "asset",
        "method": "plot_orders",
        "subplot": True,
    },
    "trades": {
        "title": "交易",
        "scope": "asset",
        "method": "plot_trades",
        "subplot": True,
    },
    "trade_pnl": {
        "title": "交易盈亏",
        "scope": "asset",
        "method": "plot_trade_pnl",
        "subplot": True,
    },
    "positions": {
        "title": "持仓区间",
        "scope": "asset",
        "method": "plot_positions",
        "subplot": False,
    },
    "position_pnl": {
        "title": "持仓盈亏",
        "scope": "asset",
        "method": "plot_position_pnl",
        "subplot": False,
    },
    "asset_flow": {
        "title": "资产数量变化",
        "scope": "asset",
        "method": "plot_asset_flow",
        "subplot": True,
    },
    "cash_flow": {
        "title": "现金流",
        "scope": "portfolio",
        "method": "plot_cash_flow",
        "subplot": True,
    },
    "assets": {
        "title": "持仓数量",
        "scope": "asset",
        "method": "plot_assets",
        "subplot": True,
    },
    "cash": {
        "title": "现金余额",
        "scope": "portfolio",
        "method": "plot_cash",
        "subplot": True,
    },
    "asset_value": {
        "title": "持仓市值",
        "scope": "portfolio",
        "method": "plot_asset_value",
        "subplot": True,
    },
    "value": {
        "title": "组合净值",
        "scope": "portfolio",
        "method": "plot_value",
        "subplot": True,
    },
    "cum_returns": {
        "title": "累计收益",
        "scope": "portfolio",
        "method": "plot_cum_returns",
        "subplot": True,
    },
    "drawdowns": {
        "title": "回撤区间",
        "scope": "portfolio",
        "method": "plot_drawdowns",
        "subplot": True,
    },
    "underwater": {
        "title": "水下回撤",
        "scope": "portfolio",
        "method": "plot_underwater",
        "subplot": True,
    },
    "gross_exposure": {
        "title": "总敞口",
        "scope": "portfolio",
        "method": "plot_gross_exposure",
        "subplot": True,
    },
    "net_exposure": {
        "title": "净敞口",
        "scope": "portfolio",
        "method": "plot_net_exposure",
        "subplot": True,
    },
}


DEFAULT_PORTFOLIO_CHARTS = (
    "nav_comparison",
    "cum_returns",
    "underwater",
    "cash",
    "gross_exposure",
    "net_exposure",
)

DEFAULT_ASSET_CHARTS = (
    "orders",
    "trade_pnl",
    "asset_flow",
    "assets",
)


def available_plotly_charts() -> Dict[str, ChartMetadata]:
    """Return a copy of all supported vectorbt Plotly chart definitions."""
    return {name: metadata.copy() for name, metadata in PLOTLY_CHARTS.items()}


def build_nav_curves(pf: Any) -> pd.DataFrame:
    """Return normalized strategy, benchmark, and excess wealth curves.

    Excess NAV is relative wealth, ``strategy_nav / benchmark_nav``.  It starts
    at one and compounds correctly, unlike subtracting two cumulative-return
    series.
    """
    value = pf.value()
    if isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            raise ValueError("NAV 对比要求组合价值是单列或共享现金后的 Series")
        value = value.iloc[:, 0]
    value = value.astype(float)
    if value.empty or pd.isna(value.iloc[0]) or value.iloc[0] == 0.0:
        raise ValueError("组合首日价值必须是非零有效数值")

    curves = pd.DataFrame(
        {"strategy_nav": value / float(value.iloc[0])},
        index=value.index,
    )
    benchmark_close = getattr(pf, "_qs_benchmark_close", None)
    if benchmark_close is not None:
        benchmark_close = pd.Series(benchmark_close, copy=False).reindex(value.index).ffill()
        first_valid = benchmark_close.first_valid_index()
        if first_valid is None or first_valid != value.index[0]:
            raise ValueError("基准指数在回测首日缺少收盘价，无法构造可比 NAV")
        benchmark_nav = benchmark_close / float(benchmark_close.iloc[0])
    else:
        benchmark_returns = getattr(pf, "_qs_benchmark_returns", None)
        if benchmark_returns is None:
            return curves
        benchmark_returns = (
            pd.Series(benchmark_returns, copy=False)
            .reindex(value.index)
            .fillna(0.0)
        )
        benchmark_nav = (1.0 + benchmark_returns).cumprod()

    curves["benchmark_nav"] = benchmark_nav
    curves["excess_nav"] = curves["strategy_nav"] / curves["benchmark_nav"]
    return curves


def _add_nav_traces(
    pf: Any,
    fig: Any,
    *,
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> None:
    from plotly import graph_objects as go

    curves = build_nav_curves(pf)
    trace_kwargs = {}
    if row is not None and col is not None:
        trace_kwargs = {"row": row, "col": col}
    benchmark_symbol = getattr(pf, "_qs_benchmark_symbol", "Benchmark")
    names = {
        "strategy_nav": "策略 NAV",
        "benchmark_nav": f"基准 NAV ({benchmark_symbol})",
        "excess_nav": "超额 NAV",
    }
    colors = {
        "strategy_nav": "#1f77b4",
        "benchmark_nav": "#7f7f7f",
        "excess_nav": "#d62728",
    }
    for curve in curves.columns:
        fig.add_trace(
            go.Scatter(
                x=curves.index,
                y=curves[curve],
                name=names[curve],
                mode="lines",
                line={"color": colors[curve]},
            ),
            **trace_kwargs,
        )


def _validate_charts(charts: Iterable[str], *, dashboard: bool) -> tuple[str, ...]:
    selected = tuple(charts)
    if not selected:
        raise ValueError("charts 不能为空")
    unknown = [chart for chart in selected if chart not in PLOTLY_CHARTS]
    if unknown:
        raise ValueError(f"未知 Plotly 图表: {', '.join(unknown)}")
    if dashboard:
        standalone = [
            chart for chart in selected if not PLOTLY_CHARTS[chart]["subplot"]
        ]
        if standalone:
            raise ValueError(
                "以下图表只能单独绘制，不能加入组合仪表板: "
                + ", ".join(standalone)
            )
    return selected


def build_plotly_chart(
    pf: Any,
    chart: str,
    *,
    column: Optional[Any] = None,
    title: Optional[str] = None,
    **plot_kwargs: Any,
) -> Any:
    """Build one vectorbt Plotly figure.

    Asset-scoped charts require ``column``.  vectorbt's record and asset-flow
    plotters can select one symbol directly even when portfolio cash is shared.
    """
    _validate_charts((chart,), dashboard=False)
    metadata = PLOTLY_CHARTS[chart]
    target = pf
    if metadata["scope"] == "asset":
        if column is None:
            raise ValueError(f"{chart} 是单标的图表，必须指定 column")

    if chart == "nav_comparison":
        from plotly import graph_objects as go

        fig = go.Figure()
        _add_nav_traces(pf, fig)
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="NAV",
            hovermode="x unified",
        )
    else:
        method = getattr(target, metadata["method"])
        fig = method(column=column, **plot_kwargs)
    if title:
        fig.update_layout(title=title)
    return fig


def build_plotly_dashboard(
    pf: Any,
    charts: Optional[Sequence[str]] = None,
    *,
    column: Optional[Any] = None,
    title: Optional[str] = None,
    **plot_kwargs: Any,
) -> Any:
    """Build a multi-row interactive vectorbt Plotly dashboard.

    Without ``column``, the default is a portfolio-level dashboard that is
    safe for cash-sharing multi-asset portfolios.  With ``column``, grouping is
    disabled and the default switches to an asset-level trading dashboard.
    """
    if charts is None:
        charts = DEFAULT_PORTFOLIO_CHARTS if column is None else DEFAULT_ASSET_CHARTS
    selected = _validate_charts(charts, dashboard=True)
    asset_charts = [
        chart for chart in selected if PLOTLY_CHARTS[chart]["scope"] == "asset"
    ]
    if asset_charts and column is None:
        raise ValueError(
            "仪表板包含单标的图表，必须指定 column: " + ", ".join(asset_charts)
        )

    portfolio_charts = [
        chart for chart in selected if PLOTLY_CHARTS[chart]["scope"] == "portfolio"
    ]
    if column is not None and portfolio_charts:
        raise ValueError(
            "单标的仪表板不能混入共享现金的组合级图表: "
            + ", ".join(portfolio_charts)
        )

    if column is None and "nav_comparison" not in selected:
        fig = pf.plot(subplots=list(selected), **plot_kwargs)
    else:
        # Portfolio.plot deliberately skips asset subplots on a cash-sharing
        # grouped portfolio.  Compose the figures row by row while still
        # delegating every trace to vectorbt's own Plotly methods.
        from plotly.subplots import make_subplots

        subplot_titles = [PLOTLY_CHARTS[chart]["title"] for chart in selected]
        fig = make_subplots(
            rows=len(selected),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=min(0.08, 0.3 / max(len(selected), 1)),
            subplot_titles=subplot_titles,
        )
        for row, chart in enumerate(selected, start=1):
            if chart == "nav_comparison":
                _add_nav_traces(pf, fig, row=row, col=1)
            else:
                method = getattr(pf, PLOTLY_CHARTS[chart]["method"])
                method(
                    column=column,
                    fig=fig,
                    add_trace_kwargs={"row": row, "col": 1},
                )
        fig.update_layout(height=max(320 * len(selected), 500))
        if column is None:
            fig.update_yaxes(title_text="NAV", row=1, col=1)
            fig.update_layout(hovermode="x unified")
    if title:
        fig.update_layout(title=title)
    return fig


def export_plotly_dashboard(
    pf: Any,
    output: Union[str, Path],
    charts: Optional[Sequence[str]] = None,
    *,
    column: Optional[Any] = None,
    title: Optional[str] = None,
    include_plotlyjs: Union[bool, str] = True,
    **plot_kwargs: Any,
) -> Path:
    """Build and export an offline-capable Plotly dashboard to HTML."""
    output_path = Path(output)
    if output_path.suffix.lower() != ".html":
        output_path = output_path / "portfolio_dashboard.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = build_plotly_dashboard(
        pf,
        charts=charts,
        column=column,
        title=title,
        **plot_kwargs,
    )
    fig.write_html(
        str(output_path),
        include_plotlyjs=include_plotlyjs,
        full_html=True,
    )
    return output_path
