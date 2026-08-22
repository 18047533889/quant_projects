# vectorbt_qs CLI 入口
# 用法:
#   python -m vectorbt_qs backtest --benchmark B001
#   python -m vectorbt_qs backtest --positions path/to/target_positions.parquet
#   python -m vectorbt_qs run config.yaml
#   python -m vectorbt_qs batch config.batch.yaml
#   python -m vectorbt_qs accurate-batch --positions ... --barra-root ... --output-root ...
#   python -m vectorbt_qs accurate-benchmark --positions ... --barra-root ... --output-root ...

import argparse
import copy
import json
import sys
import os
from pathlib import Path

# 确保项目根在 path 中
_PROJ = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJ.parent))
sys.path.insert(0, str(_PROJ))
sys.path.insert(0, str(_PROJ / "vectorbt"))

import pandas as pd


# ============================================================
# Benchmark 注册表（路径映射）
# ============================================================
BENCHMARK_ROOT = Path(
    os.environ.get("VECTORBT_QS_BENCHMARK_ROOT", _PROJ.parent / "benchmarks")
)

BENCHMARKS = {
    "B001": BENCHMARK_ROOT / "gtja191_alpha191_topn/results/gtja191_alpha191_topn_v1/target_positions.parquet",
    "B002": BENCHMARK_ROOT / "gtja191_alpha191_meanvar_hist/results/gtja191_alpha191_meanvar_hist_v1_1/target_positions.parquet",
    "B002v1": BENCHMARK_ROOT / "gtja191_alpha191_meanvar_hist/results/gtja191_alpha191_meanvar_hist_v1/riskfolio/target_positions.parquet",
    "B003": BENCHMARK_ROOT / "gtja191_composite169_ic_topn/results/gtja191_composite169_ic_topn_v1/target_positions.parquet",
    "B004": BENCHMARK_ROOT / "gtja191_composite169_ic_meanvar_hist/results/gtja191_composite169_ic_meanvar_hist_v1/target_positions.parquet",
    "B005": BENCHMARK_ROOT / "linear_ridge_h1_topn/results/linear_ridge_h1_topn_v1/target_positions.parquet",
    "B006": BENCHMARK_ROOT / "linear_ridge_h1_meanvar_hist/results/linear_ridge_h1_meanvar_hist_v1/target_positions.parquet",
    "B007": BENCHMARK_ROOT / "tree_xgboost_h1_topn/results/tree_xgboost_h1_topn_v1/target_positions.parquet",
    "B008": BENCHMARK_ROOT / "tree_xgboost_h1_meanvar_hist/results/tree_xgboost_h1_meanvar_hist_v1/target_positions.parquet",
}
BENCHMARKS_BY_ID = {benchmark_id.upper(): path for benchmark_id, path in BENCHMARKS.items()}

DEFAULT_CONFIG = {
    "init_cash": 10_000_000,
    "benchmark_index": "000300.SH",
}
DEFAULT_START_DATE = "2024-01-01"


def _resolve_path(value, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _normalize_run_range(start=None, end=None) -> tuple[str, str | None]:
    """Normalize public CLI/YAML date bounds and validate their order."""
    start_value = DEFAULT_START_DATE if start in {None, ""} else start
    start_date = pd.Timestamp(start_value)
    if pd.isna(start_date):
        raise ValueError("start 不是有效日期")
    if start_date.tz is not None:
        start_date = start_date.tz_localize(None)
    start_date = start_date.normalize()

    end_text = None
    if end not in {None, ""}:
        end_date = pd.Timestamp(end)
        if pd.isna(end_date):
            raise ValueError("end 不是有效日期")
        if end_date.tz is not None:
            end_date = end_date.tz_localize(None)
        end_date = end_date.normalize()
        if end_date < start_date:
            raise ValueError("end 不能早于 start")
        end_text = str(end_date.date())
    return str(start_date.date()), end_text


def _pop_run_range(config: dict) -> tuple[str, str | None]:
    """Remove start/end from YAML backtest config before engine dispatch."""
    return _normalize_run_range(
        config.pop("start", DEFAULT_START_DATE),
        config.pop("end", None),
    )


def _run_exposure_analysis(
    pf,
    exposure_cfg: dict,
    out_dir: Path,
    *,
    base_dir: Path,
    default_benchmark: str | None,
):
    """Run optional B/f analysis without changing the Portfolio return type."""
    if not exposure_cfg.get("enabled", False):
        return None
    root_value = exposure_cfg.get("risk_model_root")
    if not root_value:
        raise ValueError("analysis.exposure.enabled=true 时必须配置 risk_model_root")
    risk_model_root = _resolve_path(root_value, base_dir)
    benchmark_index = exposure_cfg.get("benchmark_index", default_benchmark)
    benchmark_data_root = exposure_cfg.get("benchmark_data_root")
    if benchmark_index and benchmark_data_root is None:
        default_data_root = _PROJ.parent / "lqtp_data"
        if default_data_root.is_dir():
            benchmark_data_root = default_data_root
    if benchmark_data_root is not None:
        benchmark_data_root = _resolve_path(benchmark_data_root, base_dir)

    from vectorbt_qs.mvp.analysis import (
        analyze_portfolio_exposure,
        export_exposure_dashboard,
        export_industry_exposure_pngs,
        export_style_exposure_pngs,
    )

    print(f"[exposure] 风险模型: {risk_model_root}")
    result = analyze_portfolio_exposure(
        pf,
        risk_model_root=risk_model_root,
        benchmark_index=benchmark_index,
        benchmark_data_root=benchmark_data_root,
        weight_source=exposure_cfg.get("weight_source", "realized_close"),
        missing_policy=exposure_cfg.get("missing_policy", "report_unknown"),
        min_portfolio_coverage=float(
            exposure_cfg.get("min_portfolio_coverage", 0.98)
        ),
        min_benchmark_coverage=float(
            exposure_cfg.get("min_benchmark_coverage", 0.995)
        ),
        include_attribution=bool(exposure_cfg.get("include_attribution", True)),
    )
    exposure_dir = result.export(out_dir / "exposure")
    print(f"[exposure] 分析结果已保存: {exposure_dir}")
    if exposure_cfg.get("plot", True):
        pngs = export_style_exposure_pngs(
            result,
            exposure_dir,
            dpi=int(exposure_cfg.get("png_dpi", 150)),
        )
        industry_pngs = export_industry_exposure_pngs(
            result,
            exposure_dir,
            top_n=int(exposure_cfg.get("industry_top_n", 10)),
            dpi=int(exposure_cfg.get("png_dpi", 150)),
        )
        dashboard = export_exposure_dashboard(
            result,
            exposure_dir / "exposure_dashboard.html",
            include_plotlyjs=exposure_cfg.get("include_plotlyjs", "cdn"),
        )
        print(
            "[exposure] 风格暴露 PNG 已保存: "
            + ", ".join(str(path) for path in pngs.values())
        )
        print(
            "[exposure] 行业暴露 PNG 已保存: "
            + ", ".join(str(path) for path in industry_pngs.values())
        )
        print(f"[exposure] 交互式报告已保存: {dashboard}")
    return result


# ============================================================
# 子命令: backtest
# ============================================================
def cmd_backtest(args):
    """运行单次回测"""
    from vectorbt_qs.mvp.engine.runner import run_backtest, portfolio_report

    # 加载 target_positions
    if args.benchmark:
        path = BENCHMARKS_BY_ID.get(args.benchmark.upper())
        if path is None:
            print(f"未知 benchmark: {args.benchmark}")
            print(f"可用: {', '.join(sorted(BENCHMARKS))}")
            sys.exit(1)
        if not path.exists():
            print(f"Benchmark 文件不存在: {path}")
            sys.exit(1)
        print(f"[benchmark] {args.benchmark} → {path}")
    elif args.positions:
        path = Path(args.positions)
        if not path.exists():
            print(f"文件不存在: {path}")
            sys.exit(1)
    else:
        print("请指定 --benchmark 或 --positions")
        sys.exit(1)

    tw = pd.read_parquet(path)
    print(f"  权重: {tw.shape[0]} 天 × {tw.shape[1]} 只")
    print(f"  日期: {tw.index.min().date()} ~ {tw.index.max().date()}")

    # 回测配置
    cfg = {**DEFAULT_CONFIG}
    if args.init_cash is not None:
        cfg["init_cash"] = args.init_cash
    if args.fees is not None:
        cfg["fees"] = args.fees
    if args.slippage is not None:
        cfg["slippage"] = args.slippage

    # 执行
    start, end = _normalize_run_range(args.start, args.end)
    pf = run_backtest(
        "ashare",
        tw,
        start=start,
        end=end,
        config=cfg,
    )

    # 输出
    stats = portfolio_report(pf)
    keys = [
        "Start Value", "End Value", "Total Return [%]",
        "Sharpe Ratio", "Max Drawdown [%]", "Total Trades",
    ]
    print("\n绩效:")
    for k in keys:
        v = stats.get(k, "N/A")
        print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.risk_model_root:
        out_dir = Path(args.output_dir or "examples/output").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_exposure_analysis(
            pf,
            {
                "enabled": True,
                "risk_model_root": args.risk_model_root,
                "benchmark_data_root": args.risk_data_root,
                "benchmark_index": cfg.get("benchmark_index"),
                "include_attribution": args.exposure_attribution,
                "plot": args.plot,
            },
            out_dir,
            base_dir=Path.cwd(),
            default_benchmark=cfg.get("benchmark_index"),
        )

    # 可选图表
    if args.plot:
        out_dir = Path(args.output_dir or "examples/output")
        out_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                        gridspec_kw={"height_ratios": [3, 1]})
        from vectorbt_qs.mvp.visualization import (
            build_nav_curves,
            export_plotly_dashboard,
        )

        nav_curves = build_nav_curves(pf)
        nav_handles = []
        nav_labels = []
        strategy_line, = ax1.plot(
            nav_curves.index,
            nav_curves["strategy_nav"],
            color="#1f77b4",
            linewidth=1.2,
        )
        nav_handles.append(strategy_line)
        nav_labels.append("Strategy NAV")
        if "benchmark_nav" in nav_curves:
            benchmark_symbol = getattr(pf, "_qs_benchmark_symbol", "Benchmark")
            benchmark_line, = ax1.plot(
                nav_curves.index,
                nav_curves["benchmark_nav"],
                color="#7f7f7f",
                linewidth=1.0,
            )
            excess_line, = ax1.plot(
                nav_curves.index,
                nav_curves["excess_nav"],
                color="#d62728",
                linewidth=1.0,
            )
            nav_handles.extend([benchmark_line, excess_line])
            nav_labels.extend([
                f"Benchmark NAV ({benchmark_symbol})",
                "Excess NAV",
            ])
        ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax1.set_ylabel("Net Value")
        ax1.set_title("Equity Curve", fontsize=14, fontweight="bold")
        ax1.legend(nav_handles, nav_labels, loc="best")
        ax1.grid(True, alpha=0.3)

        dd = pf.drawdown()
        ax2.fill_between(dd.index, dd.values * 100, 0, color="#d62728", alpha=0.6)
        ax2.set_ylabel("Drawdown %")
        ax2.set_xlabel("Date")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        path = out_dir / "equity_curve.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        dashboard_path = export_plotly_dashboard(
            pf,
            out_dir / "portfolio_dashboard.html",
            title="Portfolio Overview",
            include_plotlyjs="cdn",
        )
        print(f"\n图表已保存: {path}")
        print(f"交互式报告已保存: {dashboard_path}")


# ============================================================
# 子命令: run (配置文件模式)
# ============================================================
def cmd_run(args):
    """配置文件驱动的回测"""
    import yaml

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    from vectorbt_qs.mvp.engine.runner import run_backtest, portfolio_report

    # 加载仓位
    positions_path = Path(cfg["input"]["positions"])
    if not positions_path.is_absolute():
        positions_path = config_path.parent / positions_path
    tw = pd.read_parquet(positions_path)
    print(f"[config] {config_path}")
    print(f"  权重: {tw.shape[0]} 天 × {tw.shape[1]} 只")

    # 回测配置
    bt_cfg = {**DEFAULT_CONFIG, **cfg.get("backtest", {})}
    start, end = _pop_run_range(bt_cfg)
    market = cfg.get("market", "ashare")

    # 执行
    pf = run_backtest(
        market,
        tw,
        start=start,
        end=end,
        config=bt_cfg,
    )

    # 输出（默认 = examples/output/{config文件名}）
    stats = portfolio_report(pf)
    out_dir = cfg.get("output", {}).get("dir")
    if out_dir:
        out_dir = Path(out_dir)
        if not out_dir.is_absolute():
            out_dir = _PROJ / out_dir
    else:
        out_dir = _PROJ / "examples" / "output" / config_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_dir / "performance_stats.csv")
    print(f"  绩效已保存: {out_dir / 'performance_stats.csv'}")

    exposure_cfg = (cfg.get("analysis", {}) or {}).get("exposure", {}) or {}
    _run_exposure_analysis(
        pf,
        exposure_cfg,
        out_dir,
        base_dir=config_path.parent.resolve(),
        default_benchmark=bt_cfg.get("benchmark_index"),
    )

    # 可视化（默认开启；config 中可设 output.plot: false 关闭）
    if cfg.get("output", {}).get("plot", True):
        _generate_charts(pf, stats, out_dir, bt_cfg["init_cash"])


# ============================================================
# 子命令: batch (参数网格批量回测)
# ============================================================
def _resolve_batch_positions(raw_positions, config_path: Path):
    """Load one or more named target-weight matrices from batch YAML."""
    if isinstance(raw_positions, (str, os.PathLike)):
        entries = [(Path(raw_positions).stem, raw_positions)]
    elif isinstance(raw_positions, list):
        entries = [(Path(value).stem, value) for value in raw_positions]
    elif isinstance(raw_positions, dict):
        entries = list(raw_positions.items())
    else:
        raise TypeError(
            "batch 的 input.positions 必须是路径、路径列表或 {名称: 路径} 映射"
        )
    if not entries:
        raise ValueError("batch 的 input.positions 不能为空")

    weights = {}
    sources = {}
    for raw_label, raw_path in entries:
        label = str(raw_label).strip()
        if not label:
            raise ValueError("batch 输入名称不能为空")
        if label in weights:
            raise ValueError(
                f"batch 输入名称重复: {label}；请改用 {{名称: 路径}} 显式命名"
            )
        path = _resolve_path(raw_path, config_path.parent)
        if not path.exists():
            raise FileNotFoundError(f"目标权重文件不存在: {path}")
        frame = pd.read_parquet(path)
        weights[label] = frame
        sources[label] = str(path)
        print(
            f"[batch/input] {label}: {frame.shape[0]} 天 × "
            f"{frame.shape[1]} 只 → {path}"
        )
    return weights, sources


def _export_batch_results(
    result,
    out_dir: Path,
    *,
    plot: bool,
    run_directories: dict[str, Path] | None = None,
):
    """Persist the exact manifest, per-run statistics and comparison NAV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parameters_path = out_dir / "batch_parameters.csv"
    performance_path = out_dir / "performance_comparison.csv"
    nav_path = out_dir / "nav_curves.parquet"
    configs_path = out_dir / "effective_configs.json"
    comparison = result.compare()
    performance = result.parameters.join(comparison, how="left")
    result.parameters.to_csv(parameters_path, encoding="utf-8-sig")
    performance.to_csv(performance_path, encoding="utf-8-sig")
    with open(configs_path, "w", encoding="utf-8") as file:
        json.dump(
            result.configs,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    for run_id, portfolio in result.portfolios.items():
        run_dir = (
            run_directories[run_id]
            if run_directories is not None
            else out_dir / "runs" / run_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        comparison.loc[run_id].to_csv(
            run_dir / "performance_stats.csv",
            encoding="utf-8-sig",
        )

    nav = result.nav_curves()
    if not nav.empty:
        nav.to_parquet(nav_path)

    if plot and not nav.empty:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 8))
        for column in nav:
            ax.plot(nav.index, nav[column], linewidth=1.0, label=column)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title("Batch Strategy NAV Comparison")
        ax.set_xlabel("Date")
        ax.set_ylabel("NAV")
        ax.grid(True, alpha=0.3)
        if len(nav.columns) <= 20:
            ax.legend(loc="best", ncol=max(1, min(4, len(nav.columns) // 6 + 1)))
        fig.tight_layout()
        fig.savefig(
            out_dir / "batch_nav_curves.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

        import plotly.graph_objects as go

        figure = go.Figure()
        for column in nav:
            figure.add_trace(
                go.Scatter(
                    x=nav.index,
                    y=nav[column],
                    mode="lines",
                    name=column,
                )
            )
        figure.add_hline(y=1.0, line_dash="dash", line_color="gray")
        figure.update_layout(
            title="Batch Strategy NAV Comparison",
            xaxis_title="Date",
            yaxis_title="NAV",
            hovermode="x unified",
            template="plotly_white",
        )
        figure.write_html(
            out_dir / "batch_nav_curves.html",
            include_plotlyjs="cdn",
        )

    print(f"[batch] 参数清单: {parameters_path}")
    print(f"[batch] 完整配置: {configs_path}")
    print(f"[batch] 绩效对比: {performance_path}")
    if not nav.empty:
        print(f"[batch] NAV 数据: {nav_path}")
    return comparison


def _configure_local_ashare_data() -> Path:
    """Use the existing local lqtp_data without adding another required input."""
    configured = os.environ.get("VECTORBT_QS_ASHARE_DATA_ROOT")
    data_root = (
        Path(configured).resolve()
        if configured
        else (_PROJ.parent / "lqtp_data").resolve()
    )
    if not data_root.is_dir():
        raise FileNotFoundError(
            "standard_accurate_v2 要求本地 lqtp_data；"
            f"未找到 {data_root}。可设置 VECTORBT_QS_ASHARE_DATA_ROOT。"
        )
    # data_access reads these dynamically when creating its first store.
    # Child processes inherit them on Windows spawn.
    os.environ["ASHARE_PARQUET_ROOT"] = str(data_root)
    os.environ["DATA_ACCESS_SKIP_COS_MIRROR"] = "1"
    os.environ["VECTORBT_QS_ASHARE_DATA_ROOT"] = str(data_root)
    from vectorbt_qs.mvp.data.adapter import set_data_root

    set_data_root("ashare", str(data_root))
    return data_root


def _attach_benchmark_view(portfolio, benchmark_index: str):
    """Shallow-clone one trading result and attach an analysis benchmark."""
    from vectorbt_qs.mvp.data.adapter import load_benchmark_close

    view = copy.copy(portfolio)
    index = pd.DatetimeIndex(portfolio.wrapper.index)
    benchmark_close = load_benchmark_close(
        "ashare",
        benchmark_index,
        start=str(index.min().date()),
        end=str(index.max().date()),
    ).reindex(index).ffill()
    view._qs_benchmark_close = benchmark_close
    view._qs_benchmark_returns = benchmark_close.pct_change(fill_method=None)
    view._qs_benchmark_symbol = benchmark_index
    return view


def _expand_standard_accurate_benchmarks(execution_result):
    """Expand trading simulations into benchmark-specific report views."""
    from vectorbt_qs.mvp.engine import (
        BatchBacktestResult,
        apply_parameter_overrides,
        expand_parameter_grid,
        standard_accurate_v2_base_config,
        standard_accurate_v2_grid,
    )

    return _expand_accurate_benchmark_views(
        execution_result,
        base_config=standard_accurate_v2_base_config(),
        report_grid=standard_accurate_v2_grid(),
    )


def _expand_accurate_benchmark_views(
    execution_result,
    *,
    base_config: dict,
    report_grid: dict,
):
    """Attach each analysis benchmark without repeating trading simulations."""
    from vectorbt_qs.mvp.engine import (
        BatchBacktestResult,
        apply_parameter_overrides,
        expand_parameter_grid,
    )

    trading_fields = ("price_type", "costs.commission", "freq")
    source_by_key = {
        tuple(row[field] for field in trading_fields): run_id
        for run_id, row in execution_result.parameters.iterrows()
    }
    portfolios = {}
    configs = {}
    rows = []
    for run_number, overrides in enumerate(
        expand_parameter_grid(report_grid),
        start=1,
    ):
        run_id = f"run_{run_number:04d}"
        source_key = tuple(overrides[field] for field in trading_fields)
        source_id = source_by_key[source_key]
        view = _attach_benchmark_view(
            execution_result.portfolios[source_id],
            str(overrides["benchmark_index"]),
        )
        view._qs_source_run_id = source_id
        portfolios[run_id] = view
        configs[run_id] = apply_parameter_overrides(
            base_config,
            overrides,
        )
        rows.append(
            {
                "run_id": run_id,
                "positions": execution_result.parameters.loc[
                    source_id,
                    "positions",
                ],
                **overrides,
                "source_run_id": source_id,
                "status": "backtest_success",
                "error": "",
            }
        )
    parameters = pd.DataFrame(rows).set_index("run_id")
    parameters.index.name = "run_id"
    return BatchBacktestResult(
        portfolios=portfolios,
        parameters=parameters,
        configs=configs,
        errors={},
    )


def _preflight_standard_accurate_inputs(
    weights: pd.DataFrame,
    barra_path: Path,
    data_root: Path,
    *,
    start: str,
    end: str | None,
    benchmarks: tuple[str, ...] | None = None,
    minimum_benchmark_coverage: float | None = None,
) -> dict:
    """Fail fast on risk schemas and target/benchmark exposure coverage."""
    import pyarrow.parquet as pq

    from vectorbt_qs.mvp.analysis import RiskModelStore
    from vectorbt_qs.mvp.analysis.weights import load_benchmark_weights
    from vectorbt_qs.mvp.engine import (
        STANDARD_ACCURATE_V2_MIN_BENCHMARK_COVERAGE,
        standard_accurate_v2_benchmarks,
    )

    dividend_root = data_root / "StockDividend"
    if not dividend_root.is_dir():
        raise FileNotFoundError(
            f"准确回测要求公司行为目录: {dividend_root}"
        )
    dividend_files = dividend_root.rglob("*.parquet")
    dividend_sample = next(dividend_files, None)
    if dividend_sample is None:
        raise FileNotFoundError(
            f"公司行为目录没有 parquet 文件: {dividend_root}"
        )
    dividend_columns = set(pq.read_schema(dividend_sample).names)
    required_dividend_columns = {
        "TradeDate",
        "Symbol",
        "CashDividend",
        "StockDividend",
        "StockTransfer",
    }
    missing_dividend_columns = required_dividend_columns - dividend_columns
    if missing_dividend_columns:
        raise ValueError(
            "StockDividend schema 缺少字段: "
            + ", ".join(sorted(missing_dividend_columns))
        )

    store = RiskModelStore(barra_path)
    selected = weights.loc[weights.index >= pd.Timestamp(start)]
    if end is not None:
        selected = selected.loc[selected.index <= pd.Timestamp(end)]
    if selected.empty:
        raise ValueError("指定日期范围内没有目标权重")
    exposure_weights = selected.loc[
        (selected.index >= store.date_min)
        & (selected.index <= store.date_max)
    ].ffill().fillna(0.0)
    if exposure_weights.empty:
        raise ValueError(
            "回测区间与风险模型区间没有交集："
            f"{store.date_min.date()} ~ {store.date_max.date()}"
        )

    def validate_coverage(frame: pd.DataFrame, minimum: float, label: str) -> None:
        invested = frame["total_abs_weight"] > 1e-12
        failed = frame.loc[
            invested & frame["coverage_weight"].lt(float(minimum))
        ]
        if not failed.empty:
            date = failed.index[0]
            raise ValueError(
                f"预检失败：{label} {date.date()} 风险暴露覆盖率 "
                f"{failed.iloc[0]['coverage_weight']:.4%} "
                f"低于门槛 {float(minimum):.4%}"
            )

    target_agg = store.aggregate_exposure(
        exposure_weights,
        missing_policy="report_unknown",
    )
    validate_coverage(target_agg.coverage, 0.98, "目标组合")
    if benchmarks is None:
        benchmarks = standard_accurate_v2_benchmarks()
    benchmark_minimum = (
        STANDARD_ACCURATE_V2_MIN_BENCHMARK_COVERAGE
        if minimum_benchmark_coverage is None
        else float(minimum_benchmark_coverage)
    )
    for benchmark in benchmarks:
        benchmark_weights = load_benchmark_weights(
            data_root,
            benchmark,
            exposure_weights.index,
        )
        benchmark_agg = store.aggregate_exposure(
            benchmark_weights,
            missing_policy="report_unknown",
        )
        validate_coverage(
            benchmark_agg.coverage,
            benchmark_minimum,
            f"基准 {benchmark}",
        )

    backtest_min = selected.index.min().normalize()
    backtest_max = selected.index.max().normalize()
    truncated = backtest_min < store.date_min or backtest_max > store.date_max
    if truncated:
        print(
            "[preflight] 注意：风险分析将截断到 "
            f"{store.date_min.date()} ~ {store.date_max.date()}；"
            f"回测目标区间为 {backtest_min.date()} ~ {backtest_max.date()}"
        )
    print(
        "[preflight] 风险模型 schema、目标组合及"
        f"{len(benchmarks)} 个基准覆盖率检查通过"
    )
    return {
        "risk_model_version": store.version,
        "risk_model_date_min": str(store.date_min.date()),
        "risk_model_date_max": str(store.date_max.date()),
        "backtest_target_date_min": str(backtest_min.date()),
        "backtest_target_date_max": str(backtest_max.date()),
        "exposure_truncated": bool(truncated),
        "corporate_action_source": str(dividend_root),
        "corporate_action_schema_validated": True,
    }


def _deliver_accurate_profile_results(
    result,
    *,
    profile_name: str,
    positions_path: Path,
    barra_path: Path,
    data_root: Path,
    output_path: Path,
    output_folder_name: str,
    base_dir: Path,
    start: str,
    end: str | None,
    workers: int,
    preflight: dict,
    minimum_benchmark_coverage: float,
    folder_name_builder=None,
):
    """Export reports and exposure analysis for one frozen accurate profile."""
    from vectorbt_qs.mvp.visualization import build_nav_curves

    portfolio_root = output_path / output_folder_name
    result.parameters["positions"] = positions_path.stem
    result.parameters.insert(1, "positions_path", str(positions_path))
    result.parameters.insert(2, "start", start)
    result.parameters.insert(3, "end", end or "")
    result.parameters.insert(4, "workers", workers)

    run_directories: dict[str, Path] = {}
    folder_names: list[str] = []
    for run_id, row in result.parameters.iterrows():
        if folder_name_builder is None:
            folder_name = output_folder_name
            run_directory = portfolio_root
        else:
            folder_name = folder_name_builder(row)
            run_directory = portfolio_root / folder_name
        folder_names.append(folder_name)
        run_directories[run_id] = run_directory
        result.configs[run_id].update(
            {
                "_profile": profile_name,
                "_positions_path": str(positions_path),
                "_barra_root": str(barra_path),
                "_data_root": str(data_root),
                "_start": start,
                "_end": end,
                "_workers": workers,
                "_preflight": preflight,
                "_result_directory": str(run_directory),
            }
        )
    if len(folder_names) != len(set(folder_names)):
        raise ValueError("冻结参数生成了重复的结果目录名")
    result.parameters.insert(5, "result_folder", folder_names)
    result.parameters["backtest_status"] = result.parameters["status"]
    result.parameters["status"] = "report_pending"

    comparison = _export_batch_results(
        result,
        portfolio_root,
        plot=True,
        run_directories=run_directories,
    )

    exposure_cfg = {
        "enabled": True,
        "risk_model_root": str(barra_path),
        "benchmark_data_root": str(data_root),
        "weight_source": "realized_close",
        "missing_policy": "report_unknown",
        "min_portfolio_coverage": 0.98,
        "min_benchmark_coverage": minimum_benchmark_coverage,
        # B/f attribution is explicitly ex-post and experimental. Frozen
        # delivery exports exposure only unless a future profile opts in.
        "include_attribution": False,
        "plot": True,
        "png_dpi": 150,
        "industry_top_n": 10,
        "include_plotlyjs": "cdn",
    }
    delivery_errors: dict[str, str] = {}

    def persist_delivery_status() -> None:
        result.parameters.to_csv(
            portfolio_root / "batch_parameters.csv",
            encoding="utf-8-sig",
        )
        result.parameters.join(comparison, how="left").to_csv(
            portfolio_root / "performance_comparison.csv",
            encoding="utf-8-sig",
        )

    for run_id, portfolio in result.portfolios.items():
        run_dir = run_directories[run_id]
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            nav_curves = build_nav_curves(portfolio)
            nav_curves.to_parquet(run_dir / "nav_curves.parquet")
            with open(run_dir / "effective_config.json", "w", encoding="utf-8") as file:
                json.dump(
                    result.configs[run_id],
                    file,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            _generate_charts(
                portfolio,
                comparison.loc[run_id],
                run_dir,
                result.configs[run_id]["init_cash"],
            )
            _run_exposure_analysis(
                portfolio,
                exposure_cfg,
                run_dir,
                base_dir=base_dir,
                default_benchmark=result.configs[run_id]["benchmark_index"],
            )
            result.parameters.loc[run_id, "status"] = (
                "complete_with_exposure_truncation"
                if preflight["exposure_truncated"]
                else "complete"
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            result.parameters.loc[run_id, "status"] = "failed"
            result.parameters.loc[run_id, "error"] = message
            delivery_errors[run_id] = message
            print(f"[profile] {run_id} 报告生成失败: {message}")
        finally:
            persist_delivery_status()

    if delivery_errors:
        sample_id = next(iter(delivery_errors))
        raise RuntimeError(
            f"{len(delivery_errors)} 组报告生成失败；"
            f"首个错误 {sample_id}: {delivery_errors[sample_id]}。"
            f"完整状态见 {portfolio_root / 'batch_parameters.csv'}"
        )

    print(
        f"[profile] {profile_name} 完成，"
        f"{len(result.portfolios)} 组结果已保存至 {portfolio_root}"
    )
    return result


def _run_standard_accurate_v2(
    *,
    positions,
    barra_root,
    output_root,
    base_dir: Path,
    start=DEFAULT_START_DATE,
    end=None,
    workers: int = 1,
):
    """Run the frozen 24-point accurate grid and export named result folders."""
    from vectorbt_qs.mvp.engine import (
        STANDARD_ACCURATE_V2,
        STANDARD_ACCURATE_V2_MIN_BENCHMARK_COVERAGE,
        run_backtest_batch,
        standard_accurate_v2_base_config,
        standard_accurate_v2_execution_grid,
        standard_accurate_v2_execution_run_count,
        standard_accurate_v2_folder_name,
        standard_accurate_v2_run_count,
    )
    positions_path = _resolve_path(positions, base_dir)
    barra_path = _resolve_path(barra_root, base_dir)
    output_path = _resolve_path(output_root, base_dir)
    if not positions_path.is_file():
        raise FileNotFoundError(f"目标权重文件不存在: {positions_path}")
    if not barra_path.is_dir():
        raise FileNotFoundError(f"Barra 因子目录不存在: {barra_path}")
    start, end = _normalize_run_range(start, end)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers 必须是正整数")
    data_root = _configure_local_ashare_data()

    weights = pd.read_parquet(positions_path)
    preflight = _preflight_standard_accurate_inputs(
        weights,
        barra_path,
        data_root,
        start=start,
        end=end,
    )
    print(f"[profile] {STANDARD_ACCURATE_V2}")
    print(f"[profile] 仓位: {positions_path}")
    print(f"[profile] Barra: {barra_path}")
    print(f"[profile] 区间: {start} ~ {end or '仓位数据末日'}")
    print(f"[profile] 固定生成 {standard_accurate_v2_run_count()} 组准确回测")
    print(
        f"[profile] 实际交易计算 {standard_accurate_v2_execution_run_count()} 组，"
        "每组复用 3 个基准报告"
    )
    print(f"[profile] 回测进程数: {workers}")

    execution_result = run_backtest_batch(
        "ashare",
        weights,
        standard_accurate_v2_execution_grid(),
        base_config=standard_accurate_v2_base_config(),
        start=start,
        end=end,
        max_runs=standard_accurate_v2_execution_run_count(),
        on_error="raise",
        workers=workers,
        data_root=str(data_root),
    )
    result = _expand_standard_accurate_benchmarks(execution_result)

    return _deliver_accurate_profile_results(
        result,
        profile_name=STANDARD_ACCURATE_V2,
        positions_path=positions_path,
        barra_path=barra_path,
        data_root=data_root,
        output_path=output_path,
        output_folder_name=positions_path.stem,
        base_dir=base_dir,
        start=start,
        end=end,
        workers=workers,
        preflight=preflight,
        minimum_benchmark_coverage=(
            STANDARD_ACCURATE_V2_MIN_BENCHMARK_COVERAGE
        ),
        folder_name_builder=standard_accurate_v2_folder_name,
    )


def _positions_parent_folder_name(positions_path: Path) -> str:
    """Use the position file's containing folder, falling back to its stem."""
    folder_name = positions_path.parent.name.strip()
    return folder_name or positions_path.stem


def _run_standard_accurate_benchmark_v1(
    *,
    positions,
    barra_root,
    output_root,
    base_dir: Path,
    start=DEFAULT_START_DATE,
    end=None,
):
    """Run one canonical Accurate V2 configuration and export one folder."""
    from vectorbt_qs.mvp.engine import (
        STANDARD_ACCURATE_BENCHMARK_V1,
        STANDARD_ACCURATE_BENCHMARK_V1_MIN_BENCHMARK_COVERAGE,
        run_backtest_batch,
        standard_accurate_benchmark_v1_base_config,
        standard_accurate_benchmark_v1_execution_grid,
        standard_accurate_benchmark_v1_grid,
    )

    positions_path = _resolve_path(positions, base_dir)
    barra_path = _resolve_path(barra_root, base_dir)
    output_path = _resolve_path(output_root, base_dir)
    if not positions_path.is_file():
        raise FileNotFoundError(f"目标权重文件不存在: {positions_path}")
    if not barra_path.is_dir():
        raise FileNotFoundError(f"Barra 因子目录不存在: {barra_path}")
    start, end = _normalize_run_range(start, end)
    data_root = _configure_local_ashare_data()

    weights = pd.read_parquet(positions_path)
    report_grid = standard_accurate_benchmark_v1_grid()
    benchmark = str(report_grid["benchmark_index"][0])
    preflight = _preflight_standard_accurate_inputs(
        weights,
        barra_path,
        data_root,
        start=start,
        end=end,
        benchmarks=(benchmark,),
        minimum_benchmark_coverage=(
            STANDARD_ACCURATE_BENCHMARK_V1_MIN_BENCHMARK_COVERAGE
        ),
    )
    print(f"[profile] {STANDARD_ACCURATE_BENCHMARK_V1}")
    print(f"[profile] 仓位: {positions_path}")
    print(f"[profile] Barra: {barra_path}")
    print(f"[profile] 区间: {start} ~ {end or '仓位数据末日'}")
    print(
        "[profile] 固定配置: benchmark=000852.SH, price=vwap, "
        "freq=1D, commission=0.0005, slippage=0.001"
    )

    base_config = standard_accurate_benchmark_v1_base_config()
    execution_result = run_backtest_batch(
        "ashare",
        weights,
        standard_accurate_benchmark_v1_execution_grid(),
        base_config=base_config,
        start=start,
        end=end,
        max_runs=1,
        on_error="raise",
        workers=1,
        data_root=str(data_root),
    )
    result = _expand_accurate_benchmark_views(
        execution_result,
        base_config=base_config,
        report_grid=report_grid,
    )
    return _deliver_accurate_profile_results(
        result,
        profile_name=STANDARD_ACCURATE_BENCHMARK_V1,
        positions_path=positions_path,
        barra_path=barra_path,
        data_root=data_root,
        output_path=output_path,
        output_folder_name=_positions_parent_folder_name(positions_path),
        base_dir=base_dir,
        start=start,
        end=end,
        workers=1,
        preflight=preflight,
        minimum_benchmark_coverage=(
            STANDARD_ACCURATE_BENCHMARK_V1_MIN_BENCHMARK_COVERAGE
        ),
        folder_name_builder=None,
    )


def cmd_accurate_batch(args):
    """Three-path public interface for the frozen accurate batch profile."""
    _run_standard_accurate_v2(
        positions=args.positions,
        barra_root=args.barra_root,
        output_root=args.output_root,
        base_dir=Path.cwd(),
        start=args.start,
        end=args.end,
        workers=args.workers,
    )


def cmd_accurate_benchmark(args):
    """Three-path public interface for one canonical accurate configuration."""
    _run_standard_accurate_benchmark_v1(
        positions=args.positions,
        barra_root=args.barra_root,
        output_root=args.output_root,
        base_dir=Path.cwd(),
        start=args.start,
        end=args.end,
    )


def cmd_batch(args):
    """Run a Cartesian parameter grid from one YAML config."""
    import yaml

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)
    with open(config_path, encoding="utf-8") as file:
        cfg = yaml.safe_load(file) or {}

    batch_cfg = cfg.get("batch") or {}
    profile = batch_cfg.get("profile")
    if profile is not None:
        from vectorbt_qs.mvp.engine import (
            STANDARD_ACCURATE_V2,
            standard_accurate_v2_base_config,
            standard_accurate_v2_grid,
        )

        if profile != STANDARD_ACCURATE_V2:
            raise ValueError(
                f"未知 batch.profile={profile!r}；当前仅支持 {STANDARD_ACCURATE_V2}"
            )
        provided_grid = batch_cfg.get("grid")
        if provided_grid is not None:
            expected_grid = {
                key: list(values)
                for key, values in standard_accurate_v2_grid().items()
            }
            if provided_grid != expected_grid:
                raise ValueError(
                    f"{STANDARD_ACCURATE_V2} 的 batch.grid 是冻结矩阵；"
                    "可以完整写出用于说明，但不能增删或修改候选值"
                )
        if cfg.get("market", "ashare") != "ashare":
            raise ValueError(f"{STANDARD_ACCURATE_V2} 固定使用 market=ashare")
        backtest_cfg = dict(cfg.get("backtest") or {})
        frozen_overrides = {
            key: value
            for key, value in backtest_cfg.items()
            if key not in {"start", "end"}
        }
        expected_base = standard_accurate_v2_base_config()
        unknown_backtest = set(frozen_overrides).difference(expected_base)
        if unknown_backtest:
            raise ValueError(
                f"{STANDARD_ACCURATE_V2} 包含未知 backtest 参数: "
                f"{', '.join(sorted(unknown_backtest))}"
            )
        for key, value in frozen_overrides.items():
            if value != expected_base[key]:
                raise ValueError(
                    f"{STANDARD_ACCURATE_V2} 的 backtest.{key} 是冻结值 "
                    f"{expected_base[key]!r}，不能改为 {value!r}"
                )
        input_cfg = cfg.get("input") or {}
        output_cfg = cfg.get("output") or {}
        if "positions" not in input_cfg or "barra_root" not in input_cfg:
            raise ValueError(
                f"{STANDARD_ACCURATE_V2} 要求 input.positions 和 input.barra_root"
            )
        if "root" not in output_cfg:
            raise ValueError(f"{STANDARD_ACCURATE_V2} 要求 output.root")
        _run_standard_accurate_v2(
            positions=input_cfg["positions"],
            barra_root=input_cfg["barra_root"],
            output_root=output_cfg["root"],
            base_dir=config_path.parent,
            start=backtest_cfg.get("start", DEFAULT_START_DATE),
            end=backtest_cfg.get("end"),
            workers=int(batch_cfg.get("workers", 1)),
        )
        return

    parameter_grid = batch_cfg.get("grid")
    if parameter_grid is None:
        raise ValueError("batch 配置必须包含 batch.grid")

    from vectorbt_qs.mvp.engine import run_backtest_batch

    weights, sources = _resolve_batch_positions(
        cfg["input"]["positions"],
        config_path,
    )
    base_config = {**DEFAULT_CONFIG, **(cfg.get("backtest") or {})}
    start, end = _pop_run_range(base_config)
    market = cfg.get("market", "ashare")
    batch_data_root = (
        _configure_local_ashare_data()
        if market == "ashare" and (_PROJ.parent / "lqtp_data").is_dir()
        else None
    )
    workers = int(batch_cfg.get("workers", 1))
    result = run_backtest_batch(
        market,
        weights,
        parameter_grid,
        base_config=base_config,
        start=start,
        end=end,
        max_runs=int(batch_cfg.get("max_runs", 256)),
        on_error=batch_cfg.get("on_error", "raise"),
        workers=workers,
        data_root=str(batch_data_root) if batch_data_root is not None else None,
    )
    result.parameters.insert(
        1,
        "positions_path",
        result.parameters["positions"].map(sources),
    )
    result.parameters.insert(2, "start", start)
    result.parameters.insert(3, "end", end or "")
    result.parameters.insert(4, "workers", workers)

    output_cfg = cfg.get("output") or {}
    out_dir = output_cfg.get("dir")
    if out_dir:
        out_dir = Path(out_dir)
        if not out_dir.is_absolute():
            out_dir = _PROJ / out_dir
    else:
        out_dir = _PROJ / "examples" / "output" / f"{config_path.stem}_batch"

    comparison = _export_batch_results(
        result,
        out_dir,
        plot=bool(output_cfg.get("plot", True)),
    )

    exposure_cfg = (cfg.get("analysis", {}) or {}).get("exposure", {}) or {}
    plot_each = bool(output_cfg.get("plot_each", False))
    for run_id, portfolio in result.portfolios.items():
        run_dir = out_dir / "runs" / run_id
        _run_exposure_analysis(
            portfolio,
            exposure_cfg,
            run_dir,
            base_dir=config_path.parent,
            default_benchmark=result.configs[run_id].get("benchmark_index"),
        )
        if plot_each:
            stats = comparison.loc[run_id]
            _generate_charts(
                portfolio,
                stats,
                run_dir,
                result.configs[run_id]["init_cash"],
            )

    print(
        f"[batch] 完成: {len(result.portfolios)} 成功, "
        f"{len(result.errors)} 失败, 输出目录 {out_dir}"
    )


# ============================================================
# 可视化
# ============================================================
def _generate_charts(pf, stats, out_dir, init_cash):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    ret = stats.get("Total Return [%]", 0)
    dd = stats.get("Max Drawdown [%]", 0)
    sharpe = stats.get("Sharpe Ratio", 0)

    # --- 净值曲线 + 回撤 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    from vectorbt_qs.mvp.visualization import (
        build_nav_curves,
        export_plotly_dashboard,
    )

    nav_curves = build_nav_curves(pf)
    nav_handles = []
    nav_labels = []
    strategy_line, = ax1.plot(
        nav_curves.index,
        nav_curves["strategy_nav"],
        color="#1f77b4",
        linewidth=1.2,
    )
    nav_handles.append(strategy_line)
    nav_labels.append("Strategy NAV")
    if "benchmark_nav" in nav_curves:
        benchmark_symbol = getattr(pf, "_qs_benchmark_symbol", "Benchmark")
        benchmark_line, = ax1.plot(
            nav_curves.index,
            nav_curves["benchmark_nav"],
            color="#7f7f7f",
            linewidth=1.0,
        )
        excess_line, = ax1.plot(
            nav_curves.index,
            nav_curves["excess_nav"],
            color="#d62728",
            linewidth=1.0,
        )
        nav_handles.extend([benchmark_line, excess_line])
        nav_labels.extend([
            f"Benchmark NAV ({benchmark_symbol})",
            "Excess NAV",
        ])
    ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Net Value (x Initial)")
    ax1.set_title(f"Equity Curve  |  Return: {ret:.1f}%  DD: {dd:.1f}%  Sharpe: {sharpe:.2f}",
                  fontsize=14, fontweight="bold")
    ax1.legend(nav_handles, nav_labels, loc="best")
    ax1.grid(True, alpha=0.3)

    drawdown = pf.drawdown()
    ax2.fill_between(drawdown.index, drawdown.values * 100, 0, color="#d62728", alpha=0.6)
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "equity_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- 年度收益 ---
    returns_s = pf.returns()
    if isinstance(returns_s, pd.DataFrame):
        returns_s = returns_s.iloc[:, 0]
    yearly = returns_s.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in yearly.values]
    ax.bar(yearly.index.year.astype(str), yearly.values, color=colors, edgecolor="white")
    ax.axhline(y=0, color="gray", linewidth=0.8)
    for i, v in enumerate(yearly.values):
        ax.text(i, v + (2 if v >= 0 else -4), f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Annual Returns", fontsize=14, fontweight="bold")
    ax.set_ylabel("Return %")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "annual_returns.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    dashboard_path = export_plotly_dashboard(
        pf,
        out_dir / "portfolio_dashboard.html",
        title="Portfolio Overview",
        include_plotlyjs="cdn",
    )
    print(f"  图表已保存: {out_dir / 'equity_curve.png'}, ...")
    print(f"  交互式报告已保存: {dashboard_path}")


# ============================================================
# 子命令: list (列出可用 benchmark)
# ============================================================
def cmd_list(args):
    """列出所有可用 benchmark"""
    print("可用 Benchmarks:\n")
    print(f"{'ID':<8} {'文件':<60} {'存在'}")
    print("-" * 80)
    for bid, path in BENCHMARKS.items():
        # Keep the CLI usable in the default Windows GBK console.
        exists = "yes" if path.exists() else "no"
        print(f"{bid:<8} {str(path):<60} {exists}")


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        prog="vectorbt_qs",
        description="vectorbt_qs — 量化回测系统 v0.4",
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    # backtest
    p_bt = sub.add_parser("backtest", help="运行单次回测")
    p_bt.add_argument("--benchmark", "-b", help="Benchmark ID (B001-B008)")
    p_bt.add_argument("--positions", "-p", help="target_positions.parquet 路径")
    p_bt.add_argument("--init-cash", type=float, help="初始资金")
    p_bt.add_argument("--fees", type=float, help="兼容参数：双边统一总费率")
    p_bt.add_argument("--slippage", type=float, help="滑点")
    p_bt.add_argument(
        "--start",
        default=DEFAULT_START_DATE,
        help=f"回测开始日期（默认 {DEFAULT_START_DATE}）",
    )
    p_bt.add_argument("--end", help="回测结束日期（默认使用仓位数据末日）")
    p_bt.add_argument("--plot", action="store_true", help="生成静态图和 Plotly 交互报告")
    p_bt.add_argument("--output-dir", "-o", help="输出目录")
    p_bt.add_argument("--risk-model-root", help="启用组合暴露分析并指定 Barra-lite 目录")
    p_bt.add_argument("--risk-data-root", help="lqtp_data 根目录，用于读取指数成分权重")
    p_bt.add_argument(
        "--exposure-attribution",
        action="store_true",
        help="同时生成实验性 B×f 事后归因",
    )
    p_bt.set_defaults(func=cmd_backtest)

    # run
    p_run = sub.add_parser("run", help="配置文件驱动回测")
    p_run.add_argument("config", help="YAML 配置文件路径")
    p_run.set_defaults(func=cmd_run)

    # batch
    p_batch = sub.add_parser("batch", help="参数列表笛卡尔积批量回测")
    p_batch.add_argument("config", help="Batch YAML 配置文件路径")
    p_batch.set_defaults(func=cmd_batch)

    # frozen accurate batch
    p_accurate_batch = sub.add_parser(
        "accurate-batch",
        help="运行冻结的 standard_accurate_v2 批量回测",
    )
    p_accurate_batch.add_argument(
        "--positions",
        required=True,
        help="目标仓位 parquet 路径",
    )
    p_accurate_batch.add_argument(
        "--barra-root",
        required=True,
        help="Barra 因子文件夹路径",
    )
    p_accurate_batch.add_argument(
        "--output-root",
        required=True,
        help="回测结果根目录",
    )
    p_accurate_batch.add_argument(
        "--start",
        default=DEFAULT_START_DATE,
        help=f"回测开始日期（默认 {DEFAULT_START_DATE}）",
    )
    p_accurate_batch.add_argument(
        "--end",
        help="回测结束日期（默认使用仓位数据末日）",
    )
    p_accurate_batch.add_argument(
        "--workers",
        type=int,
        default=1,
        help="并行回测进程数（默认 1；建议 2-4）",
    )
    p_accurate_batch.set_defaults(func=cmd_accurate_batch)

    # canonical single accurate benchmark
    p_accurate_benchmark = sub.add_parser(
        "accurate-benchmark",
        help="运行冻结的单配置 Accurate 基准回测",
    )
    p_accurate_benchmark.add_argument(
        "--positions",
        required=True,
        help="目标仓位 parquet 路径",
    )
    p_accurate_benchmark.add_argument(
        "--barra-root",
        required=True,
        help="Barra 因子文件夹路径",
    )
    p_accurate_benchmark.add_argument(
        "--output-root",
        required=True,
        help="回测结果根目录",
    )
    p_accurate_benchmark.add_argument(
        "--start",
        default=DEFAULT_START_DATE,
        help=f"回测开始日期（默认 {DEFAULT_START_DATE}）",
    )
    p_accurate_benchmark.add_argument(
        "--end",
        help="回测结束日期（默认使用仓位数据末日）",
    )
    p_accurate_benchmark.set_defaults(func=cmd_accurate_benchmark)

    # list
    p_list = sub.add_parser("list", help="列出可用 benchmark")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
