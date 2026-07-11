"""阶段 4-2 A 轨时序指标计算核心实现。

所属模块:
    evaluation/indicator

对应文档:
    docs/FID_stage4_2_indicator_metrics.md

文件职责:
    读取阶段 4-1 输出的绩效时序 bundle，完成输入字段校验、核心指标计算、
    评分卡生成、Dev4/Yurui 下游摘要适配、标准产物落盘和运行日志记录。

模块边界:
    本文件不从原始因子、行情或 forward return 重新生成 RankIC、分组收益、
    多空收益等上游时序，只消费阶段 4-1 已经生成的绩效序列。
"""

from __future__ import annotations

import logging
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd

TRACK_A = "Track_A"

P0_REQUIRED_SERIES = {
    "rank_ic_series",
    "quantile_return_panel",
    "top_minus_bottom_series",
    "long_short_return_series",
    "turnover_series",
    "coverage_series",
}

P1_OPTIONAL_SERIES = {
    "ic_series",
    "kendall_tau_series",
    "benchmark_return_series",
    "cost_series",
    "style_exposure_panel",
    "library_corr_matrix",
}

SERIES_REQUIRED_COLUMNS = {
    "rank_ic_series": {"date", "factor_id", "eval_run_id", "horizon", "rank_ic_t", "valid_asset_count"},
    "ic_series": {"date", "factor_id", "eval_run_id", "horizon", "ic_t"},
    "kendall_tau_series": {"date", "factor_id", "eval_run_id", "horizon", "kendall_tau_t"},
    "quantile_return_panel": {
        "date",
        "factor_id",
        "eval_run_id",
        "horizon",
        "quantile_id",
        "gross_return",
        "net_return",
        "asset_count",
    },
    "top_minus_bottom_series": {
        "date",
        "factor_id",
        "eval_run_id",
        "horizon",
        "top_return",
        "bottom_return",
        "top_minus_bottom_t",
    },
    "long_short_return_series": {
        "date",
        "factor_id",
        "eval_run_id",
        "horizon",
        "gross_return",
        "net_return",
    },
    "turnover_series": {"date", "factor_id", "eval_run_id", "horizon", "turnover_t"},
    "coverage_series": {
        "date",
        "factor_id",
        "eval_run_id",
        "horizon",
        "valid_asset_count",
        "universe_count",
        "coverage_t",
    },
    "benchmark_return_series": {"date", "factor_id", "eval_run_id", "horizon", "benchmark_return"},
    "cost_series": {
        "date",
        "factor_id",
        "eval_run_id",
        "horizon",
        "impact_cost",
        "borrow_fee",
        "net_return_erosion",
    },
    "style_exposure_panel": {"date", "factor_id", "eval_run_id", "style_name", "exposure_value"},
    "library_corr_matrix": {"factor_id", "other_factor_id", "corr_value"},
}

DEFAULT_METHOD_CONFIG = {
    "annual_factor": 252,
    "eps": 1e-8,
    "win_rate_threshold": 0.0,
    "ic_ir_method": "mean_over_std",
    "t_stat_method": "standard",
    "pt_test_enabled": True,
    "monotonicity_method": "spearman_quantile_return_corr",
    "risk_free_rate": 0.0,
    "min_valid_observations": 60,
    "max_delay_k": 60,
    "priority_scope": "P0_P1_first",
    "fitness_turnover_lambda": 0.05,
}

SCORECARD_METRICS = [
    "rank_ic_mean",
    "rank_ic_ir",
    "rank_ic_win_rate",
    "quantile_monotonicity_score",
    "top_minus_bottom_mean",
    "pt_pvalue",
    "long_short_sharpe",
    "long_short_ir",
    "max_drawdown",
    "calmar",
    "turnover",
    "universe_coverage_mean",
    "library_corr_max",
]


@dataclass(frozen=True)
class RunPaths:
    """本地默认运行路径，供脚本快速找到 mock 输入和标准输出目录。"""

    root: Path
    input_dir: Path
    output_dir: Path
    log_dir: Path
    config_path: Path
    bundle_path: Path


def default_run_paths(root: Path) -> RunPaths:
    """根据项目根目录生成默认输入、输出、日志和配置文件路径。

    这里把传入的 root 当作 `/AutoFactorEvaluation` 项目根目录使用。
    因此脚本需要先 cd 到项目根目录再运行，默认路径全部从该根目录展开。
    """

    input_dir = root / "database" / "mock" / "evaluation" / "indicator" / "stage4_1_output"
    output_dir = root / "database" / "mock" / "evaluation" / "indicator" / "stage4_2_output"
    log_dir = root / "database" / "log" / "evaluation" / "indicator"
    return RunPaths(
        root=root,
        input_dir=input_dir,
        output_dir=output_dir,
        log_dir=log_dir,
        config_path=input_dir / "metric_method_config.json",
        bundle_path=input_dir / "performance_series_bundle.json",
    )


def ensure_dir(path: Path) -> None:
    """确保目录存在；如果不存在则递归创建。"""

    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    """以 UTF-8 JSON 格式写出对象。"""

    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    """读取 UTF-8 JSON 文件。"""

    return json.loads(path.read_text(encoding="utf-8"))


def write_table(path: Path, df: pd.DataFrame) -> None:
    """按文件后缀写出表格，支持 parquet、csv 和 json。"""

    ensure_dir(path.parent)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    elif path.suffix.lower() == ".json":
        path.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported table format: {path.suffix}")


def read_table(path: Path) -> pd.DataFrame:
    """按文件后缀读取表格，支持 parquet、csv 和 json。"""

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def _setup_logger(log_dir: Path | None) -> logging.Logger:
    """创建阶段 4-2 运行日志；未传 log_dir 时只使用空处理器。

    复用同名 logger 时先关闭并清理旧 handler，避免文件句柄泄漏（Windows 下会
    导致占用）。文件以追加模式打开，使同一日志目录被多次调用（如 pipeline 逐
    horizon 运行）时各次运行的日志都得到保留，每次以"started"行作为分隔。
    """

    logger = logging.getLogger("stage4_2_indicator_metrics")
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logger.propagate = False
    if log_dir is None:
        logger.addHandler(logging.NullHandler())
        return logger

    ensure_dir(log_dir)
    handler = logging.FileHandler(log_dir / "stage4_2_metrics.log", mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _project_relative(path: Path, root: Path | None) -> str:
    """尽量把路径写成项目根目录相对路径，避免产物里出现本机绝对路径。"""

    if root is not None:
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()

@dataclass(frozen=True)
class MetricCalculationResult:
    """阶段 4-2 统一入口函数的返回对象。

    属性：
        metrics_matrix: 内存中的指标矩阵，保留 Python 原生数值类型，便于测试和二次处理。
        summary_scorecard: 写入 JSON 的核心评分卡摘要。
        dev4_evaluation_summary: 适配下游 Dev4/Yurui 的评估摘要。
        validation_events: 输入校验和计算过程中的异常事件。
        output_dir: 本次计算产物所在目录。
    """

    metrics_matrix: pd.DataFrame
    summary_scorecard: dict[str, Any]
    dev4_evaluation_summary: dict[str, Any]
    validation_events: list[dict[str, Any]]
    output_dir: str

    def as_dict(self) -> dict[str, Any]:
        """保持旧版脚本和测试使用的 dict 返回格式。"""

        return {
            "metrics_matrix": self.metrics_matrix,
            "summary_scorecard": self.summary_scorecard,
            "dev4_evaluation_summary": self.dev4_evaluation_summary,
            "validation_events": self.validation_events,
            "output_dir": self.output_dir,
        }


def _finite_series(values: pd.Series) -> pd.Series:
    """转为数值序列，只保留有限值，同时保留原始行索引。"""

    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return numeric.dropna()


def _sample_stats(values: pd.Series) -> tuple[pd.Series, int, int, float]:
    """返回清洗后的序列、原始样本数、有效样本数和有效样本比例。"""

    clean = _finite_series(values)
    total = int(len(values))
    valid = int(len(clean))
    ratio = float(valid / total) if total else 0.0
    return clean, total, valid, ratio


def _safe_std(values: pd.Series) -> float:
    """计算样本标准差；样本不足两个时返回 0，避免产生无意义标准差。"""

    if len(values) < 2:
        return 0.0
    return float(values.std(ddof=1))


def _annualized_return(returns: pd.Series, annual_factor: float) -> float:
    """按复利累计周期收益，并用有效采样频率进行年化。"""

    values = _finite_series(returns)
    if values.empty:
        return np.nan
    growth = 1.0 + values
    if bool((growth <= 0.0).to_numpy().any()):
        # 单期亏损 >= 100%（1 + return <= 0）会使复利基为非正，分数次幂得 NaN；
        # 这里显式返回 NaN，使年化口径退化可见，而非依赖负底数静默产生 NaN。
        return np.nan
    compounded = float(np.prod(growth.to_numpy(dtype=float)))  # 样本期复利总收益：连乘每期 1 + return。
    return compounded ** (annual_factor / len(values)) - 1.0  # 把样本期总收益按有效频率折算为年化收益。


def _max_drawdown(returns: pd.Series) -> float:
    """根据多空累计净值曲线计算样本期最大回撤。"""

    values = _finite_series(returns)
    if values.empty:
        return np.nan
    equity = (1.0 + values).cumprod()  # 多空策略净值曲线。
    running_max = equity.cummax()  # 截至每个时点的历史最高净值。
    return float((equity / running_max - 1.0).min())  # 当前净值相对历史最高点的最小跌幅即最大回撤。


def _t_stat(values: pd.Series, eps: float) -> float:
    """计算均值估计的标准 t 统计量。"""

    clean = _finite_series(values)
    std = _safe_std(clean)
    if clean.empty or std <= eps:
        return np.nan
    return float(clean.mean() / (std / np.sqrt(len(clean))))  # 均值除以标准误，得到标准 t 统计量。


def _t_p_value(t_value: float, valid_count: int) -> float:
    """基于 Student t 分布计算双侧 p 值。

    小样本下 t 分布和正态近似差异明显，因此优先使用 SciPy。
    如果当前环境没有 SciPy，返回 NaN，避免静默退化为不准确的正态近似。
    """

    if pd.isna(t_value) or valid_count < 2:
        return np.nan
    try:
        from scipy import stats
    except ImportError:
        return np.nan
    return float(stats.t.sf(abs(float(t_value)), df=valid_count - 1) * 2.0)  # 双侧检验：右尾概率乘以 2。


def _spearman_corr(x: pd.Series, y: pd.Series) -> float:
    """计算 Spearman 相关，用于分位组单调性评分。"""

    aligned = pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1).dropna()
    if len(aligned) < 2:
        return np.nan
    return float(aligned.iloc[:, 0].rank().corr(aligned.iloc[:, 1].rank()))  # Spearman 等价于秩变量的 Pearson 相关。


def _horizon_days(horizon: str) -> float:
    """把 '5D' 或 5 这类 horizon 标签解析为交易日数量。"""

    match = re.search(r"\d+(\.\d+)?", str(horizon))
    return max(float(match.group(0)), 1.0) if match else 1.0


def _effective_annual_factor(config: dict[str, Any], horizon: str) -> float:
    """把日频年化因子换算为当前 horizon 对应的有效年化因子。"""

    base_annual_factor = float(config["annual_factor"])
    return base_annual_factor / _horizon_days(horizon)  # 例如 5D 信号一年约 252 / 5 次有效观测。


def _hash_file(path: Path) -> str:
    """按块读取文件并计算 SHA256，用于 manifest 记录输出快照。"""

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _path_from_ref(ref: str) -> Path:
    """把 bundle 中的普通路径或 file:// 引用转换为 Path。"""

    parsed = urlparse(ref)
    if parsed.scheme == "file":
        return Path(parsed.path)
    return Path(ref)


def _load_bundle(bundle_path: Path) -> dict[str, pd.DataFrame]:
    """根据 performance_series_bundle.json 读取全部上游绩效序列表。"""

    bundle = read_json(bundle_path)
    return {name: read_table(_path_from_ref(path)) for name, path in bundle.items()}


def validate_series_bundle(
    series_bundle: dict[str, pd.DataFrame],
    method_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """校验阶段 4-1 序列是否齐全、字段是否齐全、样本是否足够、horizon 是否一致。"""

    events: list[dict[str, Any]] = []
    min_obs = int(method_config["min_valid_observations"])

    for name in sorted(P0_REQUIRED_SERIES):
        if name not in series_bundle:
            events.append({"severity": "error", "series": name, "code": "missing_p0_series", "message": f"Missing P0 series: {name}"})

    for name in sorted(P1_OPTIONAL_SERIES):
        if name not in series_bundle:
            events.append({"severity": "warning", "series": name, "code": "missing_optional_series", "message": f"Missing optional series: {name}"})

    for name, df in series_bundle.items():
        required = SERIES_REQUIRED_COLUMNS.get(name)
        if not required:
            events.append({"severity": "warning", "series": name, "code": "unknown_series", "message": f"Unknown series: {name}"})
            continue
        missing_cols = sorted(required.difference(df.columns))
        if missing_cols:
            events.append({"severity": "error", "series": name, "code": "missing_columns", "message": f"Missing columns: {missing_cols}"})
        if "date" in df.columns:
            valid_dates = pd.to_datetime(df["date"], errors="coerce").notna().sum()
            if valid_dates < min_obs and name in P0_REQUIRED_SERIES:
                events.append(
                    {
                        "severity": "warning",
                        "series": name,
                        "code": "insufficient_observations",
                        "message": f"Valid dates {valid_dates} < min_valid_observations {min_obs}",
                    }
                )
        if "horizon" in df.columns and df["horizon"].nunique(dropna=True) > 1:
            events.append({"severity": "error", "series": name, "code": "mixed_horizon", "message": "Series contains multiple horizons"})

    horizons = {
        name: set(df["horizon"].dropna().astype(str).unique())
        for name, df in series_bundle.items()
        if "horizon" in df.columns and name in P0_REQUIRED_SERIES and not df.empty
    }
    flattened = {h for vals in horizons.values() for h in vals}
    if len(flattened) > 1:
        events.append(
            {
                "severity": "error",
                "series": "performance_series_bundle",
                "code": "horizon_mismatch",
                "message": f"P0 series horizons do not match: {horizons}",
            }
        )
    return events


class MetricBuilder:
    """指标行构造器，确保所有指标都遵循统一输出 schema。"""

    def __init__(self, factor_id: str, eval_run_id: str, horizon: str, config: dict[str, Any]) -> None:
        self.factor_id = factor_id
        self.eval_run_id = eval_run_id
        self.horizon = horizon
        self.config = config
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        family: str,
        code: str,
        value: Any,
        input_series: str | list[str],
        method: str,
        priority: str,
        status: str = "valid",
        warning: str = "",
    ) -> None:
        if isinstance(value, (np.floating, np.integer)):
            value = float(value)
        value_type = _value_type(value)
        self.rows.append(
            {
                "factor_id": self.factor_id,
                "eval_run_id": self.eval_run_id,
                "track": TRACK_A,
                "horizon": self.horizon,
                "metric_family": family,
                "metric_code": code,
                "input_series": json.dumps(input_series, ensure_ascii=False) if isinstance(input_series, list) else input_series,
                "method": method,
                "value": value,
                "value_type": value_type,
                "window": "full_sample",
                "priority": priority,
                "status": status,
                "warning": warning,
            }
        )

    def add_sample_stats(self, family: str, prefix: str, input_series: str, total: int, valid: int, ratio: float, priority: str) -> None:
        """输出显式样本数和有效样本比例，便于 review 和下游审计。"""

        self.add(family, f"{prefix}_sample_count", total, input_series, "raw_observation_count", priority)
        self.add(family, f"{prefix}_valid_count", valid, input_series, "finite_observation_count", priority)
        self.add(family, f"{prefix}_valid_ratio", ratio, input_series, "finite_over_raw_observations", priority)


def _value_type(value: Any) -> str:
    """标记指标值类型，方便下游把字符串形式的 value 还原成正确类型。"""

    if isinstance(value, (dict, list)):
        return "json"
    if value is None or pd.isna(value):
        return "null"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return "number"
    return "string"


def _identity_from_series(series_bundle: dict[str, pd.DataFrame]) -> tuple[str, str, str]:
    """读取 factor_id、eval_run_id 和 horizon。

    factor_id/eval_run_id 取自首个同时包含这两列且有有效值的表；horizon 单独取自首个
    含非空 ``horizon`` 列的表。这样可避免诸如 ``style_exposure_panel``（含 factor_id/
    eval_run_id 但无 horizon）排在前面时把 horizon 静默退化为 ``"NA"`` → 年化因子=252/1。
    """

    factor_id: str | None = None
    eval_run_id: str | None = None
    horizon: str | None = None
    for df in series_bundle.values():
        if (
            factor_id is None
            and {"factor_id", "eval_run_id"}.issubset(df.columns)
            and df["factor_id"].notna().any()
            and df["eval_run_id"].notna().any()
        ):
            factor_id = str(df["factor_id"].dropna().iloc[0])
            eval_run_id = str(df["eval_run_id"].dropna().iloc[0])
        if horizon is None and "horizon" in df.columns and df["horizon"].notna().any():
            horizon = str(df["horizon"].dropna().iloc[0])
    return (
        factor_id or "UNKNOWN_FACTOR",
        eval_run_id or "UNKNOWN_EVAL_RUN",
        horizon or "NA",
    )


def compute_metrics(series_bundle: dict[str, pd.DataFrame], method_config: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """基于阶段 4-1 绩效序列计算全部阶段 4-2 指标。

    这里刻意保持按指标族群顺序展开的线性流程，避免过度模块化。
    每个代码块先读取一类上游序列，输出样本质量指标，再输出实际绩效指标。
    """

    config = dict(DEFAULT_METHOD_CONFIG)
    config.update(method_config)
    validation_events = validate_series_bundle(series_bundle, config)
    factor_id, eval_run_id, horizon = _identity_from_series(series_bundle)
    b = MetricBuilder(factor_id, eval_run_id, horizon, config)
    eps = float(config["eps"])
    annual_factor = _effective_annual_factor(config, horizon)
    b.add("metadata", "effective_annual_factor", annual_factor, "metric_method_config", "annual_factor_divided_by_horizon_days", "P0")

    if "rank_ic_series" in series_bundle:
        rank_ic, total, valid, ratio = _sample_stats(series_bundle["rank_ic_series"]["rank_ic_t"])
        b.add_sample_stats("predictive", "rank_ic", "rank_ic_series", total, valid, ratio, "P0")
        std = _safe_std(rank_ic)
        b.add("predictive", "rank_ic_mean", rank_ic.mean(), "rank_ic_series", "mean", "P0")
        raw_ir = rank_ic.mean() / max(std, eps)  # 原始 IR：RankIC 均值除以 RankIC 样本标准差。
        b.add("predictive", "rank_ic_ir_raw", raw_ir, "rank_ic_series", "mean_over_std", "P1", warning="std protected by eps" if std <= eps else "")
        b.add("predictive", "rank_ic_ir", raw_ir * np.sqrt(annual_factor), "rank_ic_series", "mean_over_std_times_sqrt_effective_annual_factor", "P0", warning="std protected by eps" if std <= eps else "")  # 年化 IR：原始 IR 乘以有效年化频率平方根。
        b.add("predictive", "rank_ic_win_rate", float((rank_ic > float(config["win_rate_threshold"])).mean()), "rank_ic_series", "positive_ratio", "P0")
        t_value = _t_stat(rank_ic, eps)
        b.add("predictive", "rank_ic_t_stat", t_value, "rank_ic_series", "standard_t_stat", "P1", status="warning" if pd.isna(t_value) else "valid")
        b.add("predictive", "rank_ic_p_value", _t_p_value(t_value, valid), "rank_ic_series", "student_t_two_sided", "P1", status="warning" if pd.isna(t_value) else "valid")
    else:
        for code in ["rank_ic_mean", "rank_ic_ir", "rank_ic_win_rate"]:
            b.add("predictive", code, np.nan, "rank_ic_series", "missing", "P0", "missing_input", "P0 rank_ic_series missing")

    if "ic_series" in series_bundle:
        ic, total, valid, ratio = _sample_stats(series_bundle["ic_series"]["ic_t"])
        b.add_sample_stats("predictive", "ic", "ic_series", total, valid, ratio, "P1")
        std = _safe_std(ic)
        b.add("predictive", "ic_mean", ic.mean(), "ic_series", "mean", "P1")
        raw_ir = ic.mean() / max(std, eps)
        b.add("predictive", "ic_ir_raw", raw_ir, "ic_series", "mean_over_std", "P1", warning="std protected by eps" if std <= eps else "")
        b.add("predictive", "ic_ir", raw_ir * np.sqrt(annual_factor), "ic_series", "mean_over_std_times_sqrt_effective_annual_factor", "P1", warning="std protected by eps" if std <= eps else "")
        b.add("predictive", "ic_win_rate", float((ic > float(config["win_rate_threshold"])).mean()), "ic_series", "positive_ratio", "P1")
    else:
        b.add("predictive", "ic_mean", np.nan, "ic_series", "missing", "P1", "missing_input", "Optional ic_series missing")

    if "kendall_tau_series" in series_bundle and "kendall_tau_t" in series_bundle["kendall_tau_series"].columns:
        kendall, total, valid, ratio = _sample_stats(series_bundle["kendall_tau_series"]["kendall_tau_t"])
        b.add_sample_stats("predictive", "kendall_tau", "kendall_tau_series", total, valid, ratio, "P1")
        if kendall.empty:
            b.add("predictive", "kendall_tau_mean", np.nan, "kendall_tau_series", "mean", "P1", "warning", "kendall_tau_t is present but all values are NaN")
        else:
            b.add("predictive", "kendall_tau_mean", kendall.mean(), "kendall_tau_series", "mean", "P1")
            b.add("predictive", "kendall_tau_t_stat", _t_stat(kendall, eps), "kendall_tau_series", "standard_t_stat", "P1")
    else:
        b.add("predictive", "kendall_tau_mean", np.nan, "kendall_tau_series", "missing", "P1", "missing_input", "Optional kendall_tau_series missing")

    if "quantile_return_panel" in series_bundle:
        q = series_bundle["quantile_return_panel"].copy()
        curve = q.groupby("quantile_id")["net_return"].mean().sort_index()
        curve_dict = {f"Q{int(k)}": float(v) for k, v in curve.items()}
        score = _spearman_corr(pd.Series(curve.index.astype(float)), pd.Series(curve.to_numpy(dtype=float)))  # 用分位组编号和组均收益的秩相关衡量单调性。
        b.add("structure", "quantile_curve", curve_dict, "quantile_return_panel", "mean_net_return_by_quantile", "P0")
        b.add("structure", "quantile_monotonicity_score", score, "quantile_return_panel", "spearman_quantile_return_corr", "P0")
        if "top_minus_bottom_series" in series_bundle:
            tb, _total, valid, _ratio = _sample_stats(series_bundle["top_minus_bottom_series"]["top_minus_bottom_t"])
            t_value = _t_stat(tb, eps)
            b.add("structure", "pt_pvalue", _t_p_value(t_value, valid), ["quantile_return_panel", "top_minus_bottom_series"], "student_t_two_sided", "P1")
            b.add("structure", "cross_section_dispersion", float(q.groupby("date")["net_return"].std(ddof=1).mean()), "quantile_return_panel", "mean_daily_quantile_std", "P1")
    else:
        b.add("structure", "quantile_curve", np.nan, "quantile_return_panel", "missing", "P0", "missing_input", "P0 quantile_return_panel missing")
        b.add("structure", "quantile_monotonicity_score", np.nan, "quantile_return_panel", "missing", "P0", "missing_input", "P0 quantile_return_panel missing")

    if "top_minus_bottom_series" in series_bundle:
        tb, total, valid, ratio = _sample_stats(series_bundle["top_minus_bottom_series"]["top_minus_bottom_t"])
        b.add_sample_stats("structure", "top_minus_bottom", "top_minus_bottom_series", total, valid, ratio, "P0")
        b.add("structure", "top_minus_bottom_mean", tb.mean(), "top_minus_bottom_series", "mean", "P0")
        top = _finite_series(series_bundle["top_minus_bottom_series"]["top_return"])
        bottom = _finite_series(series_bundle["top_minus_bottom_series"]["bottom_return"])
        asym = float(abs(top.mean()) / max(abs(bottom.mean()), eps))
        b.add("structure", "top_bottom_symmetry", asym, "top_minus_bottom_series", "abs_top_mean_over_abs_bottom_mean", "P1")
    else:
        b.add("structure", "top_minus_bottom_mean", np.nan, "top_minus_bottom_series", "missing", "P0", "missing_input", "P0 top_minus_bottom_series missing")

    if "long_short_return_series" in series_bundle:
        ls = series_bundle["long_short_return_series"]
        net, total, valid, ratio = _sample_stats(ls["net_return"])
        b.add_sample_stats("risk", "long_short", "long_short_return_series", total, valid, ratio, "P0")
        gross = _finite_series(ls["gross_return"])
        ann_return = _annualized_return(net, annual_factor)
        ann_vol = float(net.std(ddof=1) * np.sqrt(annual_factor)) if len(net) > 1 else np.nan  # 年化波动：样本标准差乘以有效年化频率平方根。
        sharpe = (ann_return - float(config["risk_free_rate"])) / max(ann_vol, eps) if not pd.isna(ann_return) else np.nan  # Sharpe：超额年化收益除以年化波动。
        mdd = _max_drawdown(net)
        calmar = ann_return / abs(mdd) if not pd.isna(ann_return) and not pd.isna(mdd) and abs(mdd) > eps else np.nan  # Calmar 使用行业常见口径：年化收益 / 样本最大回撤绝对值。
        calmar_warning = "sample max_drawdown is zero; calmar set to null" if pd.isna(calmar) else "industry convention: annual return over sample max drawdown"
        b.add("risk", "long_short_total_return", float(np.prod(1.0 + net) - 1.0), "long_short_return_series", "compound_total_return", "P0")
        b.add("risk", "long_short_ann_return", ann_return, "long_short_return_series", "compound_annual_return", "P0")
        b.add("risk", "long_short_ann_vol", ann_vol, "long_short_return_series", "std_sqrt_annual_factor", "P0")
        b.add("risk", "long_short_sharpe", sharpe, "long_short_return_series", "ann_return_over_ann_vol", "P0", warning="vol protected by eps" if ann_vol <= eps else "")
        b.add("risk", "max_drawdown", mdd, "long_short_return_series", "equity_curve_drawdown", "P0")
        b.add("risk", "calmar", calmar, "long_short_return_series", "ann_return_over_sample_max_drawdown", "P0", warning=calmar_warning)
        if "benchmark_return_series" in series_bundle:
            bench = _finite_series(series_bundle["benchmark_return_series"]["benchmark_return"])
            aligned = pd.concat([net.reset_index(drop=True), bench.reset_index(drop=True)], axis=1).dropna()
            active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
            tracking_error = active.std(ddof=1) * np.sqrt(annual_factor)
            b.add("risk", "long_short_ir", active.mean() * annual_factor / max(tracking_error, eps), ["long_short_return_series", "benchmark_return_series"], "active_return_over_tracking_error", "P1")  # 主动收益年化后除以年化跟踪误差。
        if len(gross) == len(net):
            erosion = (gross - net).sum() / max(abs(gross.sum()), eps)
            b.add("execution", "net_return_erosion_ratio", erosion, "long_short_return_series", "sum_gross_minus_net_over_abs_gross", "P1")
    else:
        for code in ["long_short_ann_return", "long_short_sharpe", "max_drawdown", "calmar"]:
            b.add("risk", code, np.nan, "long_short_return_series", "missing", "P0", "missing_input", "P0 long_short_return_series missing")

    if "turnover_series" in series_bundle:
        turnover, total, valid, ratio = _sample_stats(series_bundle["turnover_series"]["turnover_t"])
        b.add_sample_stats("execution", "turnover", "turnover_series", total, valid, ratio, "P0")
        avg_turnover = turnover.mean()
        b.add("execution", "turnover", avg_turnover, "turnover_series", "mean", "P0")
        ann_row = next((row for row in b.rows if row["metric_code"] == "long_short_ann_return"), None)
        if ann_row is not None and isinstance(ann_row["value"], (int, float, np.floating)):
            fitness = float(ann_row["value"]) - float(config["fitness_turnover_lambda"]) * avg_turnover  # 简单 fitness：年化收益扣除换手惩罚项。
            b.add("execution", "fitness", fitness, ["long_short_return_series", "turnover_series"], "ann_return_minus_lambda_turnover", "P1")
    else:
        b.add("execution", "turnover", np.nan, "turnover_series", "missing", "P0", "missing_input", "P0 turnover_series missing")

    if "coverage_series" in series_bundle:
        coverage, total, valid, ratio = _sample_stats(series_bundle["coverage_series"]["coverage_t"])
        b.add_sample_stats("coverage", "coverage", "coverage_series", total, valid, ratio, "P0")
        b.add("coverage", "universe_coverage_mean", coverage.mean(), "coverage_series", "mean", "P0")
        b.add("coverage", "coverage_min", coverage.min(), "coverage_series", "min", "P1")
        b.add("coverage", "coverage_stability", 1.0 - coverage.std(ddof=1), "coverage_series", "one_minus_std", "P1")
    else:
        b.add("coverage", "universe_coverage_mean", np.nan, "coverage_series", "missing", "P0", "missing_input", "P0 coverage_series missing")

    if "cost_series" in series_bundle:
        cost = series_bundle["cost_series"]
        b.add("execution", "impact_cost", _finite_series(cost["impact_cost"]).mean(), "cost_series", "mean", "P2")
        b.add("execution", "borrow_fee_penalty", _finite_series(cost["borrow_fee"]).sum(), "cost_series", "sum", "P2")
    else:
        b.add("execution", "impact_cost", np.nan, "cost_series", "missing", "P2", "missing_input", "Optional cost_series missing")

    if "style_exposure_panel" in series_bundle:
        style = series_bundle["style_exposure_panel"]
        vector = style.groupby("style_name")["exposure_value"].mean().to_dict()
        vector = {str(k): float(v) for k, v in vector.items()}
        b.add("style", "style_exposure_vector", vector, "style_exposure_panel", "mean_by_style", "P1")
        b.add("style", "industry_bias_score", np.nan, "industry_exposure_panel", "missing", "P1", "missing_input", "industry_exposure_panel not provided in mock input")
    else:
        b.add("style", "style_exposure_vector", np.nan, "style_exposure_panel", "missing", "P1", "missing_input", "Optional style_exposure_panel missing")

    if "library_corr_matrix" in series_bundle:
        corr = _finite_series(series_bundle["library_corr_matrix"]["corr_value"]).abs()
        b.add("style", "library_corr_max", corr.max(), "library_corr_matrix", "max_abs_corr", "P1")
        b.add("style", "cluster_id", np.nan, "library_corr_matrix", "external_clustering_required", "P2", "missing_input", "cluster_id requires upstream similarity clustering")
    else:
        b.add("style", "library_corr_max", np.nan, "library_corr_matrix", "missing", "P1", "missing_input", "Optional library_corr_matrix missing")

    return pd.DataFrame(b.rows), validation_events


def build_summary_scorecard(metrics_matrix: pd.DataFrame) -> dict[str, Any]:
    """构造核心评分卡；白名单字段缺失时显式输出 null。"""

    valid = metrics_matrix[metrics_matrix["status"].isin(["valid", "warning"])]
    scorecard: dict[str, Any] = {}
    for code in SCORECARD_METRICS:
        row = valid[valid["metric_code"] == code]
        scorecard[code] = _json_value(row.iloc[0]["value"]) if not row.empty else None
    scorecard["ic_positive_ratio"] = scorecard["rank_ic_win_rate"]
    scorecard["quantile_monotonicity"] = scorecard["quantile_monotonicity_score"]
    scorecard["coverage"] = scorecard["universe_coverage_mean"]
    return scorecard


def build_dev4_evaluation_summary(scorecard: dict[str, Any], method_config: dict[str, Any]) -> dict[str, Any]:
    """把阶段 4-2 评分卡适配为 Yurui/Dev4 需要的 evaluation_summary 结构。"""

    rank_ic_mean = float(scorecard.get("rank_ic_mean") or 0.0)
    turnover = float(scorecard.get("turnover") or 0.0)
    # 区分"缺失"与"恰好为 0"：关键风险指标缺失时不得据此判定 net_survive。
    sharpe_raw = scorecard.get("long_short_sharpe")
    mdd_raw = scorecard.get("max_drawdown")
    sharpe = float(sharpe_raw) if sharpe_raw is not None else None
    max_drawdown = float(mdd_raw) if mdd_raw is not None else None

    survival_view = (
        "net_survive"
        if (sharpe is not None and max_drawdown is not None and sharpe > 0 and max_drawdown > -0.5)
        else "failed"
    )
    if survival_view == "failed":
        usage_role = "archive_only"
        route_recommendation = "tier4_archive"
        confidence = 0.35
        failure_reasons = ["negative_or_fragile_net_performance"]
    elif rank_ic_mean >= 0.03 and sharpe is not None and sharpe >= 1.0:
        usage_role = "core_signal"
        route_recommendation = "tier3a_core"
        confidence = 0.86
        failure_reasons = []
    else:
        usage_role = "satellite_signal"
        route_recommendation = "tier3b_satellite"
        confidence = 0.68
        failure_reasons = []

    fragility_tags = []
    if turnover >= 0.3:
        fragility_tags.append("high_turnover")

    return {
        "evaluation_protocol_key": method_config.get("evaluation_protocol_key", "cs_equity_daily_v1"),
        "detail_page_template_key": "cs_default",
        "summary_scorecard": scorecard,
        "survival_view": survival_view,
        "usage_role": usage_role,
        "fragility_tags": fragility_tags,
        "incremental_value_summary": {
            "incremental_ir": scorecard.get("long_short_ir"),
        },
        "route_recommendation": route_recommendation,
        "primary_failure_reasons": failure_reasons,
        "admission_confidence": confidence,
    }


def run_metric_calculation(input_dir: Path, output_dir: Path, log_dir: Path | None = None) -> dict[str, Any]:
    """公共入口函数：完整运行阶段 4-2 指标计算。

    入参：
        input_dir: 输入目录，必须包含 `performance_series_bundle.json`。
            `metric_method_config.json` 是可选文件；缺失时使用默认配置。
        output_dir: 输出目录，所有指标产物都会写入这里。
        log_dir: 日志目录；传入时写入 `stage4_2_metrics.log`。

    出参：
        返回兼容 dict 的结果对象，包含内存态指标矩阵、评分卡、校验事件和输出目录。
        写出的 JSON/Parquet 使用稳定 schema；`metrics_matrix.value` 统一为字符串或 null，
        `value_type` 负责告诉下游如何还原原始类型。
    """

    logger = _setup_logger(log_dir)
    project_root = Path.cwd()
    logger.info("Stage 4-2 metric calculation started")
    logger.info(
        "input_dir=%s output_dir=%s log_dir=%s",
        _project_relative(input_dir, project_root),
        _project_relative(output_dir, project_root),
        _project_relative(log_dir, project_root) if log_dir is not None else None,
    )

    ensure_dir(output_dir)
    bundle_path = input_dir / "performance_series_bundle.json"
    config_path = input_dir / "metric_method_config.json"
    series_bundle = _load_bundle(bundle_path)
    method_config = read_json(config_path) if config_path.exists() else dict(DEFAULT_METHOD_CONFIG)
    logger.info("Loaded %s input series", len(series_bundle))
    metrics_matrix, validation_events = compute_metrics(series_bundle, method_config)
    factor_id, eval_run_id, horizon = _identity_from_series(series_bundle)
    for event in validation_events:
        message = "validation_event severity=%s series=%s code=%s message=%s"
        args = (
            event.get("severity"),
            event.get("series"),
            event.get("code"),
            event.get("message"),
        )
        if event.get("severity") == "error":
            logger.error(message, *args)
        else:
            logger.warning(message, *args)
    logger.info(
        "Computed metrics rows=%s factor_id=%s eval_run_id=%s horizon=%s validation_events=%s",
        len(metrics_matrix),
        factor_id,
        eval_run_id,
        horizon,
        len(validation_events),
    )
    scorecard = build_summary_scorecard(metrics_matrix)
    dev4_summary = build_dev4_evaluation_summary(scorecard, method_config)
    logger.info(
        "Scorecard built | rank_ic_mean=%s long_short_sharpe=%s survival=%s",
        scorecard.get("rank_ic_mean"),
        scorecard.get("long_short_sharpe"),
        dev4_summary.get("survival_view"),
    )

    metrics_for_output = metrics_matrix.copy()
    metrics_for_output["value"] = metrics_for_output["value"].map(_serialized_value)

    metrics_parquet = output_dir / "metrics_matrix.parquet"
    metrics_json = output_dir / "metrics_matrix.json"
    write_table(metrics_parquet, metrics_for_output)
    write_table(metrics_json, metrics_for_output)
    logger.info("Metrics matrix written | parquet=%s json=%s", _project_relative(metrics_parquet, project_root), _project_relative(metrics_json, project_root))

    summary_payload = {
        "factor_id": factor_id,
        "eval_run_id": eval_run_id,
        "track": TRACK_A,
        "horizon": horizon,
        "summary_scorecard": scorecard,
    }
    write_json(output_dir / "summary_scorecard.json", summary_payload)
    write_json(output_dir / "dev4_evaluation_summary.json", dev4_summary)
    write_json(output_dir / "metric_validation_report.json", {"events": validation_events})
    logger.info("Scorecard / dev4 summary / validation report written")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_bundle": _project_relative(bundle_path, project_root),
        "method_config": _project_relative(config_path, project_root),
        "outputs": {
            "metrics_matrix_parquet": _project_relative(metrics_parquet, project_root),
            "metrics_matrix_json": _project_relative(metrics_json, project_root),
            "summary_scorecard": _project_relative(output_dir / "summary_scorecard.json", project_root),
            "dev4_evaluation_summary": _project_relative(output_dir / "dev4_evaluation_summary.json", project_root),
            "metric_validation_report": _project_relative(output_dir / "metric_validation_report.json", project_root),
        },
        "hashes": {
            "metrics_matrix_parquet_sha256": _hash_file(metrics_parquet),
            "summary_scorecard_sha256": _hash_file(output_dir / "summary_scorecard.json"),
        },
        "code_version": "local_prototype_v0.1",
        "protocol_snapshot": "stage4_2_track_a_v0.3",
    }
    write_json(output_dir / "metric_manifest.json", manifest)
    package = {
        "factor_id": factor_id,
        "eval_run_id": eval_run_id,
        "track": TRACK_A,
        "horizon": horizon,
        "metrics_matrix_ref": _project_relative(metrics_parquet, project_root),
        "summary_scorecard": summary_payload,
        "dev4_evaluation_summary": dev4_summary,
        "validation_report_ref": _project_relative(output_dir / "metric_validation_report.json", project_root),
        "manifest_ref": _project_relative(output_dir / "metric_manifest.json", project_root),
        "method_config": method_config,
        "input_series_refs": {
            name: _project_relative(_path_from_ref(path), project_root)
            for name, path in read_json(bundle_path).items()
        },
    }
    write_json(output_dir / "metrics_package.json", package)
    logger.info("Manifest & metrics package written")
    logger.info("Stage 4-2 metric calculation finished output_dir=%s", _project_relative(output_dir, project_root))
    return MetricCalculationResult(
        metrics_matrix=metrics_matrix,
        summary_scorecard=summary_payload,
        dev4_evaluation_summary=dev4_summary,
        validation_events=validation_events,
        output_dir=str(output_dir),
    ).as_dict()


def _json_value(value: Any) -> Any:
    """把 NumPy 标量和 NaN 转为 JSON 安全的 Python 原生值。"""

    if isinstance(value, (np.floating, np.integer)):
        value = float(value)
    if isinstance(value, (dict, list)):
        return value
    if value is None or pd.isna(value):
        return None
    return value


def _serialized_value(value: Any) -> str | None:
    """统一 JSON 和 Parquet 中 metrics_matrix.value 的落盘表示。"""

    value = _json_value(value)
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)
