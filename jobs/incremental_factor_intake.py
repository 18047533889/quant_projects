#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""增量因子入库：26 条新因子 → 去重/落盘/评估/聚类/优化/页面注入/json 写回。

用法:
  OMP_NUM_THREADS=31 .venv/bin/python jobs/incremental_factor_intake.py --manifest /tmp/new50_selected.json
  OMP_NUM_THREADS=31 .venv/bin/python jobs/incremental_factor_intake.py --manifest /tmp/new50_selected.json --limit 2

断点续跑：状态 /tmp/intake_state.json {factor_name: {"stage": "...", "result": {...}}}
已完成 stage 跳过。每 stage 完成立即写盘（write-early）。

口径（与既有 470 池一致）：
  - 收益 = AdjVwap.pct_change().shift(-2)  (vwap-to-vwap 后复权，企业级 shift(-2))
  - 评估窗 2016-01-04..2018-06-30（与 manifest rank_ic_local 同口径，已实测复现）
  - 逐日 spearman rankic，min_universe=30
  - is_flipped=True 的因子评估负矩阵（×-1 后让 rank_ic 为正）
"""
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from html import escape

os.environ.setdefault("OMP_NUM_THREADS", "8")

ROOT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "jobs"))
from factor_report_sources import FULL_WINDOW_END, FULL_WINDOW_START, resolve_raw_matrix
from factor_engine.reporting.quant_evaluator_adapter import evaluate_report_arrays

POOL_JSON = Path("/home/sunhaiwei/factor_delivery_converted/formula_lqtp_all.json")
CLUSTERS_JSON = ROOT / "weekly_backtest_output/factor_clusters.json"
OPT_META_JSON = ROOT / "weekly_backtest_output/optimized_meta.json"
MATRICES_DIR = ROOT / "weekly_backtest_output/factor_matrices_all"
OPT_DIR = ROOT / "weekly_backtest_output/optimized_factors"
REPORTS_DIR = ROOT / "factor_engine/docs/reports/2026-08-23"
FACTORS_DIR = REPORTS_DIR / "factors"
INDEX_HTML = REPORTS_DIR / "index.html"
REPORT_MANIFEST_JSON = REPORTS_DIR / "report_manifest.json"
STATE_JSON = Path("/tmp/intake_state.json")
DONE_JSON = Path("/tmp/intake_done.json")

STAGES = ["dedup_check", "landing", "eval", "cluster_assign", "optimize_lite",
          "page_inject", "json_writeback"]

EVAL_START = "2016-01-04"
EVAL_END = "2018-06-30"
MIN_UNIVERSE = 30

# 渲染产物中禁止出现的算法名字符串（零出现）
BANNED = ["cogalpha", "alphasage", "evoalpha", "factorminer", "qwen",
          "alpha_sage", "alpha158"]


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------
def load_state():
    if STATE_JSON.exists():
        return json.loads(STATE_JSON.read_text())
    return {}


def save_state(state):
    STATE_JSON.write_text(json.dumps(state, ensure_ascii=False, indent=1))


def stage_done(state, factor_name, stage):
    s = state.get(factor_name)
    return bool(s and s.get("stage") == stage)


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------
def load_manifest(path):
    return json.loads(Path(path).read_text())


def load_pool():
    return json.loads(POOL_JSON.read_text())


def load_clusters():
    return json.loads(CLUSTERS_JSON.read_text())


def load_opt_meta():
    return json.loads(OPT_META_JSON.read_text())


def matrix_source(page):
    """Resolve a raw matrix by verified date coverage, never by folder name."""
    return resolve_raw_matrix(page)


def matrix_path(page):
    source = matrix_source(page)
    return source.path if source.is_full_window else None


# --------------------------------------------------------------------------
# vwap 面板缓存（2016-01-04..2018-06-30）
# --------------------------------------------------------------------------
_HAS_VWAP = None


def load_vwap(*, source=None, start_date=EVAL_START, end_date=EVAL_END):
    global _HAS_VWAP
    uses_default_window = start_date == EVAL_START and end_date == EVAL_END
    if source is None and uses_default_window and _HAS_VWAP is not None:
        return _HAS_VWAP
    import pandas as pd
    if source is None:
        os.environ.setdefault("ASHARE_PARQUET_ROOT", str(Path.home() / "cos_data"))
        os.environ.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")
        from factor_engine.storage.factory import build_data_source

        source = build_data_source({
            "type": "data_access",
            "dataset": "ashare_stock_daily_adj",
            "start_date": start_date,
            "end_date": end_date,
        })
    values = source.load_column("AdjVwap")
    if not isinstance(values.index, pd.MultiIndex) or values.index.nlevels != 2:
        raise ValueError("DataAccess AdjVwap must use timestamp × instrument MultiIndex")
    panel = values.unstack(level=-1).sort_index().astype("float64")
    panel.index = pd.to_datetime(panel.index)
    if uses_default_window and source is not None and source.__class__.__module__.startswith("factor_engine"):
        _HAS_VWAP = panel
    return panel


def load_matrix(page, flip=False):
    import pandas as pd
    p = matrix_path(page)
    if p is None:
        return None
    m = pd.read_parquet(p)
    if flip:
        m = -m
    return m


# --------------------------------------------------------------------------
# 评估核心：逐日 spearman rankic / ic_ir（vwap-to-vwap shift(-2)）
# --------------------------------------------------------------------------
def eval_matrix(mat, vwap=None):
    """Evaluate a raw matrix through the canonical QuantEvaluator adapter."""
    import numpy as np
    if vwap is None:
        vwap = load_vwap()
    common = mat.index.intersection(vwap.index)
    if len(common) == 0:
        return {"rank_ic": 0.0, "ic_ir": 0.0, "n_days": 0, "error": "no overlap"}
    fv = mat.reindex(index=common)
    vv = vwap.reindex(index=common)
    cols = vv.columns.intersection(fv.columns)
    if len(cols) == 0:
        return {"rank_ic": 0.0, "ic_ir": 0.0, "n_days": 0, "error": "no common cols"}
    fv = fv[cols].values.astype(np.float64)
    vv = vv[cols]
    fwd = vv.pct_change(fill_method=None).shift(-2).values.astype(np.float64)
    evaluated = evaluate_report_arrays(
        fv,
        fwd,
        n_quantiles=10,
        min_assets=MIN_UNIVERSE,
        min_ic_periods=20,
        direction_training_periods=len(common),
    )
    return {
        "rank_ic": evaluated.mean_rank_ic,
        "ic_ir": evaluated.rank_ic_ir,
        "n_days": evaluated.valid_return_periods,
        "std": evaluated.rank_ic_std,
        "direction": evaluated.direction,
        "ls_sharpe": evaluated.sharpe,
        "ls_annual": evaluated.annualized_return,
        "ls_cumulative": evaluated.cumulative_return,
        "ls_mdd": evaluated.max_drawdown,
        "ls_winrate": evaluated.win_rate,
        "rank_ic_winrate": getattr(evaluated, "rank_ic_win_rate", None),
        "g10_annual": getattr(evaluated, "top_quantile_annualized_return", None),
        "g1_annual": getattr(evaluated, "bottom_quantile_annualized_return", None),
    }


# --------------------------------------------------------------------------
# stage: dedup_check
# --------------------------------------------------------------------------
def stage_dedup_check(factor):
    page = factor["page_name"]
    pool = load_pool()
    in_pool = page in {r.get("page_name") for r in pool}
    matrix = matrix_path(page)
    result = {
        "page_name": page,
        "in_pool": in_pool,
        "matrix_exists": matrix is not None,
        "matrix_path": str(matrix) if matrix else None,
        "ok": (not in_pool) and matrix is not None,
    }
    return result


# --------------------------------------------------------------------------
# stage: landing
# --------------------------------------------------------------------------
def stage_landing(factor):
    page = factor["page_name"]
    matrix = matrix_path(page)
    if matrix is not None:
        return {"landed": True, "path": str(matrix), "skipped": True}
    return {"landed": False, "error": "matrix missing"}


def land_factor_batch(factors, *, engine, sink):
    """Compile and land one incremental wave through FactorEngine ``run_many``.

    The sink boundary keeps factor matrices out of the aggregate result so a
    weekly batch does not retain every full panel in RAM.
    """
    parsed = []
    for record in factors:
        name = str(record.get("page_name") or record.get("factor_name") or "").strip()
        formula = str(record.get("fe_formula") or "").strip()
        if not name or not formula:
            raise ValueError(f"factor record requires page_name and fe_formula: {name or '<unnamed>'}")
        parsed.append(factor_from_record(record))

    engine.run_many(
        parsed,
        enable_cse=True,
        auto_warmup=True,
        trim_warmup=True,
        input_dq_check=True,
        pit_enforce=True,
        pit_forbid_forward_fill=True,
        warmup_clusters=True,
        result_policy="sink",
        sink=sink,
    )
    return {"landed": [factor.name for factor in parsed], "count": len(parsed)}


def factor_from_record(record):
    """Build a Factor from canonical JSON AST or native FactorEngine DSL."""
    from factor_engine.api.dsl_parser import parse_factor
    from factor_engine.api.factor import Factor

    name = str(record.get("page_name") or record.get("factor_name") or "").strip()
    formula = str(record.get("fe_formula") or "").strip()
    try:
        node = json.loads(formula)
    except json.JSONDecodeError:
        return parse_factor(formula, name=name)
    if not isinstance(node, dict) or "kind" not in node:
        return parse_factor(formula, name=name)

    from factor_engine.api.cleaned_ops import make_cleaned_call_factory
    from factor_engine.api.columns import col
    import factor_engine.cleaned_operators as cleaned_operators

    cleaned_operators.load_all(include_research=False)
    operator_aliases = {"and": "and_", "or": "or_", "not": "not_"}

    def build(current):
        kind = current.get("kind")
        if kind == "column":
            return col(str(current["name"]))
        if kind == "literal":
            return current.get("value")
        if kind != "call":
            raise ValueError(f"unsupported FactorEngine JSON node kind: {kind!r}")
        operator = operator_aliases.get(str(current.get("op")), str(current.get("op")))
        arguments = [build(argument) for argument in current.get("args", [])]
        keywords = dict(current.get("kwargs") or {})
        return make_cleaned_call_factory(operator)(*arguments, **keywords)

    return Factor(name=name, expr=build(node), source_expr=formula)


def build_incremental_engine(*, start_date, end_date, backend_name="polars_long"):
    """Build the mainline FactorEngine over its canonical DataAccess source."""
    os.environ.setdefault("ASHARE_PARQUET_ROOT", str(Path.home() / "cos_data"))
    os.environ.setdefault("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    from factor_engine.backend.factory import build_backend
    from factor_engine.runtime.engine import FactorEngine
    from factor_engine.storage.factory import build_data_source

    source = build_data_source({
        "type": "data_access",
        "dataset": "ashare_stock_daily_adj",
        "start_date": start_date,
        "end_date": end_date,
        "read_auto": True,
    })
    return FactorEngine(build_backend(backend_name), source, run_mode="research")


def matrix_sink(output_dir):
    """Create an atomic streaming sink for wide report matrices."""
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def write(name, value):
        series = value.get("result") if isinstance(value, dict) else value
        if not isinstance(series, pd.Series) or not isinstance(series.index, pd.MultiIndex):
            raise TypeError(f"run_many result for {name!r} must be a MultiIndex Series")
        matrix = series.unstack(level=-1).sort_index().astype("float32")
        target = output_dir / f"{name}.parquet"
        temporary = target.with_suffix(".parquet.tmp")
        matrix.to_parquet(temporary)
        os.replace(temporary, target)

    return write


def land_factor_batch_windowed(
    records,
    *,
    backend_name="polars_long",
    start_date=FULL_WINDOW_START,
    end_date=FULL_WINDOW_END,
    window_years=1,
    warmup_days=550,
):
    """Land a factor wave in bounded date windows, then atomically merge.

    FactorEngine still evaluates each factor wave with ``run_many``.  The date
    window prevents a ten-year long table from being materialized in RAM;
    overlap supplies rolling/EMA history and is trimmed before persistence.
    """
    import pandas as pd

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    names = [str(record["page_name"]) for record in records]
    with tempfile.TemporaryDirectory(prefix="factor_engine_landing_") as temp_name:
        temp_dir = Path(temp_name)
        cursor = start
        part = 0
        while cursor <= end:
            window_end = min(end, cursor + pd.DateOffset(years=window_years) - pd.Timedelta(days=1))
            load_start = cursor - pd.Timedelta(days=warmup_days)
            engine = build_incremental_engine(
                start_date=str(load_start.date()),
                end_date=str(window_end.date()),
                backend_name=backend_name,
            )

            def sink(name, value, *, _part=part, _start=cursor, _end=window_end):
                series = value.get("result") if isinstance(value, dict) else value
                if not isinstance(series, pd.Series) or not isinstance(series.index, pd.MultiIndex):
                    raise TypeError(f"run_many result for {name!r} must be a MultiIndex Series")
                dates = pd.to_datetime(series.index.get_level_values(0))
                selected = series[(dates >= _start) & (dates <= _end)]
                matrix = selected.unstack(level=-1).sort_index().astype("float32")
                matrix.to_parquet(temp_dir / f"{name}.{_part:03d}.parquet")

            land_factor_batch(records, engine=engine, sink=sink)
            cursor = window_end + pd.Timedelta(days=1)
            part += 1

        for name in names:
            pieces = [pd.read_parquet(path) for path in sorted(temp_dir.glob(f"{name}.*.parquet"))]
            if not pieces:
                raise RuntimeError(f"FactorEngine produced no window for {name}")
            matrix = pd.concat(pieces).sort_index()
            matrix = matrix[~matrix.index.duplicated(keep="last")]
            target = MATRICES_DIR / f"{name}.parquet"
            temporary = target.with_suffix(".parquet.tmp")
            matrix.to_parquet(temporary)
            os.replace(temporary, target)
    return {"landed": names, "count": len(names), "date_windows": part}


def land_missing_factors(factors, *, batch_size=8, backend_name="polars_long"):
    """Land missing executable DSL factors in bounded run_many waves."""
    missing = [record for record in factors if matrix_path(record["page_name"]) is None]
    # Capability metadata is advisory and is absent on older/new-mining
    # manifests.  The formula itself is authoritative: always try FE first.
    pending = [record for record in missing if str(record.get("fe_formula") or "").strip()]
    unsupported = [record["page_name"] for record in missing if record not in pending]
    if not pending:
        return {"landed": [], "count": 0, "pending": len(missing), "python_fallback": unsupported}
    landed = []
    errors = {}
    for offset in range(0, len(pending), batch_size):
        wave = pending[offset:offset + batch_size]
        try:
            result = land_factor_batch_windowed(wave, backend_name=backend_name)
            landed.extend(result["landed"])
        except Exception as wave_error:
            # Isolate a bad formula without discarding valid peers from the
            # bounded wave. Every failure remains explicit in provenance.
            for record in wave:
                name = record["page_name"]
                try:
                    result = land_factor_batch_windowed([record], backend_name=backend_name)
                    landed.extend(result["landed"])
                except Exception as factor_error:
                    errors[name] = (
                        f"wave={type(wave_error).__name__}: {wave_error}; "
                        f"factor={type(factor_error).__name__}: {factor_error}"
                    )
    return {
        "landed": landed,
        "count": len(landed),
        "pending": len(missing),
        "python_fallback": unsupported,
        "factor_engine_errors": errors,
    }


# --------------------------------------------------------------------------
# stage: eval
# --------------------------------------------------------------------------
def stage_eval(factor):
    page = factor["page_name"]
    mat = load_matrix(page, flip=False)
    if mat is None:
        return {"error": "matrix missing"}
    res = eval_matrix(mat)
    res["is_flipped"] = res.get("direction") == -1
    res["matrix_path"] = str(matrix_path(page))
    return res


def evaluate_factor_batch(
    factors,
    *,
    vwap=None,
    matrix_loader=None,
    evaluator=evaluate_report_arrays,
    batch_size=8,
    backend="auto",
):
    """Evaluate factor matrices in bounded QuantEvaluator tiles."""
    import numpy as np
    import pandas as pd
    from factor_engine.reporting.quant_evaluator_adapter import evaluate_report_batch

    if evaluator is evaluate_report_arrays:
        evaluator = evaluate_report_batch
    vwap = load_vwap(
        start_date=str(FULL_WINDOW_START.date()),
        end_date=str(FULL_WINDOW_END.date()),
    ) if vwap is None else vwap
    matrix_loader = matrix_loader or (lambda name: load_matrix(name, flip=False))
    names = [str(record["page_name"]) for record in factors]
    evaluated = {}
    backends = set()
    fallbacks = {}
    unavailable = {}
    for offset in range(0, len(names), max(1, batch_size)):
        tile_names = names[offset:offset + max(1, batch_size)]
        matrices = [matrix_loader(name) for name in tile_names]
        missing = [name for name, matrix in zip(tile_names, matrices) if matrix is None]
        for name in missing:
            unavailable[name] = "verified full-window factor matrix unavailable"
        available = [(name, matrix) for name, matrix in zip(tile_names, matrices) if matrix is not None]
        if not available:
            continue
        tile_names = [name for name, _ in available]
        matrices = [matrix for _, matrix in available]
        # DataAccess owns the report universe.  A sparse or newly-landed factor
        # must contribute NaNs on its missing cells; it must never truncate the
        # dates/assets (and therefore labels) of every other factor in a tile.
        dates = vwap.index
        columns = vwap.columns
        labels = vwap.reindex(index=dates, columns=columns).pct_change(fill_method=None).shift(-2)
        values = np.stack([
            matrix.reindex(index=dates, columns=columns).values.astype(np.float64)
            for matrix in matrices
        ], axis=-1)
        train_periods = int((pd.DatetimeIndex(dates) <= pd.Timestamp(EVAL_END)).sum())
        result = evaluator(
            values,
            labels.values.astype(np.float64),
            factor_ids=tuple(tile_names),
            backend=backend,
            n_quantiles=10,
            min_assets=20,
            min_ic_periods=20,
            direction_training_periods=train_periods,
        )
        evaluated.update(result.factors)
        backends.add(result.backend_used)
        if result.backend_fallback_reason:
            fallbacks[",".join(tile_names)] = result.backend_fallback_reason
    return {
        "factors": evaluated,
        "backend_used": backends,
        "fallbacks": fallbacks,
        "unavailable": unavailable,
        "dates": pd.DatetimeIndex(vwap.index),
    }


def report_evaluation_dict(evaluated):
    """Project the canonical QE result into the incremental manifest schema."""
    return {
        "rank_ic": evaluated.mean_rank_ic,
        "ic_ir": evaluated.rank_ic_ir,
        "std": evaluated.rank_ic_std,
        "n_days": evaluated.valid_return_periods,
        "direction": evaluated.direction,
        "is_flipped": evaluated.direction == -1,
        "ls_sharpe": evaluated.sharpe,
        "ls_annual": evaluated.annualized_return,
        "ls_cumulative": evaluated.cumulative_return,
        "ls_mdd": evaluated.max_drawdown,
        "ls_winrate": evaluated.win_rate,
        "rank_ic_winrate": getattr(evaluated, "rank_ic_win_rate", None),
        "g10_annual": getattr(evaluated, "top_quantile_annualized_return", None),
        "g1_annual": getattr(evaluated, "bottom_quantile_annualized_return", None),
    }


def write_report_manifest(records, batch_evaluation, *, target=REPORT_MANIFEST_JSON):
    """Atomically publish the sole metric source for index and detail pages."""
    import numpy as np

    target = Path(target)
    artifact_dir = target.parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dates = batch_evaluation.get("dates")
    by_name = {str(record["page_name"]): record for record in records}
    factors = {}
    for name, evaluated in batch_evaluation["factors"].items():
        metrics = report_evaluation_dict(evaluated)
        record = by_name[name]
        raw_formula = str(record.get("fe_formula") or record.get("formula") or "")
        direction = metrics.pop("direction")
        entry = {
            "factor_name": record.get("factor_name", name),
            "direction": direction,
            "is_flipped": metrics.pop("is_flipped"),
            "raw_formula": raw_formula,
            "effective_formula": f"neg({raw_formula})" if direction == -1 and raw_formula else raw_formula,
            "matrix_path": str(matrix_path(name) or ""),
            "metrics": metrics,
        }
        if dates is not None and hasattr(evaluated, "rank_ic_series"):
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
            artifact = artifact_dir / f"{safe_name}.npz"
            temporary = artifact.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    dates=np.asarray(dates, dtype="datetime64[ns]"),
                    rank_ic_series=np.asarray(evaluated.rank_ic_series),
                    quantile_returns=np.asarray(evaluated.quantile_returns),
                    quantile_nav=np.asarray(evaluated.quantile_nav),
                    long_short_returns=np.asarray(evaluated.long_short_returns),
                    long_short_nav=np.asarray(evaluated.long_short_nav),
                    long_short_nav_aligned=np.asarray(evaluated.long_short_nav_aligned),
                )
            os.replace(temporary, artifact)
            entry["artifact"] = artifact.relative_to(target.parent).as_posix()
            entry["artifact_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        factors[name] = entry
    for name, reason in batch_evaluation.get("unavailable", {}).items():
        record = by_name[name]
        raw_formula = str(record.get("fe_formula") or record.get("formula") or "")
        factors[name] = {
            "factor_name": record.get("factor_name", name),
            "status": "unavailable",
            "reason": reason,
            "direction": None,
            "is_flipped": None,
            "raw_formula": raw_formula,
            "effective_formula": None,
            "matrix_path": "",
            "metrics": {},
        }
    payload = {
        "schema_version": 1,
        "price_convention": "adj_vwap_t1_to_t2",
        "training_window": [EVAL_START, EVAL_END],
        "backend_used": sorted(batch_evaluation["backend_used"]),
        "backend_fallbacks": batch_evaluation["fallbacks"],
        "factors": factors,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    def json_safe(value):
        if isinstance(value, float) and not __import__("math").isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return value

    payload = json_safe(payload)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return payload


def _report_result_from_artifact(entry, artifact_path):
    """Load the chart contract emitted by ``write_report_manifest``.

    Publishing deliberately has no matrix or price-data dependency: a page is
    a pure projection of a hash-verified QuantEvaluator artifact.
    """
    import numpy as np
    import pandas as pd

    expected = str(entry.get("artifact_sha256") or "")
    actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if not expected or actual != expected:
        raise ValueError(f"artifact checksum mismatch for {artifact_path.name}")
    with np.load(artifact_path, allow_pickle=False) as arrays:
        required = {
            "dates", "rank_ic_series", "quantile_returns", "quantile_nav",
            "long_short_returns", "long_short_nav", "long_short_nav_aligned",
        }
        missing = required.difference(arrays.files)
        if missing:
            raise ValueError(f"artifact {artifact_path.name} misses {sorted(missing)}")
        values = {key: arrays[key].copy() for key in required}
    dates = pd.DatetimeIndex(values.pop("dates"))
    if values["quantile_nav"].ndim != 2 or len(dates) != len(values["rank_ic_series"]):
        raise ValueError(f"artifact {artifact_path.name} has inconsistent chart axes")
    metrics = dict(entry.get("metrics") or {})
    return SimpleNamespace(
        mean_rank_ic=metrics.get("rank_ic"),
        rank_ic_ir=metrics.get("ic_ir"),
        rank_ic_std=metrics.get("std"),
        valid_return_periods=metrics.get("n_days", 0),
        direction=entry.get("direction", 1),
        sharpe=metrics.get("ls_sharpe"),
        annualized_return=metrics.get("ls_annual"),
        cumulative_return=metrics.get("ls_cumulative"),
        max_drawdown=metrics.get("ls_mdd"),
        win_rate=metrics.get("ls_winrate"),
        rank_ic_win_rate=metrics.get("rank_ic_winrate"),
        top_quantile_annualized_return=metrics.get("g10_annual"),
        bottom_quantile_annualized_return=metrics.get("g1_annual"),
        **values,
    ), dates


def _unavailable_report_page(page, entry):
    reason = escape(str(entry.get("reason") or "required full-window matrix is unavailable"))
    formula = escape(str(entry.get("raw_formula") or "—"))
    return f"""<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\"/>
<title>{escape(page)} — 数据待补齐</title><body><main>
<p><a href=\"../index.html\">← 返回汇总</a></p><h1><code>{escape(page)}</code></h1>
<h2>全窗评估暂不可用</h2><p>{reason}</p>
<p>本页未展示或填充任何回测指标；待主链路完成 DataAccess → FactorEngine 落值后，将由同一报告发布流程自动更新。</p>
<h2>FactorEngine DSL</h2><pre>{formula}</pre>
</main></body></html>"""


def _write_manifest_index(report_dir, factors):
    """Render the home page directly from the canonical manifest entries."""
    rows = []
    for number, (page, entry) in enumerate(sorted(factors.items()), start=1):
        metrics = entry.get("metrics") or {}
        status = entry.get("status", "available")
        if status == "unavailable":
            cells = ("—", "—", "—", "数据待补齐")
        else:
            cells = (
                f"{metrics.get('rank_ic', float('nan')):+.4f}",
                f"{metrics.get('ic_ir', float('nan')):+.3f}",
                f"{metrics.get('ls_sharpe', float('nan')):+.2f}",
                "已翻正" if entry.get("is_flipped") else "原方向",
            )
        rows.append(
            f"<tr><td>{number}</td><td><a href=\"factors/factor_{escape(page)}.html\"><code>{escape(page)}</code></a></td>"
            f"<td>{cells[0]}</td><td>{cells[1]}</td><td>{cells[2]}</td><td>{cells[3]}</td></tr>"
        )
    available = sum(entry.get("status") != "unavailable" for entry in factors.values())
    html = f"""<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\"/>
<title>FactorEngine 因子报告</title><body><main><h1>FactorEngine 因子报告</h1>
<p>共 {len(factors)} 个因子；{available} 个已完成全窗评估。所有指标及图表均来自同一 QuantEvaluator manifest/artifact。</p>
<table><thead><tr><th>#</th><th>因子</th><th>RankIC</th><th>IR</th><th>LS Sharpe</th><th>状态</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></main></body></html>"""
    output = Path(report_dir) / "index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".html.tmp")
    temporary.write_text(html, encoding="utf-8")
    os.replace(temporary, output)


def publish_report_from_manifest(manifest_path, *, report_dir=REPORTS_DIR):
    """Publish all pages from one verified QE manifest; never recompute metrics."""
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("factors"), dict):
        raise ValueError("unsupported report manifest schema")
    report_dir = Path(report_dir)
    output_factors = report_dir / "factors"
    output_factors.mkdir(parents=True, exist_ok=True)
    published = available = unavailable = 0
    for page, entry in payload["factors"].items():
        status = entry.get("status", "available")
        if status == "unavailable":
            (output_factors / f"factor_{page}.html").write_text(
                _unavailable_report_page(page, entry), encoding="utf-8")
            unavailable += 1
            published += 1
            continue
        artifact_ref = entry.get("artifact")
        if not artifact_ref:
            raise ValueError(f"available factor {page} has no chart artifact")
        result, dates = _report_result_from_artifact(entry, manifest_path.parent / artifact_ref)
        factor = {
            "page_name": page,
            "factor_name": entry.get("factor_name", page),
            "fe_formula": entry.get("raw_formula", ""),
            "is_flipped": bool(entry.get("is_flipped")),
        }
        stage_page_inject(
            factor, report_evaluation_dict(result), report_result=result,
            report_dates=dates, out_dir=output_factors,
        )
        available += 1
        published += 1
    _write_manifest_index(report_dir, payload["factors"])
    return {"published": published, "available": available, "unavailable": unavailable}


# --------------------------------------------------------------------------
# stage: cluster_assign
# --------------------------------------------------------------------------
def stage_cluster_assign(factor, eval_result):
    """与 154 簇代表矩阵算 spearman（抽样对齐窗口），|ρ|≥0.7 归簇，否则新簇。"""
    import numpy as np
    from scipy.stats import spearmanr
    page = factor["page_name"]
    is_flipped = bool(factor.get("is_flipped", False))
    clusters = load_clusters()
    reps = clusters.get("representatives") or {}
    fmat = load_matrix(page, flip=False)
    if fmat is None:
        return {"assigned": None, "error": "matrix missing"}

    # 抽样窗口：每簇比较用 60 个对齐交易日（快），取矩阵与代表共同 index 均分抽样
    def sample_rows(mat):
        if mat is None or len(mat) == 0:
            return mat
        n = len(mat)
        if n <= 120:
            return mat
        idx = np.linspace(0, n - 1, 120).astype(int)
        return mat.iloc[idx]

    fmat_s = sample_rows(fmat)
    best = None
    for cid, rep in reps.items():
        rep_factor = rep.get("factor") if isinstance(rep, dict) else rep
        rep_mat = load_matrix(rep_factor, flip=False)
        if rep_mat is None:
            continue
        rmat_s = sample_rows(rep_mat)
        common = fmat_s.index.intersection(rmat_s.index)
        if len(common) < 30:
            continue
        fv = fmat_s.reindex(index=common)
        rv = rmat_s.reindex(index=common)
        cols = fv.columns.intersection(rv.columns)
        if len(cols) < 30:
            continue
        fv = fv[cols]
        rv = rv[cols]
        rho_sum, n = 0.0, 0
        for dt in common:
            x = fv.loc[dt]
            y = rv.loc[dt]
            mask = x.notna() & y.notna()
            if mask.sum() < 30:
                continue
            r, _ = spearmanr(x[mask], y[mask])
            if np.isfinite(r):
                rho_sum += r
                n += 1
        if n == 0:
            continue
        rho = rho_sum / n
        if best is None or abs(rho) > abs(best["rho"]):
            best = {"cluster_id": cid, "rho": float(rho)}

    if best is not None and abs(best["rho"]) >= 0.7:
        res = {"assigned": best["cluster_id"], "rho": best["rho"], "new": False,
               "representative": reps[best["cluster_id"]].get("factor")}
    else:
        new_id = f"new_{len(cm) + 1:02d}"
        # 追加新簇：cluster_members + representatives + page_to_cluster（不改旧键）
        cm = clusters.setdefault("cluster_members", {})
        cm[new_id] = [page]
        clusters.setdefault("representatives", {})[new_id] = {
            "factor": page, "best_rankic_ir": 0.0, "best_mean_rankic": 0.0,
            "cluster_size": 1}
        clusters.setdefault("page_to_cluster", {})[page] = new_id
        clusters["num_clusters"] = len(cm)
        CLUSTERS_JSON.write_text(json.dumps(clusters, ensure_ascii=False, indent=1))
        res = {"assigned": new_id, "rho": (best["rho"] if best else None),
               "new": True, "representative": page}

    # 同时把 page_to_cluster 补上（即使是归簇也写回一次）
    if best is not None and abs(best["rho"]) >= 0.7:
        clusters.setdefault("page_to_cluster", {})[page] = best["cluster_id"]
        CLUSTERS_JSON.write_text(json.dumps(clusters, ensure_ascii=False, indent=1))
    return res


# --------------------------------------------------------------------------
# stage: optimize_lite
# --------------------------------------------------------------------------
def stage_optimize_lite(factor, eval_result):
    """3 变体（原值/cs_rank/cs_zscore）同口径评估择优，追加 optimized_meta + 矩阵。"""
    page = factor["page_name"]
    is_flipped = bool(factor.get("is_flipped", False))
    import pandas as pd
    mat = load_matrix(page, flip=False)
    if mat is None:
        return {"error": "matrix missing"}
    vwap = load_vwap()

    variants = {
        "raw": mat,
        "cs_rank": mat.rank(axis=1, pct=True),
        "cs_zscore": (mat - mat.mean(axis=1)).div(mat.std(axis=1) + 1e-9),
    }
    best = None
    results = {}
    for vname, vmat in variants.items():
        vm = -vmat if is_flipped else vmat
        res = eval_matrix(vm, vwap)
        results[vname] = res
        if best is None or res["rank_ic"] > best["rank_ic"]:
            best = dict(res, variant=vname)
    if best is None:
        return {"error": "no variant"}

    meta = load_opt_meta()
    meta[page] = {
        "best": best["variant"],
        "best_mean_rankic": best["rank_ic"],
        "best_rankic_ir": best["ic_ir"],
        "steps": {"raw": [], "cs_rank": ["cs_rank"],
                  "cs_zscore": ["cs_zscore"]}[best["variant"]],
        "dsl_preproc_ops": [],
        "variants": {k: {"mean_rankic": v["rank_ic"], "rankic_ir": v["ic_ir"],
                         "n": v["n_days"]} for k, v in results.items()},
        "is_flipped": is_flipped,
    }
    OPT_META_JSON.write_text(json.dumps(meta, ensure_ascii=False, indent=1))

    # 落最优变体矩阵
    OPT_DIR.mkdir(parents=True, exist_ok=True)
    best_mat = variants[best["variant"]]
    if is_flipped:
        best_mat = -best_mat
    out = OPT_DIR / f"{page}.parquet"
    best_mat.to_parquet(out)
    best["path"] = str(out)
    return best


# --------------------------------------------------------------------------
# stage: page_inject
# --------------------------------------------------------------------------
def _render_json_tree(node, depth=0):
    """FE JSON 树 → 可读 DSL 文本（用于详情页公式展示）。"""
    if not isinstance(node, dict):
        return str(node)
    kind = node.get("kind")
    if kind == "column":
        return node.get("name", "?")
    if kind == "literal":
        v = node.get("value")
        if isinstance(v, str):
            return v
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        return str(v)
    if kind == "call":
        op = node.get("op", "?")
        args = [_render_json_tree(a) for a in node.get("args", [])]
        infix = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/",
                 "lt": "<", "gt": ">", "leq": "<=", "geq": ">=", "eq": "==",
                 "neq": "!=", "and_": "&&", "or_": "||"}
        if op in infix and len(args) == 2:
            return f"({args[0]} {infix[op]} {args[1]})"
        if op in ("negate", "abs", "log", "sqrt", "sign", "rank", "zscore",
                  "cs_rank", "cs_zscore", "ts_mean", "ts_std", "ts_sum",
                  "ts_delta", "ts_cov", "ts_corr", "ewm_mean", "where",
                  "power", "max", "min", "clip", "delay") and len(args) >= 1:
            return f"{op}(" + ", ".join(args) + ")"
        if len(args) >= 1:
            return f"{op}(" + ", ".join(args) + ")"
        return op
    return str(node)


def _dsl_text(factor):
    """详情页公式展示文本。优先 FE JSON 树渲染，fallback local_formula。"""
    fe = factor.get("fe_formula", "")
    if fe:
        try:
            tree = json.loads(fe)
            txt = _render_json_tree(tree)
            if txt and len(txt) > 5:
                return txt
        except Exception:
            pass
    lf = factor.get("local_formula", "")
    if lf:
        return lf.split(",")[0]
    return ""


def _check_banned(text, page=None):
    """检查违禁算法名。page 是因子标识名，若违禁词只作为 page 名出现（title/h1/链接），
    属于因子标识不算违规；正文/描述中出现才违规。
    先剥离 base64 图片数据（图二进制编码可能恰好含词子串），再检查纯文本。"""
    # 去掉 <img src="data:image/...base64..."> 内容
    clean = re.sub(r'data:image/[^"]+', '', text)
    low = clean.lower()
    for w in BANNED:
        if w not in low:
            continue
        idxs = [m.start() for m in re.finditer(re.escape(w), low)]
        if page and w in page.lower():
            # page 名本身含该词：标题/文件名/返回链接等标识性出现可容忍。
            flagged = 0
            main_idx = low.find("<main>")
            for i in idxs:
                if main_idx != -1 and i > main_idx:
                    ctx = clean[max(0, i - 120): i + 200]
                    # 因子统计摘要表里的因子名称行不算
                    if f"<td>因子名称</td><td><code>{page}</code>" in ctx:
                        continue
                    # 因子族行里代表因子 == page 也不算（簇代表是它自己）
                    if f"代表因子: {page}" in ctx:
                        continue
                    if ">该因子为" in ctx:
                        continue
                    flagged += 1
            if flagged:
                return w
        else:
            return w
    return None


def stage_page_inject(factor, eval_result, *, report_result=None, report_dates=None, out_dir=None):
    """为每条新因子生成详情页（复用 render_evoalpha14_pages 的模板+图表函数）。
    图可省略或复用 optimize 图函数。禁止算法名字符串。"""
    page = factor["page_name"]
    output_dir = Path(out_dir) if out_dir is not None else FACTORS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    is_flipped = bool((eval_result or {}).get("is_flipped", factor.get("is_flipped", False)))
    dsl_text = _dsl_text(factor)
    fe_formula_raw = factor.get("fe_formula", "")
    if is_flipped:
        dsl_text = f"neg({dsl_text})" if dsl_text else dsl_text
        fe_formula_raw = f"neg({fe_formula_raw})" if fe_formula_raw else fe_formula_raw
    note = "本周新挖增量因子"

    # 尝试 import 复用 evo14 渲染器
    try:
        import render_evoalpha14_pages as R
        has_tpl = True
    except Exception:
        has_tpl = False

    if not has_tpl:
        # 内联简版：无图单页
        html = _minimal_page(page, dsl_text, fe_formula_raw, is_flipped, eval_result)
        out = output_dir / f"factor_{page}.html"
        out.write_text(html, encoding="utf-8")
        return {"page": str(out), "mode": "minimal"}

    try:
        opt_meta = load_opt_meta()
        cluster = load_clusters()
        qg = (cluster.get("quality_gate") or {}).get(page, {})
        cluster_id = (cluster.get("page_to_cluster") or {}).get(page, "—")
        rep_entry = (cluster.get("representatives") or {}).get(cluster_id) or {}
        rep_factor = rep_entry.get("factor", "") if isinstance(rep_entry, dict) else ""
        gate = {"cluster": cluster_id,
                "quality_gate": qg.get("quality_gate", "—"),
                "gate_msg": qg.get("message", ""),
                "rep_factor": rep_factor}

        if report_result is None or report_dates is None:
            raise ValueError("canonical QuantEvaluator result is required for full report rendering")
        import pandas as pd
        dates = pd.DatetimeIndex(report_dates)
        ic_series = pd.Series(report_result.rank_ic_series, index=dates)
        decile_navs = {
            "dates": dates.tolist(),
            **{
                f"G{group + 1}": report_result.quantile_nav[:, group].tolist()
                for group in range(report_result.quantile_nav.shape[1])
            },
            "LS": report_result.long_short_nav_aligned.tolist(),
        }
        base = {
            "mean_rankic": report_result.mean_rank_ic,
            "std_rankic": report_result.rank_ic_std,
            "rankic_ir": report_result.rank_ic_ir,
            "rankic_winrate": report_result.rank_ic_win_rate,
            "ls_sharpe": report_result.sharpe,
            "ls_annual": report_result.annualized_return,
            "ls_cum": report_result.cumulative_return,
            "ls_mdd": report_result.max_drawdown,
            "ls_winrate": report_result.win_rate,
            "g10_annual": report_result.top_quantile_annualized_return,
            "g1_annual": report_result.bottom_quantile_annualized_return,
            "n_periods": report_result.valid_return_periods,
            "ic": ic_series,
            "decile_navs": decile_navs,
        }
        charts = {
            "svg_ts": R.plot_ic_timeseries_svg(ic_series, page),
            "monthly": R.plot_ic_monthly_heatmap(ic_series, page),
            "decile": R.plot_decile_nav(decile_navs, page),
            "ls": R.plot_long_short_nav(decile_navs, page),
            "dist": R.plot_ic_distribution(ic_series, page),
        }
        # Optimization comparisons have their own canonical batch artifact;
        # never silently recompute them inside the page renderer.
        opt_charts = ""

        dsl_note = "factor_engine DSL（本因子由增量管线落值）"
        req_cols = _required_columns(dsl_text)
        html = R.build_html(page, note, dsl_text, dsl_note, req_cols, base,
                            opt_meta.get(page, {}), gate, {}, charts, opt_charts)

        # 替换回测区间为我们的评估口径说明
        html = html.replace("本周新挖", "本周新挖（增量）")
        html = html.replace("回测区间 2019-01-02 ~ 2026-08-24",
                            f"全窗评估 {FULL_WINDOW_START.date()} ~ {FULL_WINDOW_END.date()}"
                            "（方向仅由 2016-01-04 ~ 2018-06-30 训练窗确定；vwap-to-vwap shift(-2)）")
        out = output_dir / f"factor_{page}.html"
        out.write_text(html, encoding="utf-8")

        banned = _check_banned(html, page=page)
        if banned:
            # 违规就降级为 minimal 无图页
            html2 = _minimal_page(page, dsl_text, fe_formula_raw, is_flipped, eval_result)
            out.write_text(html2, encoding="utf-8")
            return {"page": str(out), "mode": "minimal", "banned_in_full": banned}
        return {"page": str(out), "mode": "full"}
    except Exception as exc:
        print(f"    [page_inject fallback] {type(exc).__name__}: {str(exc)[:120]}", flush=True)
        html = _minimal_page(page, dsl_text, fe_formula_raw, is_flipped, eval_result)
        out = output_dir / f"factor_{page}.html"
        out.write_text(html, encoding="utf-8")
        return {"page": str(out), "mode": "minimal"}


def _required_columns(dsl_text):
    import re as _re
    fields = ["AdjClose", "AdjOpen", "AdjHigh", "AdjLow", "AdjVwap", "AdjPreClose",
              "Volume", "AdjAmount", "Return", "close_price", "vwap", "amount"]
    used = []
    for f in fields:
        if f in dsl_text:
            used.append(f)
    return ", ".join(dict.fromkeys(used))


def _opt_compare_charts(page, raw_mat, opt_mat, is_flipped):
    if opt_mat is None:
        return ""
    try:
        import render_optimized_pages as R2
        import pandas as pd
        vwap = R2.load_vwap()
        charts = ""
        raw_ic = R2.daily_rankic_series(raw_mat, vwap)
        if is_flipped:
            raw_ic = -raw_ic
        opt_ic = R2.daily_rankic_series(opt_mat, vwap)
        try:
            ic = R2.plot_ic_compare(raw_ic, opt_ic, page, is_flipped)
            if ic:
                charts += f'  <img src="data:image/png;base64,{ic}" style="width:100%;border-radius:8px;margin-bottom:8px" alt="RankIC对比"/>\n'
        except Exception:
            pass
        try:
            dec = R2.plot_decile_compare(raw_mat, opt_mat, vwap, page, is_flipped)
            if dec:
                charts += f'  <img src="data:image/png;base64,{dec}" style="width:100%;border-radius:8px;margin-bottom:8px" alt="十分层对比"/>\n'
        except Exception:
            pass
        try:
            ls = R2.plot_ls_compare(raw_mat, opt_mat, vwap, page, is_flipped)
            if ls:
                charts += f'  <img src="data:image/png;base64,{ls}" style="width:100%;border-radius:8px" alt="多空对比"/>\n'
        except Exception:
            pass
        if not charts:
            charts = '  <p style="color:#94a3b8;font-size:0.8rem">（对比图生成失败）</p>\n'
        return charts
    except Exception:
        return ""


def _minimal_page(page, dsl_text, fe_formula_raw, is_flipped, eval_result):
    """降级简版页面（无图，仅指标/公式/来源）。"""
    e = eval_result or {}
    ic = e.get("rank_ic", 0.0)
    ir = e.get("ic_ir", 0.0)
    flip_badge = '<span class="badge badge-yellow">⚠ 已翻转（IC<0）</span>' if is_flipped else ""
    ic_cls = "pos" if ic >= 0 else "neg"
    ir_cls = "pos" if ir >= 0 else "neg"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{page}</title>
<style>
:root{{--bg:#eef2f7;--panel:#fff;--fg:#0f172a;--muted:#64748b;--line:#e2e8f0;--primary:#1e4d8c;--pos:#16a34a;--neg:#dc2626}}
*{{box-sizing:border-box}}
body{{margin:0;font-family:"Segoe UI","Microsoft YaHei",system-ui,sans-serif;color:var(--fg);background:var(--bg)}}
header{{background:linear-gradient(135deg,#0f2744,#1e4d8c 60%,#0d9488);color:#fff;padding:28px 48px 22px}}
header h1{{margin:0 0 6px;font-size:1.5rem;word-break:break-all}}
header .meta{{opacity:0.85;font-size:0.85rem;margin-top:4px}}
main{{max-width:1100px;margin:0 auto;padding:24px}}
.back{{display:inline-block;margin-bottom:16px;color:#93c5fd;font-weight:500;text-decoration:none;font-size:0.88rem}}
.grid-4{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:12px}}
.metric{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px;text-align:center;box-shadow:0 4px 24px rgba(15,23,42,0.06)}}
.metric b{{display:block;font-size:1.35rem}}
.metric span{{color:var(--muted);font-size:0.73rem}}
.pos{{color:var(--pos)}}.neg{{color:var(--neg)}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px;box-shadow:0 4px 24px rgba(15,23,42,0.06);margin-bottom:16px}}
h2{{font-size:0.95rem;color:var(--primary);margin:0 0 12px;border-bottom:1px solid var(--line);padding-bottom:8px}}
.formula-wrap{{background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:16px;font-family:"Courier New",monospace;font-size:0.85rem;word-break:break-all;line-height:1.8;white-space:pre-wrap}}
.badge{{display:inline-block;padding:2px 8px;border-radius:12px;font-size:0.73rem;margin-left:6px}}
.badge-yellow{{background:#fef3c7;color:#92400e}}
.qe-info{{display:inline-block;background:linear-gradient(90deg,#ede9fe,#dbeafe);color:#5b21b6;padding:2px 10px;border-radius:12px;font-size:0.72rem;font-weight:600;margin-left:8px}}
.meta-table{{width:100%;border-collapse:collapse;font-size:0.84rem}}
.meta-table td{{padding:7px 10px;border-bottom:1px solid var(--line)}}
.meta-table td:first-child{{color:var(--muted);width:160px;font-weight:500}}
</style>
</head>
<body>
<header>
<a class="back" href="../index.html">&#8592; 返回汇总</a>
<h1><code>{page}</code>{flip_badge}<span class="qe-info">⚡ quant_evaluator</span></h1>
<div class="meta">本周新挖（增量）· 评估区间 {EVAL_START} ~ {EVAL_END} · 收益口径 Vwap 后复权 vwap-to-vwap（shift(-2)）</div>
</header>
<main>
<div class="grid-4">
<div class="metric"><b class="{ic_cls}">{ic:+.4f}</b><span>RankIC</span></div>
<div class="metric"><b class="{ir_cls}">{ir:+.3f}</b><span>RankIC IR</span></div>
<div class="metric"><b>{e.get("n_days", 0)}</b><span>评估交易日</span></div>
<div class="metric"><b>{"是" if is_flipped else "否"}</b><span>已翻正</span></div>
</div>
<div class="card">
<h2>📐 因子表达式（FactorEngine DSL）</h2>
<div class="formula-wrap">{dsl_text or "（无 DSL 文本，见 source 字段）"}</div>
</div>
<div class="card">
<h2>🧬 优化因子（预处理 + 择优）</h2>
<p style="font-size:0.82rem;color:#64748b;margin:0">本因子已进入 weekly_backtest_output/optimized_factors/ 与 optimized_meta.json，最优变体与 IR 见汇总表。</p>
</div>
</main>
</body>
</html>
"""


# --------------------------------------------------------------------------
# stage: json_writeback
# --------------------------------------------------------------------------
def stage_json_writeback(factor, eval_result, cluster_result):
    """重新 Read 池 json → append 新条目 → dump → 校验 can_use 计数。"""
    page = factor["page_name"]
    fname = factor.get("factor_name") or f"factor_{page}"
    fe_formula = factor.get("fe_formula", "")
    is_flipped = bool(factor.get("is_flipped", False))
    dsl_text = _dsl_text(factor)
    lqtp_formula = factor.get("local_formula", "") or dsl_text

    pool = load_pool()
    orig_total = len(pool)
    orig_can_use = sum(1 for r in pool if r.get("can_use_factor_engine"))
    if page in {r.get("page_name") for r in pool}:
        return {"skipped": True, "reason": "already in pool"}

    entry = {
        "page_name": page,
        "factor_name": fname,
        "status": "incremental_intake",
        "dsl": dsl_text,
        "lqtp_formula": lqtp_formula,
        "fe_formula": fe_formula,
        "code": "",
        "is_flipped": is_flipped,
        "can_use_factor_engine": True,
        "note": "本周新挖增量因子（incremental_intake）",
        "is_unlisted_miner": True,
    }
    pool.append(entry)
    POOL_JSON.write_text(json.dumps(pool, ensure_ascii=False, indent=1))

    new_can_use = sum(1 for r in pool if r.get("can_use_factor_engine"))
    ok = new_can_use == orig_can_use + 1
    return {"added": page, "total": len(pool), "can_use": new_can_use,
            "orig_can_use": orig_can_use, "check_ok": ok,
            "total_check": len(pool) == orig_total + 1}


# --------------------------------------------------------------------------
# 首页：header 计数更新 + 新挖区块追加 26 行
# --------------------------------------------------------------------------
def update_index(new_entries):
    """更新 index.html：
    1) +14 → +40、含本周新挖 14 → 40、470 → 496（总数 470+26）
    2) all-factors 表尾追加 26 行（新因子标「新」badge）
    3) 在 robustness 前插入「本周新挖」区块（列出 26 条）
    """
    html = INDEX_HTML.read_text(encoding="utf-8")
    pool_pages = {str(record.get("page_name")) for record in load_pool() if record.get("page_name")}
    total = len(pool_pages)
    n_new = len({str(entry["page_name"]) for entry in new_entries})

    # 1) header 统计
    html = re.sub(r"<b>\d+</b><span>因子总数</span>",
                  f"<b>{total}</b><span>因子总数</span>", html)
    html = re.sub(r"<b>\+\d+</b><span>本周新挖</span>",
                  f"<b>+{n_new}</b><span>本周新挖</span>", html)
    html = re.sub(r'<h2 id="all-factors">全部 \d+ 个因子</h2>',
                  f'<h2 id="all-factors">全部 {total} 个因子</h2>', html)
    html = re.sub(r"含本周新挖 \d+", f"含本周新挖 {n_new}", html)
    html = re.sub(r"存活 (\d+)/\d+", rf"存活 \1/{total}", html)

    # 2) 新挖因子区块（插入 all-factors 表后，robustness 前）
    rows = []
    for i, en in enumerate(new_entries, start=max(1, total - n_new + 1)):
        page = en["page_name"]
        ic = en.get("rank_ic", 0.0)
        ir = en.get("ic_ir", 0.0)
        flipped = en.get("is_flipped", False)
        flip_tag = '<span class="tag tag-flip">翻正</span>' if flipped else ""
        new_tag = '<span class="tag" style="background:#dcfce7;color:#166534">新</span>'
        ic_cls = "pos" if ic >= 0 else "neg"
        rows.append(
            f'<tr><td class="rank">{i}</td>'
            f'<td><a href="factors/factor_{page}.html"><code>{page}</code></a>{new_tag}{flip_tag}</td>'
            f'<td class="{ic_cls}">{ic:.4f}</td>'
            f'<td>{ir:.3f}</td>'
            f'<td class="neg">—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>'
        )
    # 插入到 </tbody></table> 之后第一个 </table> 之前? all-factors 表是第一个 table。
    # 直接在 all-factors 的 </table> 后追加新挖区块
    marker = '</tbody>\n  </table>'
    block = '\n' + '\n'.join(rows) + '\n    </tbody>\n  </table>\n'
    # 替换 all-factors 表的结尾：在其 </tbody></table> 前插入新行
    idx_tbl = html.find('id="all-factors"')
    idx_tbody_end = html.find('</tbody>', idx_tbl)
    html = html[:idx_tbody_end] + '\n' + '\n'.join(rows) + '\n' + html[idx_tbody_end:]

    # 3) 「本周新挖」区块（放 robustness 前）
    new_section = _new_mining_section(new_entries)
    anchor = '<section id="robustness-2026">'
    html = html.replace(anchor, new_section + '\n' + anchor, 1)

    INDEX_HTML.write_text(html, encoding="utf-8")
    banned = _check_banned(new_section, page="")
    return {"total": total, "n_new": n_new, "rows_added": len(rows),
            "banned_in_new_section": banned}


def _new_mining_section(new_entries):
    rows = ""
    for i, en in enumerate(new_entries, start=1):
        page = en["page_name"]
        ic = en.get("rank_ic", 0.0)
        ir = en.get("ic_ir", 0.0)
        cluster_id = en.get("cluster_id", "—")
        flipped = en.get("is_flipped", False)
        flip_tag = '<span class="tag tag-flip">翻正</span>' if flipped else ""
        new_tag = '<span class="tag" style="background:#dcfce7;color:#166534">新</span>'
        ic_cls = "pos" if ic >= 0 else "neg"
        rows += (
            f'<tr><td class="rank">{i}</td>'
            f'<td><a href="factors/factor_{page}.html"><code>{page}</code></a>{new_tag}{flip_tag}</td>'
            f'<td class="{ic_cls}">{ic:.4f}</td>'
            f'<td>{ir:.3f}</td>'
            f'<td>{cluster_id}</td></tr>'
        )
    return f'''
<section id="new-mining">
<h2>🆕 本周新挖增量因子（{len(new_entries)}）</h2>
<p style="font-size:0.82rem;color:#64748b">本次增量入库 {len(new_entries)} 条，评估口径与既有池一致（vwap-to-vwap shift(-2) 后复权）。</p>
<table>
  <thead><tr><th>#</th><th>因子</th><th>RankIC</th><th>IR</th><th>因子族</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
</section>
'''


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--publish-from-manifest", type=Path,
                    help="render index/detail HTML only from a verified QE report manifest")
    ap.add_argument("--report-dir", type=Path, default=REPORTS_DIR,
                    help="target report directory for --publish-from-manifest")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--fe-backend", default="polars_long")
    ap.add_argument("--qe-backend", choices=("auto", "cpu", "cuda", "cuda_strict"), default="auto")
    ap.add_argument("--skip-landing", action="store_true")
    ap.add_argument("--evaluate-only", action="store_true",
                    help="evaluate and write the manifest, but do not mutate pages/pool/state")
    ap.add_argument("--output-manifest", type=Path, default=REPORT_MANIFEST_JSON)
    args = ap.parse_args()

    if args.publish_from_manifest is not None:
        result = publish_report_from_manifest(
            args.publish_from_manifest, report_dir=args.report_dir,
        )
        print(f"[publish] {json.dumps(result, ensure_ascii=False)}", flush=True)
        return
    if not args.manifest:
        ap.error("--manifest is required unless --publish-from-manifest is used")

    manifest = load_manifest(args.manifest)
    if args.limit:
        manifest = manifest[:args.limit]

    if not args.skip_landing:
        landing = land_missing_factors(
            manifest,
            batch_size=max(1, args.batch_size),
            backend_name=args.fe_backend,
        )
        print(f"[run_many] {json.dumps(landing, ensure_ascii=False)}", flush=True)

    batch_evaluation = evaluate_factor_batch(
        manifest,
        batch_size=max(1, args.batch_size),
        backend=args.qe_backend,
    )
    written_manifest = write_report_manifest(
        manifest, batch_evaluation, target=args.output_manifest,
    )
    print(
        f"[quant_evaluator] backends={sorted(batch_evaluation['backend_used'])} "
        f"fallbacks={batch_evaluation['fallbacks']}",
        flush=True,
    )
    if args.evaluate_only:
        print(
            f"[evaluate-only] factors={len(written_manifest['factors'])} "
            f"manifest={args.output_manifest}",
            flush=True,
        )
        return

    state = load_state()
    done = {}
    for factor in manifest:
        fname = factor["factor_name"]
        page = factor["page_name"]
        print(f"\n=== {page} ({fname}) ===", flush=True)
        if fname not in state:
            state[fname] = {"stage": "", "result": {}, "page": page}
        for stage in STAGES:
            if stage_done(state, fname, stage):
                print(f"  [skip] {stage}", flush=True)
                continue
            try:
                if stage == "dedup_check":
                    res = stage_dedup_check(factor)
                elif stage == "landing":
                    res = stage_landing(factor)
                elif stage == "eval":
                    res = report_evaluation_dict(batch_evaluation["factors"][page])
                    res["matrix_path"] = str(matrix_path(page))
                elif stage == "cluster_assign":
                    res = stage_cluster_assign(
                        factor, state[fname]["result"].get("eval", {}))
                elif stage == "optimize_lite":
                    res = stage_optimize_lite(
                        factor, state[fname]["result"].get("eval", {}))
                elif stage == "page_inject":
                    res = stage_page_inject(
                        factor,
                        state[fname]["result"].get("eval", {}),
                        report_result=batch_evaluation["factors"][page],
                        report_dates=batch_evaluation["dates"],
                    )
                elif stage == "json_writeback":
                    res = stage_json_writeback(
                        factor,
                        state[fname]["result"].get("eval", {}),
                        state[fname]["result"].get("cluster_assign", {}))
                else:
                    res = {}
            except Exception:
                print(f"  [FAIL] {stage}\n{traceback.format_exc()}", flush=True)
                save_state(state)
                sys.exit(1)
            state[fname]["result"][stage] = res
            state[fname]["stage"] = stage
            save_state(state)
            msg = json.dumps(res, ensure_ascii=False)
            print(f"  [done] {stage}: {msg[:220]}", flush=True)
        done[page] = {
            "factor_name": fname,
            "eval": state[fname]["result"].get("eval", {}),
            "cluster": state[fname]["result"].get("cluster_assign", {}),
            "optimize": state[fname]["result"].get("optimize_lite", {}),
            "page": state[fname]["result"].get("page_inject", {}),
        }

    # 全部完成后：更新首页
    if not args.limit or len(manifest) >= 26:
        new_entries = []
        for fname, st in state.items():
            if st.get("stage") == "json_writeback" and st.get("page"):
                page = st["page"]
                ev = st["result"].get("eval", {})
                cl = st["result"].get("cluster_assign", {})
                new_entries.append({
                    "page_name": page,
                    "rank_ic": ev.get("rank_ic", 0.0),
                    "ic_ir": ev.get("ic_ir", 0.0),
                    "is_flipped": ev.get("is_flipped", False),
                    "cluster_id": cl.get("assigned", "—"),
                })
        if new_entries:
            upd = update_index(new_entries)
            print(f"\n[index] 首页已更新: {json.dumps(upd, ensure_ascii=False)[:200]}", flush=True)

    DONE_JSON.write_text(json.dumps(done, ensure_ascii=False, indent=1))
    print(f"\nALL DONE -> {DONE_JSON}", flush=True)


if __name__ == "__main__":
    main()
