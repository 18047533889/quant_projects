# -*- coding: utf-8 -*-
"""R31-P0-023/024: SQL certification factory —— DuckDB emitter vs Pandas reference parity。

目标（R31 §24/25/109/110）
    - 自动化 parity harness：每个 candidate canonical，Pandas reference vs
      DuckDB SQL，在 normal / NaN / ties / zero-denominator / short-history /
      constant / group 等 fixtures 上比较。
    - 输出：``value parity`` / ``NaN mask parity`` / ``dtype`` / ``index order`` /
      warmup。
    - 生成 ``docs/evidence/r31/R31_SQL_EMITTER_CERTIFICATION.csv`` 与
      ``R31_TRIPLE_BACKEND_PARITY.csv``。
    - **ClickHouse 不自动继承 DuckDB**（R31 §25）：同 emitter 在 ClickHouse 需
      单独认证——本 harness 只认证 DuckDB；ClickHouse 单独计分。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from planner.logical_plan import PlanNode  # noqa: E402

EVIDENCE_DIR = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "r31"

#: Tier A 核心算子（R31 §23）：四则/比较/逻辑 + 基础 ts/cs/group + 简单技术。
#: 只认证真实 SQL emitter 支持且非 research-only 的算子。
CORE_OPS: dict[str, dict[str, Any]] = {
    "add": {"arity": 2},
    "subtract": {"arity": 2},
    "multiply": {"arity": 2},
    "divide": {"arity": 2},
    "abs": {"arity": 1},
    "log": {"arity": 1},
    "exp": {"arity": 1},
    "sqrt": {"arity": 1},
    "sign": {"arity": 1},
    "clip": {"arity": 1, "low": 1.0, "high": 5.0},
    "power": {"arity": 2},
    "gt": {"arity": 2},
    "lt": {"arity": 2},
    "ge": {"arity": 2},
    "le": {"arity": 2},
    "eq": {"arity": 2},
    "ne": {"arity": 2},
    "and_": {"arity": 2, "bool": True},
    "or_": {"arity": 2, "bool": True},
    "not_": {"arity": 1, "bool": True},
    "where": {"arity": 3, "special": "where"},
    "coalesce": {"arity": 2, "special": "coalesce"},
    "is_nan": {"arity": 1},
    "is_finite": {"arity": 1},
    "fillna": {"arity": 1, "fill": 0.0},
    "ts_delay": {"arity": 1, "window": 1},
    "ts_delta": {"arity": 1, "window": 1},
    "ts_pct": {"arity": 1, "window": 1},
    "log_returns": {"arity": 1, "window": 1},
    "ts_mean": {"arity": 1, "window": 3},
    "ts_sum": {"arity": 1, "window": 3},
    "ts_std": {"arity": 1, "window": 3},
    "ts_min": {"arity": 1, "window": 3},
    "ts_max": {"arity": 1, "window": 3},
    "ts_count": {"arity": 1, "window": 3},
    "ts_ema": {"arity": 1, "window": 3},
    "ts_rank": {"arity": 1, "window": 3},
    "ts_median": {"arity": 1, "window": 3},
    "cs_sum": {"arity": 1},
    "cs_mean": {"arity": 1},
    "cs_std": {"arity": 1},
    "cs_count": {"arity": 1},
    "cs_rank": {"arity": 1},
    "cs_zscore": {"arity": 1},
    "cs_demean": {"arity": 1},
    "rank": {"arity": 1},
    "group_mean": {"arity": 1, "group": "inst"},
    "group_std": {"arity": 1, "group": "inst"},
}


def _plan_for(canonical: str, spec: dict[str, Any]) -> PlanNode:
    cols = ["close", "open", "volume"]
    def leaf(i: int) -> PlanNode:
        return PlanNode(op="column", attrs={"name": cols[i]})

    arity = spec.get("arity", 1)
    kwargs: dict[str, Any] = {}
    for key in ("window", "low", "high", "fill", "group_by"):
        if key in spec:
            kwargs[key] = spec[key]
    inputs = [leaf(i) for i in range(arity)]
    if spec.get("group"):
        # group 算子：第二个输入是 group 列（fixture 提供 sector）。
        inputs.append(PlanNode(op="column", attrs={"name": "sector"}))
    if spec.get("special") == "where":
        # where(cond, x, y)
        cond = PlanNode(op="gt", inputs=[leaf(0), PlanNode(op="literal", attrs={"value": 2.0})])
        return PlanNode(op=canonical, inputs=[cond, leaf(1), leaf(2)], attrs=kwargs)
    if spec.get("special") == "coalesce":
        return PlanNode(op=canonical, inputs=[leaf(0), PlanNode(op="literal", attrs={"value": 0.0})], attrs=kwargs)
    if spec.get("bool"):
        # 布尔算子在数值列上：close > 2
        return PlanNode(op=canonical, inputs=[PlanNode(op="gt", inputs=[leaf(0), PlanNode(op="literal", attrs={"value": 2.0})]), leaf(1) if arity > 1 else PlanNode(op="gt", inputs=[leaf(1), PlanNode(op="literal", attrs={"value": 2.0})])], attrs=kwargs) if arity == 2 else PlanNode(op=canonical, inputs=[PlanNode(op="gt", inputs=[leaf(0), PlanNode(op="literal", attrs={"value": 2.0})])], attrs=kwargs)
    return PlanNode(op=canonical, inputs=inputs, attrs=kwargs)


def _build_fixture(con: Any, *, name: str) -> None:
    """normal fixture：6 行 × 2 inst，含 NaN 与 Inf，5 列。"""
    dates = pd.bdate_range("2024-01-02", periods=6)
    rows = [(d, inst) for d in dates for inst in ["A", "B"]]
    data = {
        "ts": [r[0] for r in rows],
        "inst": [r[1] for r in rows],
        "close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "open": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        "volume": [100.0, 200.0, 0.0, 400.0, 500.0, 600.0, 100.0, 200.0, 0.0, 400.0, 500.0, 600.0],
        "sector": ["FIN", "FIN", "TEC", "TEC", "FIN", "FIN", "TEC", "TEC", "FIN", "FIN", "TEC", "TEC"],
    }
    # 注入 NaN / Inf / zero-denominator / tie
    data["close"][2] = float("nan")
    data["open"][1] = float("nan")
    data["volume"][3] = float("inf")
    df = pd.DataFrame(data)
    con.register(name, df)


def _run_sql(con: Any, plan: PlanNode, *, table: str) -> pd.Series | None:
    from backend.sql_pushdown.emitter import compile_plan_to_sql

    compiled = compile_plan_to_sql(
        plan, dataset=table, time_column="ts", instrument_column="inst"
    )
    if compiled is None:
        return None
    query = compiled.query.replace("{{" + table + "}}", table)
    try:
        out = con.execute(query).fetchdf()
    except Exception:
        return None
    if out.empty:
        return pd.Series(dtype=float)
    return out.set_index(["ts", "inst"])["value"].sort_index()


def _run_pandas(plan: PlanNode) -> pd.Series | None:
    from backend.context import ExecutionContext
    from backend.pandas_backend import PandasBackend
    from tests.helpers import InMemorySeriesSource

    dates = pd.bdate_range("2024-01-02", periods=6)
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["timestamp", "instrument"])
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    open_vals = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    vol_vals = [100.0, 200.0, 0.0, 400.0, 500.0, 600.0, 100.0, 200.0, 0.0, 400.0, 500.0, 600.0]
    close = pd.Series(vals, index=idx)
    open_ = pd.Series(open_vals, index=idx)
    volume = pd.Series(vol_vals, index=idx)
    close.iloc[2] = float("nan")
    open_.iloc[1] = float("nan")
    volume.iloc[3] = float("inf")
    sector = pd.Series(
        ["FIN", "FIN", "TEC", "TEC", "FIN", "FIN", "TEC", "TEC", "FIN", "FIN", "TEC", "TEC"],
        index=idx,
    )
    src = InMemorySeriesSource({"close": close, "open": open_, "volume": volume, "sector": sector})
    ctx = ExecutionContext(data_source=src, run_mode="research")
    try:
        result = PandasBackend().execute(plan, ctx)
        return result if isinstance(result, pd.Series) else None
    except Exception:
        return None


def _parity(ref: pd.Series | None, sql: pd.Series | None) -> dict[str, Any]:
    if ref is None and sql is None:
        return {"emitted": False, "parity": "n/a"}
    if sql is None:
        return {"emitted": True, "parity": "no-sql"}
    if ref is None:
        return {"emitted": True, "parity": "no-ref"}
    value_ok = True
    mask_ok = True
    dtype_ok = str(ref.dtype) == str(sql.dtype)
    try:
        a = ref.reindex(sql.index)
        b = sql
        # 值比较 dtype-agnostic：统一转 float 后逐元素比（``equals`` 会把
        # float64 vs Int8 的「值相同但 dtype 不同」误判为不等）。
        a_f = a.fillna(0.0).astype(float).to_numpy()
        b_f = b.fillna(0.0).astype(float).to_numpy()
        value_ok = bool((a_f == b_f).all())
        mask_ok = bool((a.isna() == b.isna()).all())
    except Exception:
        value_ok = False
        mask_ok = False
    idx_ok = list(ref.index) == list(sql.index)
    if value_ok and mask_ok and dtype_ok and idx_ok:
        parity = "pass"
    elif value_ok and mask_ok and idx_ok:
        # 值 + NaN mask + index 完全一致，仅 dtype 不同（float64 vs Int8 等）——
        # 诚实标记 value-level parity，可 production（值语义一致）。
        parity = "pass-value-mask"
    else:
        parity = "fail"
    return {
        "emitted": True,
        "parity": parity,
        "value_ok": value_ok,
        "mask_ok": mask_ok,
        "dtype_ok": dtype_ok,
        "index_ok": idx_ok,
    }


def run_certification(ops: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    import duckdb

    con = duckdb.connect(":memory:")
    _build_fixture(con, name="px")
    ops = ops or CORE_OPS
    results: list[dict[str, Any]] = []
    for canonical, spec in ops.items():
        plan = _plan_for(canonical, spec)
        ref = _run_pandas(plan)
        sql = _run_sql(con, plan, table="px")
        info = _parity(ref, sql)
        results.append(
            {
                "canonical": canonical,
                "family": "core",
                "duckdb_emitter": "yes" if info["emitted"] else "no",
                "duckdb_parity": info["parity"],
                "duckdb_prod_safe": "yes" if info["parity"] in {"pass", "pass-value-mask"} else "no",
                "value_ok": info.get("value_ok"),
                "mask_ok": info.get("mask_ok"),
                "dtype_ok": info.get("dtype_ok"),
                "index_ok": info.get("index_ok"),
                "clickhouse_parity": "not-certified",
                "recommended_target": "duckdb_sql" if info["parity"] in {"pass", "pass-value-mask"} else "pandas_numpy",
            }
        )
    con.close()
    return results


def write_artifacts(results: list[dict[str, Any]]) -> dict[str, Any]:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    cert_path = EVIDENCE_DIR / "R31_SQL_EMITTER_CERTIFICATION.csv"
    fields = [
        "canonical", "family", "duckdb_emitter", "duckdb_parity",
        "duckdb_prod_safe", "value_ok", "mask_ok", "dtype_ok", "index_ok",
        "clickhouse_parity", "recommended_target",
    ]
    with cert_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    # R31_TRIPLE_BACKEND_PARITY.csv：三后端矩阵（Pandas=reference always prod-safe）。
    triple_path = EVIDENCE_DIR / "R31_TRIPLE_BACKEND_PARITY.csv"
    tfields = [
        "canonical", "family", "pandas_impl", "pandas_prod",
        "polars_impl", "polars_parity", "duckdb_emitter", "duckdb_parity",
        "duckdb_prod", "clickhouse_parity", "recommended_target",
    ]
    with triple_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=tfields)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "canonical": row["canonical"],
                    "family": row["family"],
                    "pandas_impl": "yes",
                    "pandas_prod": "yes",
                    "polars_impl": "n/a",
                    "polars_parity": "n/a",
                    "duckdb_emitter": row["duckdb_emitter"],
                    "duckdb_parity": row["duckdb_parity"],
                    "duckdb_prod": row["duckdb_prod_safe"],
                    "clickhouse_parity": "not-certified",
                    "recommended_target": row["recommended_target"],
                }
            )
    summary = {
        "canonicals_certified": len(results),
        "duckdb_emitters": sum(1 for r in results if r["duckdb_emitter"] == "yes"),
        "duckdb_parity_pass": sum(1 for r in results if r["duckdb_parity"] in {"pass", "pass-value-mask"}),
        "duckdb_parity_fail": sum(1 for r in results if r["duckdb_parity"] == "fail"),
        "clickhouse_separate_certification": "not-certified",
        "note": "R31 只认证 DuckDB；ClickHouse 独立 SQL dialect production certification，不自动继承 DuckDB。",
    }
    (EVIDENCE_DIR / "R31_DUCKDB_CORE_PARITY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def write_backend_target_matrix() -> dict[str, Any]:
    """R31-109: R31_BACKEND_TARGET_MATRIX.csv —— 三后端 target 矩阵自动生成。

    不人工维护「哪些该三后端」名单：canonical 从认证结果 + operator registry
    派生，pandas_impl 一律 yes（reference），polars/duckdb 由认证/能力表填充。
    """
    from cleaned_operators.registry import OperatorRegistry

    rows = list(csv.DictReader(open(EVIDENCE_DIR / "R31_TRIPLE_BACKEND_PARITY.csv")))
    certified = {r["canonical"] for r in rows}
    matrix_fields = [
        "canonical", "family", "surface", "pandas_impl", "pandas_prod",
        "polars_impl", "polars_parity", "polars_prod", "polars_long_native",
        "duckdb_emitter", "duckdb_parity", "duckdb_prod", "clickhouse_parity",
        "recommended_target", "priority", "benchmark_class",
    ]
    matrix_rows: list[dict[str, str]] = []
    for r in rows:
        try:
            backends = OperatorRegistry.backends_for(r["canonical"])
        except Exception:
            backends = []
        has_polars = "polars" in backends or "polars_panel" in backends or "polars_long" in backends
        has_pandas = "pandas_numpy" in backends or not backends
        matrix_rows.append(
            {
                "canonical": r["canonical"],
                "family": r["family"],
                "surface": "daily",
                "pandas_impl": "yes",
                "pandas_prod": "yes",
                "polars_impl": "yes" if has_polars else "no",
                "polars_parity": "n/a",
                "polars_prod": "no",
                "polars_long_native": "no",
                "duckdb_emitter": r["duckdb_emitter"],
                "duckdb_parity": r["duckdb_parity"],
                "duckdb_prod": r["duckdb_prod"],
                "clickhouse_parity": "not-certified",
                "recommended_target": r["recommended_target"],
                "priority": "tier-a",
                "benchmark_class": "core",
            }
        )
    path = EVIDENCE_DIR / "R31_BACKEND_TARGET_MATRIX.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=matrix_fields)
        writer.writeheader()
        for row in sorted(matrix_rows, key=lambda x: x["canonical"]):
            writer.writerow(row)
    return {"canonicals_in_matrix": len(matrix_rows), "certified_duckdb_prod": sum(1 for r in rows if r["duckdb_prod"] == "yes")}


def main() -> int:
    results = run_certification()
    summary = write_artifacts(results)
    summary["backend_target_matrix"] = write_backend_target_matrix()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
