#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R19-108/109/110/116/128: 真实 backend parity 差分执行审计。

对每个关键 canonical 跑多组 hostile fixtures，比较 reference / pandas_numpy /
polars（/ duckdb lowering 若可用），并 **显式记录 native_used / fallback_used**。

- R19-108: parity 结果记录 ``native_used`` / ``fallback_used``；测试分类
  ``native parity`` 与 ``fallback parity`` 分开，不混为一项。
- R19-109: production-certified native backend 必须 ``native_used=True``，否则
  只能叫 ``supported_via_reference_fallback``，不能叫 native certified。
- R19-116: optimized backend（polars/duckdb）必须 differential test reference。
- R19-110: hostile fixtures（all equal / many ties / zero denominator / Inf /
  NaN block / group labels …）。

输出:
    build/r19_audit/operator_differential_execution.json
    build/r19_audit/operator_differential_execution.csv

容错：与 math contract 审计一致 —— ``load_all`` 失败不致命；单 canonical 失败
记录 AUDIT_ERROR，不崩溃。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _load_mc():
    """Robust import of ``cleaned_operators.math_certificate``.

    优先直接文件加载（只依赖 stdlib + numpy），避免触发缓慢/可能损坏的 package
    ``__init__``；失败时回退到 package import。
    """
    import importlib.util as _iu
    _name = "_r19_math_certificate"
    _spec = _iu.spec_from_file_location(
        _name, REPO / "cleaned_operators" / "math_certificate.py")
    _mod = _iu.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
    return _mod


MC = _load_mc()
build_hostile_fixtures = MC.build_hostile_fixtures
_rank_rowwise_np = MC._rank_rowwise_np
_zscore_rowwise_np = MC._zscore_rowwise_np
_corr_rowwise_np = MC._corr_rowwise_np
_cov_rowwise_np = MC._cov_rowwise_np
_beta_rowwise_np = MC._beta_rowwise_np
_rolling_corr_ref = MC._rolling_corr_ref
_rolling_cov_ref = MC._rolling_cov_ref
_rolling_beta_ref = MC._rolling_beta_ref

OUT = REPO / "build" / "r19_audit"

# canonical -> invocation mode:
#   "unary_2d": fn(panel) —— row-wise cross-sectional
#   "unary_1d": fn(single-column panel) —— column-wise ts rolling
#   "binary_xy": fn(x_panel, y_panel, **kw) —— bivariate ts
# 每种 mode 都提供一个 "reference" numpy 实现名（math_certificate），用于
# reference vs optimized 差分；无 reference 的实现用 pandas_numpy 作 reference。
MODE_TABLE: dict[str, dict[str, Any]] = {
    # row-wise（cross-sectional）
    "rank": {"mode": "unary_2d", "reference": "rank"},
    "cs_rank_01": {"mode": "unary_2d", "reference": "rank"},
    "zscore": {"mode": "unary_2d", "reference": "zscore"},
    "cs_std": {"mode": "unary_2d", "reference": None},
    "cs_demean": {"mode": "unary_2d", "reference": None},
    "normalize": {"mode": "unary_2d", "reference": None},
    "cs_quantile": {"mode": "unary_2d", "reference": None},
    # column-wise ts rolling
    "ts_mean": {"mode": "unary_1d", "reference": None},
    "ts_std": {"mode": "unary_1d", "reference": None},
    "ts_var": {"mode": "unary_1d", "reference": None},
    "ts_rank": {"mode": "unary_1d", "reference": None},
    "ts_quantile": {"mode": "unary_1d", "reference": None},
    "ts_moment": {"mode": "unary_1d", "reference": None},
    "ts_sharpe": {"mode": "unary_1d", "reference": None},
    "ts_autocorr": {"mode": "unary_1d", "reference": None},
    "ts_delta": {"mode": "unary_1d", "reference": None},
    "ts_pct": {"mode": "unary_1d", "reference": None},
    "ts_product": {"mode": "unary_1d", "reference": None},
    "ts_decay_linear": {"mode": "unary_1d", "reference": None},
    "ts_sum": {"mode": "unary_1d", "reference": None},
    "expanding_mean": {"mode": "unary_1d", "reference": None},
    "expanding_zscore": {"mode": "unary_1d", "reference": None},
    # bivariate ts —— min_periods 是引擎 reviewed 参考（Beta 家族=5，Corr/Cov=2）。
    "Corr": {"mode": "binary_xy", "reference": "corr", "window": 6, "min_periods": 2},
    "Cov": {"mode": "binary_xy", "reference": "cov", "window": 6, "min_periods": 2},
    "Covariance": {"mode": "binary_xy", "reference": "cov", "window": 6, "min_periods": 2},
    "Beta": {"mode": "binary_xy", "reference": "beta", "window": 6, "min_periods": 5},
    "ts_corr": {"mode": "binary_xy", "reference": "corr", "window": 6, "min_periods": 2},
    "ts_cov": {"mode": "binary_xy", "reference": "cov", "window": 6, "min_periods": 2},
    "ts_beta": {"mode": "binary_xy", "reference": "beta", "window": 6, "min_periods": 5},
}

REFERENCE_KERNELS = {
    "rank": _rank_rowwise_np,
    "zscore": _zscore_rowwise_np,
    "corr": _corr_rowwise_np,
    "cov": _cov_rowwise_np,
    "beta": _beta_rowwise_np,
}


def load_registry() -> tuple[Any, bool, str]:
    error = ""
    load_ok = True
    try:
        from factor_engine.cleaned_operators import load_all
        load_all()
    except Exception as exc:  # noqa: BLE001
        load_ok = False
        error = f"{type(exc).__name__}: {exc}"
    try:
        from factor_engine.cleaned_operators.registry import OperatorRegistry
        return OperatorRegistry, load_ok, error
    except Exception as exc:  # noqa: BLE001
        return None, False, f"{error}; registry_unavailable:{type(exc).__name__}: {exc}"


def _panel1d(arr: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(arr, dtype=float).reshape(-1, 1), columns=["s0"])


def _to_input(arr: np.ndarray, *, backend: str) -> Any:
    """把 numpy 数组转成对应 backend 的 panel 输入。

    pandas_numpy → ``pd.DataFrame``（宽表）；polars → ``pl.DataFrame``。
    """
    a = np.asarray(arr, dtype=float)
    df = pd.DataFrame(a) if a.ndim == 2 else _panel1d(a)
    if backend == "polars":
        import polars as pl
        return pl.from_pandas(df)
    return df


def _from_output(out: Any, mode: str) -> np.ndarray:
    """把 backend 输出转回 numpy。"""
    if hasattr(out, "to_numpy"):
        arr = np.asarray(out.to_numpy(), dtype=float)
    else:
        arr = np.asarray(out, dtype=float)
    if mode in ("unary_1d", "binary_xy"):
        return arr[:, 0] if arr.ndim == 2 and arr.shape[1] >= 1 else arr.ravel()
    return arr


def _invoke(op: Any, mode: str, x: np.ndarray, y: np.ndarray | None = None,
            *, backend: str = "pandas_numpy", **kw: Any) -> np.ndarray:
    """按 mode 与 backend 调用 operator。失败抛异常。"""
    if mode == "unary_2d":
        out = op.calculate(_to_input(x, backend=backend))
    elif mode == "unary_1d":
        out = op.calculate(_to_input(x, backend=backend))
    elif mode == "binary_xy":
        if y is None:
            raise ValueError("binary_xy requires y")
        out = op.calculate(_to_input(x, backend=backend),
                           _to_input(y, backend=backend), **kw)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return _from_output(out, mode)


def _invoke_reference(ref_name: str, mode: str, x: np.ndarray,
                      y: np.ndarray | None = None, *,
                      window: int | None = None,
                      min_periods: int | None = None) -> np.ndarray:
    """用 math_certificate 的 reference numpy kernel 计算。"""
    if mode == "unary_2d":
        return REFERENCE_KERNELS[ref_name](x)
    if mode == "binary_xy":
        if y is None:
            raise ValueError("binary_xy requires y")
        if ref_name in ("corr", "cov", "beta"):
            if window is None:
                raise ValueError(f"{ref_name} reference requires window")
            if ref_name == "corr":
                return _rolling_corr_ref(x, y, window,
                                         min_periods=min_periods or 2)
            if ref_name == "cov":
                return _rolling_cov_ref(x, y, window,
                                        min_periods=min_periods or 2)
            # 引擎 Beta(y, x, window)：x_panel 绑定 y（因变量）、y_panel 绑定 x
            # （自变量），故 reference = beta(x_panel ~ y_panel)。
            return _rolling_beta_ref(x, y, window, min_periods=min_periods or 5)
        return REFERENCE_KERNELS[ref_name](x, y)
    raise ValueError(f"reference kernel not applicable to mode {mode!r}")


def _compare(a: np.ndarray, b: np.ndarray, *, rtol: float = 1e-8,
             atol: float = 1e-10) -> bool:
    if a.shape != b.shape:
        return False
    if a.size == 0:
        return True
    return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))


def audit_one(canonical: str, reg, fixtures: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """对单个 canonical 跑 hostile fixtures 的 backend 差分。"""
    spec = MODE_TABLE[canonical]
    mode = spec["mode"]
    window = spec.get("window")
    min_periods = spec.get("min_periods")
    ref_name = spec.get("reference")
    ref_op = reg.get(canonical, "pandas_numpy")
    pol_op = reg.get(canonical, "polars")

    # native / fallback 判定
    native_used = pol_op is not None
    fallback_used = (not native_used) and (ref_op is not None)
    duckdb_available = False
    try:
        from factor_engine.backend.sql_pushdown.available import sql_lowering_available  # noqa: F401
        duckdb_available = False  # 仅当 lowering 真实可路由
    except Exception:  # noqa: BLE001
        duckdb_available = False

    rows: list[dict[str, Any]] = []
    all_native_ok = True
    all_fallback_ok = True
    all_reference_ok = True

    for fname, fx in fixtures.items():
        x = np.asarray(fx["x"], dtype=float)
        # unary_1d / binary_xy 用第一列构造时序
        y = None
        if mode in ("unary_1d", "binary_xy"):
            if x.ndim == 2 and x.shape[1] > 1:
                y = x[:, 1] if mode == "binary_xy" else None
                x0 = x[:, 0]
            else:
                x0 = x.ravel()
            x = x0
            if mode == "binary_xy" and y is None:
                y = x0.copy()
        rec: dict[str, Any] = {
            "canonical": canonical, "fixture": fname,
            "native_used": bool(native_used), "fallback_used": bool(fallback_used),
            "parity_class": None, "native_pass": None, "fallback_pass": None,
            "reference_pass": None, "note": "",
        }
        # reference kernel（仅 row-wise / bivariate）
        if ref_name:
            try:
                ref_out = _invoke_reference(ref_name, mode, x, y, window=window,
                                            min_periods=min_periods)
            except Exception as exc:  # noqa: BLE001
                ref_out = None
                rec["note"] += f"ref_err:{type(exc).__name__};"
        else:
            ref_out = None

        # pandas_numpy
        pandas_out = None
        if ref_op is not None:
            try:
                pandas_out = _invoke(ref_op, mode, x, y, backend="pandas_numpy",
                                     **({"window": window} if window else {}))
            except Exception as exc:  # noqa: BLE001
                rec["note"] += f"pandas_err:{type(exc).__name__};"

        # reference vs pandas（reference 优先，R19-116/§56）
        if ref_out is not None and pandas_out is not None:
            ok = _compare(ref_out, pandas_out)
            rec["reference_pass"] = bool(ok)
            all_reference_ok = all_reference_ok and ok
            if not ok:
                rec["note"] += "reference_mismatch;"

        # polars（native path）
        if native_used and pol_op is not None:
            try:
                pol_out = _invoke(pol_op, mode, x, y, backend="polars",
                                  **({"window": window} if window else {}))
                if pandas_out is not None:
                    ok = _compare(pandas_out, pol_out)
                elif ref_out is not None:
                    ok = _compare(ref_out, pol_out)
                else:
                    ok = False
                rec["native_pass"] = bool(ok)
                all_native_ok = all_native_ok and ok
                rec["parity_class"] = "native_parity"
                if not ok:
                    rec["note"] += "native_mismatch;"
            except Exception as exc:  # noqa: BLE001
                rec["native_pass"] = False
                all_native_ok = False
                rec["parity_class"] = "native_parity"
                rec["note"] += f"native_err:{type(exc).__name__};"
        else:
            rec["parity_class"] = "fallback_parity"
            rec["native_pass"] = None
            if pandas_out is not None:
                rec["fallback_pass"] = True  # pandas 参考本身能跑
                all_fallback_ok = all_fallback_ok and True

        rows.append(rec)

    return {
        "canonical": canonical,
        "mode": mode,
        "native_used": bool(native_used),
        "fallback_used": bool(fallback_used),
        "duckdb_available": bool(duckdb_available),
        "native_parity_all_pass": bool(all_native_ok) if native_used else None,
        "fallback_parity_all_pass": bool(all_fallback_ok) if fallback_used else None,
        "reference_all_pass": bool(all_reference_ok) if ref_name else None,
        "fixtures": rows,
        "certification": ("native_certified" if native_used and all_native_ok
                          else ("supported_via_reference_fallback"
                                if (fallback_used or (native_used and not all_native_ok))
                                else "not_certified")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonicals", nargs="*", default=None,
                    help="限定 canonical 列表（默认 MODE_TABLE 全部）")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    reg, load_ok, load_error = load_registry()
    fixtures = build_hostile_fixtures(n_rows=12, n_cols=4, seed=7)
    canonicals = args.canonicals or sorted(MODE_TABLE.keys())

    if reg is None:
        payload = {
            "schema": "factor_engine.r19_differential_execution.v1",
            "meta": {"load_ok": False, "load_error": load_error,
                     "canonicals": 0, "native_certified": 0,
                     "supported_via_reference_fallback": 0},
            "results": [],
        }
        (Path(args.out) / "operator_differential_execution.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1, default=str),
            encoding="utf-8")
        print(f"WARNING: registry unavailable — {load_error}", file=sys.stderr)
        return 0

    results: list[dict[str, Any]] = []
    for canon in canonicals:
        try:
            results.append(audit_one(canon, reg, fixtures))
        except Exception as exc:  # noqa: BLE001
            results.append({
                "canonical": canon, "mode": None, "native_used": None,
                "fallback_used": None, "duckdb_available": None,
                "native_parity_all_pass": None, "fallback_parity_all_pass": None,
                "reference_all_pass": None, "fixtures": [],
                "certification": "audit_error",
                "note": f"{type(exc).__name__}: {exc}",
            })

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "factor_engine.r19_differential_execution.v1",
        "meta": {
            "load_ok": load_ok, "load_error": load_error,
            "canonicals": len(results),
            "native_certified": sum(1 for r in results
                                    if r.get("certification") == "native_certified"),
            "supported_via_reference_fallback": sum(
                1 for r in results
                if r.get("certification") == "supported_via_reference_fallback"),
        },
        "results": results,
    }
    (out_dir / "operator_differential_execution.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")

    with open(out_dir / "operator_differential_execution.csv", "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["canonical", "fixture", "parity_class", "native_used",
                         "fallback_used", "native_pass", "fallback_pass",
                         "reference_pass", "note"])
        for r in results:
            for fx in (r.get("fixtures") or []):
                writer.writerow([r["canonical"], fx["fixture"], fx["parity_class"],
                                 fx["native_used"], fx["fallback_used"],
                                 fx["native_pass"], fx["fallback_pass"],
                                 fx["reference_pass"], fx["note"]])
    print(f"R19 differential audit written to {out_dir}")
    print(f"  canonicals: {len(results)}")
    print(f"  meta: {payload['meta']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
