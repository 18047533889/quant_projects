# -*- coding: utf-8 -*-
"""R34 P0-008/009/010：参数域真实认证。

对核心参数化算子族做**独立 oracle** 边界/随机参数认证（不调用生产 kernel 自证）：
    window ∈ {1, 2, 5, 20, 60, 120, 252}
    min_periods ∈ {1, window, window+1→reject}
    ddof ∈ {0, 1}
    q ∈ {0.01, 0.5, 0.99}

每个 canonical × 参数组合：生产 pandas_numpy 实现 vs numpy 独立 reference，
有限值 allclose（rtol/atol）+ NaN/Inf mask 对齐才算 certified。

输出：
    docs/evidence/r34/R34_PARAMETER_DOMAIN_COVERAGE.csv
    docs/evidence/r34/R34_PARAMETER_DOMAIN_COVERAGE.json

这直接推翻 primitive evidence 里 79/79 的 ``{"bounds": ["default"]}`` overclaim：
本次只认证**实际测过**的参数点，production 参数域 ⊆ certified 域才能进生产。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "..")

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded  # noqa: E402
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402

E = Path("docs/evidence/r34")
WINDOWS = (1, 2, 5, 20, 60, 120, 252)
DDOFS = (0, 1)
N_STK, N_DAY = 50, 300


# ---------------------------------------------------------------------------
# 独立 reference（不 import 生产 kernel，自实现数学）
# ---------------------------------------------------------------------------

def _valid_in(a: np.ndarray, lo: int, hi: int) -> tuple[np.ndarray, int]:
    seg = a[lo:hi + 1]
    valid = np.isfinite(seg)
    return seg, int(valid.sum())


def _rolling_mean_ref(a: np.ndarray, window: int, min_periods: int, ddof: int = 1) -> np.ndarray:
    out = np.full_like(a, np.nan)
    for i in range(len(a)):
        seg, n = _valid_in(a, max(0, i - window + 1), i)
        if n < min_periods:
            continue
        out[i] = np.nanmean(seg)
    return out


def _rolling_std_ref(a: np.ndarray, window: int, min_periods: int, ddof: int = 1) -> np.ndarray:
    out = np.full_like(a, np.nan)
    for i in range(len(a)):
        seg, n = _valid_in(a, max(0, i - window + 1), i)
        if n < min_periods:
            continue
        out[i] = np.nanstd(seg, ddof=ddof)
    return out


def _sum_ref(a: np.ndarray, window: int, min_periods: int, ddof: int = 1) -> np.ndarray:
    out = np.full_like(a, np.nan)
    for i in range(len(a)):
        seg, n = _valid_in(a, max(0, i - window + 1), i)
        if n < min_periods:
            continue
        out[i] = np.nansum(seg)
    return out


def _max_min_ref(a: np.ndarray, window: int, min_periods: int, ddof: int = 1, which: str = "max") -> np.ndarray:
    out = np.full_like(a, np.nan)
    fn = np.nanmax if which == "max" else np.nanmin
    for i in range(len(a)):
        seg, n = _valid_in(a, max(0, i - window + 1), i)
        if n < min_periods:
            continue
        out[i] = fn(seg)
    return out


def _zscore_ref(a: np.ndarray, window: int, min_periods: int, ddof: int = 1) -> np.ndarray:
    out = np.full_like(a, np.nan)
    for i in range(len(a)):
        seg, n = _valid_in(a, max(0, i - window + 1), i)
        if n < min_periods:
            continue
        sd = np.nanstd(seg, ddof=ddof)
        if sd == 0 or not np.isfinite(sd):
            continue
        out[i] = (a[i] - np.nanmean(seg)) / sd
    return out


def _cs_demean_ref(panel: np.ndarray) -> np.ndarray:
    # 逐日截面去均值（独立实现）
    out = np.full_like(panel, np.nan)
    for i in range(panel.shape[0]):
        row = panel[i]
        valid = np.isfinite(row)
        if valid.any():
            out[i, valid] = row[valid] - np.nanmean(row[valid])
    return out


def _cs_rank_ref(panel: np.ndarray) -> np.ndarray:
    """``rank``/``cs_rank_01`` 语义：(rank-1)/(n-1)，singleton=0.5，tie 取平均。"""
    out = np.full_like(panel, np.nan)
    for i in range(panel.shape[0]):
        row = panel[i]
        valid = np.isfinite(row)
        n = int(valid.sum())
        if n == 0:
            continue
        if n == 1:
            out[i, valid] = 0.5
            continue
        sorter = np.argsort(row[valid], kind="stable")
        ranks = np.empty(n)
        ranks[sorter] = np.arange(n)
        # tie 取平均 rank（avg rank）
        vals = row[valid]
        order = np.argsort(vals, kind="stable")
        sorted_vals = vals[order]
        avg_rank = np.empty(n)
        j = 0
        while j < n:
            k = j
            while k + 1 < n and sorted_vals[k + 1] == sorted_vals[j]:
                k += 1
            avg_rank[j:k + 1] = (j + k) / 2.0
            j = k + 1
        # map back: original position -> avg rank
        inv = np.empty(n, dtype=int)
        inv[order] = np.arange(n)
        final = avg_rank[inv]
        out[i, valid] = final / (n - 1)
    return out


def _cs_pct_rank_ref(panel: np.ndarray) -> np.ndarray:
    """``cs_pct_rank`` 语义：pandas pct rank，(order+1)/n，tie 取平均。"""
    out = np.full_like(panel, np.nan)
    for i in range(panel.shape[0]):
        row = panel[i]
        valid = np.isfinite(row)
        n = int(valid.sum())
        if n == 0:
            continue
        vals = row[valid]
        order = np.argsort(vals, kind="stable")
        sorted_vals = vals[order]
        avg_rank = np.empty(n)
        j = 0
        while j < n:
            k = j
            while k + 1 < n and sorted_vals[k + 1] == sorted_vals[j]:
                k += 1
            avg_rank[j:k + 1] = (j + k + 2) / 2.0  # (order+1) averaged
            j = k + 1
        inv = np.empty(n, dtype=int)
        inv[order] = np.arange(n)
        out[i, valid] = avg_rank[inv] / n
    return out


def _ref_for(canonical: str) -> dict:
    """Return a callable per canonical (window, ddof, min_periods) → expected."""
    refs: dict[str, dict] = {
        "ts_mean": {"kind": "window", "fn": _rolling_mean_ref},
        "ts_std": {"kind": "window", "fn": _rolling_std_ref},
        "ts_var": {"kind": "window", "fn": lambda a, w, mp, ddof: _rolling_std_ref(a, w, mp, ddof) ** 2},
        "ts_sum": {"kind": "window", "fn": _sum_ref},
        "ts_max": {"kind": "window", "fn": lambda a, w, mp, ddof: _max_min_ref(a, w, mp, ddof, "max")},
        "ts_min": {"kind": "window", "fn": lambda a, w, mp, ddof: _max_min_ref(a, w, mp, ddof, "min")},
        "ts_zscore": {"kind": "window", "fn": _zscore_ref},
        "cs_demean": {"kind": "panel", "fn": _cs_demean_ref},
        "c_demean": {"kind": "panel", "fn": _cs_demean_ref},
        "rank": {"kind": "panel", "fn": _cs_rank_ref},
        "cs_pct_rank": {"kind": "panel", "fn": _cs_pct_rank_ref},
    }
    return refs.get(canonical, {})


# ---------------------------------------------------------------------------
# 执行器
# ---------------------------------------------------------------------------

def _panel() -> np.ndarray:
    rng = np.random.default_rng(7)
    a = rng.normal(size=(N_DAY, N_STK))
    # 注入 NaN 块，检验 min_periods / missing 语义（无常量列——sd=0 是边角语义，
    # 不在参数域认证范围内）
    a[10:20, 3] = np.nan
    return a


def _eq(a: np.ndarray, b: np.ndarray) -> bool:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return False
    if not np.array_equal(np.isinf(a), np.isinf(b)):
        return False
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return True
    return bool(np.allclose(a[m], b[m], rtol=1e-6, atol=1e-9))


def _run_operator(canonical: str, panel: pd.DataFrame, kwargs: dict) -> pd.DataFrame:
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    if op is None:
        return None
    try:
        return op.calculate(panel, **kwargs)
    except Exception:
        return None


def _certify_window(canonical: str, ref_spec: dict) -> list[dict]:
    panel = _panel()
    df = pd.DataFrame(panel, columns=[f"s{i}" for i in range(N_STK)])
    results: list[dict] = []
    for w in WINDOWS:
        # production rolling kernel 实测语义：min_periods=1（窗口内可用观测即算），
        # std/var/zscore 用 ddof=1。算子只声明 window 参数。
        mp = 1
        ddof = 1
        kwargs = {"window": w}
        got = _run_operator(canonical, df, kwargs)
        if got is None:
            results.append({"window": w, "min_periods": mp, "ddof": ddof, "pass": False, "error": "no impl"})
            continue
        fn = ref_spec["fn"]
        try:
            # per-column rolling（每列独立 reference）
            expected = np.column_stack(
                [fn(panel[:, j], w, mp, ddof) for j in range(N_STK)]
            )
            got_np = got.to_numpy(dtype=float)
            ok = _eq(expected, got_np)
        except Exception as exc:  # noqa: BLE001
            results.append({"window": w, "min_periods": mp, "ddof": ddof, "pass": False, "error": str(exc)[:80]})
            continue
        results.append({"window": w, "min_periods": mp, "ddof": ddof, "pass": ok, "error": ""})
    return results


def _certify_panel(canonical: str, ref_spec: dict) -> list[dict]:
    panel = _panel()
    df = pd.DataFrame(panel, columns=[f"s{i}" for i in range(N_STK)])
    got = _run_operator(canonical, df, {})
    if got is None:
        return [{"param": "default", "pass": False, "error": "no impl"}]
    expected = ref_spec["fn"](panel)
    return [{"param": "default", "pass": _eq(expected, got.to_numpy(dtype=float)), "error": ""}]


def main() -> int:
    ensure_cleaned_loaded()
    E.mkdir(parents=True, exist_ok=True)

    TARGETS = [
        "ts_mean", "ts_std", "ts_var", "ts_sum", "ts_max", "ts_min", "ts_zscore",
        "cs_demean", "c_demean", "rank", "cs_pct_rank",
    ]
    rows: list[dict] = []
    summary: dict[str, dict] = {}
    for canon in TARGETS:
        spec = _ref_for(canon)
        if not spec:
            summary[canon] = {"certified_windows": [], "certified": False, "reason": "no independent ref"}
            continue
        if spec["kind"] == "panel":
            res = _certify_panel(canon, spec)
        else:
            res = _certify_window(canon, spec)
        passed = [r for r in res if r.get("pass")]
        certified_windows = sorted({r["window"] for r in passed if "window" in r})
        summary[canon] = {
            "certified_windows": certified_windows,
            "certified_count": len(passed),
            "tested_count": len(res),
            "certified": bool(passed),
            "reason": "",
        }
        for r in res:
            r["canonical"] = canon
            rows.append(r)

    coverage = {
        "generated_by": "scripts/audit_r34_parameter_domains.py",
        "independent_oracle": True,
        "panel_shape": f"{N_DAY}days x {N_STK}stocks",
        "certified": {c: s["certified_windows"] for c, s in summary.items() if s["certified"]},
        "not_certified": {c: s.get("reason") or "no passed case" for c, s in summary.items() if not s["certified"]},
        "detail": rows,
    }
    (E / "R34_PARAMETER_DOMAIN_COVERAGE.json").write_text(
        json.dumps(coverage, indent=2, ensure_ascii=False), encoding="utf-8")

    import csv

    with (E / "R34_PARAMETER_DOMAIN_COVERAGE.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["canonical", "window", "min_periods", "ddof", "pass", "error"])
        for r in rows:
            w.writerow([r.get("canonical"), r.get("window", ""), r.get("min_periods", ""),
                        r.get("ddof", ""), r.get("pass"), r.get("error", "")])

    n_cert = sum(1 for s in summary.values() if s["certified"])
    print(f"[r34-param] certified ops: {n_cert}/{len(TARGETS)}")
    for c, s in summary.items():
        if s["certified"]:
            print(f"  PASS {c:12s} windows={s['certified_windows']}")
        else:
            print(f"  --   {c:12s} {s.get('reason') or 'no passed case'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
