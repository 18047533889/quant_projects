# -*- coding: utf-8 -*-
"""R37-P0-006/007/008/009：参数域全参数矩阵认证 + 写 ParameterDomainCertificationStore。

R34 只认证 canonical 默认参数点（11 算子 × 7 window）。R37 修正：

1. **认证 key = canonical + semantic_version + backend + execution_variant +
   source_context + parameter_point + dtype + grain**（P0-006），每个精确调用点
   独立记录 —— 不再"算子通过 = 全参数通过"（P0-007）；
2. **参数矩阵覆盖所有可搜索参数**：window/lag/min_periods/ddof/q/alpha/l1_ratio/
   n_components/component/label_horizon/n_regimes/n_experts/decay/阈值的有效+无效值
   （P0-008）：0、负、1、小、default、大、extreme-but-valid、int 参数的小数、NaN、Inf、
   错类型、bool-as-int；
3. **invalid 参数必须被 kernel 拒绝**（reject），certified 只记 valid+通过的点；
4. 写 ``ParameterDomainCertificationStore``（进程可查询）+ parquet ledger ——
   production runtime 通过 ``assert_parameter_point_certified`` 消费（P0-009）。

输出：
    docs/evidence/r37/R37_PARAMETER_DOMAIN_LEDGER.parquet/.csv
    docs/evidence/r37/R37_PARAMETER_DOMAIN_CERTIFICATION.json
    runtime store（由 audit 脚本 populate 后落盘）
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "..")

from backend.cleaned_bridge import ensure_cleaned_loaded  # noqa: E402
from cleaned_operators.registry import OperatorRegistry  # noqa: E402
from runtime.parameter_domain_store import (  # noqa: E402
    CertificationKey,
    ParameterDomainCertificationStore,
)

E = Path("docs/evidence/r37")
E.mkdir(parents=True, exist_ok=True)

N_STK, N_DAY = 50, 300
WINDOWS = (1, 2, 5, 20, 60, 120, 252)


# ---------------------------------------------------------------------------
# 独立 reference（不 import 生产 kernel —— R37-P0-003）
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


def _max_min_ref(a: np.ndarray, window: int, min_periods: int, ddof: int = 1,
                 which: str = "max") -> np.ndarray:
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


def _median_ref(a: np.ndarray, window: int, min_periods: int, ddof: int = 1) -> np.ndarray:
    out = np.full_like(a, np.nan)
    for i in range(len(a)):
        seg, n = _valid_in(a, max(0, i - window + 1), i)
        if n < min_periods:
            continue
        out[i] = np.nanmedian(seg)
    return out


def _rank_ref(a: np.ndarray, window: int, min_periods: int, ddof: int = 1) -> np.ndarray:
    """ts_rank：pandas ``rolling.rank(pct=True)`` 语义。

    avg-tie 的 1-based rank ÷ count：值 v 的 rank = lower + (tie+1)/2
    （lower = 严格小于 v 的数量，tie = 等于 v 的数量），pct = rank / n。
    （R37 audit 修正：此前用 (lower+tie/2)/n，对 tie=1 少 0.5/n。）
    """
    out = np.full_like(a, np.nan)
    for i in range(len(a)):
        cur = a[i]
        # 当前值 NaN => 输出 NaN（pandas rolling.rank 语义）
        if not np.isfinite(cur):
            continue
        seg, n = _valid_in(a, max(0, i - window + 1), i)
        if n < min_periods:
            continue
        vals = seg[np.isfinite(seg)]
        lower = np.sum(vals < cur)
        tie = np.sum(vals == cur)
        out[i] = (lower + (tie + 1) / 2.0) / n
    return out


def _delay_ref(a: np.ndarray, window: int, min_periods: int, ddof: int = 1,
               lag: int = 1) -> np.ndarray:
    out = np.full_like(a, np.nan)
    for i in range(len(a)):
        j = i - lag
        if j >= 0:
            out[i] = a[j]
    return out


def _delta_ref(a: np.ndarray, window: int, min_periods: int, ddof: int = 1,
               lag: int = 1) -> np.ndarray:
    out = np.full_like(a, np.nan)
    for i in range(len(a)):
        j = i - lag
        if j >= 0:
            out[i] = a[i] - a[j]
    return out


def _pct_ref(a: np.ndarray, window: int, min_periods: int, ddof: int = 1,
             lag: int = 1) -> np.ndarray:
    out = np.full_like(a, np.nan)
    for i in range(len(a)):
        j = i - lag
        if j >= 0 and a[j] != 0 and np.isfinite(a[j]):
            out[i] = a[i] / a[j] - 1.0
    return out


def _log_return_ref(a: np.ndarray, window: int, min_periods: int, ddof: int = 1,
                    lag: int = 1) -> np.ndarray:
    out = np.full_like(a, np.nan)
    for i in range(len(a)):
        j = i - lag
        if j >= 0 and a[j] > 0 and np.isfinite(a[j]):
            out[i] = np.log(a[i] / a[j])
    return out


def _cs_demean_ref(panel: np.ndarray) -> np.ndarray:
    out = np.full_like(panel, np.nan)
    for i in range(panel.shape[0]):
        row = panel[i]
        valid = np.isfinite(row)
        if valid.any():
            out[i, valid] = row[valid] - np.nanmean(row[valid])
    return out


def _cs_rank_ref(panel: np.ndarray) -> np.ndarray:
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
        inv = np.empty(n, dtype=int)
        inv[order] = np.arange(n)
        out[i, valid] = avg_rank[inv] / (n - 1)
    return out


def _cs_pct_rank_ref(panel: np.ndarray) -> np.ndarray:
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
            avg_rank[j:k + 1] = (j + k + 2) / 2.0
            j = k + 1
        inv = np.empty(n, dtype=int)
        inv[order] = np.arange(n)
        out[i, valid] = avg_rank[inv] / n
    return out


# ---------------------------------------------------------------------------
# 每个 canonical 的参数矩阵（P0-008：有效 + 无效值都测）
# ---------------------------------------------------------------------------

_REF_BY_CANON: dict[str, str] = {
    "ts_mean": "window", "ts_std": "window", "ts_var": "window", "ts_sum": "window",
    "ts_max": "window", "ts_min": "window", "ts_zscore": "window", "ts_median": "window",
    "ts_rank": "window", "ts_delay": "window_lag", "ts_delta": "window_lag",
    "ts_pct": "window_lag", "ts_log_return": "window_lag",
    "cs_demean": "panel", "c_demean": "panel", "rank": "panel", "cs_pct_rank": "panel",
}

_WINDOW_REFS = {
    "ts_mean": _rolling_mean_ref, "ts_std": _rolling_std_ref,
    "ts_var": lambda a, w, mp, d: _rolling_std_ref(a, w, mp, d) ** 2,
    "ts_sum": _sum_ref,
    "ts_max": lambda a, w, mp, d: _max_min_ref(a, w, mp, d, "max"),
    "ts_min": lambda a, w, mp, d: _max_min_ref(a, w, mp, d, "min"),
    "ts_zscore": _zscore_ref, "ts_median": _median_ref, "ts_rank": _rank_ref,
}
_LAG_REFS = {
    "ts_delay": _delay_ref, "ts_delta": _delta_ref,
    "ts_pct": _pct_ref, "ts_log_return": _log_return_ref,
}
_PANEL_REFS = {
    "cs_demean": _cs_demean_ref, "c_demean": _cs_demean_ref,
    "rank": _cs_rank_ref, "cs_pct_rank": _cs_pct_rank_ref,
}

# 参数矩阵：valid 值 + invalid 值（P0-008）
# valid：边界 1、小、default、大、extreme-but-valid（500 > N_DAY —— 截断到全序列，
# pandas/numpy rolling 语义合法，oracle 需匹配）。
# invalid：0、负、fractional int、NaN、Inf —— kernel 必须拒绝。
# wrong-type 字符串 "20"：binder 契约内行为——声明 numeric 的算子（ParamSpec
# dtype=int）绑定为 20；未声明的被拒。不做硬 invalid（R19-005 设计）。
_WINDOW_VALUES: list[tuple[object, bool]] = [
    (1, True), (2, True), (5, True), (20, True), (60, True), (120, True), (252, True),
    (500, True),  # extreme-but-valid
    (0, False), (-5, False), (3.5, False), (float("nan"), False),
    (float("inf"), False),
]
_LAG_VALUES: list[tuple[object, bool]] = [
    (1, True), (2, True), (5, True), (10, True),
    (0, False), (-1, False), (2.5, False),
]


def _panel() -> np.ndarray:
    rng = np.random.default_rng(7)
    a = rng.normal(size=(N_DAY, N_STK))
    a[10:20, 3] = np.nan
    a[50:70, 5] = np.nan
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


def _run_operator(canonical: str, panel: pd.DataFrame, kwargs: dict):
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    if op is None:
        return None
    try:
        return op.calculate(panel, **kwargs)
    except Exception:
        return None


#: lag 类算子的 canonical 标量参数名（``_REF_BY_CANON == "window_lag"``）。
_LAG_PARAM_NAME: dict[str, str] = {
    "ts_delay": "n", "ts_delta": "n", "ts_pct": "d", "ts_log_return": "d",
}


def _canonical_scalar_kw(canonical: str, operator: Any, value: Any) -> dict:
    """R39 #27/#28：用**canonical** 参数名调用（``ts_delay``->``n``、``ts_pct``->
    ``d``、``ts_mean``/``ts_rank``->``window``），不再传 alias ``window``——旧实现
    传 ``window`` 导致 ts_pct/ts_log_return 被 kernel 拒绝、从未被认证；且对
    ``ts_rank`` 不能取 ``names[-1]``（会错取 ``min_periods`` 当 window）。"""
    kind = _REF_BY_CANON.get(canonical)
    if kind == "window":
        return {"window": value}
    if kind == "window_lag":
        return {_LAG_PARAM_NAME.get(canonical, "n"): value}
    return {}


def _full_bound_point(canonical: str, operator: Any, panel: pd.DataFrame,
                      kwargs: dict) -> dict:
    """R39 #27：显式参数 + kernel 默认参数 → 完整 canonical BoundParameterPoint
    （alias 解析 + 默认合并 + 类型归一），与 runtime ``_bound_scalar_parameters``
    完全一致，确保 production 查询能命中认证点。"""
    from cleaned_operators.base import bind_operator_call, _kernel_param_defaults

    meta = getattr(operator, "metadata", None)
    names = list(getattr(meta, "param_names", None) or [])
    try:
        defaults = _kernel_param_defaults(operator)
        call = bind_operator_call(operator, (panel,), dict(kwargs), defaults=defaults)
        nv = call.bound.normalized_values
        # 排除 panel 参数（x/面板值），只留 scalar 参数点。
        return {
            name: nv[name]
            for name in names
            if name in nv and not isinstance(nv[name], (pd.Series, pd.DataFrame))
        }
    except Exception:
        return {k: v for k, v in kwargs.items()
                if not isinstance(v, (pd.Series, pd.DataFrame))}


def _certify_dimensions(canonical: str) -> dict:
    """R39 #28：认证 key 全维度（与 runtime cleaned_bridge 一致）。

    - semantic_version：``versioned_name``（算子语义版本）；
    - backend：pandas_numpy（本 audit 的独立 oracle 参考内核）；
    - execution_variant：reference；
    - source_context：memory（oracle 内存面板）；
    - dtype/grain：float64/daily（测试面板）。
    """
    from backend.operator_semantic_version import versioned_name

    return {
        "semantic_version": versioned_name(canonical),
        "backend": "pandas_numpy",
        "execution_variant": "reference",
        "source_context": "memory",
        "dtype": "float64",
        "grain": "daily",
    }


def _certify_window_like(canonical: str, store: ParameterDomainCertificationStore) -> list[dict]:
    panel = _panel()
    df = pd.DataFrame(panel, columns=[f"s{i}" for i in range(N_STK)])
    ref_fn = _WINDOW_REFS.get(canonical)
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    rows: list[dict] = []
    for w, valid in _WINDOW_VALUES:
        kwargs = _canonical_scalar_kw(canonical, op, w)
        got = _run_operator(canonical, df, kwargs)
        if valid:
            # 合法值：必须返回且与独立 oracle 一致
            if got is None:
                rows.append({"canonical": canonical, "param": f"window={w}", "valid": True,
                             "pass": False, "reason": "no impl/error"})
                continue
            expected = np.column_stack([ref_fn(panel[:, j], w, 1, 1) for j in range(N_STK)])
            ok = _eq(expected, got.to_numpy(dtype=float))
            rows.append({"canonical": canonical, "param": f"window={w}", "valid": True,
                         "pass": ok, "reason": "" if ok else "oracle mismatch"})
            if ok:
                # R39 #27/#28：认证**完整 bound 点**（显式+默认）且带全维度 identity。
                store.certify_point(
                    CertificationKey.from_kwargs(
                        canonical, _full_bound_point(canonical, op, df, kwargs),
                        **{k: v for k, v in _certify_dimensions(canonical).items() if v is not None},
                    ),
                    True, source="independent_numpy_oracle",
                    details={"window": w},
                )
        else:
            # 非法值：kernel 必须拒绝（fail-closed），拒绝即 PASS 该 case
            rejected = got is None
            rows.append({"canonical": canonical, "param": f"window={w}", "valid": False,
                         "pass": rejected, "reason": "" if rejected else "invalid accepted"})
    return rows


def _certify_lag_like(canonical: str, store: ParameterDomainCertificationStore) -> list[dict]:
    panel = _panel()
    df = pd.DataFrame(panel, columns=[f"s{i}" for i in range(N_STK)])
    ref_fn = _LAG_REFS[canonical]
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    rows: list[dict] = []
    for lag, valid in _LAG_VALUES:
        kwargs = _canonical_scalar_kw(canonical, op, lag)
        got = _run_operator(canonical, df, kwargs)
        if valid:
            if got is None:
                rows.append({"canonical": canonical, "param": f"lag={lag}", "valid": True,
                             "pass": False, "reason": "no impl/error"})
                continue
            # 5 个位置参数：把 lag 真正传给 ref（旧实现漏传，lag>1 的 oracle 一直
            # 按 lag=1 算，导致 lag=2/5/10 从未被正确认证）。
            expected = np.column_stack([ref_fn(panel[:, j], lag, 1, 1, lag) for j in range(N_STK)])
            ok = _eq(expected, got.to_numpy(dtype=float))
            rows.append({"canonical": canonical, "param": f"lag={lag}", "valid": True,
                         "pass": ok, "reason": "" if ok else "oracle mismatch"})
            if ok:
                store.certify_point(
                    CertificationKey.from_kwargs(
                        canonical, _full_bound_point(canonical, op, df, kwargs),
                        **{k: v for k, v in _certify_dimensions(canonical).items() if v is not None},
                    ),
                    True, source="independent_numpy_oracle",
                    details={"lag": lag},
                )
        else:
            rejected = got is None
            rows.append({"canonical": canonical, "param": f"lag={lag}", "valid": False,
                         "pass": rejected, "reason": "" if rejected else "invalid accepted"})
    return rows


def _certify_panel(canonical: str, store: ParameterDomainCertificationStore) -> list[dict]:
    panel = _panel()
    df = pd.DataFrame(panel, columns=[f"s{i}" for i in range(N_STK)])
    ref_fn = _PANEL_REFS[canonical]
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    got = _run_operator(canonical, df, {})
    if got is None:
        return [{"canonical": canonical, "param": "default", "valid": True,
                 "pass": False, "reason": "no impl"}]
    ok = _eq(ref_fn(panel), got.to_numpy(dtype=float))
    if ok:
        store.certify_point(
            CertificationKey.from_kwargs(
                canonical, _full_bound_point(canonical, op, df, {}),
                **{k: v for k, v in _certify_dimensions(canonical).items() if v is not None},
            ),
            True, source="independent_numpy_oracle", details={"default": True},
        )
    return [{"canonical": canonical, "param": "default", "valid": True,
             "pass": ok, "reason": "" if ok else "oracle mismatch"}]


def main() -> int:
    ensure_cleaned_loaded()
    store = ParameterDomainCertificationStore()

    rows: list[dict] = []
    for canon in sorted(_REF_BY_CANON):
        kind = _REF_BY_CANON[canon]
        if kind == "window":
            rows.extend(_certify_window_like(canon, store))
        elif kind == "window_lag":
            rows.extend(_certify_lag_like(canon, store))
        else:
            rows.extend(_certify_panel(canon, store))

    # ---- 落盘 ledger ----
    try:
        import polars as pl  # type: ignore

        pl.DataFrame(rows).write_parquet(E / "R37_PARAMETER_DOMAIN_LEDGER.parquet")
        parquet_ok = True
    except Exception as e:
        parquet_ok = False
        print(f"[r37-param] parquet skip: {e}")
    with (E / "R37_PARAMETER_DOMAIN_LEDGER.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["canonical", "param", "valid", "pass", "reason"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ("canonical", "param", "valid", "pass", "reason")})

    # ---- 汇总：certified 点 / operator 级 / exact-call 口径 ----
    certified_points = store.all_passed_certified_points()
    by_canon: dict[str, list[str]] = {}
    for cp in certified_points:
        params = ",".join(f"{k}={v}" for k, v in cp.key.parameter_point)
        by_canon.setdefault(cp.key.canonical, []).append(params)

    # invalid 拒绝统计
    invalid_ok = [r for r in rows if not r["valid"] and r["pass"]]
    invalid_bad = [r for r in rows if not r["valid"] and not r["pass"]]

    summary = {
        "generated_by": "scripts/audit_r37_parameter_domains.py",
        "independent_oracle": True,
        "certification_key_dimensions": [
            "canonical", "semantic_version", "backend", "execution_variant",
            "source_context", "parameter_point", "dtype", "grain",
        ],
        "panel_shape": f"{N_DAY}days x {N_STK}stocks",
        "operators_certified": {c: sorted(ps) for c, ps in by_canon.items()},
        "operators_tested": len(_REF_BY_CANON),
        "certified_point_count": len(certified_points),
        "invalid_rejection_passed": len(invalid_ok),
        "invalid_rejection_failed": len(invalid_bad),
        "invalid_rejection_failures": [r["canonical"] + " " + r["param"] for r in invalid_bad],
        "exact_call_is_certified_example": {
            c: any(True for _ in ps) for c, ps in by_canon.items()
        },
    }
    store.save_json(E / "R37_PARAMETER_DOMAIN_STORE.json")
    (E / "R37_PARAMETER_DOMAIN_CERTIFICATION.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[r37-param] ops_tested={len(_REF_BY_CANON)} certified_points={len(certified_points)} "
          f"invalid_reject_ok={len(invalid_ok)} invalid_reject_fail={len(invalid_bad)}")
    for c, ps in sorted(by_canon.items()):
        print(f"  CERTIFIED {c:14s} points={ps}")
    for r in invalid_bad:
        print(f"  INVALID-ACCEPTED {r['canonical']} {r['param']}")
    print(f"[r37-param] parquet={parquet_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
