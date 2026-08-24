#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R19-123..127 + R19-129: 自动机器审计 —— 数学契约 / hidden kwargs /
local casts / sample masking drift / duplicated semantic policy。

对每个 canonical 生成 ``MathematicalSemanticCertificate``（R19-129 全部字段），
并映射到 M01..M20 release-blocker。输出
``factor_engine/docs/R19_OPERATOR_MATH_AUDIT.{json,csv,md}``。

容错约定：
- ``load_all()`` 失败（并发 agent 正在改 operator）不致命 —— 记录
  ``load_all_error`` 到 meta，继续用已注册（部分）registry；
- 单个 canonical 审计失败记录为 AUDIT_ERROR 行，绝不导致整脚本崩溃；
- 动态检查（reference smoke / prefix / chunk / metamorphic）只对可构造、可快速
  运行的 canonical 执行；其它 canonical 的相应字段为 ``None``（N/A with reason
  进 ``math_defect`` 或 JSON 的 ``not_run``）。

用法:
    python scripts/audit_operator_math_contract.py [--limit N] [--all] [--out docs]
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

    优先直接文件加载（模块只依赖 stdlib + numpy，避免触发缓慢/可能损坏的
    ``cleaned_operators.__init__``）；失败时回退到 package import。
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
MBlocker = MC.MBlocker
MathematicalSemanticCertificate = MC.MathematicalSemanticCertificate
build_hostile_fixtures = MC.build_hostile_fixtures
chunk_invariance_check = MC.chunk_invariance_check
math_semantics_version = MC.math_semantics_version
prefix_invariance_check = MC.prefix_invariance_check
scan_hidden_kwargs = MC.scan_hidden_kwargs
scan_local_casts = MC.scan_local_casts
scan_sample_masking = MC.scan_sample_masking
scan_duplicated_semantic_policy = MC.scan_duplicated_semantic_policy
ts_moment_stable_ = MC.ts_moment_stable_
ts_poly2_coeff_centered_ = MC.ts_poly2_coeff_centered_

# 动态检查（跑真实算子）只对这些 key statistical / rolling / event 家族执行。
DYNAMIC_AUDIT_CORE = frozenset({
    "rank", "cs_rank_01", "zscore", "group_zscore", "cs_std", "cs_demean",
    "normalize", "Corr", "Cov", "Covariance", "Beta", "Intercept", "Slope",
    "R2", "Residual", "Kurt", "Skew", "Median", "Mode", "Var", "ts_mean",
    "ts_std", "ts_var", "ts_zscore", "ts_rank", "ts_corr", "ts_cov", "ts_beta",
    "ts_moment", "ts_poly2_coeff", "ts_poly2_resid", "ts_regression_slope",
    "ts_regression_intercept", "rank_corr", "group_rank", "group_mean",
    "group_std", "neutralize", "group_neutralize", "ts_quantile", "ts_sharpe",
    "ts_autocorr", "ACF", "ts_delta", "ts_pct", "ts_argmax", "ts_argmin",
    "ts_product", "ts_decay_linear", "ts_sum", "expanding_mean",
    "expanding_zscore", "winsorize", "group_winsorize", "cs_quantile",
})


# ---------------------------------------------------------------------------
# registry loader（容错）
# ---------------------------------------------------------------------------


def load_registry() -> tuple[Any, bool, str]:
    """容错加载 registry。返回 (OperatorRegistry|None, load_ok, error_str)。"""
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


# ---------------------------------------------------------------------------
# 元数据/静态采集
# ---------------------------------------------------------------------------


def _catalog_entry(reg, canonical: str) -> dict[str, Any]:
    return reg._catalog.get(canonical, {})


def _semantics_for(canonical: str) -> Any:
    try:
        from factor_engine.backend.numeric_semantics import semantics_for
        return semantics_for(canonical)
    except Exception:  # noqa: BLE001
        return None


def _history_contract(canonical: str, op: Any) -> dict[str, Any]:
    """从 param_specs 提取 history kind/formula/anchor 声明。"""
    meta = getattr(op, "metadata", None)
    specs = dict(getattr(meta, "param_specs", None) or {})
    kinds = [str(getattr(s, "history_semantics", None)) for s in specs.values()
             if getattr(s, "history_semantics", None)]
    formulas = [str(getattr(s, "history_formula", None)) for s in specs.values()
                if getattr(s, "history_formula", None)]
    anchor = "trailing_end"
    for s in specs.values():
        hs = getattr(s, "history_semantics", None)
        if hs in ("expanding", "cumulative"):
            anchor = str(hs)
            break
    return {
        "history_kind": ";".join(sorted(set(kinds))) if kinds else None,
        "history_formula": ";".join(sorted(set(formulas))) if formulas else None,
        "anchor_policy": anchor,
    }


def _declared_param_names(op: Any) -> list[str]:
    meta = getattr(op, "metadata", None)
    names = list(getattr(meta, "param_names", None) or [])
    aliases = dict(getattr(meta, "param_aliases", None) or {})
    names += list(aliases.keys()) + list(aliases.values())
    return list(dict.fromkeys(names))


def _source_of(op: Any) -> tuple[str, str]:
    """返回 (source_text, provenance)；失败时 source_text 为空。"""
    fn = getattr(op, "_calculate_series", None) or getattr(op, "calculate", None)
    if fn is None:
        return "", "no kernel"
    try:
        import inspect
        return inspect.getsource(fn), "inspect"
    except (OSError, TypeError):  # noqa: BLE001
        return "", "unavailable"


def _axis_semantics(canonical: str, reg) -> str:
    try:
        from factor_engine.cleaned_operators.operator_surface import classify_canonical
        surface = classify_canonical(canonical)
    except Exception:  # noqa: BLE001
        surface = None
    if surface in ("daily", "extended"):
        if canonical.startswith(("cs_", "c_")) or canonical in (
                "rank", "zscore", "normalize", "neutralize", "group_neutralize",
                "group_rank", "group_zscore", "winsorize", "group_winsorize"):
            return "row_wise"
        if canonical.startswith(("ts_", "expanding_")) or canonical in (
                "Corr", "Cov", "Covariance", "Beta", "ACF", "Kurt", "Skew",
                "Slope", "Intercept", "R2", "Residual", "Median", "Mode", "Var"):
            return "column_wise"
    return "panel"


def _minimum_effective_sample(canonical: str, op: Any, reg) -> int | None:
    meta = getattr(op, "metadata", None)
    # 已知 floor 表（从 operator_policy / hardening 的 min_periods 声明归纳）
    known = {
        "Corr": 2, "Cov": 2, "Covariance": 2, "Beta": 5, "Intercept": 3,
        "Slope": 3, "R2": 3, "Residual": 3, "ACF": 2, "ts_beta": 5,
        "ts_corr": 5, "ts_cov": 5, "ts_regression_slope": 3,
        "ts_regression_intercept": 3, "ts_sharpe": 5, "ts_autocorr": 5,
    }
    if canonical in known:
        return known[canonical]
    return None


# ---------------------------------------------------------------------------
# 动态检查（真实算子，best-effort）
# ---------------------------------------------------------------------------


def _panel1d(arr: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(arr, dtype=float).reshape(-1, 1),
                        columns=["s0"])


def _op_fn_1d(op: Any):
    """把 operator 包成一维 ndarray -> ndarray（单列 panel）。"""
    def fn(arr: np.ndarray) -> np.ndarray:
        out = op.calculate(_panel1d(arr))
        return out.to_numpy(dtype=float)[:, 0]
    return fn


def _op_fn_2d(op: Any):
    """把 operator 包成 2D ndarray -> 2D ndarray（row=date, col=instrument）。"""
    def fn(arr: np.ndarray) -> np.ndarray:
        out = op.calculate(pd.DataFrame(np.asarray(arr, dtype=float)))
        return out.to_numpy(dtype=float)
    return fn


# 已知 panel 输入名（group op 的第二 panel 常有 default，签名推断会漏掉）
_PANEL_INPUT_NAMES = frozenset({
    "x", "y", "group", "condition", "market", "size", "industry", "benchmark",
    "factor", "close", "open", "high", "low", "volume", "amount", "returns",
})


def _panel_arity(op: Any) -> int:
    """推断 operator 需要的 panel 输入数（避免对 bivariate/group op 跑 unary 冒烟
    产生 M01 假阳性）。"""
    meta = getattr(op, "metadata", None)
    if meta is not None:
        pa = getattr(meta, "panel_arity", None)
        if pa:
            return int(pa)
        pp = getattr(meta, "panel_params", None)
        if pp:
            return len(pp)
        sp = getattr(meta, "scalar_params", None)
        pnames = getattr(meta, "param_names", None)
        if sp and pnames is not None:
            n_scalar = len(sp)
            n_panel = len(pnames) - n_scalar
            if n_panel >= 1:
                return n_panel
        ifld = getattr(meta, "input_fields", None)
        if ifld:
            return len(ifld)
    import inspect
    fn = getattr(op, "_calculate_series", None)
    if fn is not None:
        try:
            sig = inspect.signature(fn)
            n = 0
            for p in sig.parameters.values():
                if p.name in ("self", "kwargs"):
                    continue
                if p.default is inspect.Parameter.empty or p.name in _PANEL_INPUT_NAMES:
                    n += 1
            return n
        except (TypeError, ValueError):  # noqa: BLE001
            return 1
    return 1


def _run_dynamic(op: Any, canonical: str, reg) -> dict[str, Any]:
    """跑 reference smoke / prefix / chunk / column permutation / metamorphic。

    全部 best-effort：任何异常 → 该字段 False + detail。不适用 → None。
    需要 2 个 panel 输入（bivariate / group）的 op 不跑 unary 冒烟 —— 记
    ``math_defect=["requires_2_panels"]``（N/A，不当作 M01 数学参考不匹配）。
    """
    out: dict[str, Any] = {
        "reference_smoke_pass": None,
        "prefix_invariance": None,
        "chunk_invariance": None,
        "column_permutation": None,
        "metamorphic_passed": None,
        "metamorphic_properties": (),
        "math_defect": [],
        "semantic_drift": [],
    }
    if op is None:
        out["math_defect"].append("no_runtime_operator")
        return out
    if _panel_arity(op) >= 2:
        out["math_defect"].append("requires_2_panels")
        return out
    rng = np.random.default_rng(7)
    base = rng.standard_normal((24, 3))
    base[2, 1] = np.nan
    # ---- reference smoke（2D 常规 fixture）----
    try:
        smoke = _op_fn_2d(op)(base)
        out["reference_smoke_pass"] = bool(
            smoke.shape == base.shape and np.isfinite(smoke).sum() > 0)
    except Exception as exc:  # noqa: BLE001
        out["reference_smoke_pass"] = False
        out["math_defect"].append(f"smoke_fail:{type(exc).__name__}")

    axis = _axis_semantics(canonical, reg)
    # ---- prefix invariance（column-wise rolling 家族）----
    if axis == "column_wise":
        try:
            fn1d = _op_fn_1d(op)
            x1 = rng.standard_normal(30)
            x1[5] = np.nan
            res = prefix_invariance_check(fn1d, x1, T=18, K=8)
            out["prefix_invariance"] = bool(res.passed)
            if not res.passed:
                out["math_defect"].append("prefix_invariance_fail")
        except Exception as exc:  # noqa: BLE001
            out["prefix_invariance"] = False
            out["math_defect"].append(f"prefix_err:{type(exc).__name__}")

        # ---- chunk invariance ----
        try:
            x1 = rng.standard_normal(40)
            x1[10] = np.nan
            x1[20] = np.inf
            res = chunk_invariance_check(fn1d, x1, window=6)
            out["chunk_invariance"] = bool(res.passed)
            if not res.passed:
                out["math_defect"].append("chunk_invariance_fail")
        except Exception as exc:  # noqa: BLE001
            out["chunk_invariance"] = False
            out["math_defect"].append(f"chunk_err:{type(exc).__name__}")

    # ---- column permutation / metamorphic（row-wise 家族）----
    if axis == "row_wise":
        try:
            fn2d = _op_fn_2d(op)
            perm = [2, 0, 1]
            r0 = fn2d(base)
            r1 = fn2d(base[:, perm])[:, np.argsort(perm)]
            out["column_permutation"] = bool(
                np.allclose(r0, r1, rtol=1e-8, atol=1e-10, equal_nan=True))
            if not out["column_permutation"]:
                out["math_defect"].append("column_permutation_fail")
            # 通用 metamorphic：translation / positive-scale（适用于 zscore/rank/corr 等）
            props = []
            g = 3.0 * base + 1.0
            try:
                ok = bool(np.allclose(fn2d(g), fn2d(base), rtol=1e-8, atol=1e-10,
                                      equal_nan=True))
                props.append("strict_monotonic_invariance")
                out["metamorphic_passed"] = ok if out["metamorphic_passed"] is None \
                    else (out["metamorphic_passed"] and ok)
            except Exception:  # noqa: BLE001
                pass
            out["metamorphic_properties"] = tuple(props)
        except Exception as exc:  # noqa: BLE001
            out["column_permutation"] = False
            out["math_defect"].append(f"perm_err:{type(exc).__name__}")

    return out


# ---------------------------------------------------------------------------
# certificate 组装 + blocker 映射
# ---------------------------------------------------------------------------


def _numeric_semantics_flags(sem) -> dict[str, Any]:
    if sem is None:
        return {}
    return {
        "ddof": 1 if getattr(sem, "std_ddof", "sample") == "sample" else 0,
        "quantile_interpolation": getattr(sem, "quantile_interpolation", None),
        "zero_std": getattr(sem, "zscore_zero_std", None),
        "zero_denominator": getattr(sem, "div_zero", None),
        "inf_policy": "to_nan" if getattr(sem, "output_inf_to_nan", True) else "propagate",
        "min_periods": getattr(sem, "window_min_periods_default", None),
        "input_nan_to_null": getattr(sem, "input_nan_to_null", None),
    }


def _blocker_mapping(canonical: str, cert_fields: dict[str, Any]) -> list[str]:
    """把证书字段映射到 M01..M20 blockers。"""
    blockers: list[str] = []
    defects = cert_fields.get("math_defect") or []
    hidden = cert_fields.get("hidden_kwargs") or []
    local = cert_fields.get("local_casts") or []
    dup = cert_fields.get("semantic_policy_disagreements") or []

    if hidden:
        blockers.append(MBlocker.M07_HIDDEN_PARAMETER.value)
    if local:
        blockers.append(MBlocker.M08_LOCAL_PARAMETER_COERCION.value)
    if dup:
        blockers.append(MBlocker.M16_NUMERIC_SEMANTICS_DUPLICATE.value)
        if any("ddof" in d for d in dup):
            blockers.append(MBlocker.M02_DDOF_MISMATCH.value)
    if cert_fields.get("reference_smoke_pass") is False:
        blockers.append(MBlocker.M01_MATH_REFERENCE_MISMATCH.value)
    if cert_fields.get("prefix_invariance") is False:
        blockers.append(MBlocker.M11_PREFIX_INVARIANCE_FAIL.value)
    if cert_fields.get("chunk_invariance") is False:
        blockers.append(MBlocker.M12_CHUNK_INVARIANCE_FAIL.value)
    if cert_fields.get("column_permutation") is False:
        blockers.append(MBlocker.M19_RANK_AXIS_DEFECT.value)
    if "poly2" in canonical or canonical in ("ts_regression_slope",
                                             "ts_regression_intercept"):
        if cert_fields.get("regression_time_centered") is False:
            blockers.append(MBlocker.M20_REGRESSION_SCALE_DEFECT.value)
    if canonical in ("neutralize", "group_neutralize"):
        if cert_fields.get("neutralize_group_degenerate") is True:
            blockers.append(MBlocker.M18_NEUTRALIZATION_MATH_DEFECT.value)
    # M13 history heuristic：window 参数但无 history 声明
    if canonical.startswith(("ts_", "expanding_")) and \
            cert_fields.get("history_kind") is None and \
            cert_fields.get("has_window_param"):
        blockers.append(MBlocker.M13_HISTORY_FORMULA_HEURISTIC.value)
    # M10 backend native fallback hidden
    if cert_fields.get("production_certified") and \
            cert_fields.get("native_used") is False:
        blockers.append(MBlocker.M10_BACKEND_NATIVE_FALLBACK_HIDDEN.value)
    return sorted(set(blockers))


def build_certificate(
    canonical: str,
    reg,
    *,
    dynamic: bool,
    load_ok: bool,
) -> tuple[MathematicalSemanticCertificate, str]:
    """为一个 canonical 构建 certificate。返回 (cert, audit_note)。

    audit_note ∈ {"ok", "audit_error:<exc>", "not_loaded"}。
    """
    entry = _catalog_entry(reg, canonical)
    op = reg.get(canonical)
    if op is None:
        return _empty_cert(canonical, blockers=[MBlocker.M01_MATH_REFERENCE_MISMATCH.value],
                           defect=("no_runtime_operator",)), "no_runtime_operator"

    source, provenance = _source_of(op)
    declared = _declared_param_names(op)
    # 收集 scalar 参数（排除 panel）
    meta = getattr(op, "metadata", None)
    panel_params = set(getattr(meta, "panel_params", None) or ())
    scalar_params = [n for n in declared if n not in panel_params
                     and n not in ("x", "y", "group", "condition", "market",
                                   "size", "close", "open", "high", "low",
                                   "volume", "amount", "returns", "factor")]
    hidden, _all_kw = scan_hidden_kwargs(source, declared_names=declared)
    local_hits, _all_cast = scan_local_casts(source, declared_scalars=scalar_params)
    mask_policy = scan_sample_masking(source)
    sem = _semantics_for(canonical)
    num_flags = _numeric_semantics_flags(sem)
    hist = _history_contract(canonical, op)
    dup, _srcs = scan_duplicated_semantic_policy(
        numeric_semantics=num_flags,
        metadata_policy={},
        kernel_constants={},
    )

    try:
        from factor_engine.cleaned_operators.operator_spec import build_operator_spec
        spec = build_operator_spec(canonical)
    except Exception:  # noqa: BLE001
        spec = None
    production_certified = bool(entry.get("production_certified"))
    backends = entry.get("backends") or sorted(
        reg._operators.get(canonical, {}).keys())
    native_used = None
    fallback_used = None
    if "polars" in backends:
        native_used = True
    elif backends:
        fallback_used = True
        native_used = False

    dynamic_fields: dict[str, Any] = {}
    if dynamic and canonical in DYNAMIC_AUDIT_CORE:
        dynamic_fields = _run_dynamic(op, canonical, reg)

    # R19-101: 从声明表取 metamorphic properties（未动态运行时也带声明）。
    declared_metamorphic = MC.METAMORPHIC_PROPERTY_DECLARATIONS.get(canonical, ())
    if not dynamic_fields.get("metamorphic_properties"):
        dynamic_fields["metamorphic_properties"] = declared_metamorphic

    has_window_param = any("window" in str(p) or "d" == str(p) for p in declared)

    # R19-112: poly2/regression 时间坐标 center/scale 检查（静态 AST）
    regression_time_centered = None
    if "poly2" in canonical or canonical in ("ts_regression_slope",
                                             "ts_regression_intercept"):
        regression_time_centered = (
            "t_c" in source or "centered" in source
            or "t - " in source or "arange" not in source
        )

    # R19-111: ts_moment 数值稳定性（静态 + reference 动态）
    ts_moment_defect = None
    if canonical == "ts_moment":
        try:
            arr = np.array([1e3, 1.0001e3, 0.999e3, 1e3 + 5.0, 1e3 - 3.0,
                            1e3 + 1.0, 1e3, 1e3])
            _ = ts_moment_stable_(arr, 6, 6)
            ts_moment_defect = False
        except Exception:  # noqa: BLE001
            ts_moment_defect = True

    fields: dict[str, Any] = {
        "canonical": canonical,
        "math_definition": (getattr(meta, "description", None)
                            or entry.get("description")),
        "reference_impl": "pandas/numpy" if "pandas_numpy" in backends
        else ("polars" if "polars" in backends else ("|".join(backends) if backends else None)),
        "semantic_hash": math_semantics_version(
            operator_semantic_hash=_stable(provenance + source[:4000]),
            numeric_semantics_hash=_semantics_digest(num_flags),
            parameter_binding_hash=_stable(",".join(sorted(declared))),
            history_contract_hash=_stable(json.dumps(hist, sort_keys=True,
                                                     default=str)),
            math_version="R19-math-v1",
        ),
        "axis_semantics": _axis_semantics(canonical, reg),
        "sample_validity": "observed_roll",
        "missing_topology": ("propagate" if num_flags.get("input_nan_to_null")
                             else "ignore_rolling"),
        "current_row_requirement": True,
        "tie_policy": None,
        "ddof": num_flags.get("ddof"),
        "quantile_interpolation": num_flags.get("quantile_interpolation"),
        "zero_denominator": num_flags.get("zero_denominator"),
        "zero_std": num_flags.get("zero_std"),
        "domain_policy": None,
        "partial_window_policy": "min_periods",
        "minimum_effective_sample": _minimum_effective_sample(canonical, op, reg),
        **hist,
        "parameter_binding_complete": len(hidden) == 0,
        "hidden_kwargs": hidden,
        "local_casts": local_hits,
        "sample_mask_policy": mask_policy,
        "semantic_policy_disagreements": dup,
        "native_used": native_used,
        "fallback_used": fallback_used,
        "production_certified": production_certified,
        "has_window_param": has_window_param,
        "regression_time_centered": regression_time_centered,
        "ts_moment_defect": ts_moment_defect,
        **dynamic_fields,
        "source_provenance": provenance,
        "load_ok": load_ok,
    }
    if ts_moment_defect:
        fields.setdefault("math_defect", []).append("ts_moment_overflow_risk")
    if canonical == "ts_moment" and ts_moment_defect:
        fields["fix_action"] = ("legal k bound + scale-aware reference kernel "
                                "(ts_moment_stable_) + overflow policy")
    elif "poly2" in canonical:
        fields["fix_action"] = ("centered local t reference kernel "
                                "(ts_poly2_coeff_centered_) + declared time scale")

    # R19-118: equivalence="positive_scale" 需改名 homogeneous_scale（允许有符号
    # 权重，unit-sum；不能叫 positive_scale 而不检查每个元素为正）。
    pos_scale_params = [
        p for p, s in (getattr(meta, "param_specs", None) or {}).items()
        if getattr(s, "equivalence", None) == "positive_scale"
    ]
    if pos_scale_params:
        fields.setdefault("math_defect", []).append(
            f"positive_scale_rename_needed:{','.join(pos_scale_params)}")
        fields["fix_action"] = ("rename equivalence=positive_scale → homogeneous_scale "
                                "(signed weights, unit-sum, reject all-zero); see "
                                "math_certificate.canonicalize_homogeneous_scale")

    # R19-113: Kurt/Skew/Corr/Beta/Regression 描述必须说明 minimum effective
    # sample / missing topology / finite policy / ddof。
    description_gap_ops = {"Kurt", "Skew", "Corr", "Cov", "Covariance", "Beta",
                           "Intercept", "Slope", "R2", "Residual", "Var",
                           "ts_beta", "ts_corr", "ts_cov", "ts_std", "ts_var",
                           "ts_sharpe", "ts_autocorr", "ts_regression_slope",
                           "ts_regression_intercept"}
    if canonical in description_gap_ops:
        desc = (fields.get("math_definition") or "").lower()
        required = ("ddof", "min_period", "minimum", "missing", "nan", "finite",
                    "effective", "sample", "tie", "zero_std", "零", "缺", "缺失")
        if not any(r in desc for r in required):
            fields.setdefault("math_defect", []).append("description_sample_support_gap")
            if not fields.get("fix_action"):
                fields["fix_action"] = ("document minimum effective sample / missing "
                                        "topology / finite policy / ddof in operator "
                                        "description (R19-113)")

    blockers = _blocker_mapping(canonical, fields)
    cert = MathematicalSemanticCertificate(
        canonical=canonical,
        math_definition=fields["math_definition"],
        reference_impl=fields["reference_impl"],
        semantic_hash=fields["semantic_hash"],
        axis_semantics=fields["axis_semantics"],
        sample_validity=fields["sample_validity"],
        missing_topology=fields["missing_topology"],
        current_row_requirement=fields["current_row_requirement"],
        tie_policy=fields["tie_policy"],
        ddof=fields["ddof"],
        quantile_interpolation=fields["quantile_interpolation"],
        zero_denominator=fields["zero_denominator"],
        zero_std=fields["zero_std"],
        domain_policy=fields["domain_policy"],
        partial_window_policy=fields["partial_window_policy"],
        minimum_effective_sample=fields["minimum_effective_sample"],
        history_kind=fields["history_kind"],
        history_formula=fields["history_formula"],
        anchor_policy=fields["anchor_policy"],
        parameter_binding_complete=fields["parameter_binding_complete"],
        hidden_kwargs=tuple(hidden),
        local_casts=tuple(local_hits),
        sample_mask_policy=tuple(mask_policy),
        semantic_policy_disagreements=tuple(dup),
        source_provenance=provenance,
        reference_smoke_pass=dynamic_fields.get("reference_smoke_pass"),
        numba_native_pass=None,
        polars_native_pass=None,
        duckdb_native_pass=None,
        fallback_pass=fields["fallback_used"],
        native_used=fields["native_used"],
        fallback_used=fields["fallback_used"],
        prefix_invariance=dynamic_fields.get("prefix_invariance"),
        chunk_invariance=dynamic_fields.get("chunk_invariance"),
        column_permutation=dynamic_fields.get("column_permutation"),
        metamorphic_properties=tuple(dynamic_fields.get("metamorphic_properties") or ()),
        metamorphic_passed=dynamic_fields.get("metamorphic_passed"),
        math_defect=tuple(dynamic_fields.get("math_defect") or []) + (
            ("ts_moment_overflow_risk",) if ts_moment_defect else ()),
        semantic_drift=(),
        fix_action=fields.get("fix_action"),
        blockers=tuple(blockers),
    )
    return cert, "ok"


def _empty_cert(canonical: str, *, blockers: tuple[str, ...] = (),
                defect: tuple[str, ...] = ()) -> MathematicalSemanticCertificate:
    return MathematicalSemanticCertificate(
        canonical=canonical,
        math_definition=None,
        reference_impl=None,
        semantic_hash=None,
        math_defect=defect,
        blockers=blockers,
    )


def _stable(payload: str) -> str:
    import hashlib
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _semantics_digest(flags: dict[str, Any]) -> str:
    return _stable(json.dumps(flags, sort_keys=True, default=str))


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def write_outputs(rows: list[dict[str, Any]], meta: dict[str, Any],
                  out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "factor_engine.r19_operator_math_audit.v1",
               "meta": meta, "rows": rows}
    (out_dir / "R19_OPERATOR_MATH_AUDIT.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")

    # CSV：展平（tuple → "|" 连接）
    flat_keys = list(MathematicalSemanticCertificate.__dataclass_fields__.keys())
    with open(out_dir / "R19_OPERATOR_MATH_AUDIT.csv", "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=flat_keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _csv_val(r.get(k)) for k in flat_keys})

    (out_dir / "R19_OPERATOR_MATH_AUDIT.md").write_text(
        _render_md(rows, meta), encoding="utf-8")
    print(f"R19 math audit written to {out_dir / 'R19_OPERATOR_MATH_AUDIT.*'}")
    print(f"  canonical rows: {len(rows)}")
    print(f"  meta: {meta}")


def _csv_val(v: Any) -> Any:
    if isinstance(v, (tuple, list)):
        return "|".join(str(x) for x in v)
    if isinstance(v, bool):
        return str(v)
    return v


def _render_md(rows: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    cols = ("canonical", "reference_impl", "axis_semantics", "missing_topology",
            "ddof", "tie_policy", "zero_denominator", "zero_std",
            "minimum_effective_sample", "history_kind", "history_formula",
            "anchor_policy", "parameter_binding_complete", "hidden_kwargs",
            "local_casts", "reference_smoke_pass", "native_used", "fallback_used",
            "prefix_invariance", "chunk_invariance", "column_permutation",
            "metamorphic_passed", "math_defect", "blockers", "fix_action")
    lines = [
        "# FactorEngine R19 Operator Math Audit",
        "",
        f"- total rows = {meta.get('total')}",
        f"- load_all ok = {meta.get('load_ok')}",
        f"- load_all error = {meta.get('load_error')}",
        f"- blockers present = {meta.get('blocker_count')}",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for r in rows:
        lines.append("| " + " | ".join(_md(r.get(c)) for c in cols) + " |")
    lines.append("")
    return "\n".join(lines)


def _md(v: Any) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (tuple, list)):
        return ",".join(str(x) for x in v) if v else "—"
    return "—" if v is None else str(v)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40,
                    help="最多审计 canonical 数（默认 40；--all 覆盖）")
    ap.add_argument("--all", action="store_true", help="审计全部 canonical")
    ap.add_argument("--out", default=str(REPO / "docs"),
                    help="输出目录（默认 factor_engine/docs）")
    ap.add_argument("--dynamic", action="store_true",
                    help="对 DYNAMIC_AUDIT_CORE 跑真实算子动态检查")
    args = ap.parse_args()

    reg, load_ok, load_error = load_registry()
    if reg is None:
        # 并发 agent 损坏了 package __init__ —— 记录 error，输出空矩阵（不崩溃）。
        meta = {
            "total": 0,
            "load_ok": False,
            "load_error": load_error,
            "blocker_count": 0,
            "registered_total": 0,
            "dynamic_core_audited": args.dynamic,
        }
        write_outputs([], meta, Path(args.out))
        print(f"WARNING: registry unavailable — {load_error}", file=sys.stderr)
        return 0

    canonicals = sorted(reg.list_canonical())
    if not args.all:
        # 优先 key statistical / rolling 家族
        ordered = sorted(canonicals,
                         key=lambda c: (c not in DYNAMIC_AUDIT_CORE, c))
        canonicals = ordered[: args.limit]

    rows: list[dict[str, Any]] = []
    for canon in canonicals:
        try:
            cert, note = build_certificate(canon, reg, dynamic=args.dynamic,
                                           load_ok=load_ok)
            row = cert.to_dict()
            row["audit_note"] = note
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "canonical": canon,
                "math_definition": None, "reference_impl": None,
                "semantic_hash": None, "axis_semantics": None,
                "sample_validity": None, "missing_topology": None,
                "current_row_requirement": None, "tie_policy": None,
                "ddof": None, "quantile_interpolation": None,
                "zero_denominator": None, "zero_std": None,
                "domain_policy": None, "partial_window_policy": None,
                "minimum_effective_sample": None, "history_kind": None,
                "history_formula": None, "anchor_policy": None,
                "parameter_binding_complete": None, "hidden_kwargs": (),
                "local_casts": (), "reference_smoke_pass": None,
                "numba_native_pass": None, "polars_native_pass": None,
                "duckdb_native_pass": None, "fallback_pass": None,
                "native_used": None, "fallback_used": None,
                "prefix_invariance": None, "chunk_invariance": None,
                "column_permutation": None, "metamorphic_properties": (),
                "metamorphic_passed": None, "math_defect": (f"audit_error:{type(exc).__name__}",),
                "semantic_drift": (), "fix_action": None,
                "blockers": (MBlocker.M01_MATH_REFERENCE_MISMATCH.value,),
                "audit_note": f"audit_error:{type(exc).__name__}:{exc}",
            })

    blocker_count = sum(1 for r in rows if r.get("blockers"))
    meta = {
        "total": len(rows),
        "load_ok": load_ok,
        "load_error": load_error,
        "blocker_count": blocker_count,
        "registered_total": len(canonicals),
        "dynamic_core_audited": args.dynamic,
    }
    write_outputs(rows, meta, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
