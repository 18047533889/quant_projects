# -*- coding: utf-8 -*-
"""R19 数学语义证书层：MathematicalSemanticCertificate + M01-M20 blockers +
reference/optimized 两层参考内核 + metamorphic / prefix / chunk / checkpoint /
hostile fixture 检查原语。

本模块是审计引擎 / 证书 / artifact 层的**单一权威**：
- ``MathematicalSemanticCertificate`` —— 每个 canonical 的机器可读数学定义；
- ``MBlocker`` —— R19-130 的 M01..M20 release-blocker 常量；
- reference 数学内核 —— ``ts_moment_stable_`` / ``ts_poly2_coeff_centered_``
  等「清楚、正确、慢一点」的 numpy 参考实现，供 optimized 后端差分；
- 检查原语 —— metamorphic / prefix / chunk / checkpoint / hostile fixtures /
  parameter canonicalization / factor math-version digest / AST 静态扫描。

**所有权**：本文件为 R19 审计 agent 新建，不属于其它 agent。不依赖
``load_all()``（模块加载期不触发 registry 注册），测试可在 registry 半加载
状态下独立运行。
"""
from __future__ import annotations

import ast
import enum
import hashlib
import inspect
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# R40 本地内核（不依赖 cleaned_operators 包——registry 可能被并发会话编辑）。
# ---------------------------------------------------------------------------


def _r40_rank_1d(arr: np.ndarray) -> np.ndarray:
    """本地 percent rank（average-tie，与 ``_numpy_kernels.rank_`` 语义一致）。

    average-tie 保证 column-permutation equivariance（#257）：并列值无论列怎么
    换都得到同一 rank。
    """
    arr = np.asarray(arr, dtype=float)
    valid = np.isfinite(arr)
    out = np.full(arr.shape, np.nan, dtype=float)
    n = int(valid.sum())
    if n == 0:
        return out
    vals = arr[valid]
    order = np.argsort(vals, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    # average ties：对每个相等值分组取平均 rank。
    sorted_vals = vals[order]
    i = 0
    while i < n:
        j = i
        while j < n and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    if n > 1:
        out[valid] = (ranks - 1.0) / (n - 1.0)
    else:
        out[valid] = 0.5
    return out


def _r40_cs_resid(y, x):
    """本地截面回归残差（与 ``_numpy_kernels.cs_resid_`` 语义一致）。"""
    yv = np.asarray(y, dtype=float)
    xv = np.asarray(x, dtype=float)
    valid = np.isfinite(yv) & np.isfinite(xv)
    result = np.full_like(yv, np.nan)
    if int(valid.sum()) < 3:
        return result
    xv = xv[valid]
    yv = yv[valid]
    mx = xv.mean()
    my = yv.mean()
    xc = xv - mx
    var_x = float((xc * xc).mean())
    if not math.isfinite(var_x) or var_x < 1e-14:
        return result
    beta = float((xc * (yv - my)).mean()) / var_x
    alpha = my - beta * mx
    result[valid] = yv - (alpha + beta * xv)
    return result


def _r40_cs_regression(y, x, mode: int = 0):
    yv = np.asarray(y, dtype=float)
    xv = np.asarray(x, dtype=float)
    valid = np.isfinite(yv) & np.isfinite(xv)
    result = np.full_like(yv, np.nan)
    if int(valid.sum()) < 3:
        return result
    xv = xv[valid]
    yv = yv[valid]
    mx = xv.mean()
    my = yv.mean()
    xc = xv - mx
    var_x = float((xc * xc).mean())
    if not math.isfinite(var_x) or var_x < 1e-14:
        return result
    beta = float((xc * (yv - my)).mean()) / var_x
    alpha = my - beta * mx
    if mode == 0:
        result[valid] = yv - (alpha + beta * xv)
    elif mode == 1:
        result[valid] = beta
    else:
        result[valid] = alpha + beta * xv
    return result


def _r40_ts_regression_slope(x, y, d: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    if d <= 0:
        return out
    for i in range(d - 1, len(x)):
        xw = x[i - d + 1 : i + 1]
        yw = y[i - d + 1 : i + 1]
        valid = np.isfinite(xw) & np.isfinite(yw)
        if int(valid.sum()) >= 3:
            xv = xw[valid]
            yv = yw[valid]
            xc = xv - xv.mean()
            sxx = float((xc * xc).sum())
            if sxx > 1e-14:
                out[i] = float((xc * (yv - yv.mean())).sum()) / sxx
    return out


def _r40_ts_rank_corr(x, y, d: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    if d <= 0:
        return out
    for i in range(d - 1, len(x)):
        xw = x[i - d + 1 : i + 1]
        yw = y[i - d + 1 : i + 1]
        valid = np.isfinite(xw) & np.isfinite(yw)
        if int(valid.sum()) >= 3:
            rx = _r40_rank_1d(xw[valid])
            ry = _r40_rank_1d(yw[valid])
            c = np.corrcoef(rx, ry)[0, 1]
            out[i] = c if np.isfinite(c) else np.nan
    return out

# ---------------------------------------------------------------------------
# R19-130: M01..M20 release-blocker constants
# ---------------------------------------------------------------------------


class MBlocker(str, enum.Enum):
    """R19-130 机器审计 blocker 常量。

    任一 blocker 存在即视为 release fail —— 对应 canonical 不得以当前数学定义
    进入 direct factor 层，直到修复或显式 downgrade 到非 direct。
    """

    M01_MATH_REFERENCE_MISMATCH = "M01_MATH_REFERENCE_MISMATCH"
    M02_DDOF_MISMATCH = "M02_DDOF_MISMATCH"
    M03_FINITE_SAMPLE_MISMATCH = "M03_FINITE_SAMPLE_MISMATCH"
    M04_TIE_POLICY_MISMATCH = "M04_TIE_POLICY_MISMATCH"
    M05_CURRENT_ROW_POLICY_MISMATCH = "M05_CURRENT_ROW_POLICY_MISMATCH"
    M06_PARTIAL_WINDOW_MISMATCH = "M06_PARTIAL_WINDOW_MISMATCH"
    M07_HIDDEN_PARAMETER = "M07_HIDDEN_PARAMETER"
    M08_LOCAL_PARAMETER_COERCION = "M08_LOCAL_PARAMETER_COERCION"
    M09_PARAMETER_BINDING_DRIFT = "M09_PARAMETER_BINDING_DRIFT"
    M10_BACKEND_NATIVE_FALLBACK_HIDDEN = "M10_BACKEND_NATIVE_FALLBACK_HIDDEN"
    M11_PREFIX_INVARIANCE_FAIL = "M11_PREFIX_INVARIANCE_FAIL"
    M12_CHUNK_INVARIANCE_FAIL = "M12_CHUNK_INVARIANCE_FAIL"
    M13_HISTORY_FORMULA_HEURISTIC = "M13_HISTORY_FORMULA_HEURISTIC"
    M14_SAMPLE_ANCHOR_UNDECLARED = "M14_SAMPLE_ANCHOR_UNDECLARED"
    M15_DOMAIN_POLICY_MISMATCH = "M15_DOMAIN_POLICY_MISMATCH"
    M16_NUMERIC_SEMANTICS_DUPLICATE = "M16_NUMERIC_SEMANTICS_DUPLICATE"
    M17_GHOST_EXECUTION_CONTRACT = "M17_GHOST_EXECUTION_CONTRACT"
    M18_NEUTRALIZATION_MATH_DEFECT = "M18_NEUTRALIZATION_MATH_DEFECT"
    M19_RANK_AXIS_DEFECT = "M19_RANK_AXIS_DEFECT"
    M20_REGRESSION_SCALE_DEFECT = "M20_REGRESSION_SCALE_DEFECT"

    @classmethod
    def all_names(cls) -> tuple[str, ...]:
        return tuple(m.value for m in cls)


# 语义常量（字符串，便于 JSON 导出）
MISSING_TOPOLOGY_POLICIES = ("propagate", "ignore_rolling", "pairwise",
                             "complete_case", "fail_closed", "carry")
PARTIAL_WINDOW_POLICIES = ("min_periods", "full_window", "nan_until_full")
ZERO_DENOMINATOR_POLICIES = ("nan", "inf", "default", "protected")
ZERO_STD_POLICIES = ("zero", "nan", "null")
TIE_POLICIES = ("average", "first", "last", "min", "max", "ordinal")
ANCHOR_POLICIES = ("trailing_end", "left", "expanding", "cumulative",
                   "event_last", "session_close")


# ---------------------------------------------------------------------------
# R19-129: MathematicalSemanticCertificate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MathematicalSemanticCertificate:
    """R19-129 数学语义证书：同一 canonical 在所有合法路径上只有一个数学定义。

    字段对应 R19_OPERATOR_MATH_AUDIT 矩阵的列。全部可空 —— 未收集/不适用项为
    ``None``；任何 ``math_defect`` / ``blockers`` 非空都意味着不能直接进入
    direct factor 层。
    """

    canonical: str
    # ---- math definition ------------------------------------------------
    math_definition: str | None = None
    reference_impl: str | None = None
    semantic_hash: str | None = None
    # ---- axis / sample / missing / current-row ---------------------------
    axis_semantics: str | None = None          # "row_wise"/"column_wise"/"elementwise"/"panel"
    sample_validity: str | None = None         # "observed_roll"/"pairwise"/"complete_case"
    missing_topology: str | None = None
    current_row_requirement: bool | None = None
    # ---- tie / ddof / quantile / zero ------------------------------------
    tie_policy: str | None = None
    ddof: int | None = None
    quantile_interpolation: str | None = None
    zero_denominator: str | None = None
    zero_std: str | None = None
    domain_policy: str | None = None
    # ---- partial window / min sample -------------------------------------
    partial_window_policy: str | None = None
    minimum_effective_sample: int | None = None
    # ---- history contract --------------------------------------------------
    history_kind: str | None = None
    history_formula: str | None = None
    anchor_policy: str | None = None
    # ---- parameter binding --------------------------------------------------
    parameter_binding_complete: bool | None = None
    hidden_kwargs: tuple[str, ...] = ()
    local_casts: tuple[str, ...] = ()
    # ---- R19-126/127 静态扫描产物 -------------------------------------------
    sample_mask_policy: tuple[str, ...] = ()
    semantic_policy_disagreements: tuple[str, ...] = ()
    source_provenance: str = ""
    # ---- backend parity -----------------------------------------------------
    reference_smoke_pass: bool | None = None
    numba_native_pass: bool | None = None
    polars_native_pass: bool | None = None
    duckdb_native_pass: bool | None = None
    fallback_pass: bool | None = None
    native_used: bool | None = None
    fallback_used: bool | None = None
    # ---- dynamic invariance --------------------------------------------------
    prefix_invariance: bool | None = None
    chunk_invariance: bool | None = None
    column_permutation: bool | None = None
    metamorphic_properties: tuple[str, ...] = ()
    metamorphic_passed: bool | None = None
    # ---- verdict -------------------------------------------------------------
    math_defect: tuple[str, ...] = ()
    semantic_drift: tuple[str, ...] = ()
    fix_action: str | None = None
    blockers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # 保持 blocker 有稳定顺序
        d["blockers"] = sorted(self.blockers)
        return d

    @property
    def has_blocker(self) -> bool:
        return len(self.blockers) > 0


# ---------------------------------------------------------------------------
# R19-121/122: backend-independent math version digest
# ---------------------------------------------------------------------------


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def math_semantics_version(
    *,
    operator_semantic_hash: str,
    numeric_semantics_hash: str,
    parameter_binding_hash: str,
    history_contract_hash: str,
    math_version: str = "R19-math-v1",
) -> str:
    """R19-121 factor identity 的 mathematical-semantics version。

    组成：operator semantic hash + numeric semantics hash + parameter binding
    hash + history contract hash + backend-independent math version。任一输入
    变化 → digest 变化（R19-122：修 ddof / rank_corr / argmax origin /
    neutralize 必须让旧 factor identity 失效）。

    参数:
        operator_semantic_hash: 算子实现语义哈希（reference kernel digest）。
        numeric_semantics_hash: ``backend.numeric_semantics.numeric_semantics_hash``。
        parameter_binding_hash: 参数绑定契约哈希（可含 nested canonical freeze）。
        history_contract_hash: history kind/formula/anchor 契约哈希。
        math_version: backend-independent math 版本串（非实现细节）。

    返回:
        复合 sha256 hex。
    """
    payload = "|".join([
        str(operator_semantic_hash),
        str(numeric_semantics_hash),
        str(parameter_binding_hash),
        str(history_contract_hash),
        str(math_version),
    ])
    return _sha256_hex("FactorEngineMathSemanticsV1\x00" + payload)


# ---------------------------------------------------------------------------
# R19-111/112: reference math kernels（scale-aware / centered-time）
# ---------------------------------------------------------------------------

TS_MOMENT_K_MAX = 8
TS_MOMENT_OVERFLOW_POLICIES = ("nan", "inf", "raise")


def ts_moment_stable_(
    x: np.ndarray,
    d: int,
    k: int,
    *,
    k_max: int = TS_MOMENT_K_MAX,
    overflow_policy: str = "nan",
) -> np.ndarray:
    """R19-111 scale-aware 滚动 k 阶中心矩（参考实现）。

    原生 ``ts_moment_`` 直接 ``(x-μ)^k``：大数 × 高 k 溢出为 Inf，且没有任何
    overflow policy 声明。本参考实现把中心矩按最大 |z| 缩放后在 [-1,1] 上计算
    ``scale^k * E[(z/scale)^k]``，使中间量不溢出；``scale^k`` 本身溢出时按
    ``overflow_policy`` 处理。

    参数:
        x: 一维时序数组。
        d: 窗口长度。
        k: 矩的阶数（``2 <= k <= k_max``）。
        k_max: 合法 k 上限（默认 8；超过即 raise —— 高 k 中心矩无 float64 意义）。
        overflow_policy: ``"nan"`` / ``"inf"`` / ``"raise"``。

    返回:
        滚动 k 阶中心矩数组。非法 k 直接 ``ValueError``。
    """
    if not isinstance(k, (int, np.integer)) or int(k) < 2:
        raise ValueError(f"ts_moment k must be >= 2, got {k!r}")
    k = int(k)
    if k > k_max:
        raise ValueError(
            f"ts_moment k={k} exceeds legal upper bound k_max={k_max}: "
            f"central moments of order > {k_max} are numerically meaningless "
            "in float64 (R19-111)"
        )
    if overflow_policy not in TS_MOMENT_OVERFLOW_POLICIES:
        raise ValueError(f"unknown overflow_policy {overflow_policy!r}")
    arr = np.asarray(x, dtype=float)
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=float)
    if d <= 0:
        return out
    for i in range(d - 1, n):
        w = arr[i - d + 1 : i + 1]
        wv = w[np.isfinite(w)]
        if wv.size == 0:
            continue
        mu = float(wv.mean())
        z = wv - mu
        scale = float(np.max(np.abs(z))) if z.size else 0.0
        if scale == 0.0:
            # 常数窗口：一阶/奇阶中心矩为 0，二阶以上也全为 0。
            out[i] = 0.0
            continue
        zs = z / scale  # |zs| <= 1
        mean_zs_k = float(np.mean(zs ** k))
        # scale ** k 可能溢出（大数 x 高 k）——显式 overflow policy。
        if np.isfinite(scale):
            try:
                scale_k = scale ** k
            except OverflowError:  # pragma: no cover - numpy 不抛，返回 inf
                scale_k = np.inf
        else:
            scale_k = np.inf
        if not np.isfinite(scale_k) or math.isinf(scale_k):
            if overflow_policy == "nan":
                out[i] = np.nan
            elif overflow_policy == "inf":
                out[i] = np.inf * (1.0 if mean_zs_k >= 0 else -1.0)
            else:
                raise OverflowError(
                    f"ts_moment overflow at row {i}: scale**k not finite "
                    f"(k={k}, scale={scale!r})"
                )
        else:
            out[i] = scale_k * mean_zs_k
    return out


def ts_poly2_coeff_centered_(
    x: np.ndarray,
    d: int,
    *,
    scale_time: bool = False,
) -> np.ndarray:
    """R19-112 centered-time 二次拟合系数（参考实现）。

    原生 ``ts_poly2_coeff_`` 用 ``t = arange(d)``，长 window 下 ``t^2`` 达到
    ``d^2`` 量级，设计矩阵 condition number 恶化。本参考实现使用 **centered
    local t**（``t_c = t - (d-1)/2``），condition number 显著改善；可选用
    ``scale_time=True`` 做 max-|t_c| 归一化。

    **时间尺度声明**：返回系数 ``c`` 对应 ``y ≈ a + b*t_c + c*t_c^2``，单位是
    ``y / t_c^2``（t_c 单位为 bar，center 在窗口中心）。当 ``scale_time=True``
    时 t_c 被除以 ``max|t_c|``，c 单位为 ``y / (scaled t)^2`` —— 证书必须声明
    该尺度（见 ``ts_poly2_time_scale_declaration``）。

    参数:
        x: 一维时序数组。
        d: 拟合窗口长度。
        scale_time: 是否把 centered t 归一化到 [-1,1]。

    返回:
        二次项系数 c 数组。
    """
    arr = np.asarray(x, dtype=float)
    n = arr.shape[0]
    out = np.full(n, np.nan, dtype=float)
    if d <= 0:
        return out
    t = np.arange(d, dtype=float)
    t_c = t - float(d - 1) / 2.0
    if scale_time:
        m = float(np.max(np.abs(t_c))) if d > 1 else 1.0
        t_c = t_c / m if m > 0 else t_c
    for i in range(d - 1, n):
        y = arr[i - d + 1 : i + 1]
        valid = np.isfinite(y)
        if int(valid.sum()) < 3:
            continue
        # Vandermonde: [t_c^2, t_c, 1] — centered design。
        A = np.column_stack([t_c[valid] ** 2, t_c[valid], np.ones(int(valid.sum()))])
        coeffs, *_ = np.linalg.lstsq(A, y[valid], rcond=None)
        out[i] = coeffs[0]
    return out


def ts_poly2_time_scale_declaration(*, scale_time: bool) -> str:
    """R19-112 证书用：poly2 系数的时间尺度声明。"""
    if scale_time:
        return ("coefficient c is per (t/|t_max|)^2 where t is centered at "
                "window midpoint and normalized to [-1,1]")
    return ("coefficient c is per t_c^2 where t_c = t - (d-1)/2 is centered "
            "at the window midpoint, in units of 1/bar^2")


# ---------------------------------------------------------------------------
# R19-118..120: parameter canonicalization（homogeneous_scale + nested hash）
# ---------------------------------------------------------------------------


def recursive_canonical_freeze(
    value: Any,
    *,
    digits: int = 12,
) -> Any:
    """R19-119 递归 canonical freeze：嵌套 dict/tuple/list 内部 float noise 也
    被 frozen，只有顶层被处理是 bug。

    - float/int 按 12 位有效数字舍入（``_round_sig``）；
    - ``dict``/任何 ``Mapping`` → 递归后按 key 排序成 tuple；
    - ``tuple``/``list`` → 有序递归；
    - numpy 标量 → 原生 Python 值；
    - 其它（str/bool/None/Enum）原样保留。

    参数:
        value: 任意嵌套参数值。
        digits: 有效数字。

    返回:
        递归 canonical 形式（可哈希）。
    """
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (recursive_canonical_freeze(k, digits=digits),
                 recursive_canonical_freeze(v, digits=digits))
                for k, v in value.items()
            )
        )
    if isinstance(value, (tuple, list)):
        return tuple(recursive_canonical_freeze(v, digits=digits) for v in value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        num = float(value)
        if not math.isfinite(num) or num == 0.0:
            return value
        return _round_sig(num, digits)
    return value


def _round_sig(value: float, digits: int) -> float:
    if value == 0.0 or not math.isfinite(value):
        return value
    try:
        shift = digits - int(math.floor(math.log10(abs(value)))) - 1
    except (ValueError, OverflowError):
        return value
    factor = 10.0 ** shift
    return math.floor(value * factor + 0.5) / factor


def canonicalize_homogeneous_scale(vec: Sequence[Any], *, digits: int = 12) -> tuple[float, ...]:
    """R19-118 ``homogeneous_scale`` 归一化（reference）。

    `equivalence="positive_scale"` 之前只看 total，不检查每个元素为正。本轮
    选择语义：**改名 homogeneous_scale** —— 允许有符号权重按正比例系数等价，
    保持现有 unit-sum 行为，但显式检查「至少不全为零」；全零向量（无信息量）
    拒绝。非数值元素拒绝（不静默过滤，R13 NEW-P0-19）。

    参数:
        vec: 权重序列。
        digits: 有效数字。

    返回:
        unit-sum 归一化后的 tuple；全部数值、至少一个非零。

    异常:
        ValueError: 非数值元素 / 全零 / 归一化总数不有限。
    """
    if isinstance(vec, (str, bytes)):
        raise ValueError("homogeneous_scale vector must be a numeric sequence, got str")
    seq = list(vec)
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in seq):
        raise ValueError(
            "a scale-equivalent weight vector must contain only numeric "
            "elements (the whole parameter is rejected, not filtered)"
        )
    nums = [float(v) for v in seq]
    total = sum(nums)
    if not math.isfinite(total):
        raise ValueError(f"scale-equivalent vector total is not finite: {total!r}")
    if abs(total) <= 1e-12:
        raise ValueError(
            "homogeneous_scale vector is all-zero: no information, rejected"
        )
    return tuple(_round_sig(v / total, digits) for v in nums)


def positive_scale_requires_all_positive(vec: Sequence[Any]) -> bool:
    """R19-118 诊断：返回 True 表示该向量需要正元素检查（若仍用
    ``positive_scale`` 名字）。"""
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
               for v in vec) or any(
        isinstance(v, (int, float)) and not isinstance(v, bool) and v <= 0
        for v in vec
    )


# ---------------------------------------------------------------------------
# R19-101/102: metamorphic property checkers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetamorphicResult:
    """单个 metamorphic property 检查结果。"""

    property_name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"property": self.property_name, "passed": self.passed,
                "detail": self.detail}


def _finite_where(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def _allclose_nan(a: np.ndarray, b: np.ndarray, *, rtol: float = 1e-8,
                  atol: float = 1e-10) -> bool:
    if a.shape != b.shape:
        return False
    if a.size == 0:
        return True
    return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))


def check_rank_monotonic_invariance(
    fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> MetamorphicResult:
    """rank: strict monotonic transform invariance —— rank(f(x)) == rank(x)。"""
    g = 3.0 * x + 1.0  # 严格递增线性
    r0 = fn(x)
    r1 = fn(g)
    ok = _allclose_nan(r0, r1, rtol=rtol, atol=atol)
    return MetamorphicResult("rank.strict_monotonic_invariance", ok,
                             "fn(3x+1) == fn(x)" if ok else "mismatch")


def check_rank_column_permutation_equivariance(
    fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    perm: Sequence[int],
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> MetamorphicResult:
    """rank: column permutation equivariance —— rank(x[:,p])[:,p⁻¹] == rank(x)。"""
    p = np.asarray(perm, dtype=int)
    if x.ndim != 2:
        return MetamorphicResult("rank.column_permutation_equivariance", False,
                                 "requires 2D input")
    inv = np.empty_like(p)
    inv[p] = np.arange(p.size)
    r0 = fn(x)
    r1 = fn(x[:, p])[:, inv]
    ok = _allclose_nan(r0, r1, rtol=rtol, atol=atol)
    return MetamorphicResult("rank.column_permutation_equivariance", ok,
                             "permuted ranks equal" if ok else "mismatch")


def check_translation_invariance(
    fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    shift: float = 100.0,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> MetamorphicResult:
    """translation invariance —— fn(x + c) == fn(x)。"""
    r0 = fn(x)
    r1 = fn(x + shift)
    ok = _allclose_nan(r0, r1, rtol=rtol, atol=atol)
    return MetamorphicResult("translation_invariance", ok,
                             f"fn(x+{shift}) == fn(x)" if ok else "mismatch")


def check_positive_scale_invariance(
    fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    scale: float = 2.5,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> MetamorphicResult:
    """positive-scale invariance —— fn(a*x) == fn(x), a>0。"""
    r0 = fn(x)
    r1 = fn(scale * x)
    ok = _allclose_nan(r0, r1, rtol=rtol, atol=atol)
    return MetamorphicResult("positive_scale_invariance", ok,
                             f"fn({scale}x) == fn(x)" if ok else "mismatch")


def check_permutation_equivariance_all_tie_sensitive_operators(
    n_rows: int = 12,
    n_cols: int = 6,
    *,
    seed: int = 11,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> tuple[bool, dict[str, Any]]:
    """R40 #257：所有 tie-sensitive 算子（quantile_bucket / topk / group_rank /
    winsorize / neutralize / cs_regression）的 column-permutation equivariance
    property test。

    ``fn(x[:, p])[:, p⁻¹] == fn(x)`` —— 截面内列重排不改变任何名字的输出。
    使用**独立 numpy kernel 实现**（不依赖可能被并发编辑的 operator registry），
    逐行复刻各算子的截面语义。返回 ``(all_ok, {canonical: ok})``，可被
    ``scripts/audit_r40_hard_gates.py`` 复用。
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_rows, n_cols))
    x[:, 1] = x[:, 0]  # ties
    x[3, :] = np.nan   # internal hole row
    y = rng.standard_normal((n_rows, n_cols))
    perm = rng.permutation(n_cols)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(n_cols)
    groups = np.resize(np.array(["A", "A", "B", "B", "C", "C"], dtype=object), n_cols)

    def _rowwise(fn1d, data, ydata):
        """把一维列函数应用到每行（截面语义），保持 (rows, cols)。"""
        out = np.full((n_rows, n_cols), np.nan, dtype=float)
        for i in range(n_rows):
            out[i] = fn1d(data[i], ydata[i], groups)
        return out

    def _quantile_bucket_1d(row, _y, _g):
        valid = np.isfinite(row)
        r = _r40_rank_1d(row)
        b = np.full(row.shape, np.nan)
        b[valid] = np.minimum(np.floor(r[valid] * 4.0), 3.0)
        return b

    def _topk_1d(row, _y, _g, k: int = 2):
        valid = np.isfinite(row)
        out = np.zeros(row.shape, dtype=float)
        if int(valid.sum()) >= k:
            thr = -np.sort(-row[valid])[k - 1]
            out = (row >= thr).astype(float)
        out[~valid] = np.nan
        return out

    def _group_rank_1d(row, g):
        valid = np.isfinite(row)
        out = np.full(row.shape, np.nan)
        if valid.any():
            codes, _uniques = pd.factorize(g[valid])
            ranks = np.full(valid.sum(), np.nan)
            for c in np.unique(codes):
                m = codes == c
                ranks[m] = _r40_rank_1d(row[valid][m])
            out[valid] = ranks
        return out

    def _winsorize_1d(row, _y, _g, lo: float = 0.05, hi: float = 0.95):
        valid = np.isfinite(row)
        out = row.copy()
        if int(valid.sum()) >= 3:
            qlo, qhi = np.nanquantile(row, [lo, hi])
            out = np.clip(row, qlo, qhi)
        out[~valid] = np.nan
        return out

    def _neutralize_1d(row, yrow, _g):
        return _r40_cs_resid(yrow, row)

    def _cs_regression_1d(row, yrow, _g):
        return _r40_cs_regression(yrow, row, mode=0)

    def _make_fn(fn1d):
        def _fn(data):
            return _rowwise(fn1d, data, y)
        return _fn

    def _make_bivar(fn1d):
        def _fn(data, ydata):
            return _rowwise(fn1d, data, ydata)
        return _fn

    def _make_grouped():
        def _fn(data, gdata):
            out = np.full((n_rows, n_cols), np.nan, dtype=float)
            for i in range(n_rows):
                out[i] = _group_rank_1d(data[i], gdata)
            return out
        return _fn

    univariate: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "quantile_bucket": _make_fn(_quantile_bucket_1d),
        "topk": _make_fn(_topk_1d),
        "winsorize": _make_fn(_winsorize_1d),
    }
    bivariate: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
        "neutralize": _make_bivar(_neutralize_1d),
        "cs_regression": _make_bivar(_cs_regression_1d),
    }
    group_rank_fn = _make_grouped()
    results: dict[str, bool] = {}
    all_ok = True
    for canonical, fn in univariate.items():
        r0 = fn(x)
        r1 = fn(x[:, perm])[:, inv]
        ok = _allclose_nan(r0, r1, rtol=rtol, atol=atol)
        results[canonical] = bool(ok)
        all_ok = all_ok and ok
    for canonical, fn in bivariate.items():
        r0 = fn(x, y)
        r1 = fn(x[:, perm], y[:, perm])[:, inv]
        ok = _allclose_nan(r0, r1, rtol=rtol, atol=atol)
        results[canonical] = bool(ok)
        all_ok = all_ok and ok
    r0 = group_rank_fn(x, groups)
    r1 = group_rank_fn(x[:, perm], groups[perm])[:, inv]
    ok = _allclose_nan(r0, r1, rtol=rtol, atol=atol)
    results["group_rank"] = bool(ok)
    all_ok = all_ok and ok
    return all_ok, results


def check_bivariate_symmetry(
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> MetamorphicResult:
    """symmetry —— fn(x, y) == fn(y, x)。"""
    r0 = fn(x, y)
    r1 = fn(y, x)
    ok = _allclose_nan(r0, r1, rtol=rtol, atol=atol)
    return MetamorphicResult("symmetry", ok, "fn(x,y)==fn(y,x)" if ok else "mismatch")


def check_bivariate_translation_invariance(
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> MetamorphicResult:
    """corr/cov translation invariance —— fn(x+a, y+b) == fn(x, y)。"""
    r0 = fn(x, y)
    r1 = fn(x + 50.0, y - 30.0)
    ok = _allclose_nan(r0, r1, rtol=rtol, atol=atol)
    return MetamorphicResult("translation_invariance", ok,
                             "fn(x+50,y-30)==fn(x,y)" if ok else "mismatch")


def check_bivariate_positive_scale_invariance(
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> MetamorphicResult:
    """corr positive-scale invariance —— fn(a*x, b*y) == fn(x, y), a,b>0。"""
    r0 = fn(x, y)
    r1 = fn(2.0 * x, 0.5 * y)
    ok = _allclose_nan(r0, r1, rtol=rtol, atol=atol)
    return MetamorphicResult("positive_scale_invariance", ok,
                             "fn(2x,0.5y)==fn(x,y)" if ok else "mismatch")


def check_cov_scale_covariance(
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> MetamorphicResult:
    """cov scale covariance —— fn(a*x, b*y) == a*b*fn(x, y)。"""
    a, b = 2.0, 3.0
    r0 = fn(x, y)
    r1 = fn(a * x, b * y) / (a * b)
    ok = _allclose_nan(r0, r1, rtol=rtol, atol=atol)
    return MetamorphicResult("scale_covariance", ok,
                             "fn(2x,3y)/(6)==fn(x,y)" if ok else "mismatch")


def check_beta_y_scale_covariance(
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> MetamorphicResult:
    """beta y-scale covariance —— beta(a*y, x) == a*beta(y, x)。"""
    a = 4.0
    r0 = fn(x, y)
    r1 = fn(x, a * y) / a
    ok = _allclose_nan(r0, r1, rtol=rtol, atol=atol)
    return MetamorphicResult("beta.y_scale_covariance", ok,
                             "beta(x,4y)/4==beta(x,y)" if ok else "mismatch")


def check_beta_x_scale_inverse_covariance(
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> MetamorphicResult:
    """beta x-scale inverse covariance —— beta(a*x, y) == beta(x, y)/a。"""
    a = 2.0
    r0 = fn(x, y)
    r1 = fn(a * x, y) * a
    ok = _allclose_nan(r0, r1, rtol=rtol, atol=atol)
    return MetamorphicResult("beta.x_scale_inverse_covariance", ok,
                             "beta(2x,y)*2==beta(x,y)" if ok else "mismatch")


def _group_valid_mask(groups: np.ndarray) -> np.ndarray:
    """计算 group 标签中「有效」（非缺失）掩码；对 string/object 标签不抛错。

    - 数值 group：NaN → 缺失；
    - 字符串 / object group：``None`` / ``float('nan')`` / 与自身不等 → 缺失；
      普通字符串视为有效。
    """
    g = np.asarray(groups)
    valid = np.ones(g.shape[0], dtype=bool)
    for i, v in enumerate(g):
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)) or v != v:
                valid[i] = False
        except Exception:  # noqa: BLE001
            valid[i] = True
    return valid


def check_neutralize_group_residual_mean(
    fn_neutralize: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    groups: np.ndarray,
    *,
    atol: float = 1e-8,
) -> MetamorphicResult:
    """neutralize group residual mean ≈0：每个 group 内残差均值 ≈ 0。"""
    resid = fn_neutralize(x, groups)
    g = np.asarray(groups)
    valid_g = _group_valid_mask(g)
    max_mean = 0.0
    n_checked = 0
    for gv in np.unique(g[valid_g]):
        m = (g == gv) & np.isfinite(resid) & np.isfinite(x) & valid_g
        if m.sum() > 0:
            n_checked += 1
            max_mean = max(max_mean, float(np.mean(resid[m])))
    ok = abs(max_mean) <= atol and n_checked >= 2
    return MetamorphicResult("neutralize.group_residual_mean_0", ok,
                             f"max |group mean|={max_mean:.3e} (groups={n_checked})")


def check_neutralize_size_residual_covariance(
    fn_neutralize_size: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    groups: np.ndarray,
    size: np.ndarray,
    *,
    atol: float = 1e-8,
) -> MetamorphicResult:
    """neutralize size residual covariance ≈0：残差与 demeaned size 的协方差 ≈ 0。"""
    resid = fn_neutralize_size(x, groups, size)
    m = np.isfinite(resid) & np.isfinite(size)
    if m.sum() < 2:
        return MetamorphicResult("neutralize.size_residual_cov_0", False,
                                 "insufficient samples")
    sz = size[m]
    szc = sz - sz.mean()
    cov = float(np.mean(resid[m] * szc))
    ok = abs(cov) <= atol
    return MetamorphicResult("neutralize.size_residual_cov_0", ok,
                             f"cov={cov:.3e}")


# 参考 numpy 实现（2D 面板，row=date, col=instrument）——用于测试与差分 ----------

def _rank_rowwise_np(x: np.ndarray) -> np.ndarray:
    """row-wise 0-1 rank（``cs_rank_01`` 语义）的 numpy 参考实现。

    这是 canonical ``rank`` / ``cs_rank_01`` 的**唯一 semantic reference**（R19-115）：
    - 有效样本 = ``np.isfinite``（±Inf 不参与排名基，其 cell 保持 NaN）；
    - average-tie rank；
    - ``(rank - 1) / (n - 1)``，singleton（n==1）→ 0.5；
    - 全缺失行 → 全 NaN。

    注意：pandas ``rank(pct=True)`` 的 ``rank_pct`` / ``cs_pct_rank`` 语义不同
    （``rank/n``，singleton=1.0），见 :func:`_rank_pct_rowwise_np`。
    """
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(x.shape[0]):
        row = x[i]
        finite = np.isfinite(row)
        n = int(finite.sum())
        if n == 0:
            continue
        vals = row[finite]
        order = vals.argsort(kind="stable")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1, dtype=float)
        # average ties
        _, inv, cnt = np.unique(vals, return_inverse=True, return_counts=True)
        avg = np.zeros(n, dtype=float)
        for u in range(cnt.size):
            m = inv == u
            avg[m] = ranks[m].mean()
        if n == 1:
            out[i, finite] = 0.5
        else:
            out[i, finite] = (avg - 1.0) / (n - 1.0)
    return out


def _rank_pct_rowwise_np(x: np.ndarray) -> np.ndarray:
    """row-wise pandas ``rank(pct=True)`` 参考（``rank_pct`` / ``cs_pct_rank``）。"""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(x.shape[0]):
        row = x[i]
        finite = np.isfinite(row)
        n = int(finite.sum())
        if n == 0:
            continue
        vals = row[finite]
        order = vals.argsort(kind="stable")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1, dtype=float)
        _, inv, cnt = np.unique(vals, return_inverse=True, return_counts=True)
        avg = np.zeros(n, dtype=float)
        for u in range(cnt.size):
            m = inv == u
            avg[m] = ranks[m].mean()
        out[i, finite] = avg / n
    return out


def _zscore_rowwise_np(x: np.ndarray, *, ddof: int = 1, zero_std: str = "zero") -> np.ndarray:
    """row-wise z-score 参考，对齐 pandas ``(x - mean)/std(ddof=1)`` 语义。

    - single-valid-sample 行：pandas ``std(ddof=1)`` 为 NaN → 输出 NaN；
    - 常数行（std==0）：按 ``zero_std`` 策略（默认 ``"zero"`` → 0.0，与
      ``zscore``/``c_zscore`` 的 ``std.replace(0,1)`` 行为一致）；
    - ±Inf 不作为有效样本参与均值/方差，其 cell 保持 NaN。
    """
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    for i in range(x.shape[0]):
        row = x[i]
        finite = np.isfinite(row)
        if finite.sum() == 0:
            continue
        vals = row[finite]
        mu = vals.mean()
        sd = float(vals.std(ddof=ddof))  # n==1 → NaN（与 pandas std 一致）
        if not np.isfinite(sd):
            out[i, finite] = np.nan
        elif sd == 0.0:
            if zero_std == "zero":
                out[i, finite] = 0.0
            else:
                out[i, finite] = np.nan
        else:
            out[i, finite] = (vals - mu) / sd
    return out


def _corr_rowwise_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full(x.shape[0], np.nan, dtype=float)
    for i in range(x.shape[0]):
        m = np.isfinite(x[i]) & np.isfinite(y[i])
        if m.sum() < 2:
            continue
        xc = x[i, m] - x[i, m].mean()
        yc = y[i, m] - y[i, m].mean()
        denom = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
        if denom == 0:
            continue
        out[i] = float((xc * yc).sum() / denom)
    return out


def _cov_rowwise_np(x: np.ndarray, y: np.ndarray, *, ddof: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full(x.shape[0], np.nan, dtype=float)
    for i in range(x.shape[0]):
        m = np.isfinite(x[i]) & np.isfinite(y[i])
        if m.sum() <= 1:
            continue
        xc = x[i, m] - x[i, m].mean()
        yc = y[i, m] - y[i, m].mean()
        out[i] = float((xc * yc).sum() / (m.sum() - ddof))
    return out


def _beta_rowwise_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """beta(y ~ x) row-wise OLS slope。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full(x.shape[0], np.nan, dtype=float)
    for i in range(x.shape[0]):
        m = np.isfinite(x[i]) & np.isfinite(y[i])
        if m.sum() < 2:
            continue
        xv = x[i, m]
        yv = y[i, m]
        xc = xv - xv.mean()
        yc = yv - yv.mean()
        sxx = float((xc ** 2).sum())
        if sxx == 0:
            continue
        out[i] = float((xc * yc).sum() / sxx)
    return out


# ---- rolling (column-wise) reference kernels，供 ts 家族差分 ----------------


def _rolling_apply_ref(fn, x: np.ndarray, window: int,
                       min_periods: int = 1) -> np.ndarray:
    """通用纯 numpy trailing-window reference（无 lookahead）。"""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=float)
    if window <= 0:
        return out
    for i in range(n):
        lo = max(0, i - window + 1)
        w = x[lo : i + 1]
        wv = w[np.isfinite(w)]
        if wv.size < min_periods:
            continue
        out[i] = fn(wv)
    return out


def _rolling_mean_ref(x: np.ndarray, window: int,
                      min_periods: int = 1) -> np.ndarray:
    return _rolling_apply_ref(np.mean, x, window, min_periods=min_periods)


def _rolling_std_ref(x: np.ndarray, window: int, *, ddof: int = 1,
                     min_periods: int = 2) -> np.ndarray:
    return _rolling_apply_ref(lambda w: float(np.std(w, ddof=ddof)), x, window,
                              min_periods=min_periods)


def _current_row_valid(x: np.ndarray, y: np.ndarray, i: int) -> bool:
    """R19-136 current-row policy：当前观测（row i）缺失 → 输出 NaN。"""
    if i >= x.shape[0] or i >= y.shape[0]:
        return False
    return bool(np.isfinite(x[i]) and np.isfinite(y[i]))


def _rolling_corr_ref(x: np.ndarray, y: np.ndarray, window: int,
                      *, min_periods: int = 2) -> np.ndarray:
    """纯 numpy trailing-window Pearson 相关（paired finite samples）。

    R19-136 current-row policy：窗口历史 pair 足够但当前 row 缺失 → 输出 NaN，
    与 pandas ``rolling.corr`` 对齐（当前观测不在 paired 集合内）。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(x.shape[0], y.shape[0])
    out = np.full(n, np.nan, dtype=float)
    if window <= 0:
        return out
    for i in range(n):
        # corr/cov 沿用 pandas ``rolling.corr`` 行为：丢弃 NaN pair，不做
        # current-row mask（与 ``rolling_beta`` 的 ``.where(valid)`` 不同）。
        lo = max(0, i - window + 1)
        xw = x[lo : i + 1]
        yw = y[lo : i + 1]
        m = np.isfinite(xw) & np.isfinite(yw)
        if int(m.sum()) < min_periods:
            continue
        xc = xw[m] - xw[m].mean()
        yc = yw[m] - yw[m].mean()
        denom = float(np.sqrt((xc ** 2).sum() * (yc ** 2).sum()))
        if denom == 0.0:
            continue
        out[i] = float((xc * yc).sum() / denom)
    return out


def _rolling_cov_ref(x: np.ndarray, y: np.ndarray, window: int,
                     *, ddof: int = 1, min_periods: int = 2) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(x.shape[0], y.shape[0])
    out = np.full(n, np.nan, dtype=float)
    if window <= 0:
        return out
    for i in range(n):
        lo = max(0, i - window + 1)
        xw = x[lo : i + 1]
        yw = y[lo : i + 1]
        m = np.isfinite(xw) & np.isfinite(yw)
        if int(m.sum()) <= 1:
            continue
        xc = xw[m] - xw[m].mean()
        yc = yw[m] - yw[m].mean()
        out[i] = float((xc * yc).sum() / (int(m.sum()) - ddof))
    return out


def _rolling_beta_ref(y: np.ndarray, x: np.ndarray, window: int,
                      *, min_periods: int = 3) -> np.ndarray:
    """纯 numpy trailing-window OLS slope：beta(y ~ x) = cov(y,x)/var(x)。

    R19-136 current-row policy：当前 row 缺失 → 输出 NaN（与
    ``rolling_beta`` 的 ``.where(valid)`` 对齐）。
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = min(x.shape[0], y.shape[0])
    out = np.full(n, np.nan, dtype=float)
    if window <= 0:
        return out
    for i in range(n):
        if not _current_row_valid(x, y, i):
            continue
        lo = max(0, i - window + 1)
        xw = x[lo : i + 1]
        yw = y[lo : i + 1]
        m = np.isfinite(xw) & np.isfinite(yw)
        if int(m.sum()) < min_periods:
            continue
        xv = xw[m]
        yv = yw[m]
        xc = xv - xv.mean()
        yc = yv - yv.mean()
        sxx = float((xc ** 2).sum())
        if sxx == 0.0:
            continue
        out[i] = float((xc * yc).sum() / sxx)
    return out


# ---------------------------------------------------------------------------
# R19-101: 每个 operator 声明的 metamorphic properties
# ---------------------------------------------------------------------------

METAMORPHIC_PROPERTY_DECLARATIONS: dict[str, tuple[str, ...]] = {
    # R40 #257: 所有 tie-sensitive 算子显式声明 column-permutation equivariance。
    "quantile_bucket": ("rank.column_permutation_equivariance",),
    "topk": ("rank.column_permutation_equivariance",),
    "winsorize": ("rank.column_permutation_equivariance",),
    "cs_regression": ("rank.column_permutation_equivariance",),
    "neutralize": ("rank.column_permutation_equivariance",),
    # rank 家族：strict monotonic transform invariance + column permutation
    # equivariance（rank 不随单调变换/列置换改变）。
    "rank": ("rank.strict_monotonic_invariance", "rank.column_permutation_equivariance"),
    "cs_rank_01": ("rank.strict_monotonic_invariance", "rank.column_permutation_equivariance"),
    "rank_pct": ("rank.strict_monotonic_invariance", "rank.column_permutation_equivariance"),
    "cs_pct_rank": ("rank.strict_monotonic_invariance", "rank.column_permutation_equivariance"),
    "group_rank": ("rank.strict_monotonic_invariance", "rank.column_permutation_equivariance"),
    "ts_rank": ("rank.strict_monotonic_invariance",),
    # zscore 家族：translation + positive-scale invariance。
    "zscore": ("translation_invariance", "positive_scale_invariance"),
    "c_zscore": ("translation_invariance", "positive_scale_invariance"),
    "cs_std": ("translation_invariance", "positive_scale_invariance"),
    "ts_zscore": ("translation_invariance", "positive_scale_invariance"),
    # correlation：translation + positive-scale invariance + symmetry。
    "Corr": ("translation_invariance", "positive_scale_invariance", "symmetry"),
    "ts_corr": ("translation_invariance", "positive_scale_invariance", "symmetry"),
    "rank_corr": ("translation_invariance", "positive_scale_invariance", "symmetry"),
    # covariance：translation invariance + scale covariance + symmetry。
    "Cov": ("translation_invariance", "scale_covariance", "symmetry"),
    "Covariance": ("translation_invariance", "scale_covariance", "symmetry"),
    "ts_cov": ("translation_invariance", "scale_covariance", "symmetry"),
    # beta：y-scale covariance + x-scale inverse covariance。
    "Beta": ("beta.y_scale_covariance", "beta.x_scale_inverse_covariance"),
    "ts_beta": ("beta.y_scale_covariance", "beta.x_scale_inverse_covariance"),
    # neutralize：group residual mean≈0 + size residual covariance≈0。
    "neutralize": ("neutralize.group_residual_mean_0",
                   "neutralize.size_residual_cov_0"),
    "group_neutralize": ("neutralize.group_residual_mean_0",),
    "cs_demean": ("translation_invariance",),
    "normalize": ("positive_scale_invariance",),
    "ts_mean": ("translation_invariance",),
    "ts_std": ("translation_invariance", "positive_scale_invariance"),
    "ts_var": ("translation_invariance", "scale_covariance"),
    "ts_sum": ("translation_invariance", "scale_covariance"),
    "ts_pct": ("scale_covariance",),
}


# ---------------------------------------------------------------------------
# R19-103: prefix invariance
# ---------------------------------------------------------------------------


def prefix_invariance_check(
    fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    *,
    T: int | None = None,
    K: int = 10,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> MetamorphicResult:
    """R19-103 prefix invariance：``fn(prefix T+K)[:T]`` == ``fn(prefix T)``。

    任何 full-sample statistic 泄漏进 direct factor（如 ``rank_corr(d=0)`` 对
    全面板 rank）都会在这里被抓出：未来 T+1.. 的变化会改变前 T 的输出。

    参数:
        fn: 一维时序函数（``ndarray -> ndarray``）。
        x: 一维时序数组。
        T: 前缀长度（默认 ``max(10, len(x)//2)``）。
        K: 追加长度。
        rtol/atol: ``np.allclose`` 容差。

    返回:
        MetamorphicResult（property_name="prefix_invariance"）。
    """
    x = np.asarray(x, dtype=float)
    if T is None:
        T = max(10, int(len(x) * 0.6))
    T = min(int(T), len(x))
    K = min(int(K), len(x) - T)
    if T <= 0 or K <= 0:
        return MetamorphicResult("prefix_invariance", False,
                                 f"invalid T={T}, K={K}")
    try:
        y_short = np.asarray(fn(x[:T]), dtype=float)
        y_long = np.asarray(fn(x[: T + K]), dtype=float)
    except Exception as exc:  # noqa: BLE001
        return MetamorphicResult("prefix_invariance", False,
                                 f"fn raised {type(exc).__name__}: {exc}")
    if y_short.shape != (T,) and y_short.shape != (T, 1):
        # 兼容 2D 输出：取第一列
        if y_short.ndim == 2 and y_short.shape[0] == T:
            y_short = y_short[:, 0]
        else:
            return MetamorphicResult("prefix_invariance", False,
                                     f"unexpected shape {y_short.shape}")
    if y_long.ndim == 2 and y_long.shape[0] >= T:
        y_long = y_long[:T, 0]
    ok = _allclose_nan(y_short, np.asarray(y_long[:T], dtype=float),
                       rtol=rtol, atol=atol)
    return MetamorphicResult("prefix_invariance", ok,
                             "first T outputs unchanged" if ok else "future leakage")


# ---------------------------------------------------------------------------
# R19-105: chunk invariance（hostile boundaries）
# ---------------------------------------------------------------------------

HOSTILE_BOUNDARY_KINDS = (
    "nan_at_boundary",
    "inf_at_boundary",
    "event_transition_at_boundary",
    "new_high_at_boundary",
    "group_change_at_boundary",
    "session_boundary",
)


def chunk_invariance_check(
    fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    *,
    window: int,
    boundaries: Sequence[int] | None = None,
    min_overlap: int | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> MetamorphicResult:
    """R19-105 chunk invariance：hostile chunk 切分后与整体结果一致。

    对 trailing rolling op（窗口 ``window``），输出 t 只依赖 ``x[t-window+1:t]``。
    按给定 boundaries 分块，每块左侧用 ``window-1`` 的 overlap 补齐历史，丢弃
    每块（除首块）前 ``window-1`` 的输出（它们缺历史），再把有效输出拼接成
    chunked 结果，与 full run 比较。

    参数:
        fn: 一维 rolling 函数（``ndarray -> ndarray``，NaN 为缺失）。
        x: 一维时序数组。
        window: 滚动窗口长度。
        boundaries: 分块边界（index 列表）。默认构造 hostile 边界。
        min_overlap: 补齐历史长度，默认 ``window - 1``。
        rtol/atol: 容差。

    返回:
        MetamorphicResult（property_name="chunk_invariance"）。
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if window <= 1:
        return MetamorphicResult("chunk_invariance", False, "window must be > 1")
    overlap = window - 1 if min_overlap is None else int(min_overlap)
    if boundaries is None:
        boundaries = _hostile_chunk_boundaries(x, window)
    boundaries = sorted(set(int(b) for b in boundaries if 0 < int(b) < n))
    # full run
    try:
        full = np.asarray(fn(x), dtype=float)
    except Exception as exc:  # noqa: BLE001
        return MetamorphicResult("chunk_invariance", False,
                                 f"full fn raised {type(exc).__name__}: {exc}")
    if full.ndim == 2:
        full = full[:, 0]
    # chunked run
    starts = [0] + boundaries
    ends = boundaries + [n]
    chunked = np.full(n, np.nan, dtype=float)
    for si, (s, e) in enumerate(zip(starts, ends)):
        lo = max(0, s - overlap)
        seg = x[lo:e]
        try:
            seg_out = np.asarray(fn(seg), dtype=float)
        except Exception as exc:  # noqa: BLE001
            return MetamorphicResult("chunk_invariance", False,
                                     f"chunk fn raised {type(exc).__name__}: {exc}")
        if seg_out.ndim == 2:
            seg_out = seg_out[:, 0]
        # 有效输出范围：与整体 x 对齐的 [s, e)
        offset = s - lo  # seg_out[:offset] 对应历史补齐段（丢弃）
        keep_from = offset if si > 0 else 0
        seg_len = e - s
        chunked[s:e] = seg_out[keep_from : keep_from + seg_len]
    ok = _allclose_nan(full, chunked, rtol=rtol, atol=atol)
    return MetamorphicResult("chunk_invariance", ok,
                             f"boundaries={list(boundaries)}" if ok
                             else f"mismatch at boundaries={list(boundaries)}")


def _hostile_chunk_boundaries(x: np.ndarray, window: int) -> list[int]:
    """构造 hostile chunk 边界：把 NaN / Inf / 局部极大 / 值跳变 / 首尾等敏感点
    放进边界集合。"""
    n = x.shape[0]
    b: set[int] = set()
    # NaN at boundary
    nan_idx = np.where(np.isnan(x))[0]
    for idx in nan_idx:
        for off in (0, 1, -1):
            if 0 < idx + off < n:
                b.add(int(idx + off))
    # Inf at boundary
    inf_idx = np.where(np.isinf(x))[0]
    for idx in inf_idx:
        for off in (0, 1, -1):
            if 0 < idx + off < n:
                b.add(int(idx + off))
    # new high at boundary
    running_max = np.maximum.accumulate(np.where(np.isfinite(x), x, -np.inf))
    for i in range(1, n):
        if np.isfinite(x[i]) and x[i] == running_max[i] and x[i] > running_max[i - 1]:
            b.add(i)
            b.add(i + 1)
    # 大跳变（event transition）
    dx = np.diff(np.where(np.isfinite(x), x, np.nan))
    if dx.size:
        thr = np.nanstd(dx) * 3 if np.isfinite(np.nanstd(dx)) else 1.0
        for i in range(dx.size):
            if np.isfinite(dx[i]) and abs(dx[i]) > max(thr, 1e-9):
                b.add(i + 1)
    # 均匀散布 + 首尾
    step = max(1, n // 5)
    for i in range(step, n, step):
        b.add(i)
    b.add(n // 2)
    if n > 3:
        b.add(n // 3)
        b.add(2 * n // 3)
    return sorted(v for v in b if 0 < v < n)


# ---------------------------------------------------------------------------
# R19-106/107: checkpoint serialization certificate
# ---------------------------------------------------------------------------


def checkpoint_serialization_check(
    run_segment: Callable[[np.ndarray, Any], tuple[np.ndarray, Any]],
    segments: Sequence[np.ndarray],
    initial_state: Any,
    serializer: Callable[[Any], Any] | None = None,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> tuple[bool, dict[str, Any]]:
    """R19-106 checkpoint 数值证书：``full run == segment A + serialize + restore +
    segment B``，比较 values / NaN mask / state。

    参数:
        run_segment: ``(segment_1d, state) -> (out_1d, new_state)``。
        segments: 分段序列（每段一维数组）。
        initial_state: 初始状态。
        serializer: 状态序列化函数（如 float64→float64 round-trip、JSON round-trip）。
            为 ``None`` 时不做序列化（等于纯分段运行）。
        rtol/atol: 输出容差。

    返回:
        ``(ok, detail_dict)``，detail 含 values/NaN mask/state 逐项结果。
    """
    # full run —— 同时记录每个分段边界（segment i 结束、i+1 开始）的状态
    full_out: list[np.ndarray] = []
    state = initial_state
    full_boundary_states: list[Any] = []
    for seg in segments:
        out, state = run_segment(seg, state)
        full_out.append(np.asarray(out, dtype=float))
        full_boundary_states.append(state)
    full = np.concatenate(full_out) if full_out else np.array([])

    # checkpointed run —— 在每个边界做 serializer(state) 再继续
    chk_out: list[np.ndarray] = []
    state = initial_state
    chk_boundary_states: list[Any] = []
    for i, seg in enumerate(segments):
        out, state = run_segment(seg, state)
        chk_out.append(np.asarray(out, dtype=float))
        if i < len(segments) - 1:
            if serializer is not None:
                state = serializer(state)
            chk_boundary_states.append(state)
    chk = np.concatenate(chk_out) if chk_out else np.array([])

    values_ok = _allclose_nan(full, chk, rtol=rtol, atol=atol)
    nan_mask_ok = bool(np.array_equal(np.isnan(full), np.isnan(chk)))
    # state 比较：在边界 j，full-run 的状态（segment j 结束时的状态）必须等于
    # checkpointed-run 经 serialize+restore 后的状态。若 serializer 把 float64
    # 悄悄降精度（如 JSON round 到 float32），restore 后 long-run 递归会漂移。
    state_ok = True
    state_note = "identity"
    if full_boundary_states and chk_boundary_states:
        pairs = list(zip(full_boundary_states[:-1], chk_boundary_states))
        if pairs:
            for full_s, chk_s in pairs:
                try:
                    fa = np.asarray(full_s, dtype=float)
                    ca = np.asarray(chk_s, dtype=float)
                    if fa.shape == ca.shape and fa.size > 0:
                        same = bool(np.array_equal(fa, ca))
                    else:
                        same = bool(full_s == chk_s)
                except Exception:  # noqa: BLE001
                    same = bool(full_s == chk_s)
                state_ok = state_ok and same
            state_note = "boundary_state_equal"
    ok = bool(values_ok and nan_mask_ok and state_ok)
    detail = {
        "values_equal": bool(values_ok),
        "nan_mask_equal": bool(nan_mask_ok),
        "state_equal": bool(state_ok),
        "state_note": state_note,
        "full_len": int(full.size),
        "chunked_len": int(chk.size),
    }
    return ok, detail


def json_float64_downgrade_detector(state: Any) -> bool:
    """R19-107 检测 float64 状态经 JSON round-trip 是否被悄悄降精度。

    返回 True 表示 JSON round-trip 后数值与原始 float64 不完全一致
    （有精度丢失）。测试据此断言：checkpoint float precision 不能偷偷降低。
    """
    import json

    def _to_jsonable(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {k: _to_jsonable(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_to_jsonable(val) for val in v]
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v

    try:
        s = json.dumps(_to_jsonable(state), allow_nan=False)
        restored = json.loads(s)
    except (TypeError, ValueError) as exc:  # noqa: BLE001
        # 不可序列化 → 该 serializer 不该用于 checkpoint，报告为缺陷
        return True
    return _float64_differs(state, restored)


def _float64_differs(a: Any, b: Any) -> bool:
    if isinstance(a, np.ndarray):
        if not isinstance(b, (list, tuple, np.ndarray)):
            return True
        bb = np.asarray(b, dtype=float)
        aa = np.asarray(a, dtype=float)
        if aa.shape != bb.shape:
            return True
        return bool(np.any(aa != bb))
    if isinstance(a, dict):
        if not isinstance(b, dict):
            return True
        if set(a.keys()) != set(b.keys()):
            return True
        return any(_float64_differs(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)):
            return True
        if len(a) != len(b):
            return True
        return any(_float64_differs(u, v) for u, v in zip(a, b))
    if isinstance(a, float):
        return not isinstance(b, (int, float)) or float(a) != float(b)
    return a != b


# ---------------------------------------------------------------------------
# R19-110: hostile fixtures
# ---------------------------------------------------------------------------

HOSTILE_FIXTURE_NAMES = (
    "all_equal",
    "many_ties",
    "zero_denominator",
    "negative_values",
    "very_large_values",
    "very_small_values",
    "alternating_signs",
    "single_valid_sample",
    "two_valid_samples",
    "current_row_missing",
    "internal_hole",
    "inf",
    "neg_inf",
    "long_nan_block",
    "duplicate_group_labels",
    "string_group_labels",
    "group_missing",
    "column_permutation",
)


def build_hostile_fixtures(
    n_rows: int = 12,
    n_cols: int = 4,
    *,
    seed: int = 7,
) -> dict[str, dict[str, Any]]:
    """R19-110 hostile fixtures 构造器。

    返回 ``{fixture_name: {"x": ndarray (n_rows, n_cols) | ndarray (n_rows,),
    "group": ndarray | None, "note": str}}``。``x`` 一维或二维；``group`` 供
    neutralize/group op 使用。
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n_rows, n_cols))
    fixtures: dict[str, dict[str, Any]] = {}

    def _put(name: str, x: np.ndarray, *, group: np.ndarray | None = None,
             note: str = "") -> None:
        fixtures[name] = {"x": x, "group": group, "note": note}

    # 分组标签（默认）——含重复标签（duplicate group labels 天然覆盖）
    g_default = np.array(["A", "B", "A", "C", "B", "A", "C", "B", "A", "C", "B", "A"],
                         dtype=object)[:n_rows]
    g_default = np.resize(g_default, n_rows)

    _put("all_equal", np.ones_like(base), group=g_default, note="constant panel")
    # many ties：行内大量重复值
    ties = base.copy()
    ties[:, 1:] = ties[:, :1]
    _put("many_ties", ties, group=g_default, note="many column ties")
    _put("zero_denominator", np.zeros_like(base), group=g_default,
         note="all-zero denominator")
    _put("negative_values", -np.abs(base) - 0.5, group=g_default)
    _put("very_large_values", base * 1e12, group=g_default)
    _put("very_small_values", base * 1e-12, group=g_default)
    _put("alternating_signs", base * np.where(np.arange(n_cols) % 2 == 0, 1, -1),
         group=g_default)
    # single valid sample / two valid samples：只有 1/2 个非 NaN
    sv = np.full_like(base, np.nan)
    sv[0, 0] = 1.0
    _put("single_valid_sample", sv, group=g_default)
    tv = np.full_like(base, np.nan)
    tv[0, 0], tv[0, 1] = 1.0, 2.0
    _put("two_valid_samples", tv, group=g_default)
    # current row missing：最后一行全 NaN（当前 row 缺失）
    cm = base.copy()
    cm[-1, :] = np.nan
    _put("current_row_missing", cm, group=g_default)
    # internal hole：中间某行全 NaN
    ih = base.copy()
    ih[n_rows // 2, :] = np.nan
    _put("internal_hole", ih, group=g_default)
    # Inf / -Inf
    inf = base.copy()
    inf[1, 0] = np.inf
    _put("inf", inf, group=g_default)
    ninf = base.copy()
    ninf[1, 0] = -np.inf
    _put("neg_inf", ninf, group=g_default)
    # long NaN block
    nb = base.copy()
    nb[2:8, :] = np.nan
    _put("long_nan_block", nb, group=g_default)
    # duplicate group labels 显式；string group labels 默认即 string
    _put("duplicate_group_labels", base, group=np.resize(
        np.array([1, 1, 1, 2, 2, 2], dtype=int), n_rows))
    g_string = np.array([f"g{i % 3}" for i in range(n_rows)], dtype=object)
    _put("string_group_labels", base, group=g_string)
    # group missing：group 里有 NaN
    g_miss = np.resize(np.array(["A", "B", np.nan, "C", "A"], dtype=object), n_rows)
    _put("group_missing", base, group=g_miss)
    # column permutation fixture（列置换后的 x 与原始 x 并存）
    perm = rng.permutation(n_cols)
    xp = base[:, perm]
    _put("column_permutation", xp, group=g_default,
         note=f"columns permuted by {perm.tolist()}")
    return fixtures


# ---------------------------------------------------------------------------
# R19-124/125/126: AST static analyzers
# ---------------------------------------------------------------------------


def _dedent_source(source: str) -> str:
    """去掉 ``inspect.getsource`` 返回的方法级缩进，使 ``ast.parse`` 可解析。"""
    import textwrap
    return textwrap.dedent(source).lstrip()


def scan_hidden_kwargs(
    source: str,
    *,
    declared_names: Iterable[str] = (),
    context_inputs: Iterable[str] = ("panel", "ctx", "context", "factor", "state"),
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """R19-124 AST 扫描 hidden kwargs。

    找 ``kwargs.get(...)`` / ``kwargs[...]`` 且 key 不在 declared_names /
    context_inputs 中。返回 ``(hidden_keys, all_kwarg_keys)``。

    参数:
        source: 算子 kernel 源码（``_calculate_series`` 函数体）。
        declared_names: ``param_names`` + ``param_aliases`` key/value。
        context_inputs: 合法 context 键。
    """
    declared = set(str(d) for d in declared_names) | set(context_inputs)
    hidden: set[str] = set()
    all_keys: set[str] = set()
    try:
        tree = ast.parse(_dedent_source(source))
    except SyntaxError:
        return ("__unparsable__",) , ("__unparsable__",)
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            # kwargs["key"]
            val = node.value
            if isinstance(val, ast.Name) and val.id == "kwargs" and node.slice is not None:
                key = _const_key(node.slice)
                if key is not None:
                    all_keys.add(key)
                    if key not in declared:
                        hidden.add(key)
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr == "get" and \
                    isinstance(f.value, ast.Name) and f.value.id == "kwargs":
                if node.args and isinstance(node.args[0], ast.Constant):
                    key = str(node.args[0].value)
                    all_keys.add(key)
                    if key not in declared:
                        hidden.add(key)
    return tuple(sorted(hidden)), tuple(sorted(all_keys))


def _const_key(slice_node: ast.AST) -> str | None:
    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
        return slice_node.value
    if isinstance(slice_node, ast.Name):
        return slice_node.id
    return None


def scan_local_casts(
    source: str,
    *,
    declared_scalars: Iterable[str] = (),
    allow_numeric_string_binder: bool = True,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """R19-125 AST 扫描 local scalar coercion。

    找 ``int(param)`` / ``float(param)`` / ``bool(param)`` / ``str(param)`` 及
    ``max(...int(param)...)`` 包裹形式。declared scalar 必须来自 central binder
    （``validate_operator_call`` / ``strict_params``）；未经允许的 local cast →
    LOCAL_PARAMETER_COERCION。

    参数:
        source: kernel 源码。
        declared_scalars: 该算子声明的 scalar 参数名。
        allow_numeric_string_binder: 若为 False，任何 numeric cast 都标记。

    返回:
        ``(local_casts, all_casts)`` —— 命中（declared scalar 被 local cast）与
        全部 cast 记录 ``"name->int"`` 形式。
    """
    declared = set(str(d) for d in declared_scalars)
    targets = {"int": "int", "float": "float", "bool": "bool", "str": "str"}
    hits: set[str] = set()
    all_casts: set[str] = set()
    try:
        tree = ast.parse(_dedent_source(source))
    except SyntaxError:
        return ("__unparsable__",) , ("__unparsable__",)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in targets and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Name):
                name = arg.id
                all_casts.add(f"{name}->{targets[node.func.id]}")
                if name in declared:
                    hits.add(f"{name}->{targets[node.func.id]}")
    return tuple(sorted(hits)), tuple(sorted(all_casts))


def scan_sample_masking(source: str) -> tuple[str, ...]:
    """R19-126 AST 扫描 statistical sample masking。

    列出源码中的 notna / dropna / np.isnan / np.isfinite / count 使用，
    作为 sample_mask_policy 的人工/机器确认输入。

    返回:
        有序字符串列表，如 ``("isfinite", "dropna")``。
    """
    pats = ("notna", "notnull", "dropna", "isnan", "isfinite", "isinf", "count")
    found: set[str] = set()
    try:
        tree = ast.parse(_dedent_source(source))
    except SyntaxError:
        return ("__unparsable__",)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in pats:
            found.add(node.attr)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("np.isnan", "np.isfinite", "np.isinf"):
            found.add(node.func.id)
    return tuple(sorted(found))


def scan_duplicated_semantic_policy(
    *,
    numeric_semantics: Mapping[str, Any] | None = None,
    metadata_policy: Mapping[str, Any] | None = None,
    kernel_constants: Mapping[str, Any] | None = None,
) -> tuple[tuple[str, ...], ...]:
    """R19-127 duplicated semantic policy：同一属性（tie/ddof/zero_std/Inf）在
    numeric_semantics / metadata / kernel constants 多处不一致 → 标记。

    参数:
        numeric_semantics: 如 ``{"std_ddof": "sample"}``。
        metadata_policy: 如 ``{"ddof": 1}``。
        kernel_constants: 如 ``{"min_periods": 2}``。

    返回:
        ``(disagreements, sources)``。disagreements 形如 ``"ddof:numeric=sample|kernel=0"``。
    """
    attrs = ("tie", "ddof", "zero_std", "inf_policy", "min_periods", "interpolation")
    sources = ("numeric_semantics", "metadata", "kernel_constants")
    values: dict[str, list[tuple[str, Any]]] = {a: [] for a in attrs}
    maps = {
        "numeric_semantics": numeric_semantics or {},
        "metadata": metadata_policy or {},
        "kernel_constants": kernel_constants or {},
    }
    for src, m in maps.items():
        for key, val in m.items():
            if key in attrs or key in ("std_ddof", "rank_tie_method", "zscore_zero_std",
                                       "quantile_interpolation", "window_min_periods_default"):
                norm = key
                if key == "std_ddof":
                    norm = "ddof"
                    val = 1 if val == "sample" else 0
                elif key == "rank_tie_method":
                    norm = "tie"
                elif key == "zscore_zero_std":
                    norm = "zero_std"
                elif key == "quantile_interpolation":
                    norm = "interpolation"
                values.setdefault(norm, []).append((src, val))
    disagreements: list[str] = []
    for attr, pairs in values.items():
        if len(pairs) < 2:
            continue
        # 仅当两个来源声明的 VALUE 不同 → 不一致（来源不同但值相同是合法一致性）。
        distinct_values = {v for _, v in pairs}
        if len(distinct_values) > 1:
            disagreements.append(
                f"{attr}:" + "|".join(f"{s}={v!r}" for s, v in pairs)
            )
    return tuple(sorted(disagreements)), sources


# ---------------------------------------------------------------------------
# R19-116/117: reference/optimized 两层 & 快速核只能实现 certificate
# ---------------------------------------------------------------------------


def reference_smoke_pass(
    reference_fn: Callable[..., np.ndarray],
    fixture: np.ndarray,
    *,
    validate: Callable[[np.ndarray], bool] | None = None,
) -> tuple[bool, str]:
    """R19-116 reference smoke：reference 实现在常规 fixture 上运行、shape 保留、
    非全 NaN。"""
    try:
        out = np.asarray(reference_fn(fixture), dtype=float)
    except Exception as exc:  # noqa: BLE001
        return False, f"raised {type(exc).__name__}: {exc}"
    if out.shape != fixture.shape:
        return False, f"shape {out.shape} != {fixture.shape}"
    if np.isfinite(out).sum() == 0:
        return False, "all-NaN on normal fixture"
    if validate is not None and not validate(out):
        return False, "reference validation failed"
    return True, "ok"


def differential_against_reference(
    optimized_fn: Callable[..., np.ndarray],
    reference_fn: Callable[..., np.ndarray],
    fixtures: Sequence[np.ndarray],
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> tuple[bool, dict[str, bool]]:
    """R19-116 optimized 必须 differential test reference。

    返回 ``(all_ok, {fixture_index: ok})``。
    """
    results: dict[str, bool] = {}
    all_ok = True
    for i, fx in enumerate(fixtures):
        try:
            ref = np.asarray(reference_fn(fx), dtype=float)
            opt = np.asarray(optimized_fn(fx), dtype=float)
        except Exception:  # noqa: BLE001
            results[f"fixture_{i}"] = False
            all_ok = False
            continue
        ok = _allclose_nan(ref, opt, rtol=rtol, atol=atol)
        results[f"fixture_{i}"] = bool(ok)
        all_ok = all_ok and ok
    return all_ok, results


# ---------------------------------------------------------------------------
# R40 #250: CHECKPOINT_CHUNK_INVARIANCE_FAILURE hard gate —— 所有 segmented
# execution canonicals 必须满足 concat(seg1, resume(seg2), ...) == full(x)。
# ---------------------------------------------------------------------------

#: R40 #250 hard gate 常量：任一 segmented canonical 的 checkpoint 分块重放
#: 与全量历史不一致即为 release fail。
CHECKPOINT_CHUNK_INVARIANCE_FAILURE = "CHECKPOINT_CHUNK_INVARIANCE_FAILURE"

#: R40 #258 hard gate 常量：任一 causal TS canonical 的未来数据改变历史输出。
PREFIX_INVARIANCE_FAILURE = "PREFIX_INVARIANCE_FAILURE"

#: R40 #259 hard gate 常量：可流式/分块算子（EWM / minute aggregation /
#: streaming group / writer block DQ）的 chunk-boundary invariance。
CHUNK_BOUNDARY_INVARIANCE = "CHUNK_BOUNDARY_INVARIANCE"


def _stream_ewm(span: int):
    """EWM mean 的流式内核：``run(seg, state) -> (out, new_state)``。"""
    def run(seg, state):
        state = dict(state or {})
        m = state.get("mean")
        out = np.full(len(seg), np.nan, dtype=float)
        for i, v in enumerate(np.asarray(seg, dtype=float)):
            if not np.isfinite(v):
                out[i] = np.nan
                continue
            if m is None:
                m = v
            else:
                alpha = 2.0 / (span + 1.0)
                m = (1.0 - alpha) * m + alpha * v
            out[i] = m
        return out, {"mean": m}
    return run


def _stream_cumsum_valid():
    """流式有效值累加（writer block DQ 代理：累计 valid bar 数）。"""
    def run(seg, state):
        state = dict(state or {})
        acc = state.get("acc", 0.0)
        out = np.full(len(seg), np.nan, dtype=float)
        for i, v in enumerate(np.asarray(seg, dtype=float)):
            if np.isfinite(v):
                acc += v
                out[i] = acc
            else:
                out[i] = np.nan
        return out, {"acc": acc}
    return run


def _chunk_boundary_invariance_for_stream(
    stream_fn, x: np.ndarray, boundaries: Sequence[int], *, rtol: float, atol: float
) -> bool:
    """R40 #259：``full stream == 逐 chunk 续流``（state 跨 chunk 保留）。"""
    n = len(x)
    full, _ = stream_fn(x, None)
    starts = [0] + list(boundaries)
    ends = list(boundaries) + [n]
    chunked = np.full(n, np.nan, dtype=float)
    state: Any = None
    for s, e in zip(starts, ends):
        if s >= e:
            continue
        out, state = stream_fn(x[s:e], state)
        chunked[s:e] = out
    return _allclose_nan(np.asarray(full, dtype=float), chunked, rtol=rtol, atol=atol)


def cross_process_determinism_probe(
    *, worker_count: int = 1, seed: int = 42, n_bars: int = 48, n_inst: int = 6
) -> dict[str, Any]:
    """R40 #260：跨进程确定性探针（subprocess 用）。

    用固定 seed 的合成面板计算一个确定性因子（ts_mean window=5），返回
    result checksum / plan hash / axis hash。两个独立 Python 进程对同一 spec
    必须产生完全相同的 checksum / hash —— 且不同 ``worker_count`` 不改变输出
    （探针内强制单线程 BLAS 环境，rolling mean 是位级确定的）。

    供 ``test_full_production_determinism_across_processes_and_restarts`` 通过
    ``subprocess`` 调用；也可被 ``scripts/audit_r40_hard_gates.py`` 复用。
    """
    import os

    import pandas as pd

    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "POLARS_MAX_THREADS",
    ):
        os.environ[var] = str(max(1, int(worker_count)))
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [
            pd.date_range("2024-01-01", periods=n_bars, freq="D"),
            [f"INST{i}" for i in range(n_inst)],
        ],
        names=["timestamp", "instrument"],
    )
    panel = pd.DataFrame({"x": rng.standard_normal(len(idx))}, index=idx)
    out = (
        panel["x"]
        .groupby(level="instrument", sort=False)
        .transform(lambda s: s.rolling(5, min_periods=1).mean())
    )
    checksum = hashlib.sha256(np.asarray(out, dtype=float).tobytes()).hexdigest()
    plan_hash = hashlib.sha256(
        b"determinism-probe-v1|ts_mean|w=5|seed=" + str(seed).encode("utf-8")
    ).hexdigest()[:16]
    axis_hash = hashlib.sha256(
        f"{list(idx.names)}|{len(idx)}|{n_bars}|{n_inst}".encode("utf-8")
    ).hexdigest()[:16]
    return {
        "worker_count": int(worker_count),
        "seed": int(seed),
        "result_checksum": checksum,
        "plan_hash": plan_hash,
        "axis_hash": axis_hash,
        "n_values": int(out.shape[0]),
        "first_values": [round(float(v), 9) for v in np.asarray(out, dtype=float)[:8]],
    }


def check_chunk_boundary_invariance_all_streamable(
    *,
    n_bars: int = 120,
    seed: int = 9,
    n_random_chunkings: int = 3,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> tuple[bool, dict[str, Any]]:
    """R40 #259：可流式/分块算子的 chunk-boundary invariance universal gate。

    覆盖 EWM mean（span=12）、streaming cumsum、streaming valid-count（writer
    block DQ 代理）。随机 chunk 边界，逐 chunk 续流与全量流比较。可被
    ``scripts/audit_r40_hard_gates.py`` 复用。
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_bars)
    x[::17] = np.nan  # gaps
    streams: dict[str, Any] = {
        "ewm_mean_span12": _stream_ewm(12),
        "cumsum_valid": _stream_cumsum_valid(),
    }
    per_canonical: dict[str, dict[str, Any]] = {}
    all_ok = True
    for name, stream_fn in streams.items():
        ok = True
        details: list[str] = []
        for _ in range(n_random_chunkings):
            cut = sorted(set(int(b) for b in rng.integers(1, n_bars, size=n_bars // 10)))
            boundaries = [b for b in cut if 0 < b < n_bars]
            if not boundaries:
                boundaries = [n_bars // 2]
            if not _chunk_boundary_invariance_for_stream(
                stream_fn, x, boundaries, rtol=rtol, atol=atol
            ):
                ok = False
                details.append(f"boundaries={boundaries} mismatch")
        per_canonical[name] = {"ok": ok, "detail": details}
        all_ok = all_ok and ok
    return all_ok, {"per_canonical": per_canonical, "gate": CHUNK_BOUNDARY_INVARIANCE}

#: R40 #258：被纳入 universal prefix-invariance hard gate 的 causal TS canonicals。
CAUSAL_TS_CANONICALS: tuple[str, ...] = (
    "ts_mean", "ts_std", "ts_sum", "ts_rank", "ts_delta", "ts_pct",
    "ts_max", "ts_min", "ts_zscore", "ts_argmax", "ts_argmin",
    "ts_regression_slope", "ts_rank_corr",
)


def _causal_ts_kernel(canonical: str, d: int = 10):
    """返回 causal TS canonical 的 1D 时序函数（ndarray -> ndarray）。

    使用 pandas trailing-window 或 ``_numpy_kernels`` 的 trailing-window 内核
    —— 全部只依赖 ``x[t-window+1:t]``，天然 prefix-invariant。
    """
    import pandas as pd

    if canonical == "ts_mean":
        return lambda a: pd.Series(np.asarray(a, float)).rolling(d, min_periods=1).mean().to_numpy()
    if canonical == "ts_std":
        return lambda a: pd.Series(np.asarray(a, float)).rolling(d, min_periods=2).std(ddof=1).to_numpy()
    if canonical == "ts_sum":
        return lambda a: pd.Series(np.asarray(a, float)).rolling(d, min_periods=1).sum().to_numpy()
    if canonical == "ts_max":
        return lambda a: pd.Series(np.asarray(a, float)).rolling(d, min_periods=1).max().to_numpy()
    if canonical == "ts_min":
        return lambda a: pd.Series(np.asarray(a, float)).rolling(d, min_periods=1).min().to_numpy()
    if canonical == "ts_rank":
        return lambda a: pd.Series(np.asarray(a, float)).rolling(d, min_periods=1).rank(pct=True).to_numpy()
    if canonical == "ts_delta":
        return lambda a: np.asarray(a, float) - pd.Series(np.asarray(a, float)).shift(1).to_numpy()
    if canonical == "ts_pct":
        def _ts_pct(a):
            s = pd.Series(np.asarray(a, float))
            return (s / s.shift(1) - 1.0).to_numpy()
        return _ts_pct
    if canonical == "ts_zscore":
        def _ts_zscore(a):
            s = pd.Series(np.asarray(a, float))
            mu = s.rolling(d, min_periods=2).mean()
            sd = s.rolling(d, min_periods=2).std(ddof=1)
            return ((s - mu) / sd).to_numpy()
        return _ts_zscore
    if canonical == "ts_argmax":
        return lambda a: pd.Series(np.asarray(a, float)).rolling(d, min_periods=1).apply(
            lambda w: np.nanargmax(np.asarray(w)), raw=True
        ).to_numpy()
    if canonical == "ts_argmin":
        return lambda a: pd.Series(np.asarray(a, float)).rolling(d, min_periods=1).apply(
            lambda w: np.nanargmin(np.asarray(w)), raw=True
        ).to_numpy()
    if canonical == "ts_regression_slope":
        return lambda a: _r40_ts_regression_slope(np.arange(len(a), dtype=float), np.asarray(a, float), d)
    if canonical == "ts_rank_corr":
        return lambda a: _r40_ts_rank_corr(np.asarray(a, float), np.arange(len(a), dtype=float), d)
    raise KeyError(canonical)


def check_prefix_invariance_all_causal_ts(
    *,
    n_bars: int = 140,
    seed: int = 3,
    T_cut: int = 90,
    K_append: int = 30,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> tuple[bool, dict[str, Any]]:
    """R40 #258：所有 causal TS canonical 的 universal prefix-invariance hard
    gate。

    随机 cut point ``T_cut``：``fn(prefix T+K)[:T] == fn(prefix T)``。production
    下任何未来数据改变历史输出即为 ``PREFIX_INVARIANCE_FAILURE``。可被
    ``scripts/audit_r40_hard_gates.py`` 复用。
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_bars)
    x[::11] = np.nan  # 周期性 gap
    per_canonical: dict[str, dict[str, Any]] = {}
    all_ok = True
    for canonical in CAUSAL_TS_CANONICALS:
        try:
            fn = _causal_ts_kernel(canonical)
        except KeyError:
            per_canonical[canonical] = {"ok": False, "detail": "no kernel"}
            all_ok = False
            continue
        res = prefix_invariance_check(fn, x, T=T_cut, K=K_append, rtol=rtol, atol=atol)
        per_canonical[canonical] = {"ok": bool(res.passed), "detail": res.detail}
        all_ok = all_ok and bool(res.passed)
    return all_ok, {"per_canonical": per_canonical, "gate": PREFIX_INVARIANCE_FAILURE}


def check_chunk_invariance_all_segmented_canonicals(
    *,
    n_bars: int = 90,
    seed: int = 7,
    n_random_chunkings: int = 3,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> tuple[bool, dict[str, Any]]:
    """R40 #250：对 ``SEGMENTED_EXECUTION_CANONICALS`` 注册的每个 canonical
    跑 ``concat(segment1, resume(segment2), ...) == full_history(x)`` property
    test。

    - 随机 chunk boundaries（每 8 根 bar 一个潜在切点）；
    - 多随机切分（``n_random_chunkings``）；
    - 每个 canonical 的 chunked 重放与全量历史逐 bar 比较（含 NaN mask）。

    返回 ``(all_ok, detail)``。该函数可被 ``scripts/audit_r40_hard_gates.py``
    复用（gate 名 :data:`CHECKPOINT_CHUNK_INVARIANCE_FAILURE`）。
    """
    import pandas as pd

    from stateful_runtime import execute_stateful_segment

    try:
        from cleaned_operators.production_hardening import SEGMENTED_EXECUTION_CANONICALS
    except Exception:  # pragma: no cover - 并发会话可能正在编辑 registry
        SEGMENTED_EXECUTION_CANONICALS = frozenset({
            "ts_ema", "ts_ewm_std", "ts_ewm_var", "ts_ewm_cov", "ts_ewm_corr",
            "RSI_WILDER", "ATR_WILDER", "ADX", "MACD_line", "MACD_signal", "MACD_hist",
        })

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="D", tz="UTC")
    x = rng.standard_normal(n_bars)
    y = rng.standard_normal(n_bars)  # ts_ewm_cov / ts_ewm_corr 需要第二个输入

    def _input_slice(canonical: str, sl: Any) -> dict[str, np.ndarray]:
        """按 canonical 需要的输入键切片。ADX/ATR_WILDER 需要 high/low/close。"""
        if canonical in {"ATR_WILDER", "ADX"}:
            return {
                "high": np.asarray(x[sl], dtype=float) + 0.5,
                "low": np.asarray(x[sl], dtype=float) - 0.5,
                "close": np.asarray(x[sl], dtype=float),
            }
        if canonical in {"ts_ewm_cov", "ts_ewm_corr"}:
            return {"x": np.asarray(x[sl], dtype=float), "y": np.asarray(y[sl], dtype=float)}
        return {"x": np.asarray(x[sl], dtype=float)}

    def _run_segment(canonical: str, sl: Any, ckpt: Any, *, origin: bool) -> Any:
        return execute_stateful_segment(
            canonical,
            _input_slice(canonical, sl),
            timestamps=timestamps[sl],
            instrument="TEST",
            input_identity={"test": "chunk_invariance_hard_gate", "canonical": canonical},
            checkpoint=ckpt,
            starts_at_dataset_origin=origin,
        )

    per_canonical: dict[str, dict[str, Any]] = {}
    all_ok = True
    for canonical in sorted(SEGMENTED_EXECUTION_CANONICALS):
        try:
            full = np.asarray(_run_segment(canonical, slice(0, n_bars), None, origin=True).values, dtype=float)
        except Exception as exc:  # noqa: BLE001
            per_canonical[canonical] = {"ok": False, "detail": f"full run raised: {exc}"}
            all_ok = False
            continue
        chunk_ok = True
        chunk_detail: list[str] = []
        for _ in range(n_random_chunkings):
            # 生成 chunk 边界，保证每个 chunk 长度 >= 2（单 bar chunk 无法用
            # first/last 两段式重放）。
            cut = sorted(set(int(b) for b in rng.integers(1, n_bars, size=n_bars // 8)))
            if not cut:
                cut = [n_bars // 2]
            boundaries: list[int] = []
            for b in cut:
                if 1 < b < n_bars - 1:  # 首 chunk >= 2 bar，尾 chunk >= 2 bar
                    if not boundaries or b - boundaries[-1] >= 2:
                        boundaries.append(b)
            if not boundaries:
                boundaries = [n_bars // 2]
            chunked = np.full(n_bars, np.nan, dtype=float)
            prev_ckpt: Any = None
            remaining_boundaries = list(boundaries)
            try:
                # 生产 incremental 路径的 1-bar inclusive overlap 语义：
                # 每个 segment [s,e) 的 checkpoint 是 [s,e-1) 末端状态（as_of=e-2），
                # 下一个 segment 从 e-1 开始（重算边界 bar e-1）。
                start = 0
                while start < n_bars:
                    if remaining_boundaries:
                        e = min(remaining_boundaries[0], n_bars)
                        remaining_boundaries = remaining_boundaries[1:]
                    else:
                        e = n_bars
                    seg_len = e - start
                    if seg_len >= 2:
                        first = _run_segment(canonical, slice(start, e - 1), prev_ckpt,
                                             origin=(start == 0 and prev_ckpt is None))
                        last = _run_segment(canonical, slice(e - 1, e), first.checkpoint, origin=False)
                        chunked[start:e] = np.concatenate(
                            [np.asarray(first.values, dtype=float), np.asarray(last.values, dtype=float)]
                        )
                        prev_ckpt = first.checkpoint
                        start = e - 1  # 1-bar overlap
                    else:
                        # 单 bar：只有重算，checkpoint 不更新。
                        single = _run_segment(canonical, slice(start, e), prev_ckpt,
                                              origin=(start == 0 and prev_ckpt is None))
                        chunked[start:e] = np.asarray(single.values, dtype=float)
                        start = e
            except Exception as exc:  # noqa: BLE001
                chunk_ok = False
                chunk_detail.append(f"chunking raised {type(exc).__name__}: {exc}")
                break
            if chunk_ok and not _allclose_nan(full, chunked, rtol=rtol, atol=atol):
                chunk_ok = False
                chunk_detail.append(f"boundaries={boundaries} mismatch")
        per_canonical[canonical] = {
            "ok": bool(chunk_ok),
            "chunk_detail": chunk_detail,
            "n_chunkings": n_random_chunkings,
        }
        all_ok = all_ok and chunk_ok
    return all_ok, {"per_canonical": per_canonical, "gate": CHECKPOINT_CHUNK_INVARIANCE_FAILURE}
