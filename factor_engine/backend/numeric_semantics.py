# -*- coding: utf-8
"""跨后端数值语义契约：Pandas / PolarsLong / DuckDB 对齐依据。

各 backend 实现应引用本模块常量，parity 测试覆盖 edge cases。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

NullPolicy = Literal["propagate", "ignore", "coerce_to_null"]
InfPolicy = Literal["propagate", "to_nan", "to_null"]
DivZeroPolicy = Literal["inf", "null", "default", "protected"]
StdDdof = Literal["sample", "population"]
RankNullPolicy = Literal["ignore", "bottom", "error"]
ZscoreZeroStdPolicy = Literal["zero", "nan", "null"]


@dataclass(frozen=True)
class MomentConvention:
    """R40 #253：矩（mean/std/var/skew/kurt）的完整约定 —— 进入 backend
    parity evidence，禁止依赖 backend 默认。

    旧 ``StdDdof`` 只有 sample/population 一个维度；实际 moment 语义还取决于
    bias（总体矩 vs 样本矩）、fisher（Fisher 定义 skew/kurt vs Pearson）、
    NaN policy（propagate vs skip）与 finite policy（±Inf 是否算样本）。
    """

    ddof: int = 1
    bias: bool = False
    fisher: bool = True
    nan_policy: Literal["propagate", "omit"] = "omit"
    finite_policy: Literal["exclude", "include"] = "exclude"

    @classmethod
    def from_std_ddof(cls, ddof: Literal["sample", "population"]) -> "MomentConvention":
        return cls(ddof=1 if ddof == "sample" else 0)

    def identity_payload(self) -> dict[str, object]:
        return {
            "ddof": self.ddof,
            "bias": self.bias,
            "fisher": self.fisher,
            "nan_policy": self.nan_policy,
            "finite_policy": self.finite_policy,
        }

    def moment_ddof(self) -> int:
        return 1 if self.ddof == 1 else 0


#: 全局默认 moment 约定（sample ddof=1, bias=False, fisher=True, 跳过 NaN/±Inf）。
DEFAULT_MOMENT_CONVENTION = MomentConvention()


def moment_convention_for(canon: str) -> MomentConvention:
    """算子级 moment 约定；未覆盖时返回全局默认。

    与 ``semantics_for`` 的 ``std_ddof`` 对齐（sample -> ddof=1，
    population -> ddof=0）。
    """
    from factor_engine.backend.numeric_semantics import std_ddof_value

    try:
        ddof = std_ddof_value(canon)
    except Exception:  # pragma: no cover - defensive
        ddof = 1
    return MomentConvention(ddof=ddof)


@dataclass(frozen=True)
class NumericSemantics:
    """单算子或全局默认数值语义。"""

    input_nan_to_null: bool = True
    output_inf_to_nan: bool = True
    output_inf_to_null: bool = False
    div_zero: DivZeroPolicy = "protected"
    rank_ignore_nan: bool = True
    std_ddof: StdDdof = "sample"
    zscore_zero_std: ZscoreZeroStdPolicy = "zero"
    quantile_interpolation: str = "linear"
    window_min_periods_default: int = 1


# 全局默认（panel / long-table 通用）
DEFAULT_SEMANTICS = NumericSemantics()

# 算子级覆盖（R19-072: 禁止 literal dict 重复 key —— 一律通过
# ``register_numeric_semantics`` 显式注册。历史上 line 46/72 的 ``"divide"`` 重复 key
# 被后者静默覆盖；本机制使重复注册在构建期 hard fail。每个 retained statistical
# operator 在此显式声明它真正依赖的关键 policy，全局 default 不作替代。）
OPERATOR_SEMANTICS: dict[str, NumericSemantics] = {}


def register_numeric_semantics(canonical: str, semantics: NumericSemantics) -> None:
    """显式注册一个算子的数值语义；重复注册同一 canonical 直接 hard fail。

    R19-072: 所有 production 数值语义必须通过本函数注册，杜绝 literal dict 重复 key
    被静默覆盖。传入的必须是 canonical 名称（alias 解析在 ``semantics_for`` 运行时完成，
    本函数刻意不 import operator registry，保持模块可独立加载）。
    """
    if not isinstance(semantics, NumericSemantics):
        raise TypeError(
            f"numeric semantics for {canonical!r} must be NumericSemantics, "
            f"got {type(semantics).__name__}"
        )
    existing = OPERATOR_SEMANTICS.get(canonical)
    if existing is not None:
        raise ValueError(
            f"duplicate numeric semantics for {canonical!r}: existing={existing!r} "
            f"new={semantics!r} —— repeated registration is forbidden (R19-072)"
        )
    OPERATOR_SEMANTICS[canonical] = semantics


def _build_operator_semantics() -> None:
    """模块加载期通过 ``register_numeric_semantics`` 构建 OPERATOR_SEMANTICS。"""
    _r = register_numeric_semantics
    _r("protected_div", NumericSemantics(div_zero="null"))
    _r("safe_div_null", NumericSemantics(div_zero="null"))
    _r("div_or_default", NumericSemantics(div_zero="default"))
    _r("div_or_null", NumericSemantics(div_zero="null"))
    _r("protected_log", NumericSemantics(input_nan_to_null=True))
    _r("log_fill_invalid", NumericSemantics(input_nan_to_null=False, div_zero="default"))
    _r("protected_sqrt", NumericSemantics(input_nan_to_null=True))
    # divide: raw ``x / y``，除零 → Inf，且保留 Inf（output_inf_to_nan=False）。
    _r("divide", NumericSemantics(div_zero="inf", output_inf_to_nan=False))
    _r("rank", NumericSemantics(rank_ignore_nan=True))
    _r("rank_pct", NumericSemantics(rank_ignore_nan=True))
    _r("group_rank", NumericSemantics(rank_ignore_nan=True))
    _r("ts_rank", NumericSemantics(rank_ignore_nan=True))
    _r("cs_pct_rank", NumericSemantics(rank_ignore_nan=True))
    _r("zscore", NumericSemantics(zscore_zero_std="zero", std_ddof="sample"))
    _r("group_zscore", NumericSemantics(zscore_zero_std="zero", std_ddof="sample"))
    _r("group_std", NumericSemantics(std_ddof="sample"))
    _r("ts_std", NumericSemantics(std_ddof="sample"))
    _r("ts_var", NumericSemantics(std_ddof="sample"))
    _r("ts_zscore", NumericSemantics(zscore_zero_std="zero", std_ddof="sample"))
    _r("winsorize", NumericSemantics(rank_ignore_nan=True))
    _r("group_winsorize", NumericSemantics(rank_ignore_nan=True))
    _r("cs_quantile", NumericSemantics(quantile_interpolation="linear"))
    _r("c_percentile", NumericSemantics(quantile_interpolation="linear"))
    _r("ts_quantile", NumericSemantics(quantile_interpolation="linear"))
    _r("group_percentile", NumericSemantics(quantile_interpolation="linear"))
    _r("nan_to_num", NumericSemantics(output_inf_to_nan=False, output_inf_to_null=False))
    _r("normalize", NumericSemantics())
    _r("ts_pct", NumericSemantics(input_nan_to_null=True))
    _r("and_", NumericSemantics())
    _r("or_", NumericSemantics())
    _r("not_", NumericSemantics())
    _r("where", NumericSemantics())
    # log: 负数 → NULL；0 → -Inf，且 -Inf 保留（log_zero_returns_negative_infinity）。
    _r("log", NumericSemantics(input_nan_to_null=True, output_inf_to_nan=False))
    # exp: 溢出（exp(1000)=Inf）允许保留，见 ``exp_overflow_policy``。
    _r("exp", NumericSemantics(output_inf_to_nan=False))
    # asin/acos: 定义域 [-1,1]，越界 → NaN（见 ``trig_domain_unit_interval_nan``）。
    _r("asin", NumericSemantics())
    _r("acos", NumericSemantics())
    # power: 实数域策略（负数底数 & 非整数指数 → NaN），见 ``power_real_domain_policy``。
    _r("power", NumericSemantics())


_build_operator_semantics()


def semantics_for(canon: str) -> NumericSemantics:
    """查询算子级数值语义，未覆盖时返回全局默认。

    参数:
        canon: 算子 canonical 名称或别名。

    返回:
        对应的 ``NumericSemantics`` 配置。
    """
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return OPERATOR_SEMANTICS.get(name, DEFAULT_SEMANTICS)


def std_ddof_value(canon: str) -> int:
    """返回标准差/方差算子的 ddof 数值。

    参数:
        canon: 算子 canonical 名称。

    返回:
        sample 策略为 ``1``，population 策略为 ``0``。
    """
    return 1 if semantics_for(canon).std_ddof == "sample" else 0


def protected_epsilon_default() -> float:
    """返回 protected 算子默认 epsilon 常量。

    返回:
        用于除零/对数保护的极小正数。
    """
    return 1e-12


def protected_div_default() -> float:
    """返回 protected_div 除零时的默认填充值。

    返回:
        除零保护策略下的默认输出。
    """
    return 0.0


def rank_ignore_nan(canon: str) -> bool:
    """截面/组内 rank 是否忽略 NaN（输出仍为 null）。"""
    from factor_engine.backend.rank_spec import rank_ignore_nan as _rank_ignore_nan

    return _rank_ignore_nan(canon)


def zscore_zero_std_fill(canon: str) -> float | None:
    """std=0 时 zscore 填充值；``null`` 策略返回 ``None``。"""
    from factor_engine.backend.cross_section_spec import zscore_zero_std_fill as _zscore_zero_std_fill

    return _zscore_zero_std_fill(canon)


def normalize_single_valid_is_null() -> bool:
    """截面 normalize：仅一个有效值时输出 NULL（非常数截面）。"""
    from factor_engine.backend.cross_section_spec import normalize_single_valid_is_null as _norm_single

    return _norm_single()


def normalize_constant_cross_section_fill() -> float:
    """截面 normalize：全部有效值相同（span=0）时输出 0.5。"""
    from factor_engine.backend.cross_section_spec import normalize_constant_cross_section_fill as _norm_const

    return _norm_const()


def nan_to_num_replaces_infinite() -> bool:
    """``nan_to_num`` 将 ±Inf 替换为与 NaN/null 相同的填充常数。"""
    return True


def truthy_null_is_false() -> bool:
    """逻辑算子 ``and_``/``or_``/``not_``：NULL/NaN 视为 false。

    R19-071: production ``where`` 采用三值逻辑（见 ``where_truth_value_policy``），
    Unknown（NaN/NULL）分支输出 NaN —— 本函数仅描述 and_/or_/not_ 的布尔上下文。
    """
    return True


def truthy_nan_is_false() -> bool:
    """NaN 在逻辑算子中视为 false（Polars NaN 非 NULL）。

    R19-071: 三值逻辑下 where 的 Unknown 分支输出 NaN（见 ``where_truth_value_policy``）。
    """
    return True


def truthy_inf_is_true() -> bool:
    """±Inf 在逻辑算子中是否视为 true。

    R19-069: production factor DSL 只允许 typed bool —— ``where`` 条件仅接受
    {0,1,NaN}，±Inf 是数据质量错误，不是合法 truthy。返回 False。
    注意：``backend/logical_semantics.py`` 的同名 legacy 函数仍返回 True，仅供
    非 production 快速路径；production 门必须在条件进入 ``where`` 前拒绝 ±Inf
    （R19-070 negative tests：2 / -1 / Inf / string 必须 reject）。
    """
    return False


def trig_domain_unit_interval_nan() -> bool:
    """``asin``/``acos`` 定义域为 [-1,1]；越界输入 → NaN（所有 backend 一致）。

    R19-095: pandas/numpy、polars、duckdb 的 asin/acos 越界一律输出 NaN/null，
    不抛错、不返回复数。
    """
    return True


def power_real_domain_policy() -> str:
    """``power(x, y)`` 实数域策略：负数底数且 y 非整数 → NaN；y 为整数 → 实数。

    R19-096: 返回 ``"real_domain"``。numpy / polars / duckdb 均遵循；任何 backend
    不得对 negative base & fractional exponent 产生复数或静默求模。
    """
    return "real_domain"


def exp_overflow_policy() -> str:
    """``exp`` 溢出策略：``exp(1000)=Inf`` 直接保留。

    R19-097: 返回 ``"allow_inf"``。pandas / polars / duckdb 的 ``exp`` 溢出均产生
    ±Inf 且保留（``OPERATOR_SEMANTICS["exp"].output_inf_to_nan=False``）。
    """
    return "allow_inf"


def where_truth_value_policy() -> str:
    """production ``where`` 条件采用三值逻辑。

    R19-071: 返回 ``"three_valued"`` —— condition ∈ {0,1,NaN}；Unknown（NaN/NULL）
    既不取 a 也不取 b，输出 NaN；±Inf 不是合法 condition（数据质量错误）。
    ``and_``/``or_``/``not_`` 布尔上下文里 NULL/NaN 仍按 false 处理
    （见 ``truthy_null_is_false``）。
    """
    return "three_valued"


def is_nan_excludes_null() -> bool:
    """``is_nan(NULL)=0``；``is_null(NULL)=1``。"""
    return True


def ts_argmax_empty_window_is_null() -> bool:
    """``ts_argmax/ts_argmin``：窗口全 NULL/无效 → NULL（非 0）。"""
    return True


def ts_argmax_index_origin() -> str:
    """arg 位置从窗口左端计 0（距当前点 w-1 的偏移）。

    R19-039..042/049 origin 收敛：该 origin 描述 ``ts_argmax_index_from_oldest``
    canonical（0 = 窗口最旧 bar）。默认 ``ts_argmax`` / ``ts_argmax_age`` 采用
    **age** 语义（0 = 当前/最新 bar，``bars_since_extreme``），见共享 kernel
    ``rolling_days_since_extreme``。
    """
    return "window_left_0"


def ts_argmax_tie_break() -> str:
    """并列极值取 latest（窗口内最新 occurrence，``hits[-1]``）。

    R19-039..042/049 tie 收敛：全 backend（pandas/gtja/polars）统一 latest——
    与共享 kernel ``rolling_argmax``/``rolling_days_since_extreme`` 一致。
    """
    return "latest"


def ts_sharpe_zero_std_is_null() -> bool:
    """``ts_sharpe``：std=0 → NULL（禁止 Inf/1e308 cap 分叉）。"""
    return True


def group_percentile_null_is_null() -> bool:
    """``group_percentile``：输入 NULL → 输出 NULL（非 0）。"""
    return True


def ts_mad_is_nonstandard() -> bool:
    """当前 ``ts_mad`` 为双重滚动近似，非标准 MAD；禁止 production。"""
    return True


def rank_tie_method(canon: str) -> str:
    """截面/组内/时序 rank 并列策略（见 ``rank_spec.RANK_SPECS``）。"""
    if canon in {"ts_argmax", "ts_argmin"}:
        return "first"
    from factor_engine.backend.rank_spec import rank_tie_method as _rank_tie_method

    return _rank_tie_method(canon)


def panel_binary_join_preserves_anchor() -> bool:
    """二元算子以左操作数为 anchor LEFT JOIN，禁止隐式删行。"""
    return True


def chunk_scan_invariance_required() -> bool:
    """production 算子须通过全量 vs 分块 overlap 扫描 parity。"""
    return True


def ts_pct_zero_prev_is_null() -> bool:
    """``ts_pct``：滞后值为 0 或 NULL 时输出 NULL（不做 forward-fill）。"""
    return True


def log_zero_returns_negative_infinity() -> bool:
    """``log``：输入为 0 时输出 -Inf；负数输出 NULL。"""
    return True


def canonical_numeric_algorithm(canon: str) -> str:
    """R40 #252：canonical 的数值稳定算法声明（pairwise / Welford / Neumaier）。

    高动态范围（1e16 + 小量 / 大市值 / long expanding）的 sum / mean / variance
    不得用朴素 numpy 顺序累加。声明进 numeric contract，parity evidence 据此
    检查实现。
    """
    from factor_engine.cleaned_operators._numpy_kernels import canonical_numeric_algorithm as _cna

    return _cna(canon)


def sql_stddev_fn_key(canon: str) -> str:
    """DuckDB/CH emitter 用：``stddev`` → sample，``stddev_pop`` → population。"""
    return "stddev" if semantics_for(canon).std_ddof == "sample" else "stddev_pop"


def _semantics_fingerprint(sem: NumericSemantics) -> str:
    """把单个 NumericSemantics 转成稳定、key-sorted 的可哈希串。"""
    import dataclasses

    return repr(sorted(dataclasses.asdict(sem).items()))


def _policy_flags() -> dict[str, str | bool]:
    """进入语义哈希的全局 truthiness / domain 策略旗标。"""
    return {
        "truthy_inf_is_true": truthy_inf_is_true(),
        "truthy_null_is_false": truthy_null_is_false(),
        "truthy_nan_is_false": truthy_nan_is_false(),
        "is_nan_excludes_null": is_nan_excludes_null(),
        "trig_domain_unit_interval_nan": trig_domain_unit_interval_nan(),
        "power_real_domain_policy": power_real_domain_policy(),
        "exp_overflow_policy": exp_overflow_policy(),
        "where_truth_value_policy": where_truth_value_policy(),
        "log_zero_returns_negative_infinity": log_zero_returns_negative_infinity(),
    }


def numeric_semantics_hash() -> str:
    """对全部数值语义做稳定哈希（算子级 + 全局默认 + truthiness/domain 策略）。

    R19-073: factor identity / evidence 必须包含该哈希 —— ddof / tie / quantile
    interpolation / zero-std / div-zero / Inf / NaN / truthiness 任一变更都会改变
    factor 语义身份。返回 sha256 十六进制串，跨进程稳定。
    """
    import hashlib

    h = hashlib.sha256()
    h.update(b"FactorEngineNumericSemanticsV1")
    h.update(b"\x00" + _semantics_fingerprint(DEFAULT_SEMANTICS).encode("utf-8"))
    for canon in sorted(OPERATOR_SEMANTICS):
        h.update(b"\x00" + canon.encode("utf-8") + b"=")
        h.update(_semantics_fingerprint(OPERATOR_SEMANTICS[canon]).encode("utf-8"))
    flags = _policy_flags()
    for key in sorted(flags):
        h.update(b"\x00" + key.encode("utf-8") + b"=" + str(flags[key]).encode("utf-8"))
    return h.hexdigest()
