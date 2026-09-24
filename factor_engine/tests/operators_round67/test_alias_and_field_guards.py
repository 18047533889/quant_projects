# -*- coding: utf-8 -*-
"""R67 守卫测试：算子别名一致性 + 算子×字段解耦 + 参数可调。

本文件只钉不变量，不改任何生产代码。四条主线：

A. 别名 = 同一个引擎 / 同一个实现，只是改名
   对 ``OperatorRegistry._aliases`` 全量遍历（331 条）断言：
     * 每个别名目标都是已注册 canonical（无悬空、无碰撞、无链式）；
     * 每个后端下 ``get(alias, backend=b) is get(canonical, backend=b)``；
     * 别名与 canonical 的 backends 集合完全一致；
     * 别名与 canonical 经 DSL 解析出的算子/参数/默认值完全一致（AST 逐字相等）；
     * 别名不得单独注册后端（``_operators`` / ``_catalog`` 里不得出现别名键，
       ``catalog.backends`` 与 ``catalog.backend_meta`` 必须与 runtime 三方一致）。

B. 算子 × 字段解耦
   算子必须能吃**任意合法字段**（close/high/low/volume/amount/turnover/vwap/returns），
   且输出随字段改变 —— 证明字段被真正消费，而不是被忽略 / 写死到某个列名。
   最小合成面板里所有数值字段共享**同一个 NaN 掩码**，因此字段间差异只能来自
   数值本身，不会由 NaN 支撑差异伪造出来。

C. 参数可调
   同一表达式换参数输出必须改变（防「参数被忽略」）；越界参数必须显式报错，
   切不可静默截断或回退默认值；别名与 canonical 的边界行为必须一致。

D. 类型 / 单位契约
   需要 ConditionBool / GroupKey 的算子必须显式报错，不得静默错算；
   多输入价格类算子的位置输入不得硬绑定到具体列名。

环境注记
--------
* 引擎资源 broker 的预算是按**主机 MemAvailable** 推导的：邻居任务把余量压低时
  每一次 read-wave 预留都会被拒（``ResourceBudgetExceeded: broker admission
  denied``）。这是「这一刻主机忙」，与本文件要证明的语义无关，故：
    - 镜像 ``factor_engine/tests/operators_matrix/helpers.py`` 的成熟做法，
      关掉进程级 resource autopilot 线程；
    - 注入 ``FACTOR_ENGINE_ADMISSION_WAIT_S`` 让调度器等余量而不是立刻判死；
    - 仅对 ``ResourceBudgetExceeded`` 做有限次重试（重试耗尽仍硬失败）。
  真正的算子缺陷（异常类型不对、静默错算、参数被忽略）不会被这条重试掩盖。
* 面板为自建最小合成面板（40 交易日 × 6 标的），零 IO、零真实数据。
"""
from __future__ import annotations

import keyword
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("FACTOR_ENGINE_HERMETIC_PROBES", "1")
# broker 等待上限刻意取小值：等待是为了吸收「这一刻主机忙」，不是把单次运行拖长；
# 真实的持续压力由 ``_execute`` 的有限次重试兜底（重试耗尽仍硬失败）。
if float(os.environ.get("FACTOR_ENGINE_ADMISSION_WAIT_S", "0") or 0) <= 0:
    os.environ["FACTOR_ENGINE_ADMISSION_WAIT_S"] = "10"

from factor_engine.api.dsl_parser import parse_expr  # noqa: E402
from factor_engine.api.factor import Factor  # noqa: E402
from factor_engine.backend.factory import build_backend  # noqa: E402
from factor_engine.cleaned_operators import load_all  # noqa: E402
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402
from factor_engine.runtime.engine import FactorEngine  # noqa: E402
from factor_engine.tests.operators_matrix.synthetic_source import (  # noqa: E402
    SyntheticPanelSource,
)

load_all()

# R65/R67 harness note: resource_autopilot starts a process-level thread on the
# first run_many and then tightens this process's own execution budget; value
# tests need no resource adaptation, so stub it like operators_matrix/helpers.py.
import factor_engine.runtime.resource_autopilot_service as _autopilot  # noqa: E402


class _NoAutopilot:
    started = False

    def summary(self):
        return {"mode": "harness-disabled"}

    def last_decision(self):
        return None


_autopilot.start_resource_autopilot = lambda *a, **k: _NoAutopilot()

#: 审计基线（下界断言，防止「遍历空跑」把所有别名断言变成空断言）。
EXPECTED_ALIAS_COUNT = 331
#: ``surface="all"`` 放行全部算子（``daily`` 会挡掉 experimental）。
SURFACE = "all"
#: 喂给算子的字段池：量纲刻意拉开（价格 ~50 / 成交量 ~1e6 / 换手率 ~0.02）。
FIELD_POOL = ("close", "high", "low", "volume", "amount", "turnover", "vwap", "returns")
#: 面板字段三元组（用户指定口径：close / high / volume，单位互不相同）。
FIELD_TRIPLE = ("close", "high", "volume")
_BROKER_ERROR = "ResourceBudgetExceeded"
_BROKER_ATTEMPTS = 3


# ===========================================================================
# 最小合成面板：字段之间只差数值，NaN 掩码完全共享
# ===========================================================================
_MINI_DATES = pd.bdate_range("2023-06-01", periods=40)
_MINI_INSTRUMENTS = [f"I{i:03d}.SZ" for i in range(3)] + [f"I{i:03d}.SH" for i in range(3)]
_MINI_INDEX = pd.MultiIndex.from_product(
    [_MINI_DATES, _MINI_INSTRUMENTS], names=["timestamp", "instrument"]
)


def _build_mini_panel() -> dict[str, pd.Series]:
    """确定性多字段面板；所有数值字段共享同一个 NaN 掩码。"""
    n_d, n_i = len(_MINI_DATES), len(_MINI_INSTRUMENTS)
    rng = np.random.default_rng(20240924)

    drift = rng.normal(0.0, 0.002, size=n_i)
    steps = rng.normal(0.0, 0.01, size=(n_d, n_i)) + drift[None, :]
    close = 50.0 * np.cumprod(1.0 + steps, axis=0)
    fields = {
        "open": close * (1.0 + rng.normal(0.0, 0.004, size=(n_d, n_i))),
        "high": close * (1.0 + np.abs(rng.normal(0.0, 0.01, size=(n_d, n_i)))),
        "low": close * (1.0 - np.abs(rng.normal(0.0, 0.01, size=(n_d, n_i)))),
        "vwap": close * (1.0 + rng.normal(0.0, 0.003, size=(n_d, n_i))),
    }
    volume = np.exp(rng.normal(13.0, 0.5, size=(n_d, n_i)))
    fields.update(
        {
            "close": close,
            "volume": volume,
            "amount": volume * close,
            "turnover": np.abs(rng.normal(0.02, 0.008, size=(n_d, n_i))),
            "returns": rng.normal(0.0, 0.015, size=(n_d, n_i)),
        }
    )

    mask = rng.random((n_d, n_i)) < 0.02
    mask[-1, :2] = True

    out: dict[str, pd.Series] = {}
    for name, arr in fields.items():
        values = np.asarray(arr, dtype="float64").copy()
        values[mask] = np.nan
        out[name] = pd.Series(values.reshape(-1), index=_MINI_INDEX, name=name)
    # 布尔面板（合法 ConditionBool：{0, 1} + 共享 NaN 掩码）。
    condition = (close > np.nanmedian(close, axis=0)[None, :]).astype("float64")
    condition[mask] = np.nan
    out["condition_panel"] = pd.Series(
        condition.reshape(-1), index=_MINI_INDEX, name="condition_panel"
    )
    return out


class _MiniPanelSource(SyntheticPanelSource):
    """``SyntheticPanelSource`` 的最小面板变体（零 IO、确定性）。

    R67 说明：基类的 ``scan_polars_long`` 产出 ``timestamp/instrument`` 列，而
    polars / SQL 长表计划的契约是 ``ts/inst``（``backend/long_frame.py`` 的
    ``_TS='ts' / _INST='inst'``），直接复用会让 polars / duckdb_sql 路径报
    ``ColumnNotFoundError: unable to find column "ts"``。这里按长表契约重写，
    使同一面板可用于跨后端复核。
    """

    _PANELS: dict[str, pd.Series] | None = None
    _LONG_CACHE: dict[str, object] = {}

    def load_column(self, name: str):
        if _MiniPanelSource._PANELS is None:
            _MiniPanelSource._PANELS = _build_mini_panel()
        return _MiniPanelSource._PANELS[name]

    def scan_polars_long(self, columns):
        import polars as pl

        merged = None
        for name in sorted(set(columns)):
            if name not in _MiniPanelSource._LONG_CACHE:
                series = self.load_column(name)
                frame = pl.DataFrame(
                    {
                        "ts": series.index.get_level_values(0),
                        "inst": series.index.get_level_values(1),
                        name: series.to_numpy(dtype="float64"),
                    }
                ).lazy()
                _MiniPanelSource._LONG_CACHE[name] = frame
            part = _MiniPanelSource._LONG_CACHE[name]
            merged = part if merged is None else merged.join(part, on=["ts", "inst"], how="inner")
        return merged.sort(["inst", "ts"])

    def execution_spec(self):
        return {
            "kind": "mini_synthetic_panel",
            "n_dates": len(_MINI_DATES),
            "n_instruments": len(_MINI_INSTRUMENTS),
        }


# ===========================================================================
# 引擎执行工具
# ===========================================================================
_BACKEND_ENGINES: dict[str, FactorEngine] = {}


def _engine(backend: str = "pandas") -> FactorEngine:
    if backend not in _BACKEND_ENGINES:
        _BACKEND_ENGINES[backend] = FactorEngine(
            build_backend(backend), _MiniPanelSource(), run_mode="research"
        )
    return _BACKEND_ENGINES[backend]


def _extract(value) -> np.ndarray:
    try:
        arr = value.to_numpy(dtype="float64", na_value=np.nan)
    except TypeError:  # polars Series.to_numpy() 不接受 na_value
        arr = value.to_numpy()
    return np.asarray(arr, dtype="float64")


def _run_batch_once(exprs: dict[str, str], backend: str) -> dict[str, tuple]:
    """整批 ``run_many``；只返回成功的条目。"""
    parsed = {n: parse_expr(e, surface=SURFACE) for n, e in exprs.items()}
    factors = [Factor(name=n, expr=e, source_expr=exprs[n]) for n, e in parsed.items()]
    res = _engine(backend).run_many(factors, market="ashare", result_policy="return")
    results = res.get("results") if isinstance(res, dict) else res
    results = results or {}
    return {
        n: ("OK", _extract(results[n])) for n in parsed if results.get(n) is not None
    }


def _run_many(exprs: dict[str, str], backend: str = "pandas") -> dict[str, tuple]:
    """执行 ``{名字: DSL 表达式}``，返回 ``{名字: ("OK", ndarray) | ("ERR", msg)}``。

    先整批执行（快）；批内任一算子抛错则只对缺失条目逐条隔离重跑。
    """
    out: dict[str, tuple] = {}
    for name, expr in exprs.items():
        try:
            parse_expr(expr, surface=SURFACE)
        except Exception as exc:  # noqa: BLE001
            out[name] = ("ERR", f"{type(exc).__name__}: {exc}")
    pending = {n: e for n, e in exprs.items() if n not in out}
    if not pending:
        return out
    try:
        out.update(_run_batch_once(pending, backend))
    except Exception:  # noqa: BLE001 - 批失败 → 逐条隔离
        pass
    for name, expr in pending.items():
        if name not in out:
            out[name] = _run_one(name, expr, backend)
    return out


def _execute(name: str, expr: str, backend: str) -> tuple:
    """执行已解析表达式（含 broker 重试）。"""
    engine = _engine(backend)
    last: tuple = ("ERR", "attempts-exhausted")
    for _ in range(_BROKER_ATTEMPTS):
        try:
            res = engine.run_many(
                [Factor(name=name, expr=parse_expr(expr, surface=SURFACE), source_expr=expr)],
                market="ashare",
                result_policy="return",
            )
            results = res.get("results") if isinstance(res, dict) else res
            value = (results or {}).get(name)
            if value is None:
                last = ("ERR", "no-result")
                continue
            return ("OK", _extract(value))
        except Exception as exc:  # noqa: BLE001
            last = ("ERR", f"{type(exc).__name__}: {exc}")
            if type(exc).__name__ != _BROKER_ERROR:
                return last
    return last


def _run_one(name: str, expr: str, backend: str = "pandas") -> tuple:
    try:
        parse_expr(expr, surface=SURFACE)
    except Exception as exc:  # noqa: BLE001
        return ("ERR", f"{type(exc).__name__}: {exc}")
    return _execute(name, expr, backend)


def _run_single(expr: str, name: str = "probe", backend: str = "pandas") -> tuple:
    return _run_one(name, expr, backend)


def _differs(a: np.ndarray, b: np.ndarray, rtol: float = 1e-9, atol: float = 1e-12) -> bool:
    """两个面板是否可区分（形状 / NaN 支撑 / 数值任一不同）。"""
    if a.shape != b.shape:
        return True
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return True
    mask = ~np.isnan(a)
    if not mask.any():
        return False
    scale = np.maximum(np.abs(a[mask]), np.abs(b[mask]))
    return bool(np.max(np.abs(a[mask] - b[mask]) - (atol + rtol * scale)) > 0.0)


def _identical(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return False
    mask = ~np.isnan(a)
    return bool(np.all(a[mask] == b[mask])) if mask.any() else True


def _registry_state():
    return OperatorRegistry._read_state()


def _legal_literal(spec) -> str | None:
    """由 ParamSpec 生成一个合法字面量；不适用（列表/复杂类型）返回 ``None``。"""
    if getattr(spec, "items", None) is not None:
        return None
    dtype = getattr(spec, "dtype", None)
    low, high = getattr(spec, "min", None), getattr(spec, "max", None)
    if dtype is int:
        value = low if (low is not None and low >= 1) else (low + 1 if low is not None else 2)
        if high is not None:
            value = min(value, high)
        return str(int(value))
    if dtype is float:
        if low is not None and high is not None and low <= 0.5 <= high:
            value = 0.5
        elif low is not None and (high is None or low <= high):
            value = low
        elif high is not None:
            value = high
        else:
            value = 1.0
        return repr(float(value))
    return None


# ===========================================================================
# A. 别名一致性
# ===========================================================================
def test_alias_table_is_populated_at_audited_scale():
    """别名表规模下界 —— 防止遍历空跑把下面所有断言变成空断言。"""
    _ops, aliases, _catalog = _registry_state()
    assert len(aliases) >= EXPECTED_ALIAS_COUNT, (
        f"别名数 {len(aliases)} < 审计基线 {EXPECTED_ALIAS_COUNT}："
        "别名被静默删除，或注册表未按预期加载"
    )


def test_alias_targets_are_registered_canonicals():
    """无悬空：每个别名目标必须是已注册 canonical（runtime 或 catalog）。"""
    operators, aliases, catalog = _registry_state()
    dangling = {
        alias: target
        for alias, target in aliases.items()
        if target not in operators and target not in catalog
    }
    assert not dangling, f"悬空别名（目标未注册）: {dangling}"


def test_alias_never_collides_with_a_canonical():
    """别名不得占用 canonical 的名字空间。"""
    operators, aliases, catalog = _registry_state()
    collide = sorted(a for a in aliases if a in operators or a in catalog)
    assert not collide, f"别名与 canonical 撞名: {collide}"


def test_alias_targets_are_flattened_not_chained():
    """别名必须直接指向 canonical（链式别名会让「同一个实现」的推断失效）。"""
    _ops, aliases, _catalog = _registry_state()
    chained = {
        alias: (target, OperatorRegistry.resolve_canonical(alias))
        for alias, target in aliases.items()
        if OperatorRegistry.resolve_canonical(alias) != target
    }
    assert not chained, f"未展平的别名链: {chained}"


def test_alias_and_canonical_expose_identical_backend_sets():
    """别名与 canonical 的可用后端集合必须完全一致。"""
    _ops, aliases, _catalog = _registry_state()
    mismatch = {}
    for alias, canonical in aliases.items():
        alias_backends = sorted(OperatorRegistry.backends_for(alias))
        canonical_backends = sorted(OperatorRegistry.backends_for(canonical))
        if alias_backends != canonical_backends:
            mismatch[alias] = (canonical, alias_backends, canonical_backends)
    assert not mismatch, f"别名/canonical 后端集合不一致: {mismatch}"


def test_alias_resolves_to_the_same_implementation_object_per_backend():
    """核心不变量：每个后端下别名与 canonical 解析出**同一个实现对象**。"""
    _ops, aliases, _catalog = _registry_state()
    checked = 0
    mismatch = []
    for alias, canonical in aliases.items():
        for backend in sorted(OperatorRegistry.backends_for(canonical)):
            op_alias = OperatorRegistry.get(alias, backend=backend, mode="any")
            op_canonical = OperatorRegistry.get(canonical, backend=backend, mode="any")
            if op_alias is None or op_canonical is None:
                mismatch.append((alias, canonical, backend, "resolved-to-None"))
                continue
            checked += 1
            if op_alias is not op_canonical:
                mismatch.append(
                    (alias, canonical, backend, f"{id(op_alias)} != {id(op_canonical)}")
                )
    assert not mismatch, f"别名实现对象与 canonical 不是同一个: {mismatch[:10]}"
    assert checked >= len(aliases), (
        f"仅实检 {checked} 对 (别名, 后端)，低于别名数 {len(aliases)}"
    )


def test_alias_and_canonical_parse_to_an_identical_operator_contract():
    """别名经 DSL 解析后必须与 canonical 落到完全相同的算子契约。

    ``parse_expr`` 的 AST ``repr`` 同时呈现算子名与全部实参 / 关键字，因此
    「别名参数契约 == canonical 参数契约」可以逐字比对。对每个算子声明的
    int/float 参数再显式赋值一次，等价于对每份 ParamSpec 做接受性比对。
    """
    _ops, aliases, catalog = _registry_state()
    identical = both_failed = 0
    mismatched, asymmetric = [], []
    for alias, canonical in aliases.items():
        if keyword.iskeyword(alias):
            # ``if`` 之类的 Python 关键字无法出现在 DSL 调用位置，不是引擎问题。
            continue
        meta = catalog.get(canonical) or {}
        param_names = list(meta.get("param_names") or ())
        specs = meta.get("param_specs") or {}
        positional = [p for p in param_names if p not in specs]
        n_args = max(1, len(positional))
        args = ", ".join(FIELD_POOL[i % len(FIELD_POOL)] for i in range(n_args))

        explicit = [
            f"{p}={_legal_literal(s)}"
            for p, s in sorted(specs.items())
            if _legal_literal(s) is not None
        ]
        variants = [args] + ([args + ", " + ", ".join(explicit)] if explicit else [])

        for variant in variants:
            try:
                repr_alias = repr(parse_expr(f"{alias}({variant})", surface=SURFACE))
                err_alias = None
            except Exception as exc:  # noqa: BLE001
                repr_alias, err_alias = None, type(exc).__name__
            try:
                repr_canonical = repr(parse_expr(f"{canonical}({variant})", surface=SURFACE))
                err_canonical = None
            except Exception as exc:  # noqa: BLE001
                repr_canonical, err_canonical = None, type(exc).__name__
            if repr_alias is not None and repr_canonical is not None:
                if repr_alias == repr_canonical:
                    identical += 1
                else:
                    mismatched.append((alias, canonical, f"{alias}({variant})"))
            elif err_alias is not None and err_canonical is not None:
                both_failed += 1
            else:
                asymmetric.append((alias, canonical, f"{alias}({variant})", err_alias, err_canonical))
    assert not mismatched, f"别名/canonical 解析出的算子契约不一致: {mismatched[:5]}"
    assert not asymmetric, (
        "别名与 canonical 在可解析性上不对称（一个能解析、一个不能）: " f"{asymmetric[:5]}"
    )
    assert identical >= 300, (
        f"仅 {identical} 条别名完成 AST 逐字比对（低于 300）：守卫可能被解析失败淹没"
    )


def test_alias_does_not_register_its_own_backend_entry():
    """别名不得在 ``_operators`` / ``_catalog`` 里另立门户。"""
    operators, aliases, catalog = _registry_state()
    populated = sorted(a for a in aliases if a in operators or a in catalog)
    assert not populated, f"别名单独注册了实现或 catalog 条目: {populated}"
    stray = {
        alias: sorted(operators[alias])
        for alias in aliases
        if operators.get(alias)
    }
    assert not stray, f"别名自带后端实现: {stray}"


def test_catalog_backends_agree_with_runtime_and_are_all_covered_by_backend_meta():
    """runtime 声明的后端必须与 ``catalog.backends`` 完全一致，且都被 ``backend_meta`` 覆盖。

    别名没有自己的 catalog 条目，所以这两条同时覆盖了「catalog 侧是否存在
    只有别名一侧存在的后端注册」中「少注册」的那一半。反方向（``backend_meta``
    多出 runtime 不存在的后端）是当前的真实缺口，见文末 ``xfail`` 用例。
    """
    operators, _aliases, catalog = _registry_state()
    bad = []
    for canonical, implementations in operators.items():
        meta = catalog.get(canonical) or {}
        runtime = set(implementations)
        catalog_backends = set(meta.get("backends") or [])
        backend_meta = set(meta.get("backend_meta") or {})
        if runtime != catalog_backends or not runtime <= backend_meta:
            bad.append(
                (canonical, sorted(runtime), sorted(catalog_backends), sorted(backend_meta))
            )
    assert not bad, (
        "runtime / catalog.backends 不一致，或 runtime 后端未被 backend_meta 覆盖 "
        f"(canonical, runtime, catalog.backends, backend_meta): {bad[:10]}"
    )


# ===========================================================================
# B/C. 算子 × 字段解耦 + 参数可调
# ===========================================================================
#: (canonical, 字段模板, 参数变更模板)；覆盖数学 / 时序 / 技术指标 / 比较 /
#: 截面 / 多元 / 条件各类。
FIELD_OPS: tuple[tuple[str, str, str | None], ...] = (
    ("ts_mean", "ts_mean({f}, window=5)", "ts_mean({f}, window=20)"),
    ("ts_std", "ts_std({f}, window=5)", "ts_std({f}, window=20)"),
    ("ts_sum", "ts_sum({f}, window=5)", "ts_sum({f}, window=20)"),
    ("ts_max", "ts_max({f}, window=5)", "ts_max({f}, window=20)"),
    ("ts_min", "ts_min({f}, window=5)", "ts_min({f}, window=20)"),
    ("ts_median", "ts_median({f}, window=5)", "ts_median({f}, window=20)"),
    ("ts_rank", "ts_rank({f}, window=5)", "ts_rank({f}, window=20)"),
    ("ts_var", "ts_var({f}, window=5)", "ts_var({f}, window=20)"),
    ("ts_delta", "ts_delta({f}, n=1)", "ts_delta({f}, n=5)"),
    ("ts_argmax", "ts_argmax({f}, window=5)", "ts_argmax({f}, window=20)"),
    ("ts_skew", "ts_skew({f}, window=8)", "ts_skew({f}, window=20)"),
    ("ts_kurt", "ts_kurt({f}, window=8)", "ts_kurt({f}, window=20)"),
    ("ts_quantile", "ts_quantile({f}, d=5, q=0.25)", "ts_quantile({f}, d=20, q=0.75)"),
    ("ts_nth_value", "ts_nth_value({f}, window=5, n=1)", "ts_nth_value({f}, window=5, n=3)"),
    ("ts_new_high", "ts_new_high({f}, window=5)", "ts_new_high({f}, window=20)"),
    ("ts_days_since_high", "ts_days_since_high({f}, window=5)", "ts_days_since_high({f}, window=20)"),
    ("ts_max_drawdown", "ts_max_drawdown({f}, window=5)", "ts_max_drawdown({f}, window=20)"),
    ("ts_trend_tstat", "ts_trend_tstat({f}, window=5)", "ts_trend_tstat({f}, window=20)"),
    ("abs", "abs({f})", None),
    ("log", "log({f})", None),
    ("sqrt", "sqrt({f})", None),
    ("neg", "neg({f})", None),
    ("power", "power({f}, y=2.0)", "power({f}, y=0.5)"),
    # clip / gt / where 的阈值刻意做成「相对该字段自身」——固定标量阈值会把
    # 量纲差 4 个数量级的字段一起饱和到同一个常数，那样就测不出字段是否被消费。
    ("clip", "clip({f} - ts_mean({f}, window=5), lo=-1.0, hi=1.0)",
     "clip({f} - ts_mean({f}, window=5), lo=-0.5, hi=0.5)"),
    ("winsorize", "winsorize({f}, lower=0.05, upper=0.95)",
     "winsorize({f}, lower=0.25, upper=0.75)"),
    ("rank", "rank({f})", None),
    ("cs_demean", "cs_demean({f})", None),
    ("scale", "scale({f}, to=1.0)", "scale({f}, to=5.0)"),
    ("gt", "gt({f}, ts_mean({f}, window=5))", None),
    ("where", "where(gt({f}, ts_mean({f}, window=5)), {f}, 0)", None),
    ("RSI_WILDER", "RSI_WILDER({f}, window=6)", "RSI_WILDER({f}, window=24)"),
    ("WMA", "WMA({f}, window=5)", "WMA({f}, window=20)"),
    ("DEMA", "DEMA({f}, window=5)", "DEMA({f}, window=20)"),
    ("TEMA", "TEMA({f}, window=5)", "TEMA({f}, window=20)"),
    ("OBV", "OBV({f}, volume)", None),
    ("ts_corr", "ts_corr({f}, volume, window=5)", "ts_corr({f}, volume, window=20)"),
    ("ts_cov", "ts_cov({f}, volume, window=5)", "ts_cov({f}, volume, window=20)"),
    ("ts_partial_corr", "ts_partial_corr({f}, volume, amount, window=5)",
     "ts_partial_corr({f}, volume, amount, window=20)"),
    ("ts_mean_if", "ts_mean_if({f}, gt({f}, ts_mean({f}, window=5)), window=5)",
     "ts_mean_if({f}, gt({f}, ts_mean({f}, window=5)), window=20)"),
)

#: 语义上与字段**无关**（全体正字段上恒为 +1）—— 单列出来断言真值相等。
FIELD_INVARIANT_BY_DESIGN: tuple[tuple[str, str], ...] = (("sign", "sign({f})"),)

_FIELD_RESULTS: dict[str, dict[str, tuple]] = {}
_PARAM_RESULTS: dict[str, tuple] = {}


def _field_results() -> dict[str, dict[str, tuple]]:
    """一次性跑完全部 (算子 × 字段三元组) 与全部参数变更组；结果缓存复用。"""
    if _FIELD_RESULTS:
        return _FIELD_RESULTS
    exprs = {
        f"{canonical}__{field}": template.format(f=field)
        for canonical, template, _ in FIELD_OPS
        for field in FIELD_TRIPLE
    }
    parallel = {
        f"{canonical}__paramB": param_template.format(f="close")
        for canonical, _template, param_template in FIELD_OPS
        if param_template is not None
    }
    results = _run_many({**exprs, **parallel})
    for canonical, _template, param_template in FIELD_OPS:
        _FIELD_RESULTS[canonical] = {
            field: results[f"{canonical}__{field}"] for field in FIELD_TRIPLE
        }
        if param_template is not None:
            _PARAM_RESULTS[canonical] = (
                f"{canonical}__close",
                f"{canonical}__paramB",
                results[f"{canonical}__close"],
                results[f"{canonical}__paramB"],
            )
    return _FIELD_RESULTS


_FIELD_IDS = [c for c, _, _ in FIELD_OPS]
_PARAM_IDS = [c for c, _, t in FIELD_OPS if t is not None]


@pytest.mark.parametrize("canonical", _FIELD_IDS)
def test_operator_accepts_any_legitimate_field(canonical: str):
    """同一算子对 close/high/volume（量纲完全不同）都必须跑出有效结果。"""
    results = _field_results()[canonical]
    broken = {field: res[1] for field, res in results.items() if res[0] != "OK"}
    assert not broken, f"{canonical} 无法在全部合法字段上执行: {broken}"
    for field, res in results.items():
        arr = res[1]
        assert arr.size > 0, f"{canonical}({field}) 输出空面板"
        assert np.isfinite(arr).any(), f"{canonical}({field}) 输出全 NaN"


@pytest.mark.parametrize("canonical", _FIELD_IDS)
def test_operator_output_tracks_the_input_field(canonical: str):
    """换字段必须换结果 —— 否则说明字段被忽略或写死到某个列名。"""
    results = _field_results()[canonical]
    close = results["close"][1]
    high = results["high"][1]
    volume = results["volume"][1]
    assert _differs(close, high), (
        f"{canonical}: close 与 high 输出完全一致 —— 输入字段未被消费（列名硬绑定 / 被忽略）"
    )
    assert _differs(close, volume), (
        f"{canonical}: close 与 volume 输出完全一致 —— 输入字段未被消费（列名硬绑定 / 被忽略）"
    )
    assert _differs(high, volume), (
        f"{canonical}: high 与 volume 输出完全一致 —— 输入字段被互串 / 被忽略"
    )


@pytest.mark.parametrize(
    "canonical,template",
    FIELD_INVARIANT_BY_DESIGN,
    ids=[c for c, _ in FIELD_INVARIANT_BY_DESIGN],
)
def test_field_invariant_operators_are_deterministic_across_fields(canonical: str, template: str):
    """``sign`` 在全体正字段上恒为 +1：这是语义恒等，不是字段被忽略。

    显式钉住它，避免「字段解耦」这组断言被误解为「所有算子都必须随字段变化」。
    """
    results = _run_many({field: template.format(f=field) for field in FIELD_TRIPLE})
    for field, res in results.items():
        assert res[0] == "OK", f"{canonical}({field}) 执行失败: {res[1]}"
    assert _identical(results["close"][1], results["high"][1])
    assert _identical(results["close"][1], results["volume"][1])


def test_all_field_templates_parse_on_the_all_surface():
    """``parse_expr(..., surface="all")`` 必须放行每个 (算子 × 字段) 组合。"""
    failed = {}
    for canonical, template, _ in FIELD_OPS:
        for field in FIELD_TRIPLE:
            expr = template.format(f=field)
            try:
                parse_expr(expr, surface=SURFACE)
            except Exception as exc:  # noqa: BLE001
                failed[f"{canonical}__{field}"] = f"{type(exc).__name__}: {exc}"
    assert not failed, f"surface='all' 未放行: {failed}"


def test_field_decoupling_also_holds_on_a_second_backend():
    """抽样跨后端复核：字段解耦不是 pandas 单后端的巧合。

    注：``SyntheticPanelSource``（以及本文件的最小面板）产出的 long frame 列名是
    ``timestamp/instrument``，而 polars 长表计划要求时间列名为 ``ts``，因此本文件
    的跨后端复核走 ``duckdb_sql``（SQL 下推）而不是 polars；polars 侧的同类缺口
    见报告「观察」。
    """
    sample = ("ts_mean", "abs", "gt")
    templates = {c: t for c, t, _ in FIELD_OPS}
    exprs = {
        f"{canonical}__{field}": templates[canonical].format(f=field)
        for canonical in sample
        for field in FIELD_TRIPLE
    }
    results = _run_many(exprs, backend="duckdb_sql")
    broken = {k: v[1] for k, v in results.items() if v[0] != "OK"}
    assert not broken, f"duckdb_sql 后端字段组合执行失败: {broken}"
    for canonical in sample:
        assert _differs(
            results[f"{canonical}__close"][1], results[f"{canonical}__volume"][1]
        ), f"duckdb_sql 后端 {canonical} 的 close/volume 输出一致 —— 字段未被消费"


@pytest.mark.parametrize("canonical", _PARAM_IDS)
def test_declared_parameters_are_actually_consumed(canonical: str):
    """同一表达式的两组参数必须产出不同结果（防「参数被忽略」）。"""
    _field_results()
    name_a, name_b, res_a, res_b = _PARAM_RESULTS[canonical]
    assert res_a[0] == "OK", f"{canonical} 基线表达式失败: {res_a[1]}"
    assert res_b[0] == "OK", f"{canonical} 参数变更表达式失败: {res_b[1]}"
    assert _differs(res_a[1], res_b[1]), (
        f"{canonical}: 参数改变但输出完全相同 —— 参数被忽略 "
        f"({name_a} vs {name_b})"
    )


#: (标签, 表达式, 必须报错的原因)
OUT_OF_RANGE_CASES: tuple[tuple[str, str, str], ...] = (
    ("ts_mean window=0", "ts_mean(close, window=0)", "ParamSpec min=1"),
    ("ts_mean window=-1", "ts_mean(close, window=-1)", "ParamSpec min=1"),
    ("ts_mean window=2.5", "ts_mean(close, window=2.5)", "int ParamSpec 拒绝 float"),
    ("ts_mean window='abc'", "ts_mean(close, window='abc')", "int ParamSpec 拒绝字符串"),
    ("ts_std ddof=-1", "ts_std(close, window=5, ddof=-1)", "ParamSpec min=0"),
    ("ts_quantile q=1.5", "ts_quantile(close, d=5, q=1.5)", "ParamSpec max=1"),
    ("ts_mean bogus=1", "ts_mean(close, window=5, bogus=1)", "未声明关键字参数"),
    ("ts_mean 参数挤位", "ts_mean(close, high, low)", "位置实参越过 arity"),
)

_OUT_OF_RANGE_IDS = [label for label, _, _ in OUT_OF_RANGE_CASES]

#: 这些用例要求「越界参数由参数/类型契约本身拒绝」，报错必须是领域异常，
#: 不能退化成资源预算错误之类的无关失败。
_DOMAIN_ERROR_TYPES = frozenset(
    {
        "OperatorParameterError",
        "TypedInputContractError",
        "DSLParseError",
        "DSLUnknownOperatorError",
        "ValueError",
        "TypeError",
    }
)


@pytest.mark.parametrize("label,expr,reason", OUT_OF_RANGE_CASES, ids=_OUT_OF_RANGE_IDS)
def test_out_of_range_parameters_fail_explicitly(label: str, expr: str, reason: str):
    """参数越界必须显式报错，绝不能静默截断 / 回退默认值 / 返回结果。"""
    status, payload = _run_single(expr, name="oob")
    assert status == "ERR", f"{label}: 越界参数被静默接受（{reason}）→ 返回了结果"
    error_type = payload.split(":", 1)[0]
    assert error_type in _DOMAIN_ERROR_TYPES, (
        f"{label}: 越界没有落到参数/类型契约上（{reason}）: {payload}"
    )


def test_zero_argument_call_never_silently_succeeds():
    """``ts_mean()`` 缺必填位置参数，必须失败（绝不能静默返回一角常数面板）。

    注记：当前它报的是 ``ResourceBudgetExceeded``（退化出的 0 字节 read wave
    被 broker 拒），而不是 DSL 的 arity 错误 —— 报错类型具有误导性，已在报告
    的「观察」里列出；这里只钉住「不得静默成功」这一条硬不变量。
    """
    status, payload = _run_single("ts_mean()", name="zero_arg")
    assert status == "ERR", f"ts_mean() 静默成功了（arity 契约失效）: {payload}"


#: (别名, canonical) —— 别名与其 canonical 必须共享同一份 ParamSpec 边界。
_ALIAS_BOUND_PAIRS: tuple[tuple[str, str], ...] = (
    ("m_var", "ts_var"),
    ("m_median", "ts_median"),
    ("m_skew", "ts_skew"),
    ("m_kurt", "ts_kurt"),
)


@pytest.mark.parametrize("alias,canonical", _ALIAS_BOUND_PAIRS, ids=[a for a, _ in _ALIAS_BOUND_PAIRS])
def test_alias_enforces_the_same_parameter_bounds_as_its_canonical(alias: str, canonical: str):
    """越界参数经别名走一遍，报错必须与 canonical 完全同一类。"""
    _ops, aliases, _catalog = _registry_state()
    assert aliases.get(alias) == canonical, f"{alias} 不再指向 {canonical}，请更新用例"

    status_alias, payload_alias = _run_single(f"{alias}(close, window=0)", name="alias_case")
    status_canonical, payload_canonical = _run_single(
        f"{canonical}(close, window=0)", name="canonical_case"
    )
    assert status_alias == status_canonical == "ERR", (
        f"{alias}/{canonical}: window=0 应当显式报错 "
        f"(alias={status_alias}/{payload_alias}, canonical={status_canonical}/{payload_canonical})"
    )
    assert payload_alias.split(":", 1)[0] == payload_canonical.split(":", 1)[0], (
        f"{alias}/{canonical}: 报错类型不一致 {payload_alias} vs {payload_canonical}"
    )


# ===========================================================================
# D. 类型 / 单位契约：必须显式报错，不得静默错算
# ===========================================================================
#: 需要 ConditionBool 输入的条件算子：(canonical, 模板, 是否带 x 输入)
CONDITION_BOOL_OPS: tuple[tuple[str, str], ...] = (
    ("ts_mean_if", "ts_mean_if(close, {cond}, window=5)"),
    ("ts_sum_if", "ts_sum_if(close, {cond}, window=5)"),
    ("ts_std_if", "ts_std_if(close, {cond}, window=5)"),
    ("ts_last_if", "ts_last_if(close, {cond}, window=5)"),
    ("ts_max_if", "ts_max_if(close, {cond}, window=5)"),
    ("ts_min_if", "ts_min_if(close, {cond}, window=5)"),
    ("ts_count_if", "ts_count_if({cond}, window=5)"),
    ("ts_true_streak", "ts_true_streak({cond})"),
    ("ts_days_since", "ts_days_since({cond}, max_lookback=5)"),
)

_CONDITION_IDS = [c for c, _ in CONDITION_BOOL_OPS]
_LEGAL_CONDITION = "gt(close, 100)"
_NON_BOOL_CONDITION = "ts_mean(volume, window=5)"


@pytest.mark.parametrize("canonical,template", CONDITION_BOOL_OPS, ids=_CONDITION_IDS)
def test_condition_bool_operators_accept_legal_conditions(canonical: str, template: str):
    """合法 ConditionBool（``gt(...)`` 产出的 {0,1,NaN} 面板）必须跑通。"""
    status, payload = _run_single(template.format(cond=_LEGAL_CONDITION), name=canonical)
    assert status == "OK", f"{canonical} 拒绝合法 ConditionBool: {payload}"


@pytest.mark.parametrize("canonical,template", CONDITION_BOOL_OPS, ids=_CONDITION_IDS)
def test_condition_bool_operators_reject_non_boolean_conditions(canonical: str, template: str):
    """非布尔面板喂进条件位必须显式报错 —— 而不是「非零即真」静默算下去。"""
    expr = template.format(cond=_NON_BOOL_CONDITION)
    status, payload = _run_single(expr, name=canonical)
    assert status == "ERR", (
        f"{canonical}: 非 ConditionBool 条件被静默接受（静默错算）→ 返回了结果\n  expr = {expr}"
    )
    assert "must be a" in payload and "0, 1" in payload, (
        f"{canonical}: 未给出 ConditionBool 契约的显式说明: {payload}"
    )


def test_group_key_operators_reject_numeric_columns():
    """需要 GroupKey 的算子喂数值列必须显式报错。"""
    for expr, needle in (
        ("group_rank(close, ts_mean(volume, window=5))", "GroupKey"),
        ("group_rank(close, 1)", "int"),
    ):
        status, payload = _run_single(expr, name="group_case")
        assert status == "ERR", f"GroupKey 契约未被强制: {expr} → 返回了结果"
        assert needle in payload, f"{expr}: 报错未指明契约（缺 {needle}）: {payload}"


def test_multi_input_price_operators_are_not_hard_bound_to_named_columns():
    """多输入价格类算子的位置输入必须可换任意数值字段（无列名硬绑定）。

    ``ATR_WILDER(high, low, close, ...)`` 的语义契约只要求「三个价格类输入」，
    并不要求这些输入列**必须**叫 high/low/close；若换字段就报错，即为列名硬绑定。
    """
    cases = {
        "atr_normal": "ATR_WILDER(high, low, close, window=5)",
        "atr_swapped": "ATR_WILDER(volume, amount, close, window=5)",
        "mfi_swapped": "MFI(high, low, volume, close, window=5)",
        "obv_swapped": "OBV(volume, close)",
    }
    results = _run_many(cases)
    broken = {k: v[1] for k, v in results.items() if v[0] != "OK"}
    assert not broken, f"位置输入被硬绑定到列名: {broken}"
    assert _differs(results["atr_normal"][1], results["atr_swapped"][1]), (
        "ATR_WILDER 换位后输出不变 —— 位置输入被忽略"
    )


def test_where_condition_type_is_not_enforced_documented_inconsistency():
    """``where`` 当前**不**校验 condition 的布尔类型（与同角色的 ``*_if`` 不一致）。

    本用例不是在放行这条规则，而是把现状钉住：同族的 ``ts_*_if`` /
    ``ts_count_if`` / ``ts_true_streak`` / ``ts_days_since`` 全部显式报错，
    只有 ``where`` 把任意数值当真值静默选择。若将来给 ``where`` 补上
    ConditionBool 强制，本用例会失败，提醒同步更新契约声明与本报告。
    """
    expr = "where(ts_mean(volume, window=5), close, volume)"
    status, payload = _run_single(expr, name="where_case")
    assert status == "OK", (
        "where 的 condition 契约发生了变化（现在会报错），请同步更新本用例与契约声明: "
        f"{payload}"
    )


# ===========================================================================
# 缺口登记（strict xfail：钉住「当前确实坏了」的不变量）
# ===========================================================================
@pytest.mark.xfail(
    strict=True,
    reason=(
        "R67 报告-缺口 1：catalog.backend_meta 里残留 runtime 已不存在的后端注册。"
        "ts_ewm_corr / ts_ewm_cov 的 backend_meta 声明 ['pandas_numpy','polars']，"
        "而 runtime 与 catalog.backends 只有 ['polars']。"
        "最小复现：catalog['ts_ewm_corr']['backend_meta'] vs "
        "OperatorRegistry.backends_for('ts_ewm_corr')。"
        "修复后本用例会 XPASS，请移除本标记并更新报告。"
    ),
)
def test_backend_meta_has_no_stale_backend_registration():
    """``backend_meta`` 不得残留任何 runtime 里不存在的后端（伪注册）。

    ``freeze()`` 只校验 ``catalog.backends`` vs runtime（R4-102），未覆盖
    ``backend_meta``，因此这是同一类漂移的未守卫分支。
    """
    _operators, aliases, catalog = _registry_state()
    stale = {}
    for canonical in sorted(set(catalog) | set(aliases.values())):
        meta = catalog.get(canonical) or {}
        extra = set(meta.get("backend_meta") or {}) - set(
            OperatorRegistry.backends_for(canonical)
        )
        if extra:
            stale[canonical] = (
                sorted(extra),
                sorted(OperatorRegistry.backends_for(canonical)),
            )
    assert not stale, f"backend_meta 残留无效后端注册（伪注册）: {stale}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "R67 报告-缺口 2：20 个别名已在 _aliases 登记，但没有回写进 "
        "catalog[canonical]['aliases']，按 catalog 审计会漏掉它们"
        "（如 ts_macd→MACD_line、min_of→minimum、kdj→kdj_k、psar→PSAR）。"
        "最小复现：catalog['minimum']['aliases'] 不含 'min_of'。"
        "修复后本用例会 XPASS，请移除本标记并更新报告。"
    ),
)
def test_every_alias_is_mirrored_into_the_canonical_catalog_entry():
    """别名必须同时出现在 canonical 的 catalog ``aliases`` 列表里（双向可查）。"""
    _operators, aliases, catalog = _registry_state()
    missing = {
        alias: canonical
        for alias, canonical in aliases.items()
        if alias not in ((catalog.get(canonical) or {}).get("aliases") or [])
    }
    assert not missing, f"未回写 catalog['aliases'] 的别名: {sorted(missing)[:25]}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "R67 报告-缺口 3：部分合法字段绕过调用方传入的 data_source，转而要求 COS "
        "镜像表 —— abs(turnover) / abs(float_shares) 在显式提供内存面板的情况下仍去拉 "
        "ashare_stock_valuation_daily / ashare_stock_capital_daily，镜像缺失即硬失败。"
        "最小复现：FactorEngine(build_backend('pandas'), SyntheticPanelSource()) "
        "跑 abs(turnover)。"
        "（SyntheticPanelSource 自身就实现了 turnover 面板，说明这是字段解析层越权，"
        "不是调用方缺列。）修复后本用例会 XPASS，请移除本标记并更新报告。"
    ),
)
def test_leveraged_fields_are_served_from_the_caller_data_source():
    """合法字段必须由调用方传入的 data_source 提供，不得私自改走 COS 镜像。"""
    cos_only = {}
    for field in ("turnover", "float_shares", "free_float_shares"):
        status, payload = _run_single(f"abs({field})", name=f"field_{field}")
        if status == "ERR" and "cos_mirror" in payload:
            cos_only[field] = payload[:120]
    assert not cos_only, f"合法字段被迫走 COS 镜像（绕过 data_source）: {cos_only}"


def test_catalog_alias_lists_never_point_at_the_wrong_canonical():
    """反向方向必须干净：catalog 声明的别名都要在 ``_aliases`` 里指向同一 canonical。"""
    _operators, aliases, catalog = _registry_state()
    wrong = {}
    for canonical, meta in catalog.items():
        for alias in (meta.get("aliases") or []):
            mapped = aliases.get(alias)
            if mapped != canonical:
                wrong[alias] = (canonical, mapped)
    assert not wrong, f"catalog['aliases'] 指向错误: {wrong}"
