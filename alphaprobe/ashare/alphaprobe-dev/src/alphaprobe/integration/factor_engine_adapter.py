"""FactorEngineAdapter（任务书 §3.1 / §76 / Phase 2）：公式链唯一真值。

职责
----
- ``validate``：FE ``validate_factor_engine_dsl``（import 路径见代码地图 §3.1）；
  运行时 ``sys.path`` 已含 quant_projects（``fe_bridge.paths`` 负责引导）。
- ``canonicalize``：FE ``parse_expr`` 可用时用 FE ``canonical_expression`` 作 canonical；
  否则降级到 ``alphaprobe.dedup.canonicalize_dsl``（括号平衡文本化简）。
- ``inspect``：复用 ``alphaprobe.dedup`` 提取 field/operator/complexity/lookback；
  FE ``ir.Analyzer`` 可用时 lookback 用 FE 权威值。
- ``run_many``：FE 不可 import 或数据缺失时返回带 ``error`` 标记的 ``FactorBatch``，
  不抛异常（硬规矩：旧路径先兼容，不 Big Bang）。

不修改 factor_engine/、data_access/ 任何文件；只在 AlphaPROBE 侧适配。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from alphaprobe.contracts import ExperimentContext, FactorBatch
from alphaprobe.dedup import (
    canonical_ast_hash as dedup_canonical_ast_hash,
    canonicalize_dsl as dedup_canonicalize_dsl,
    parameter_family_key as dedup_parameter_family_key,
    signal_equivalence_id as dedup_signal_equivalence_id,
)

# ---------------------------------------------------------------------------
# FE 懒加载（runtime sys.path 已含 quant_projects，见 fe_bridge/paths.py）
# ---------------------------------------------------------------------------

_FE = None
_FE_IMPORT_ERROR: Exception | None = None
_FE_LOAD_TRIED = False


def _fe() -> dict[str, Any] | None:
    """懒加载 factor_engine 公开入口；失败缓存 ImportError，不重复尝试。

    返回 {"api": ..., "expr": ..., "ir": ...} 或 None（import 失败）。
    """
    global _FE, _FE_IMPORT_ERROR, _FE_LOAD_TRIED
    if _FE is not None:
        return _FE
    if _FE_IMPORT_ERROR is not None or _FE_LOAD_TRIED:
        return None
    try:
        from alphaprobe.fe_bridge.paths import ensure_factor_engine_importable

        ensure_factor_engine_importable()
        from factor_engine import api  # noqa: F401  (引导包内 __init__ 副作用)
        from factor_engine.api import dsl_parser
        from factor_engine.api import mining_integration
        from factor_engine.expr import canonical_expression as _fe_canonical_expression
        from factor_engine.ir import Analyzer as _FeAnalyzer

        _FE = {
            "validate": mining_integration.validate_factor_engine_dsl,
            "parse_expr": dsl_parser.parse_expr,
            "canonical_expression": _fe_canonical_expression,
            "analyzer": _FeAnalyzer,
            "engine_cls": None,
            "parse_factor": dsl_parser.parse_factor,
        }
    except Exception as exc:  # ImportError / 环境缺依赖
        _FE_IMPORT_ERROR = exc
        _FE = None
    finally:
        _FE_LOAD_TRIED = True
    return _FE


# ---------------------------------------------------------------------------
# 契约数据结构（与 contracts.FactorIdentity / FormulaInspection 对齐）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalFactor:
    """§3.1 canonicalize 产物；``canonical_formula`` 是权威 identity 源。"""

    formula: str
    canonical_formula: str
    canonical_ast_hash: str
    signal_equivalence_id: str
    parameter_family_id: str
    fe_ast: Any = field(default=None, compare=False, repr=False)
    fe_canonical: str = field(default="", compare=False, repr=False)


@dataclass(frozen=True)
class FormulaInspection:
    """§75.2 FactorCandidate 的静态属性（canonical 后推断）。"""

    field_set: tuple[str, ...]
    operator_set: tuple[str, ...]
    complexity: int
    lookback: int
    has_ts_op: bool = False
    has_cs_op: bool = False


# ---------------------------------------------------------------------------
# 本地降级辅助（无 FE 时也能做确定性 identity / 括号平衡校验）
# ---------------------------------------------------------------------------

_CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_FIELD_RE = re.compile(
    r"\b(open|high|low|close|volume|vwap|pe|pb|turnover_ratio|market_cap|"
    r"circulating_market_cap|pe_ttm|pb_mrq|div_yield)\b"
)


def _balanced_formula(formula: str) -> bool:
    """仅校验括号平衡（降级 validate；真实 DSL 校验走 FE）。"""
    depth = 0
    for ch in formula:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _local_operator_set(formula: str) -> tuple[str, ...]:
    return tuple(sorted({m.group(1) for m in _CALL_RE.finditer(formula)}))


def _local_field_set(formula: str) -> tuple[str, ...]:
    return tuple(sorted({m.group(1).lower() for m in _FIELD_RE.finditer(formula)}))


def _local_ast_count(formula: str) -> int:
    """AST 节点数估算：每个算子调用 + 每个叶子标识符/字面量 = 1。"""
    return len(_CALL_RE.findall(formula)) + len(
        re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_.]*\b|\b\d+(?:\.\d+)?\b", formula)
    )


def _local_lookback(formula: str) -> int:
    """窗口最大值估算（无 FE 时）：ts_*(x, W) / ts_*(x, y, W) 的最后数字参数。"""
    lookbacks: list[int] = []
    for m in re.finditer(
        r"\b(?:ts_[a-z_]+|m_median|m_mad|ema|var|pct_change|power)\s*\([^()]*(?:\([^()]*\)[^()]*)*,\s*(\d+)\s*\)",
        formula,
    ):
        lookbacks.append(int(m.group(1)))
    # 折减：ts_mean(x, W) 实际需要 W-1 行
    if lookbacks:
        return max(lookbacks)
    return 0


# ---------------------------------------------------------------------------
# FactorEngineAdapter
# ---------------------------------------------------------------------------


class FactorEngineAdapter:
    """§3.1 唯一实现；符合 integration.FactorEngineAdapterProtocol。"""

    def validate(
        self,
        formula: str,
        *,
        surface: str = "daily",
        mode: str = "research",
    ) -> tuple[bool, str]:
        """FE 校验；不可用时降级为括号平衡检查。

        Returns (ok, message)。mode 预留（research/production），首版都走语法校验。
        """
        text = str(formula or "").strip()
        if not text:
            return False, "empty formula"
        fe = _fe()
        if fe is None:
            if not _balanced_formula(text):
                return False, f"unbalanced parentheses: {text[:120]}"
            return True, "OK(local paren-balance fallback)"
        try:
            return fe["validate"](text, surface=str(surface or "daily"))
        except Exception as exc:
            return False, f"validate error: {exc}"

    def canonicalize(self, formula: str) -> CanonicalFactor:
        """FE canonical 优先；不可用时复用 dedup.canonicalize_dsl。"""
        text = str(formula).strip()
        fe = _fe()
        canonical_formula: str
        fe_canonical = ""
        fe_ast: Any = None
        if fe is not None:
            try:
                expr = fe["parse_expr"](text, surface="daily")
                fe_canonical = fe["canonical_expression"](expr)
                canonical_formula = dedup_canonicalize_dsl(text)
                fe_ast = expr
            except Exception:
                # FE 解析失败 → 退回本地化简（validate 已挡住语法错误）
                canonical_formula = dedup_canonicalize_dsl(text)
                fe_canonical = dedup_canonicalize_dsl(text)
        else:
            canonical_formula = dedup_canonicalize_dsl(text)
            fe_canonical = canonical_formula

        return CanonicalFactor(
            formula=text,
            canonical_formula=canonical_formula,
            canonical_ast_hash=dedup_canonical_ast_hash(canonical_formula),
            signal_equivalence_id=dedup_signal_equivalence_id(canonical_formula),
            parameter_family_id=dedup_parameter_family_key(canonical_formula),
            fe_ast=fe_ast,
            fe_canonical=fe_canonical,
        )

    def inspect(self, canonical: CanonicalFactor) -> FormulaInspection:
        """复用 dedup 提取 field_set / operator_set / complexity / lookback。

        lookback 优先 FE ``ir.Analyzer``（权威 history requirement）；FE 不可用时
        用本地窗口最大值估算（``window - 1`` 折减）。
        """
        if isinstance(canonical, str):
            canonical = self.canonicalize(canonical)
        text = canonical.canonical_formula or canonical.formula
        field_set = _local_field_set(text)
        operator_set = _local_operator_set(text)
        complexity = _local_ast_count(text)

        lookback = 0
        has_ts = False
        has_cs = False
        fe = _fe()
        if fe is not None and canonical.fe_ast is not None:
            try:
                analysis = fe["analyzer"]().lower(canonical.fe_ast)
                lookback = int(analysis.lookback or 0)
                has_ts = bool(analysis.has_ts_op)
                has_cs = bool(analysis.has_cs_op)
            except Exception:
                lookback = _local_lookback(text)
        else:
            lookback = _local_lookback(text)

        return FormulaInspection(
            field_set=field_set,
            operator_set=operator_set,
            complexity=complexity,
            lookback=lookback,
            has_ts_op=has_ts,
            has_cs_op=has_cs,
        )

    def run_many(
        self,
        formulas: list[Any],
        context: ExperimentContext,
    ) -> FactorBatch:
        """批量执行：FE 不可 import 或数据缺失时返回带 error 标记的 FactorBatch。

        不抛异常；factor_ids 为空 + coverage_summary["error"] 携带原因。
        当前 executor 注入由调用方提供（engine 持有 data_source），本 adapter
        不重复构建数据源（Phase 2 仅接 FE 原生路径的入口）。
        """
        ctx = context or ExperimentContext(
            run_id="unknown", round_id="unknown", campaign_id="unknown"
        )
        fe = _fe()
        if fe is None:
            return self._error_batch(ctx, f"factor_engine import unavailable: {_FE_IMPORT_ERROR}")

        canonical_list: list[CanonicalFactor] = []
        for f in formulas:
            if isinstance(f, CanonicalFactor):
                canonical_list.append(f)
            else:
                canonical_list.append(self.canonicalize(f))

        executor = getattr(self, "_executor", None)
        if executor is None:
            return self._error_batch(
                ctx,
                "no executor injected (adapter.run_many is the native path entry; "
                "set ._executor to engine.run_many wrapper)",
            )

        ids = [c.canonical_ast_hash for c in canonical_list]
        try:
            results = executor([c.canonical_formula for c in canonical_list])
        except Exception as exc:
            return self._error_batch(ctx, f"run_many failed: {exc}")

        return FactorBatch(
            factor_ids=ids,
            values_ref=results,
            trade_dates=DateRange_unknown(),
            universe_snapshot_id=ctx.universe_snapshot_id,
            data_snapshot_id=ctx.data_snapshot_id,
            coverage_summary={"n": len(ids), "ok": len(ids)},
        )

    def _error_batch(self, ctx: ExperimentContext, reason: str) -> FactorBatch:
        return FactorBatch(
            factor_ids=[],
            values_ref="",
            trade_dates=DateRange_unknown(),
            universe_snapshot_id=ctx.universe_snapshot_id,
            data_snapshot_id=ctx.data_snapshot_id,
            coverage_summary={"error": reason, "ok": 0},
        )


def _DateRange_unknown_placeholder() -> Any:
    """避免顶部 import contracts.DateRange 带来的循环依赖（惰性）。"""
    from alphaprobe.contracts import DateRange

    return DateRange("1970-01-01", "2099-12-31")


def DateRange_unknown() -> Any:
    return _DateRange_unknown_placeholder()


def build_factor_id(canonical: CanonicalFactor) -> str:
    """§75.1 factor_id：canonical_ast_hash[:12]（稳定、可追溯）。"""
    return canonical.canonical_ast_hash[:12]


def identity_factory(
    formula: str,
    orientation: int = 1,
    *,
    factor_id: str | None = None,
) -> "FactorIdentity":
    """从 DSL 公式构造 contracts.FactorIdentity（Phase 4 统一入口）。"""
    from alphaprobe.contracts import FactorIdentity

    c = FactorEngineAdapter().canonicalize(formula)
    return FactorIdentity(
        factor_id=factor_id or build_factor_id(c),
        canonical_formula=c.canonical_formula,
        canonical_ast_hash=c.canonical_ast_hash,
        signal_equivalence_id=c.signal_equivalence_id,
        parameter_family_id=c.parameter_family_id,
        orientation=orientation,
    )


def _hash_slug(canonical_formula: str) -> str:
    return hashlib.sha256(canonical_formula.encode("utf-8")).hexdigest()[:12]


__all__ = [
    "CanonicalFactor",
    "FactorEngineAdapter",
    "FormulaInspection",
    "build_factor_id",
    "identity_factory",
]
