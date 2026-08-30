# -*- coding: utf-8 -*-
"""PIT 标签层：forward return 与特征窗口隔离（mining / backtest 侧）。

本模块提供标签窗口规格（``LabelWindowSpec``）、特征/标签时间轴对齐、
forward return 构造，以及标签 DSL 的 PiT 安全审计（禁止前视算子进入标签流水线）。
标签可使用未来收益，但须与特征 IR 在 bar 维度显式隔离（``gap_bars``）。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

from factor_engine.runtime.pit_audit import PitSafetyError, audit_ir


# ---------------------------------------------------------------------------
# R24-174..187: LabelExpr / LabelIR — the label layer has its OWN IR with
# CONTROLLED forward semantics, separate from the feature DSL (which forbids any
# forward op).  A feature formula may never carry a label's future meaning, and
# the default label formula is the single authority for the runtime builder.
# ---------------------------------------------------------------------------
class LabelOp(str, Enum):
    """R24-175: the forward ops a LabelIR is allowed to express.

    FeatureIR (R24-176) forbids every one of these; LabelIR allows exactly these.
    """

    FORWARD_RETURN = "forward_return"
    FORWARD_EXCESS_RETURN = "forward_excess_return"
    FORWARD_RESIDUAL_RETURN = "forward_residual_return"
    FORWARD_TOTAL_RETURN = "forward_total_return"

    @classmethod
    def is_forward(cls, op: str | "LabelOp") -> bool:
        try:
            cls(op)
            return True
        except ValueError:
            return False


class LabelMissingPolicy(str, Enum):
    """R24-186/187: how missing prices during the forward horizon are handled.

    ``shift_skip`` (legacy) silently skips a missing bar; ``nan`` (default,
    R24-186) fails the sample; ``terminal_return`` applies the terminal return
    policy on delisting/suspension.
    """

    NAN = "nan"
    SHIFT_SKIP = "shift_skip"
    TERMINAL_RETURN = "terminal_return"


@dataclass(frozen=True)
class LabelIR:
    """R24-174..187: an explicit label IR node.

    ``op`` is one of the controlled forward ops; ``inputs`` are the price /
    benchmark / residual panels; ``horizon_bars`` is a STRICT positive int
    (R24-181); ``gap_bars`` is a BAR count (bar clock, R24-182/183 — never a
    calendar Timedelta); ``decision_time`` makes the label start explicit
    (R24-185); ``missing_policy`` / ``terminal_return_policy`` govern delisting
    / suspension / missing close (R24-186/187).
    """

    op: LabelOp
    inputs: tuple[str, ...] = ("close",)
    horizon_bars: int = 5
    gap_bars: int = 0
    decision_time: str = "close"
    missing_policy: str = LabelMissingPolicy.NAN.value
    terminal_return_policy: str = "nan"
    return_policy: str = "compound"

    def __post_init__(self) -> None:
        # R24-181: strict positive int — ``max(1, int(horizon))`` is forbidden.
        if not isinstance(self.horizon_bars, int) or isinstance(self.horizon_bars, bool):
            raise ValueError(
                f"LabelIR.horizon_bars must be a strict positive int (R24-181); "
                f"got {self.horizon_bars!r}"
            )
        if self.horizon_bars <= 0:
            raise ValueError(
                f"LabelIR.horizon_bars must be > 0 (R24-181); got {self.horizon_bars!r}"
            )
        if not isinstance(self.gap_bars, int) or self.gap_bars < 0:
            raise ValueError(
                f"LabelIR.gap_bars must be a non-negative BAR count (R24-182/183); "
                f"got {self.gap_bars!r}"
            )
        try:
            LabelOp(self.op)
        except ValueError as exc:
            raise ValueError(f"LabelIR.op must be a LabelOp (R24-175); got {self.op!r}") from exc
        if self.missing_policy not in {p.value for p in LabelMissingPolicy}:
            raise ValueError(
                f"LabelIR.missing_policy must be one of "
                f"{[p.value for p in LabelMissingPolicy]} (R24-187)"
            )

    @property
    def label_formula(self) -> str:
        """R24-177/178: the single-authority label formula string.

        The runtime builder and every config MUST use this string — a config
        that writes ``ts_pct(close, h)`` (a BACKWARD pct-change) while the
        builder computes ``close[t+h]/close[t]-1`` is a semantic mismatch.
        """
        if self.op == LabelOp.FORWARD_RETURN:
            return f"forward_return({self.inputs[0]}, {self.horizon_bars})"
        if self.op == LabelOp.FORWARD_EXCESS_RETURN:
            return (
                f"forward_excess_return({self.inputs[0]}, {self.inputs[1]}, "
                f"{self.horizon_bars})"
            )
        return f"{self.op.value}({self.inputs[0]}, {self.horizon_bars})"


_FORWARD_OP_HINTS = (
    "forward_return", "forward_excess_return", "forward_residual_return",
    "forward_total_return", "ts_forward", "fwd_",
)


def parse_label_expr(formula: str) -> LabelIR:
    """R24-174/177: parse a label formula into a LabelIR.

    The label formula is the SINGLE authority (R24-178): ``forward_return(close,
    5)`` is a forward return; a formula like ``ts_pct(close, 5)`` (backward
    pct-change) is REJECTED as a label formula — it does not express the label's
    future semantics.
    """
    text = str(formula or "").strip()
    for hint in _FORWARD_OP_HINTS:
        if text.startswith(hint + "(") and text.endswith(")"):
            inner = text[len(hint) + 1 : -1]
            parts = [p.strip() for p in inner.split(",")]
            name = hint.rstrip("_") if hint.startswith("fwd_") else hint
            op = {
                "forward_return": LabelOp.FORWARD_RETURN,
                "forward_excess_return": LabelOp.FORWARD_EXCESS_RETURN,
                "forward_residual_return": LabelOp.FORWARD_RESIDUAL_RETURN,
                "forward_total_return": LabelOp.FORWARD_TOTAL_RETURN,
                "ts_forward": LabelOp.FORWARD_RETURN,
            }.get(name)
            if op is None:
                break
            horizon = 5
            try:
                horizon = int(parts[-1])
            except (ValueError, IndexError):
                horizon = 5
            return LabelIR(op=op, inputs=tuple(parts[:-1]) or ("close",), horizon_bars=horizon)
    raise ValueError(
        f"label formula {formula!r} is not a forward-label expression "
        "(R24-174/177); use forward_return(close, h) etc. — a backward "
        "pct-change is a FEATURE op, never a label"
    )


def assert_feature_ir_no_forward(feature_ir: Any) -> None:
    """R24-176: a FeatureIR must NOT contain any forward op.

    ``feature_ir`` is a lowered IR (``AnalysisResult.ir`` or any node with an
    ``op`` attribute); every operator name is checked against the forward hint
    set and rejected.
    """
    nodes = getattr(feature_ir, "nodes", None)
    if nodes is None and hasattr(feature_ir, "op"):
        nodes = [feature_ir]
    for node in nodes or []:
        op = str(getattr(node, "op", ""))
        low = op.lower()
        if low in {h.rstrip("_") for h in _FORWARD_OP_HINTS} or any(
            low.startswith(f) for f in _FORWARD_OP_HINTS
        ):
            raise PitSafetyError(
                [f"feature_ir_forward_forbidden: op={op!r} is forward-only "
                 "(R24-176); labels may use forward returns, features may not"]
            )


def build_label_series(
    prices: pd.DataFrame,
    *,
    ir: LabelIR,
    benchmark: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """R24-179/180: build the label from a LabelIR, corporate-action safe.

    Default ``return_policy="compound"`` — the label is the compounded
    ``Π(1+r)-1`` over the horizon (a raw ``close[t+h]/close[t]`` ratio is only
    used when the caller asserts the price series is corporate-action safe).
    """
    if ir.op in (LabelOp.FORWARD_EXCESS_RETURN,) and benchmark is None:
        raise ValueError(
            "LabelIR FORWARD_EXCESS_RETURN requires a benchmark panel (R24-175)"
        )
    h = ir.horizon_bars
    # R24-179: the default target is a return DECIMAL.  ``close[t+h]/close[t]-1``
    # is exactly the compounded Π(1+r)-1 forward return (telescoping) and is
    # corporate-action-safe by construction (a split rescales both endpoints).
    fwd = prices.shift(-h) / prices - 1.0
    if ir.op == LabelOp.FORWARD_EXCESS_RETURN:
        fwd = fwd - (benchmark.shift(-h) / benchmark - 1.0)
    return fwd


@dataclass(frozen=True)
class LabelWindowSpec:
    """标签窗口规格：预测 horizon、特征 lookback 与隔离 gap。"""

    horizon_bars: int
    feature_lookback_bars: int
    gap_bars: int = 0

    @property
    def min_separation_bars(self) -> int:
        """特征窗口结束与标签起始之间的最小 bar 间隔（``gap_bars`` 下界为 0）。"""
        return max(0, int(self.gap_bars))

    def validate_no_overlap(self) -> None:
        """校验 horizon / lookback 参数合法（不检查实际 index 重叠）。"""
        if self.horizon_bars <= 0:
            raise ValueError("horizon_bars 必须 > 0")
        if self.feature_lookback_bars < 0:
            raise ValueError("feature_lookback_bars 不能为负")


def assert_label_feature_no_overlap(
    *,
    feature_end_bar: int,
    label_start_bar: int,
    spec: LabelWindowSpec,
) -> None:
    """断言标签起始 bar 在特征窗口结束之后（含可选 gap）。

    Parameters
    ----------
    feature_end_bar : int
        特征窗口最后一根 bar 的索引。
    label_start_bar : int
        标签窗口第一根 bar 的索引。
    spec : LabelWindowSpec
        含 ``gap_bars`` 的窗口规格。

    Raises
    ------
    PitSafetyError
        ``label_start_bar < feature_end_bar + gap`` 时抛出。
    """
    spec.validate_no_overlap()
    required = feature_end_bar + spec.min_separation_bars
    if label_start_bar < required:
        raise PitSafetyError(
            [
                f"label_feature_overlap: feature_end={feature_end_bar}, "
                f"label_start={label_start_bar}, gap={spec.min_separation_bars}",
            ]
        )


def build_forward_return_series(
    close: pd.Series,
    *,
    horizon: int = 1,
) -> pd.Series:
    """构造 forward return（仅用于标签；不可作为因子特征输入）。

    Parameters
    ----------
    close : pd.Series
        MultiIndex ``(timestamp, instrument)`` 收盘价序列。
    horizon : int
        前瞻 bar 数（默认 1）。

    Returns
    -------
    pd.Series
        ``close[t+h] / close[t] - 1``，与输入同 index 结构。
    """
    # R24-181: strict positive int — ``max(1, int(horizon))`` is forbidden.
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError(
            f"forward horizon must be a strict positive int (R24-181); got {horizon!r}"
        )
    h = horizon
    panel = close.unstack("instrument")
    fwd = panel.shift(-h) / panel - 1.0
    return fwd.stack(future_stack=True)


def default_mining_label_config(
    *,
    horizon_bars: int = 5,
    feature_lookback_bars: int = 20,
    gap_bars: int = 1,
    return_column: str = "vwap",
) -> dict[str, Any]:
    """挖掘/回测默认标签配置（与特征窗口显式隔离）。

    Parameters
    ----------
    horizon_bars : int
        标签前瞻 bar 数（默认 5）。
    feature_lookback_bars : int
        特征最大 lookback（默认 20，仅写入元数据）。
    gap_bars : int
        特征结束与标签起始之间的隔离 bar 数（默认 1）。
    return_column : str
        收益计算所用价格列（默认 ``"vwap"`` —— 全局 vwap-to-vwap 口径，
        AdjVwap(t+2)/AdjVwap(t+1)-1 对齐 TargetVwapReturnH01；复权权威
        StockDailyBarAdj 上 bare ``vwap`` 解析到 AdjVwap）。

    Returns
    -------
    dict[str, Any]
        含 ``label_formula``、``horizon_bars``、``gap_bars`` 等的 JSON 可序列化配置。
    """
    spec = LabelWindowSpec(
        horizon_bars=horizon_bars,
        feature_lookback_bars=feature_lookback_bars,
        gap_bars=gap_bars,
    )
    spec.validate_no_overlap()
    # R24-177/178: the config label formula is the SINGLE authority — a
    # backward ``ts_pct(close, h)`` would NOT match the runtime forward-return
    # builder.  The authority is the LabelIR's ``label_formula`` string.
    ir = LabelIR(
        op=LabelOp.FORWARD_RETURN,
        inputs=(return_column,),
        horizon_bars=spec.horizon_bars,
        gap_bars=spec.gap_bars,
    )
    return {
        "schema_version": "factor_engine.label_pit.v2",
        "return_column": return_column,
        "horizon_bars": spec.horizon_bars,
        "feature_lookback_bars": spec.feature_lookback_bars,
        "gap_bars": spec.gap_bars,
        "label_formula": ir.label_formula,
        "label_op": ir.op.value,
        "pit_note": (
            "标签使用未来收益；特征 IR 须通过 pit_audit，且 label_start >= feature_end + gap"
        ),
    }


def validate_label_formula_for_pit(formula: str, *, enforce: bool = True) -> dict[str, Any]:
    """校验标签 DSL：禁止负 lag / Lead 等前视算子进入标签流水线。

    Parameters
    ----------
    formula : str
        标签 DSL 公式字符串。
    enforce : bool
        为 ``True`` 时违规抛出 ``PitSafetyError``；否则仅写入 report。

    Returns
    -------
    dict[str, Any]
        含 ``ok``、``pit_safe``、``violations`` 等字段的审计报告。
    """
    from factor_engine.api.dsl_parser import parse_expr
    from factor_engine.ir.analyzer import Analyzer
    from factor_engine.runtime.pit_audit import assert_pit_safe

    expr = parse_expr(str(formula or "").strip())
    analysis = Analyzer().lower(expr)
    report = assert_pit_safe(
        analysis.ir,
        enforce=enforce,
        forbid_forward_fill=True,
    )
    return {
        "ok": report.passed,
        "formula": formula,
        "pit_safe": report.passed,
        "violations": list(report.violations),
    }


def align_feature_and_label_windows(
    feature_index: pd.Index,
    label_index: pd.Index,
    *,
    spec: LabelWindowSpec,
) -> tuple[pd.Index, pd.Index]:
    """裁剪特征/标签 index，保证时间轴上无重叠区间。

    Parameters
    ----------
    feature_index : pd.Index
        特征可用时间戳 index。
    label_index : pd.Index
        标签可用时间戳 index。
    spec : LabelWindowSpec
        含 ``gap_bars`` 的窗口规格。

    Returns
    -------
    tuple[pd.Index, pd.Index]
        ``(feature_index, trimmed_label_index)``；必要时裁剪标签侧。
    """
    spec.validate_no_overlap()
    if len(feature_index) == 0 or len(label_index) == 0:
        return feature_index, label_index

    feature_end = pd.Timestamp(feature_index.max())
    label_start = pd.Timestamp(label_index.min())
    min_label_start = feature_end + pd.Timedelta(days=spec.min_separation_bars)
    if label_start < min_label_start:
        trimmed_label = label_index[label_index >= min_label_start]
        return feature_index, trimmed_label
    return feature_index, label_index
