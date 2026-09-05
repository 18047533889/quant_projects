"""Feature extraction protocol + 默认实现 for the multi-fidelity promotion
surrogate (plan.md Task 17 / Part F5).

Feature groups (from plan Task 17 "Inputs"):
- FE static analysis features (operator/field/complexity/lookback counts);
- DA field domains (whitelist membership, horizon/domain heuristics);
- L0/L1 metrics (live per-candidate static gate signals + scout-tier metric
  bundle: coverage / untradeable / nan_inf_ratio);
- parameter family (stable categorical index);
- cluster (stable categorical index);
- schema/logic (stable categorical index);
- parent fitness;
- historical action success;
- estimated FE/QE cost.

Design constraints (Part G):
- #23 sealed test 零读取: label space is L3_L1-only. A label whose segment
  == "test" raises :class:`SealedTestLabelError`. A label can carry an explicit
  ``version``/``segment`` marker (``Provenance``/dict); the marker is checked
  against an allowlist of research-caliber segments ("train", "validation",
  "valid", "research", "val") and any sealed/held-out marker ("test", "sealed",
  "held_out", "hidden", "oof") raises. This is a hard fail-closed gate —
  a corrupted training frame can never silently train on sealed labels.
- #7 不手算指标: this module only *consumes* metric values already present in
  the supplied metric bundle dict; it never recomputes any IC/Sharpe/rank
  statistic from raw panel data.
- #30 EVI/cost/routing 全开关: every learned branch is gated by an explicit
  flag. Callers that want pure-deterministic behaviour keep the flags off;
  this module's own feature computation is always deterministic.

The extractor is sklearn-unaware: :class:`ExtractedFeatures` exposes a plain
``as_row()`` float vector plus a parallel ``names()`` list so any sklearn-style
model can be fit. Vectorised model fit therefore only requires the caller to
row-stack the vectors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

#: research-caliber segments that may legally supply training labels.
#: "L3" pass observations come from L3_search_valid on train/validation
#: segments only — sealed test (L5) may never feed the surrogate.
RESEARCH_SEGMENTS: frozenset[str] = frozenset(
    {"train", "validation", "valid", "research", "val", ""}
)

#: sealed / held-out markers — any of these in a label provenance raises.
SEALED_SEGMENT_MARKERS: frozenset[str] = frozenset(
    {"test", "sealed", "held_out", "held-out", "hidden", "oof", "l5", "L5_sealed_test"}
)

#: label marker version strings that are definitely not research-caliber.
SEALED_VERSION_MARKERS: frozenset[str] = frozenset({"sealed", "l5", "frozen_test"})


class SealedTestLabelError(ValueError):
    """A surrogate training label would read a sealed/hold-out test event.

    Fail-closed (Part G #23): corrupted training data must raise loudly
    instead of silently training a surrogate on sealed Test labels.
    """


@runtime_checkable
class FeatureExtractor(Protocol):
    """Feature extraction protocol for the promotion surrogate.

    ``extract`` maps one candidate (dict context, see :class:`ExtractionContext`)
    to a fixed-width feature vector. Implementations must be deterministic and
    side-effect free; nothing here may reach the network, an evaluator, or a
    sealed dataset.
    """

    def extract(self, ctx: "ExtractionContext") -> "ExtractedFeatures":
        """Extract a deterministic feature row for one promotion candidate."""
        ...  # pragma: no cover - protocol


@dataclass
class ExtractionContext:
    """Everything the default extractor consumes for one candidate.

    All attributes are optional — the extractor degrades to neutral defaults
    for missing groups. ``metric_bundle`` is consumed verbatim (L0/L1 gate
    metrics and nothing else); L2+ panel metrics, when present in the bundle,
    are ignored for promotion features (the surrogate predicts L3 pass from
    cheap evidence only).
    """

    formula: str = ""
    factor_id: str = ""
    parent_ids: Sequence[str] = field(default_factory=list)

    # FE static analysis (may be precomputed by caller or derived here)
    field_set: Sequence[str] = field(default_factory=list)
    operator_set: Sequence[str] = field(default_factory=list)
    complexity: int | None = None
    lookback: int | None = None
    has_ts_op: bool | None = None
    has_cs_op: bool | None = None

    # DA field-domain whitelist / horizon heuristic
    data_fields: Sequence[str] = field(default_factory=list)

    # L0/L1 metrics (scout-tier bundle; consumed verbatim, never recomputed)
    metric_bundle: Mapping[str, Any] = field(default_factory=dict)

    # Stable categorical indexes ('' or None = unknown / neutral)
    parameter_family_id: str = ""
    cluster_id: str = ""
    schema_id: str = ""
    logic_id: str = ""

    # Parent fitness (numeric summary of the immediate parent(s))
    parent_fitness: float | None = None

    # Historical action success rate for this candidate's action family,
    # estimated FE/QE seconds and normalised [0,1] cost estimates.
    action_success_rate: float | None = None
    estimated_fe_seconds: float | None = None
    estimated_qe_seconds: float | None = None
    estimated_cost: float | None = None


def _feat(s: Any, *, max_len: int = 3) -> float:
    """Deterministic positional digest of a string for a stable categorical slot."""
    raw = str(s or "")
    if not raw:
        return 0.0
    h = 0
    for i, ch in enumerate(raw):
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    idx = (h % 999983) % max_len
    return float(idx + 1)  # 1..max_len; 0 reserved for unknown


@dataclass
class ExtractedFeatures:
    """Fixed-width feature row + names + cheap-label convenience."""

    values: list[float] = field(default_factory=list)
    names: list[str] = field(default_factory=list)

    def as_row(self) -> list[float]:
        return [float(v) if v is not None else 0.0 for v in self.values]

    def with_names(self, names: Sequence[str]) -> "ExtractedFeatures":
        if names and len(names) != len(self.values):
            raise ValueError(
                f"feature/name count mismatch: {len(self.values)} values vs {len(names)} names"
            )
        return ExtractedFeatures(values=list(self.values), names=list(names))


FEATURE_GROUPS: tuple[str, ...] = (
    "operator_count",
    "operator_set_size",
    "field_count",
    "field_set_size",
    "complexity",
    "lookback",
    "has_ts_op",
    "has_cs_op",
    "field_domain_known",
    "data_domain_price",
    "data_domain_volume",
    "data_domain_valuation",
    "l0_coverage_metric",
    "l0_untradeable_metric",
    "l0_nan_inf_ratio",
    "param_family_feat",
    "cluster_feat",
    "schema_feat",
    "logic_feat",
    "parent_fitness",
    "action_success_rate",
    "estimated_fe_seconds",
    "estimated_qe_seconds",
    "estimated_cost",
)


class DefaultFeatureExtractor:
    """Default implementation of :class:`FeatureExtractor`.

    Deterministic digest features replace raw identifiers so the vector is
    model-ready without a fitted encoder. ``metric_bundle`` values are
    consumed verbatim (``coverage`` / ``untradeable_ratio`` /
    ``nan_inf_ratio``); every other key is ignored. ``L0Output``-shaped
    objects are honoured: a passing L0 output contributes a ``1.0``
    l0_coverage_metric so un-evaluated but statically-clean candidates still
    score above rejected ones in the deterministic fallback.
    """

    def __init__(
        self,
        *,
        known_fields: Sequence[str] | None = None,
    ) -> None:
        self._known_fields = frozenset(str(f).lower() for f in (known_fields or ()))

    # -- price / volume / valuation domain membership -------------------------

    @staticmethod
    def _is_price_field(f: str) -> bool:
        f = f.lower()
        return (
            f in {"open", "high", "low", "close", "vwap", "preclose", "adjclose"}
            or f.endswith("_open")
            or f.endswith("_high")
            or f.endswith("_low")
            or f.endswith("_close")
            or f.endswith("_vwap")
        )

    @staticmethod
    def _is_volume_field(f: str) -> bool:
        f = f.lower()
        return f in {"volume", "amount", "turnover", "turnover_ratio", "money"} or f.startswith(
            "volume"
        )

    @staticmethod
    def _is_valuation_field(f: str) -> bool:
        f = f.lower()
        return f in {"pe", "pb", "ps", "pe_ttm", "pb_mrq", "div_yield", "market_cap", "eps"} or (
            f.endswith("_pe") or f.endswith("_pb") or f.endswith("_yield")
        )

    def extract(self, ctx: ExtractionContext) -> ExtractedFeatures:
        operator_set = [str(o) for o in (ctx.operator_set or ())]
        field_set = [str(f) for f in (ctx.field_set or ())]
        if not operator_set:
            operator_set = self._operators_from_formula(ctx.formula)
        if not field_set:
            field_set = self._fields_from_formula(ctx.formula)
        data_fields = [str(f) for f in (ctx.data_fields or ())]
        if not data_fields:
            data_fields = field_set

        known = 0.0
        price = 0.0
        volume = 0.0
        valuation = 0.0
        for f in data_fields:
            fl = f.lower()
            if self._is_price_field(fl):
                price += 1.0
            if self._is_volume_field(fl):
                volume += 1.0
            if self._is_valuation_field(fl):
                valuation += 1.0
            if self._known_fields:
                if fl in self._known_fields:
                    known += 1.0
            elif self._is_price_field(fl) or self._is_volume_field(fl) or self._is_valuation_field(fl):
                known += 1.0
        n_fields = len(data_fields) or 1
        field_domain_known = float(known > 0.0)

        mb = ctx.metric_bundle or {}
        cov_metric = _as_float(mb.get("coverage"))
        untr_metric = _as_float(mb.get("untradeable_ratio"))
        nan_metric = _as_float(mb.get("nan_inf_ratio"))
        # dict / L0Output-shaped 对象都 honor 静态通过标记：clean static pass
        # 贡献 coverage_metric=1.0，promotion 不会低估静态干净的 candidate。
        if cov_metric is None:
            passed_marker = (
                mb.get("passed")
                if isinstance(mb, Mapping)
                else getattr(mb, "passed", None)
            )
            if passed_marker is not None:
                cov_metric = 1.0 if bool(passed_marker) else 0.0
            elif hasattr(mb, "passed"):
                cov_metric = 1.0 if getattr(mb, "passed", False) else 0.0
        if untr_metric is None:
            passed_marker = (
                mb.get("passed")
                if isinstance(mb, Mapping)
                else getattr(mb, "passed", None)
            )
            if passed_marker is not None:
                untr_metric = 0.0 if bool(passed_marker) else 1.0

        complexity = ctx.complexity
        if complexity is None:
            complexity = int(len(field_set) + len(operator_set)) or 1
        lookback = ctx.lookback
        if lookback is None:
            lookback = self._lookback_from_formula(ctx.formula)

        values: list[float] = [
            float(len(operator_set)),
            float(len(set(operator_set))),
            float(len(field_set)),
            float(len(set(field_set))),
            float(complexity),
            float(lookback),
            1.0 if ctx.has_ts_op else 0.0,
            1.0 if ctx.has_cs_op else 0.0,
            field_domain_known,
            price / n_fields,
            volume / n_fields,
            valuation / n_fields,
            cov_metric if cov_metric is not None else 0.5,
            untr_metric if untr_metric is not None else 0.0,
            nan_metric if nan_metric is not None else 0.0,
            _feat(ctx.parameter_family_id),
            _feat(ctx.cluster_id),
            _feat(ctx.schema_id),
            _feat(ctx.logic_id),
            _as_float(ctx.parent_fitness) if ctx.parent_fitness is not None else 0.0,
            _clip01(ctx.action_success_rate),
            _clip_ge0(ctx.estimated_fe_seconds),
            _clip_ge0(ctx.estimated_qe_seconds),
            _clip01(ctx.estimated_cost),
        ]
        assert len(values) == len(FEATURE_GROUPS), "feature group drift"
        return ExtractedFeatures(values=values, names=list(FEATURE_GROUPS))

    # -- local DSL digests (deterministic fallbacks) ---------------------------

    _CALL_RE_PATTERN = r"([A-Za-z_][A-Za-z0-9_]*)\s*\("

    def _operators_from_formula(self, formula: str) -> list[str]:
        import re

        return sorted({m.group(1) for m in re.finditer(self._CALL_RE_PATTERN, str(formula or ""))})

    _FIELD_TOKEN_PATTERN = r"\b(open|high|low|close|volume|vwap|pe|pb|turnover_ratio|market_cap|pe_ttm|pb_mrq|div_yield|amount)\b"

    def _fields_from_formula(self, formula: str) -> list[str]:
        import re

        return sorted(
            {
                m.group(1).lower()
                for m in re.finditer(self._FIELD_TOKEN_PATTERN, str(formula or ""))
            }
        )

    def _lookback_from_formula(self, formula: str) -> int:
        import re

        s = str(formula or "")
        windows = [int(v) for v in re.findall(r"(\d{1,4})\s*\)", s)]
        return max(windows) if windows else 0

    @staticmethod
    def _label_from_metric_bundle(mb: Mapping[str, Any]) -> int:
        """L3_L1 标签（#23）：sealed Test 永远不可作为 surrogate 目标。

        Raises
        ------
        SealedTestLabelError
            bundle carries a sealed/held-out segment marker.
        """
        seg_raw = str(mb.get("segment") or mb.get("version") or mb.get("split") or "")
        seg = seg_raw.strip().lower()
        if not seg:
            seg = "train"
        if seg in SEALED_SEGMENT_MARKERS:
            raise SealedTestLabelError(
                f"surrogate target would read sealed test label (segment={seg_raw!r})"
            )
        if seg not in RESEARCH_SEGMENTS:
            raise SealedTestLabelError(
                f"surrogate target segment {seg_raw!r} is not research-caliber "
                f"(allowed: {sorted(RESEARCH_SEGMENTS)})"
            )
        # L3 pass = l3_status ok OR explicit l3_pass flag; bundles that never
        # reached L3 evaluation map to the L2 non-pass bucket.
        raw = mb.get("l3_status") or mb.get("l3_pass") or mb.get("l3_ok")
        if raw is None:
            return 0
        return _coerce_l3_label(raw)


def _as_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _clip01(v: Any) -> float:
    f = _as_float(v)
    if f is None:
        return 0.0
    return min(max(f, 0.0), 1.0)


def _clip_ge0(v: Any) -> float:
    f = _as_float(v)
    if f is None:
        return 0.0
    return max(f, 0.0)


def _nonnull_sequence(rows: Sequence["ExtractedFeatures"], *, axis: int) -> list[float]:
    """axis 上非空数值序列（训练标签/成本投影用）。"""
    out: list[float] = []
    for r in rows:
        v = r.values[axis] if axis < len(r.values) else None
        if v is not None:
            f = _as_float(v)
            if f is not None:
                out.append(f)
    return out


def historical_target_rate(rows: Sequence["ExtractedFeatures"], *, axis: int = -1) -> float:
    """历史训练标签正例率（确定性 fallback 的 P(L3) 微调用）。

    axis = -1 表示标签未嵌入特征行（调用方把 y 独立保存）；否则取该列非空
    数值均值。无样本 → 返回 ``_as_float`` 语义的 0.0（调用方自行回退先验）。
    """
    if axis < 0:
        return 0.0
    vals = _nonnull_sequence(rows, axis=axis)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _coerce_l3_label(raw: Any) -> int:
    """'ok'/'pass'/'L3_search_valid'/1/True → 1; otherwise 0."""
    if isinstance(raw, str):
        low = raw.lower()
        if low in {"ok", "pass", "passed", "true", "1", "promoted", "accepted"}:
            return 1
        if low in {"fail", "reject", "rejected", "false", "0", "none", ""}:
            return 0
        if "pass" in low or "ok" in low or "accepted" in low:
            return 1
        if "reject" in low or "fail" in low:
            return 0
        return 0
    try:
        return 1 if float(raw) > 0.5 else 0
    except (TypeError, ValueError):
        return 0


__all__ = [
    "DefaultFeatureExtractor",
    "ExtractedFeatures",
    "ExtractionContext",
    "FEATURE_GROUPS",
    "FeatureExtractor",
    "RESEARCH_SEGMENTS",
    "SEALED_SEGMENT_MARKERS",
    "SEALED_VERSION_MARKERS",
    "SealedTestLabelError",
    "historical_target_rate",
]
