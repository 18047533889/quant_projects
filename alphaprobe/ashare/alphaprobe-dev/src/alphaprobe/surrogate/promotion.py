"""PromotionSurrogate + deterministic fallback + EVI promotion order.

plan.md Task 17 / Part F5:

    EVI = P(elite) × ExpectedGain / ExpectedCost

- **Deterministic fallback Predictor** (:class:`ThresholdPredictor`): when no
  trained surrogate exists (warmup / no history), promotion must fall back to
  the *existing funnel threshold semantics* and never guess blindly (Part G
  #13 neutral-defaults rule, Task 17 test 1).
- **Trained surrogate** (:class:`PromotionSurrogate`): any sklearn-compatible
  model may be injected (``fit(X, y)`` / ``predict_proba(X)``). History is
  accumulated with :meth:`observe`; :meth:`fit` is called explicitly once
  enough history exists. Labels may only come from research-caliber segments —
  any sealed-Test-flagged label raises :class:`SealedTestLabelError`
  (Part G #23, Task 17 test 2).
- **EVI ordering** (Task 17 test 3): candidates are ranked by
  ``EVI = P_elite × ExpectedGain / ExpectedCost`` — a high-probability,
  high-cost candidate loses to a moderate-probability, near-free one.
  Cost is *not* just logged: it enters the EVI numerator/denominator.

Model availability (Part G): this module **never imports** lightgbm/xgboost.
Fit probes ``predict_proba`` only; if the injected model lacks it we
fail-closed with ``ModelNotProbabilisticError`` (never silently punt). If the
caller has no model at all (warmup) ``recommend`` stays deterministic.

All switchable knobs live on :class:`PromotionSurrogateConfig` (#30).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from alphaprobe.surrogate.features import (
    ExtractedFeatures,
    ExtractionContext,
    FeatureExtractor,
    SealedTestLabelError,
)

#: 确定性 fallback 的保守 L3 过率先验（无观测 → 中性低门槛）。
DEFAULT_L3_PRIOR = 0.15
#: EVI 归一安全下限（防除 0）。
_EPS = 1e-12
#: surrogate 最小历史样本数（低于 → 拒绝 fit，保持确定性 fallback）。
MIN_HISTORY_FOR_FIT = 20


class ModelNotProbabilisticError(RuntimeError):
    """Injected model lacks ``predict_proba``.

    fail-closed: EVI needs P(elite); a non-probabilistic model must not be
    silently approximated.
    """


@runtime_checkable
class ProbabilisticModel(Protocol):
    """sklearn-compatible probabilistic classifier shape (duck-typed)."""

    def fit(self, X: Any, y: Any) -> "ProbabilisticModel": ...  # pragma: no cover

    def predict_proba(self, X: Any) -> Any: ...  # pragma: no cover


@dataclass
class PromotionSurrogateConfig:
    """EVI promotion 配置（#30 全开关，无硬编码行为分支）。

    Attributes
    ----------
    enabled : bool
        False = ``recommend`` 恒走确定性 fallback（ablation / 关 EVI）。
    min_history_for_fit : int
        低于此样本数拒绝 fit（保持确定性 fallback）。
    l3_prior : float
        确定性 fallback 的 P(L3 pass) 先验；历史 action success /
        L0 metric 只在该先验附近微调，不瞎猜。
    use_model : bool
        True 且已 fit → 用模型概率；否则确定性 fallback。
    gain_weight : float
        EVI 增益项系数（ExpectedGain 归一基准缩放）。
    """

    enabled: bool = True
    min_history_for_fit: int = MIN_HISTORY_FOR_FIT
    l3_prior: float = DEFAULT_L3_PRIOR
    use_model: bool = True
    gain_weight: float = 1.0


@dataclass
class PromotionSuggestion:
    """单条 promotion 建议（不写 gate，L4 接线留给后续 pipeline 集成）。"""

    factor_id: str = ""
    formula: str = ""
    p_l3_pass: float = 0.0
    expected_fitness_gain: float = 0.0
    expected_novelty_gain: float = 0.0
    expected_pool_gain: float = 0.0
    expected_cost: float = 0.0
    evi: float = 0.0
    source: str = "deterministic_fallback"  # 或 "model"

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "formula": self.formula,
            "p_l3_pass": self.p_l3_pass,
            "expected_fitness_gain": self.expected_fitness_gain,
            "expected_novelty_gain": self.expected_novelty_gain,
            "expected_pool_gain": self.expected_pool_gain,
            "expected_cost": self.expected_cost,
            "evi": self.evi,
            "source": self.source,
        }


class ThresholdPredictor:
    """确定性 fallback：回到现有 funnel 阈值语义，不瞎猜（Task 17 测试 1）。

    只消费特征里**已经算好**的数值（L0 metric / 历史 action success /
    parent fitness 摘要 / 复杂度），不做任何统计推断：
    - 历史同 action success 高 → P(L3) 从先验向上微调；
    - 复杂度高 → 向下微调（与 funnel 的复杂度拒绝语义一致）；
    - L0 静态通过（coverage_metric=1）→ 微调向上。
    调整幅度都被钳在 [0.05, 0.6]，避免在没有真实证据时给出极端概率。
    """

    def __init__(self, *, l3_prior: float = DEFAULT_L3_PRIOR) -> None:
        self.l3_prior = min(max(float(l3_prior), 0.0), 1.0)
        self._observed_positive_rate: float | None = None

    def set_observed_positive_rate(self, rate: float | None) -> None:
        """注入历史正例率（surrogate 未 fit 但已有观测时的后验先验）。"""
        if rate is None:
            self._observed_positive_rate = None
        else:
            self._observed_positive_rate = min(max(float(rate), 0.0), 1.0)

    def predict_p_l3(self, features: ExtractedFeatures) -> float:
        p = self.l3_prior
        if self._observed_positive_rate is not None:
            # 历史观测正例率 → 先验中心向观测移动（仍钳在保守带内，不瞎猜）
            p = 0.5 * p + 0.5 * self._observed_positive_rate
        named = dict(zip(features.names, features.as_row()))
        succ = named.get("action_success_rate")
        if succ is not None:
            p += 0.3 * (float(succ) - 0.5)
        coverage = named.get("l0_coverage_metric")
        if coverage is not None and float(coverage) >= 0.999:
            p += 0.1
        cx = named.get("complexity")
        if cx is not None and float(cx) > 12:
            p -= 0.1
        return min(max(p, 0.05), 0.6)

    def expected_cost(self, features: ExtractedFeatures) -> float:
        named = dict(zip(features.names, features.as_row()))
        cost = named.get("estimated_cost")
        if cost is not None and float(cost) > 0.0:
            return float(cost)
        qe = named.get("estimated_qe_seconds")
        if qe is not None and float(qe) > 0.0:
            return min(1.0, 0.1 + float(qe) / 3600.0)
        # 无成本估算 → 中性低成本（保持 EVI 不被瞎估的成本惩罚压垮）。
        return 0.15


@dataclass
class PromotionSurrogate:
    """可注入 sklearn 兼容模型的 surrogate + 确定性 fallback + EVI 建议序。

    ``observe`` 累积历史行（context 特征 + L3 结果标签）；``fit`` 显式触发
    模型训练（``n_obs >= min_history_for_fit`` 且 ``use_model`` 才可能真正 fit）；
    ``recommend`` 对一批 candidates 返回按 ``EVI`` 降序的建议。

    非 sklearn 依赖：模型经 ``ProbabilisticModel`` 鸭子协议注入，本模块不
    import lightgbm/xgboost/sklearn。无模型 / warmup → ``recommend`` 纯确定性。
    """

    extractor: FeatureExtractor = field(
        default_factory=lambda: _default_extractor()
    )
    config: PromotionSurrogateConfig = field(
        default_factory=PromotionSurrogateConfig
    )
    model: Any | None = field(default=None)
    fallback: ThresholdPredictor = field(
        default_factory=ThresholdPredictor
    )
    _history_X: list[ExtractedFeatures] = field(default_factory=list)
    _history_y: list[int] = field(default_factory=list)
    _model_fitted: bool = field(default=False, init=False)

    @property
    def is_trained(self) -> bool:
        return self._model_fitted and self.config.use_model and self.config.enabled

    @property
    def n_observations(self) -> int:
        return len(self._history_y)

    # -- 观测 / 训练 ---------------------------------------------------------

    def observe(self, ctx: ExtractionContext, label: Mapping[str, Any]) -> None:
        """累积一条历史行。label 只允许 research-caliber segment。

        Raises
        ------
        SealedTestLabelError
            label 带 sealed/hold-out/test 标记（Part G #23 fail-closed）。
        """
        y = _label_from_ctx(ctx, label)
        features = self.extractor.extract(ctx)
        self._history_X.append(features)
        self._history_y.append(y)
        # 观测即同步历史正例率到确定性 fallback 先验（不管模型开关如何，
        # 已观测历史都应让 fallback 从无信息先验向观测移动）。
        self._sync_fallback_to_history()

    def fit(self, *, force: bool = False) -> bool:
        """显式训练注入模型。返回 True = 真正 fit 了。

        - 样本不足 → 不 fit（False，保持确定性 fallback）；
        - ``use_model=False`` / ``enabled=False`` → 不 fit（#30）；
        - 模型无 ``predict_proba`` → :class:`ModelNotProbabilisticError`。

        无论是否真正 fit，只要观测到历史（且未拒绝 fit），就把历史正例率
        投影到确定性 fallback 的先验（观测后验 → fallback 不再是纯无信息
        先验，但仍钳在保守带内不瞎猜）。
        """
        if not self.config.enabled:
            return False
        if not self.config.use_model:
            return False
        if self.model is None:
            self._sync_fallback_to_history()
            return False
        if not force and len(self._history_y) < self.config.min_history_for_fit:
            self._sync_fallback_to_history()
            return False
        if not hasattr(self.model, "fit"):
            raise ModelNotProbabilisticError(
                f"injected model {type(self.model).__name__} has no fit()"
            )
        if not hasattr(self.model, "predict_proba"):
            raise ModelNotProbabilisticError(
                f"injected model {type(self.model).__name__} lacks predict_proba; "
                "EVI needs P(elite) — refusing to silently approximate"
            )
        X = _matrix_from_rows(self._history_X)
        y = self._history_y
        self.model.fit(X, y)
        self._model_fitted = True
        self._sync_fallback_to_history()
        return True

    def _sync_fallback_to_history(self) -> None:
        if self._history_y:
            rate = sum(int(v) for v in self._history_y) / len(self._history_y)
            self.fallback.set_observed_positive_rate(rate)

    def reset(self) -> None:
        """清空历史与 fitted 状态（测试/隔离用）。"""
        self._history_X.clear()
        self._history_y.clear()
        self._model_fitted = False
        self.fallback.set_observed_positive_rate(None)

    # -- 预测 ----------------------------------------------------------------

    def predict_p_l3(self, ctx: ExtractionContext) -> float:
        """P(L3 pass)：已训练且启用 → 模型概率；否则确定性 fallback。"""
        features = self.extractor.extract(ctx)
        if self.is_trained:
            # 确定性概率模型仍要经合法 [0,1] 校验（fail-closed：坏模型概率
            # 不静默放行到 EVI）。
            p = _predict_proba_positive(self.model, [features.as_row()])
            if not (0.0 <= p <= 1.0):
                return self.fallback.predict_p_l3(features)
            return float(p)
        return self.fallback.predict_p_l3(features)

    def expected_cost_of(self, ctx: ExtractionContext) -> float:
        """ExpectedCost（确定性估算；成本数据充分后由 ledger 覆盖注入）。"""
        features = self.extractor.extract(ctx)
        return self.fallback.expected_cost(features)

    # -- EVI / 建议序 ---------------------------------------------------------

    def recommend(
        self,
        candidates: Sequence[Any],
        *,
        gains: Mapping[str, float] | None = None,
        default_gain: float = 0.5,
    ) -> list[PromotionSuggestion]:
        """对一批 candidates 返回 EVI 降序建议（Task 17 测试 3）。

        candidate 可以是 :class:`ExtractionContext`、dict 或任何带
        ``extract_ctx`` 的对象。gains 可选（key = factor_id 或公式的期望增益）；
        未提供 → 用 L0/L1 特征派生中性增益。
        """
        rows: list[tuple[float, PromotionSuggestion]] = []
        for cand in candidates:
            ctx = _coerce_context(cand)
            p = self.predict_p_l3(ctx)
            cost = max(self.expected_cost_of(ctx), _EPS)
            gain = _resolve_gain(gains, ctx, default_gain)
            # EVI = P(elite) × ExpectedGain / ExpectedCost（gain_weight 缩放）
            evi = p * gain * float(self.config.gain_weight) / cost
            rows.append(
                (
                    evi,
                    PromotionSuggestion(
                        factor_id=ctx.factor_id,
                        formula=ctx.formula,
                        p_l3_pass=p,
                        expected_fitness_gain=gain,
                        expected_novelty_gain=gain * 0.5,
                        expected_pool_gain=gain * 0.3,
                        expected_cost=cost,
                        evi=evi,
                        source="model" if self.is_trained else "deterministic_fallback",
                    ),
                )
            )
        rows.sort(key=lambda r: r[0], reverse=True)
        return [s for _, s in rows]

    # -- 状态 -----------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        return {
            "trained": self.is_trained,
            "model_fitted": self._model_fitted,
            "enabled": self.config.enabled,
            "use_model": self.config.use_model,
            "n_observations": self.n_observations,
            "model": type(self.model).__name__ if self.model is not None else None,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state(),
            "config": {
                "enabled": self.config.enabled,
                "min_history_for_fit": self.config.min_history_for_fit,
                "l3_prior": self.config.l3_prior,
                "use_model": self.config.use_model,
                "gain_weight": self.config.gain_weight,
            },
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _default_extractor() -> FeatureExtractor:
    from alphaprobe.surrogate.features import DefaultFeatureExtractor

    return DefaultFeatureExtractor()


def _label_from_ctx(ctx: ExtractionContext, label: Mapping[str, Any]) -> int:
    """research-caliber segment 校验 + L3 标签提取（#23 fail-closed）。

    label 键约定：``segment`` / ``version`` / ``split`` 任一带 sealed 标记
    → raise；``l3_status`` / ``l3_pass`` 解析为 0/1。ctx 里的 segment 标记
    也参与校验（双保险）。
    """
    from alphaprobe.surrogate.features import (
        RESEARCH_SEGMENTS,
        SEALED_SEGMENT_MARKERS,
        SEALED_VERSION_MARKERS,
    )

    for key in ("segment", "split", "version"):
        raw = str(label.get(key) or "")
        if not raw:
            continue
        low = raw.strip().lower()
        marker_set = (
            SEALED_VERSION_MARKERS if key == "version" else SEALED_SEGMENT_MARKERS
        )
        if low in marker_set:
            raise SealedTestLabelError(
                f"surrogate target would read sealed test label ({key}={raw!r})"
            )
        if key != "version" and low not in RESEARCH_SEGMENTS:
            raise SealedTestLabelError(
                f"surrogate target segment {raw!r} is not research-caliber "
                f"(allowed: {sorted(RESEARCH_SEGMENTS)})"
            )
    # ctx.segment 也查一遍（ctx 可能自带头）
    ctx_seg = str(getattr(ctx, "segment", "") or "").strip().lower()
    if ctx_seg in SEALED_SEGMENT_MARKERS:
        raise SealedTestLabelError(
            f"surrogate target context marked sealed (segment={ctx_seg!r})"
        )
    raw_l3 = label.get("l3_status", label.get("l3_pass"))
    if raw_l3 is None:
        return 0
    return _coerce_l3_label(raw_l3)


def _coerce_l3_label(raw: Any) -> int:
    if isinstance(raw, str):
        low = raw.lower()
        if low in {"ok", "pass", "passed", "true", "1", "promoted", "accepted"}:
            return 1
        if low in {"fail", "reject", "rejected", "false", "0", "none", ""}:
            return 0
        if "pass" in low or "ok" in low or "accepted" in low:
            return 1
        return 0
    try:
        return 1 if float(raw) > 0.5 else 0
    except (TypeError, ValueError):
        return 0


def _predict_proba_positive(model: Any, X: Any) -> float:
    import numpy as np

    proba = model.predict_proba(X)
    arr = np.asarray(proba, dtype="float64")
    if arr.ndim == 1:  # binary 单列输出（部分包装器）
        return float(arr[0])
    row = arr[0]
    if row.shape[0] >= 2 and abs(float(row[0]) + float(row[1]) - 1.0) < 1e-6:
        return float(row[1])
    # 多类：取 index=1 列；若列数=1 → 视为正类概率
    return float(row[1]) if row.shape[0] >= 2 else float(row[0])


def _matrix_from_rows(rows: list[ExtractedFeatures]) -> Any:
    import numpy as np

    return np.asarray([r.as_row() for r in rows], dtype="float64")


def _coerce_context(cand: Any) -> ExtractionContext:
    if isinstance(cand, ExtractionContext):
        return cand
    if hasattr(cand, "extract_ctx") and callable(getattr(cand, "extract_ctx")):
        return cand.extract_ctx()
    if isinstance(cand, Mapping):
        return ExtractionContext(**{k: v for k, v in cand.items() if k in _CTX_FIELDS})
    raise TypeError(
        f"candidate must be ExtractionContext, mapping, or expose extract_ctx(); "
        f"got {type(cand).__name__}"
    )


_CTX_FIELDS = frozenset(ExtractionContext.__dataclass_fields__.keys())


def _resolve_gain(
    gains: Mapping[str, float] | None, ctx: ExtractionContext, default: float
) -> float:
    if gains:
        for key in (ctx.factor_id, ctx.formula):
            if key and key in gains:
                g = gains[key]
                try:
                    return max(float(g), 0.0)
                except (TypeError, ValueError):
                    break
    # 默认：L0 静态通过（passed 标记 / coverage>=0.5）→ 0.6；否则 0.4。
    mb = ctx.metric_bundle
    is_pass = None
    if isinstance(mb, Mapping):
        if "passed" in mb:
            is_pass = bool(mb["passed"])
        else:
            cov = mb.get("coverage")
            if cov is not None:
                try:
                    is_pass = float(cov) >= 0.5
                except (TypeError, ValueError):
                    is_pass = None
    elif hasattr(mb, "passed"):
        is_pass = bool(getattr(mb, "passed", False))
    return 0.6 if is_pass is not None and is_pass else 0.4


__all__ = [
    "DEFAULT_L3_PRIOR",
    "MIN_HISTORY_FOR_FIT",
    "ModelNotProbabilisticError",
    "ProbabilisticModel",
    "PromotionSuggestion",
    "PromotionSurrogate",
    "PromotionSurrogateConfig",
    "ThresholdPredictor",
]
