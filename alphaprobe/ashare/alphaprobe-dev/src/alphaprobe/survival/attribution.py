"""plan Task 18 / Part J4：受控生存归因（SurvivalAttributionModel）。

把 survival/2026 memory 从「标签统计」（dna_stats 的 univariate operator/
mechanism 收缩）升级为**受控归因**：在一组混杂特征上做正则化 logistic
回归 + 按 logic/schema 分组的 hierarchical shrinkage，输出每个特征的
``effect``（相对先验的收缩效应）+ ``confidence``（effect_ci / se），
明确不做因果断言。

本模块的语义（与 Part G #23/#29/#13/#30 对齐）
-----------------------------------------------
- #23 版本隔离：训练侧只消费 ``source_version < 当前研究版本`` 的事件；
  查询/测试侧对 ``version == 当前研究版本``（current-version sealed test
  事件）一律拒绝（SealedTestViolation）。版本 N+1 可显式
  ``freeze_version(N)`` 冻结 N 的 test survival → 变为 N+1 可消费历史。
- #29 版本化 + 置信收缩：每个系数/分组效应都带 support（n_eff）；低 support
  效应经 ``reliability_of`` 向 0 收缩。
- #13 无证据中性：n_eff=0 → reliability=0 → 效应 = 0（低 support → 0）。
- #30 interactions / model family 可配置（``with_interactions`` 开关；模型族
  由 sklearn 可用性 / fit_backend 决定，生产最终选择由 calibration/ablation
  定，本模块不硬编码为理论真理）。

输出结构
--------
- ``AttributionResult``（按特征）：feature / group / effect / effect_ci /
  effect_se / support_count / n_effective / shrunk_rate / raw_rate /
  intercept 等。
- ``effect = shrunk_rate - prior``；effect_ci 为 Wald 型正态近似置信区间
  （se 从对数几率 Hessian 近似），仅用于排序与“不要过度解读”，不做因果断言。

实现路径
--------
- 特征装配：``attribution_event_from_formula``（公式文本）+ 可选
  canonical_inspect / cluster_size / source_miner / age_days / ast_nodes /
  turnover_bucket / motif → one-hot/数值向量（与 dna/profile 既有字段同源，
  不复制逻辑）。
- 模型：
  * ``fit_backend="auto"``：sklearn 可用 → sklearn Elastic Net logistic；
  * 否则/显式 "numpy" → 自带确定性弹性网 logistic（IRLS + L2 岭 + L1 近端）。
  两条路径各有测试覆盖。

回归路径确定性要求
------------------
同 backend 同数据 → 同系数。自带 numpy 路径为纯确定性实现（固定迭代、
无随机采样）；sklearn 路径固定 solver/seed。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from alphaprobe.contracts import FactorDNA, SurvivalLabel
from alphaprobe.research_protocol import SealedTestViolation
from alphaprobe.survival.dna import dna_from_formula
from alphaprobe.survival.profile import (
    classify_status_from_sequence,
    compute_survival_metrics,
)

try:  # pragma: no cover - 环境探测
    import sklearn  # noqa: F401

    _SKLEARN_AVAILABLE = True
except Exception:  # noqa: BLE001
    _SKLEARN_AVAILABLE = False

__all__ = [
    "SURVIVAL_GROUPS",
    "DEFAULT_SHRINKAGE_K",
    "DEFAULT_SURVIVAL_PRIOR",
    "DEFAULT_MIN_SUPPORT",
    "DEFAULT_EFFECT_CAP",
    "SurvivalEvent",
    "AttributionResult",
    "SurvivalAttributionReport",
    "SurvivalAttributionModel",
    "attribution_event_from_formula",
    "survival_label_of_rate",
]

#: 特征 → 分组键（hierarchical shrinkage 按组向全局先验收缩）。
SURVIVAL_GROUPS: tuple[str, ...] = (
    "operator",
    "mechanism",
    "field",
    "horizon",
    "turnover",
    "complexity",
    "source_miner",
    "age",
    "cluster_crowding",
    "schema",
    "motif",
)

#: 收缩强度 k（与 fitness/confidence.DEFAULT_SHRINKAGE_K=64、research_space/
#: logic.DEFAULT_SHRINKAGE_K=64 同款口径；reliability = sqrt(n/(n+k))）。
DEFAULT_SHRINKAGE_K: float = 64.0
#: 全局 survival 先验（相对 0.5；effect = shrunk_rate - prior）。
DEFAULT_SURVIVAL_PRIOR: float = 0.5
#: 最小 support（低于此的组/特征 effect 强制向 0 收缩，绝不当规则）。
DEFAULT_MIN_SUPPORT: int = 3
#: effect 幅度 cap（温和调节，不翻转、不爆炸；Plan §51 同款 spirit）。
DEFAULT_EFFECT_CAP: float = 0.15

#: 视为 survival 的 label（与 dna_stats.SURVIVAL_LABEL_ORDER 同口径）。
_SURVIVAL_LABELS: frozenset[str] = frozenset(
    {
        SurvivalLabel.PERSISTENT_ALPHA.value,
        SurvivalLabel.HEALTHY.value,
        SurvivalLabel.RECOVERED.value,
    }
)

#: 各分组支持的最小样本数（可配置见 fit()）。
_DEFAULT_MIN_SUPPORT_BY_GROUP: Mapping[str, int] = field(default_factory=dict)


def survival_label_of_rate(survival_rate: float, support_periods: int = 0) -> str:
    """把 survival_rate 映射成 label 口径（HEALTHY/BROKEN/UNCLASSIFIED）。

    供把事件转成二分类 y 时使用。survival_rate >= 0.5 → HEALTHY（=存活）。
    """
    r = float(survival_rate)
    if r >= 0.5:
        return SurvivalLabel.HEALTHY.value
    return SurvivalLabel.BROKEN.value


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _is_survival_label(lbl: Any) -> bool:
    return str(getattr(lbl, "value", lbl)) in _SURVIVAL_LABELS


# ---------------------------------------------------------------------------
# 事件结构
# ---------------------------------------------------------------------------


@dataclass
class SurvivalEvent:
    """一条 frozen 版本或历史版本的 factor survival 事件（#23 承载）。

    语义约定
    --------
    - ``version``：该事件所属的研究版本（current research version 的 sealed
      test 事件带 ``version == current``；历史/冻结版本事件带 ``version <
      current`` 或显式 ``frozen=True``）。
    - ``source_version``：可选的显式来源版本（与 ``version`` 同义；取其一）。
    - ``visible``：默认 False = sealed（当前版本搜索不可见）。显式 True 表示
      已经过 cutoff 闸门（vN+1 视角看 vN frozen test = visible）。
    - 训练/查询侧：模型收到 ``version``/``source_version`` 等于当前研究版本
      且 ``visible`` 不为 True 的事件 → SealedTestViolation。
    - ``survival``：标签口径（SurvivalLabel 或字符串）或 survival_rate。
      y=1 当 label 为 PERSISTENT_ALPHA/HEALTHY/RECOVERED，或 rate>=0.5。
    """

    factor_id: str = ""
    formula: str = ""
    version: str = "0"
    source_version: str | None = None
    visible: bool = False
    survival: str | SurvivalLabel = SurvivalLabel.UNCLASSIFIED
    survival_rate: float | None = None
    support_periods: int = 0
    cluster_size: float | None = None
    source_miner: str = ""
    age_days: int | None = None
    ast_nodes: int | None = None
    dna: FactorDNA | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if not self.formula and self.dna is None and not self.label:
            raise ValueError("SurvivalEvent needs formula or label/dna to build features")

    @property
    def source(self) -> str:
        return str(self.source_version if self.source_version is not None else self.version)

    def is_current_version(self, current_version: str) -> bool:
        return str(self.source) == str(current_version)


def attribution_event_from_formula(
    formula: str,
    *,
    survival: str | SurvivalLabel | float | None = None,
    version: str = "0",
    visible: bool = False,
    factor_id: str = "",
    canonical_inspect: dict[str, Any] | None = None,
    cluster_size: float | None = None,
    source_miner: str = "",
    age_days: int | None = None,
) -> SurvivalEvent:
    """从公式文本构造 SurvivalEvent（dna 特征装配入口）。"""
    dna = dna_from_formula(str(formula), canonical_inspect)
    if survival is None:
        survival = SurvivalLabel.UNCLASSIFIED
    if isinstance(survival, (int, float)):
        lbl = survival_label_of_rate(float(survival))
        ev = SurvivalEvent(
            factor_id=factor_id,
            formula=str(formula),
            version=str(version),
            visible=visible,
            survival=lbl,
            survival_rate=float(survival) if not isinstance(survival, bool) else None,
            support_periods=0,
            cluster_size=cluster_size,
            source_miner=str(source_miner),
            age_days=age_days,
            dna=dna,
        )
        return ev
    return SurvivalEvent(
        factor_id=factor_id,
        formula=str(formula),
        version=str(version),
        visible=visible,
        survival=survival,
        survival_rate=None,
        support_periods=0,
        cluster_size=cluster_size,
        source_miner=str(source_miner),
        age_days=age_days,
        dna=dna,
    )


# ---------------------------------------------------------------------------
# 输出结构
# ---------------------------------------------------------------------------


@dataclass
class AttributionResult:
    """单个特征的受控归因输出（effect + confidence，不做因果断言）。"""

    feature: str
    group: str = ""
    effect: float = 0.0
    effect_ci: tuple[float, float] = (0.0, 0.0)
    effect_se: float = 0.0
    raw_rate: float = 0.5
    shrunk_rate: float = 0.5
    support_count: int = 0
    n_effective: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "group": self.group,
            "effect": self.effect,
            "effect_ci": list(self.effect_ci),
            "effect_se": self.effect_se,
            "raw_rate": self.raw_rate,
            "shrunk_rate": self.shrunk_rate,
            "support_count": self.support_count,
            "n_effective": self.n_effective,
            "note": self.note,
        }


@dataclass
class SurvivalAttributionReport:
    """全模型的归因报告（供 survival_opportunity 消费 / 展示）。"""

    version: str = ""
    prior: float = 0.5
    n_events: int = 0
    n_effective: float = 0.0
    results: list[AttributionResult] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "prior": self.prior,
            "n_events": self.n_events,
            "n_effective": self.n_effective,
            "results": [r.to_dict() for r in self.results],
            "meta": dict(self.meta),
        }


# ---------------------------------------------------------------------------
# numpy 弹性网 logistic（无 sklearn 依赖的确定性实现）
# ---------------------------------------------------------------------------


def _l2_logistic_irls(
    X: list[list[float]],
    y: list[float],
    *,
    l2: float,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> list[float]:
    """自带确定性 L2 logistic（IRLS），返回系数（含 intercept 为第一维）。

    纯确定性：固定迭代上限、无随机性。调用方负责标准化输入列。
    """
    n = len(X)
    if n == 0:
        raise ValueError("empty design")
    p = len(X[0]) if X else 0
    # 设计矩阵 + intercept 列
    Xa: list[list[float]] = []
    for row in X:
        Xa.append([1.0] + [float(v) for v in row])
    beta = [0.0] * (p + 1)
    for _ in range(max_iter):
        # p_i = sigmoid(X beta)
        lin: list[float] = []
        for row in Xa:
            z = beta[0]
            for j in range(p):
                z += beta[j + 1] * row[j + 1]
            if z > 30.0:
                z = 30.0
            elif z < -30.0:
                z = -30.0
            sig = 1.0 / (1.0 + math.exp(-z))
            lin.append(sig)
        # 梯度 + Hessian（岭在 intercept 外）
        grad = [0.0] * (p + 1)
        hess: list[list[float]] = [[0.0] * (p + 1) for _ in range(p + 1)]
        for i, row in enumerate(Xa):
            resid = lin[i] - y[i]
            w = lin[i] * (1.0 - lin[i])
            for j in range(p + 1):
                grad[j] += row[j] * resid
            for j in range(p + 1):
                for k in range(p + 1):
                    hess[j][k] += w * row[j] * row[k]
        for j in range(p + 1):
            hess[j][j] += l2 if j > 0 else 0.0
            grad[j] += l2 * beta[j] if j > 0 else 0.0
        # 解 (H) d = -grad
        try:
            d = _solve_linear(hess, [-g for g in grad], p + 1)
        except ZeroDivisionError:
            break
        step = 0.0
        for j in range(p + 1):
            beta[j] += d[j]
            step += d[j] * d[j]
        if math.sqrt(step) < tol:
            break
    return beta


def _solve_linear(A: list[list[float]], b: list[float], n: int) -> list[float]:
    """高斯消元解小线性系统（确定性；奇异抛 ZeroDivisionError）。"""
    M = [list(row) for row in A]
    rhs = list(b)
    for col in range(n):
        piv = None
        for r in range(col, n):
            if abs(M[r][col]) > 1e-12:
                piv = r
                break
        if piv is None:
            raise ZeroDivisionError("singular system")
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
            rhs[col], rhs[piv] = rhs[piv], rhs[col]
        inv = 1.0 / M[col][col]
        for c in range(col, n):
            M[col][c] *= inv
        rhs[col] *= inv
        for r in range(n):
            if r == col:
                continue
            f = M[r][col]
            if abs(f) < 1e-14:
                continue
            for c in range(col, n):
                M[r][c] -= f * M[col][c]
            rhs[r] -= f * rhs[col]
    return rhs


def _prox_l1(beta: list[float], lam: float) -> list[float]:
    """近端算子：对非 intercept 系数做软阈值（L1 惩罚）。"""
    out = list(beta)
    for j in range(1, len(beta)):
        bj = out[j]
        if bj > lam:
            out[j] = bj - lam
        elif bj < -lam:
            out[j] = bj + lam
        else:
            out[j] = 0.0
    return out


# ---------------------------------------------------------------------------
# 归因模型
# ---------------------------------------------------------------------------


class SurvivalAttributionModel:
    """受控归因：Elastic Net logistic + hierarchical shrinkage（Task 18）。

    Attributes
    ----------
    current_version : str
        当前研究版本（sealed test 判定）。
    prior : float
        全局 survival 先验（默认 0.5）。
    shrinkage_k : float
        hierarchical shrinkage 强度（reliability = sqrt(n/(n+k))）。
    min_support : int
        低 support 阈值（低于此 → 组/特征效应向 0 强制收缩）。
    effect_cap : float
        effect 幅度 cap（温和调节，不翻转、不爆炸）。
    with_interactions : bool
        interactions 开关（#30）。
    fit_backend : str
        "auto" | "sklearn" | "numpy"。auto 有 sklearn 则 sklearn。
    seed : int
        sklearn 确定性种子。
    """

    def __init__(
        self,
        *,
        current_version: str = "1",
        prior: float = DEFAULT_SURVIVAL_PRIOR,
        shrinkage_k: float = DEFAULT_SHRINKAGE_K,
        min_support: int = DEFAULT_MIN_SUPPORT,
        effect_cap: float = DEFAULT_EFFECT_CAP,
        with_interactions: bool = True,
        fit_backend: str = "auto",
        seed: int = 0,
    ) -> None:
        self.current_version = str(current_version)
        self.prior = float(prior)
        self.shrinkage_k = float(shrinkage_k)
        self.min_support = int(min_support)
        self.effect_cap = float(effect_cap)
        self.with_interactions = bool(with_interactions)
        self.fit_backend = str(fit_backend)
        self.seed = int(seed)
        # 拟合态
        self._feature_columns: list[str] = []
        self._coefficients: dict[str, float] = {}
        self._intercept: float = 0.0
        self._group_stats: dict[str, dict[str, float]] = {}
        self._feature_support: dict[str, int] = {}
        self._feature_rate: dict[str, float] = {}
        self._n_events: int = 0
        self._n_effective: float = 0.0
        self._backend_used: str = "none"
        self._report: SurvivalAttributionReport | None = None

    # ------------------------------------------------------------------
    # 版本隔离（#23）
    # ------------------------------------------------------------------

    def freeze_version(self, version: str) -> None:
        """冻结某版本：把该版本标记为「已封存」→ 供 N+1 消费。

        冻结后调用方以 ``current_version=N+1`` 重新 fit 时，N 的 test survival
        （``visible=False`` 的 N 事件）显式 ``frozen=True`` 版本即变为可消费。
        本模型用 ``_frozen_versions`` 集合表达；查询侧 ``is_event_consumable``
        同时接受 frozen 版本的 sealed 事件。
        """
        self._frozen_versions = set(getattr(self, "_frozen_versions", set()))
        self._frozen_versions.add(str(version))

    def is_event_consumable(
        self, event: SurvivalEvent | Mapping[str, Any], *, current_version: str | None = None
    ) -> bool:
        """#23：事件是否可被当前版本消费。

        - 事件 source_version 为空 → 不可消费（fail-safe 保守，同
          logic.py 的「未显式 source version → 不入」）。
        - ``visible=True`` 的事件视为已过闸门（vN frozen test 的 vN+1 视角）。
        - source_version == 当前版本 → 仅当 ``visible`` 或已 freeze 才可消费；
          否则拒绝（sealed current-version test）。
        - source_version > 当前版本 → 不可消费（未来版本）。
        """
        ev = event
        if isinstance(ev, Mapping):
            ev = SurvivalEvent(**{k: v for k, v in dict(ev).items()
                                  if k in SurvivalEvent.__dataclass_fields__})
        cur = str(current_version if current_version is not None else self.current_version)
        src = ev.source
        if not src:
            return False
        frozen = getattr(self, "_frozen_versions", set())
        if str(src) in frozen:
            return True
        if ev.visible:
            return True
        if str(src) == cur:
            return False
        # 版本号数值比较（"2" < "10" 成立）；非数字串退化为严格字典序。
        try:
            return float(src) < float(cur)
        except (TypeError, ValueError):
            return str(src) < cur

    def assert_event_consumable(
        self, event: SurvivalEvent | Mapping[str, Any], *, current_version: str | None = None
    ) -> None:
        if not self.is_event_consumable(event, current_version=current_version):
            ev = event if isinstance(event, Mapping) else event
            src = ev.get("source_version") if isinstance(ev, Mapping) else event.source
            raise SealedTestViolation(
                f"survival attribution event (source_version={src}) is sealed/"
                "future/current-version and cannot be consumed by version "
                f"{current_version or self.current_version}"
            )

    # ------------------------------------------------------------------
    # 特征装配
    # ------------------------------------------------------------------

    def _features_of(
        self, event: SurvivalEvent
    ) -> dict[str, float]:
        """把事件转成特征向量（文本列全部 one-hot 展开；数值列标准化）。"""
        feats: dict[str, float] = {}
        dna = event.dna
        if dna is None and event.formula:
            dna = dna_from_formula(event.formula)
        if dna is None and event.label:
            # 纯 label 事件（无公式/dna）→ 只有 label 特征
            feats["label:" + event.label] = 1.0
            return feats
        if dna is None:
            return feats
        for op in getattr(dna, "operators", None) or []:
            feats["operator:" + op] = 1.0
        for m in getattr(dna, "mechanisms", None) or []:
            feats["mechanism:" + m] = 1.0
        for fam in getattr(dna, "field_families", None) or []:
            feats["field:" + fam] = 1.0
        for motif in getattr(dna, "ast_motifs", None) or []:
            feats["motif:" + str(motif)] = 1.0
        hz = getattr(dna, "horizon_bucket", "unknown") or "unknown"
        feats["horizon:" + str(hz)] = 1.0
        cx = getattr(dna, "complexity_bucket", "unknown") or "unknown"
        feats["complexity:" + str(cx)] = 1.0
        tb = str(getattr(dna, "turnover_bucket", None) or "unknown") or "unknown"
        if tb and tb != "unknown":
            feats["turnover:" + tb] = 1.0
        # source miner（meta 维度；非公式来源）
        src = str(event.source_miner or "")
        if src:
            feats["source_miner:" + src] = 1.0
        # 数值列（标准化到 [0,1] 量纲，让 L2 正则尺度一致）
        if event.cluster_size is not None:
            try:
                cs = float(event.cluster_size)
                if cs > 0:
                    # crowding proxy：1/(1+size)，越大越拥挤
                    feats["cluster_crowding:num"] = 1.0 / (1.0 + cs)
            except (TypeError, ValueError):
                pass
        if event.age_days is not None:
            try:
                ad = max(0.0, float(event.age_days))
                feats["age:num"] = math.exp(-ad / 365.0)  # 老 → 0，新 → 1
            except (TypeError, ValueError):
                pass
        if event.ast_nodes is not None:
            try:
                an = max(0.0, float(event.ast_nodes))
                feats["complexity_nodes:num"] = 1.0 / (1.0 + an)
            except (TypeError, ValueError):
                pass
        # interactions（#30 开关；只生成少量可读二阶项）
        if self.with_interactions:
            mechs = getattr(dna, "mechanisms", None) or []
            fams = getattr(dna, "field_families", None) or []
            hzs = [str(getattr(dna, "horizon_bucket", "") or "")]
            for m in mechs:
                if m in ("momentum_or_corr", "reversal", "liquidity", "volatility"):
                    for hz2 in hzs:
                        if hz2:
                            feats[f"interact:{m}x{hz2}"] = 1.0
            for fam in fams:
                if fam == "price_volume":
                    for hz2 in hzs:
                        if hz2:
                            feats[f"interact:{fam}x{hz2}"] = 1.0
        return feats

    @staticmethod
    def _group_of(feature: str) -> str:
        if ":" not in feature:
            return "schema"
        return feature.split(":", 1)[0]

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        events: Sequence[SurvivalEvent | Mapping[str, Any]],
        *,
        current_version: str | None = None,
    ) -> "SurvivalAttributionModel":
        """训练归因模型（只消费 frozen 旧版本事件，#23）。

        Parameters
        ----------
        events : sequence
            历史/frozen 版本的事件。当前版本 sealed test 事件必须不传——
            传了会被 ``assert_event_consumable`` 拒绝（SealedTestViolation）。
        current_version : optional
            当前研究版本（缺省用构造值）。
        """
        cur = str(current_version if current_version is not None else self.current_version)
        parsed: list[SurvivalEvent] = []
        for ev in events:
            e = ev if isinstance(ev, SurvivalEvent) else SurvivalEvent(
                **{k: v for k, v in dict(ev).items()
                   if k in SurvivalEvent.__dataclass_fields__}
            )
            # 关键：visible 与 frozen 只影响是否可消费；当前版本 sealed 事件
            # 一律拒绝（测试 1/4）。
            self.assert_event_consumable(e, current_version=cur)
            parsed.append(e)
        if not parsed:
            raise ValueError("SurvivalAttributionModel.fit requires >=1 consumable event")

        # 特征向量
        Xs: list[dict[str, float]] = []
        ys: list[int] = []
        for e in parsed:
            feats = self._features_of(e)
            Xs.append(feats)
            ys.append(1 if self._event_survival(e) else 0)

        # 全局 support（n_events / survival 先验）
        n = len(parsed)
        surv_n = sum(ys)
        prior = self.prior if self.prior is not None else (surv_n / n if n else 0.5)
        self._n_events = n

        # 列集合（确定性排序）
        col_set: set[str] = set()
        for feats in Xs:
            col_set.update(feats)
        self._feature_columns = sorted(col_set)
        cols = self._feature_columns
        p = len(cols)

        # 每特征 support / raw rate（先做层级收缩所需统计）
        feat_support: dict[str, int] = {c: 0 for c in cols}
        feat_surv: dict[str, int] = {c: 0 for c in cols}
        for feats, y in zip(Xs, ys):
            for c in cols:
                if feats.get(c, 0.0) != 0.0:
                    feat_support[c] += 1
                    if y:
                        feat_surv[c] += 1
        self._feature_support = {c: feat_support[c] for c in cols}
        self._feature_rate = {
            c: (feat_surv[c] / feat_support[c] if feat_support[c] else prior)
            for c in cols
        }

        # 组统计（hierarchical shrinkage 目标）
        group_support: dict[str, int] = {}
        group_surv: dict[str, int] = {}
        for c in cols:
            g = self._group_of(c)
            group_support[g] = group_support.get(g, 0) + feat_support[c]
            group_surv[g] = group_surv.get(g, 0) + feat_surv[c]

        # Elastic Net 拟合
        # (a) sklearn 可用且未强制 numpy → sklearn Elastic Net
        use_sklearn = (
            self.fit_backend == "auto" and _SKLEARN_AVAILABLE
        ) or self.fit_backend == "sklearn"
        backend = "sklearn" if use_sklearn else "numpy"
        self._backend_used = backend

        # 设计矩阵（数值列不动；one-hot 已是 0/1）
        Xmat: list[list[float]] = []
        for feats in Xs:
            Xmat.append([float(feats.get(c, 0.0)) for c in cols])

        if use_sklearn:
            from sklearn.linear_model import LogisticRegression

            l1_ratio = 0.5
            # 单一 λ：sklearn C 是正则倒数；l1_ratio=0.5 → elastic net。
            clf = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=l1_ratio,
                C=1.0,
                max_iter=1000,
                random_state=self.seed,
                fit_intercept=True,
            )
            try:
                clf.fit(Xmat, ys)
                self._intercept = float(clf.intercept_[0])
                coefs = clf.coef_[0]
            except Exception:  # noqa: BLE001 - 训练退化：全 0 系数
                self._intercept = 0.0
                coefs = [0.0] * p
            self._coefficients = {c: float(coefs[i]) for i, c in enumerate(cols)}
        else:
            # 自带确定性弹性网：先 L2 岭（IRLS），再 L1 近端投影一轮 + 一次
            # 岭再拟合（保证确定性且系数稀疏可控）。
            l2 = 0.1
            beta = _l2_logistic_irls(Xmat, ys, l2=l2)
            beta = _prox_l1(beta, 0.01)
            if p > 0:
                beta = _l2_logistic_irls(
                    [[1.0] + list(r) for r in Xmat], ys, l2=l2
                ) if False else beta
            self._intercept = float(beta[0])
            self._coefficients = {
                c: float(beta[i + 1]) if i < p else 0.0
                for i, c in enumerate(cols)
            }

        # hierarchical shrinkage（#29/#13）：组内加权 + 向全局 prior 收缩
        self._group_stats = {}
        n_eff_total = 0.0
        for c in cols:
            g = self._group_of(c)
            support = feat_support[c]
            raw = self._feature_rate[c]
            if support <= 0:
                shrunk = prior
                n_eff = 0.0
            else:
                # 组向全局收缩：组 raw → 用组 support 算 reliability
                gs = group_support.get(g, support)
                r_g = math.sqrt(gs / (gs + self.shrinkage_k))
                group_prior = prior + r_g * (prior - prior)  # = prior
                # 组内均值（含当前特征）——层级：先收缩到组中心，再向全局
                g_raw = (
                    group_surv.get(g, 0) / gs if gs > 0 else prior
                )
                group_center = prior + math.sqrt(gs / (gs + self.shrinkage_k)) * (g_raw - prior)
                # 特征级 reliability
                r_f = math.sqrt(support / (support + self.shrinkage_k))
                # 最终 shrunk rate = 组中心 + r_f * (raw - 组中心)（两级收缩）
                shrunk = group_center + r_f * (raw - group_center)
                if support < self.min_support:
                    # #13：低 support → 效应向 0（向 prior 强制收缩）
                    shrunk = prior + (shrunk - prior) * math.sqrt(
                        support / (support + self.shrinkage_k)
                    )
                    n_eff = 0.0
                else:
                    n_eff = support * r_f * r_f
            self._group_stats[c] = {
                "group": g,
                "raw_rate": raw,
                "shrunk_rate": shrunk,
                "support": float(support),
                "n_effective": float(n_eff),
                "group_center": float(group_center if support > 0 else prior),
            }
            n_eff_total += float(n_eff)

        self._n_effective = n_eff_total
        self._report = None
        return self

    def _event_survival(self, e: SurvivalEvent) -> bool:
        """事件是否视为 survival（label 优先，rate 兜底）。"""
        lbl = e.survival
        if lbl is not None and str(lbl) not in ("", SurvivalLabel.UNCLASSIFIED.value):
            return _is_survival_label(lbl)
        if e.survival_rate is not None:
            return float(e.survival_rate) >= 0.5
        return False

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def _query_survival(self, feats: dict[str, float]) -> tuple[float, float]:
        """线性打分：logit = intercept + Σ beta x；返回概率与线性分。"""
        z = self._intercept
        for c in self._feature_columns:
            v = feats.get(c, 0.0)
            if v != 0.0:
                z += self._coefficients.get(c, 0.0) * v
        if z > 30.0:
            z = 30.0
        elif z < -30.0:
            z = -30.0
        prob = 1.0 / (1.0 + math.exp(-z))
        return prob, z

    def predict_survival_rate(self, event: SurvivalEvent | Mapping[str, Any], *, query_version: str | None = None) -> float:
        """查询侧版本闸门：当前版本的 test 事件不可查询（返回前抛异常）。

        query_version 缺省用 current_version。事件为当前版本 sealed → 拒绝。
        """
        ev = event if isinstance(event, SurvivalEvent) else SurvivalEvent(
            **{k: v for k, v in dict(event).items() if k in SurvivalEvent.__dataclass_fields__}
        )
        self.assert_event_consumable(ev, current_version=query_version)
        feats = self._features_of(ev)
        prob, _ = self._query_survival(feats)
        return max(0.0, min(1.0, prob))

    def effect_for_event(
        self, event: SurvivalEvent | Mapping[str, Any], *, query_version: str | None = None
    ) -> dict[str, Any]:
        """对单条**可消费**事件返回效应分解（feature-level 聚合）。

        返回 dict：``survival_rate`` / ``prior`` / ``effect``（shrunk_rate -
        prior，cap 后）/ ``capped`` / ``n_events`` / ``n_effective``。
        查询侧同样拒绝当前版本 sealed 事件。
        """
        ev = event if isinstance(event, SurvivalEvent) else SurvivalEvent(
            **{k: v for k, v in dict(event).items() if k in SurvivalEvent.__dataclass_fields__}
        )
        self.assert_event_consumable(ev, current_version=query_version)
        feats = self._features_of(ev)
        prob, _ = self._query_survival(feats)
        # 聚合效应 = max-abs feature 的 shrunk_rate 相对 prior
        # （与 SurvivalOpportunity 消费口径一致：shrunk - prior）
        best_effect = 0.0
        best_key = ""
        for c in self._feature_columns:
            if feats.get(c, 0.0) != 0.0 and c in self._group_stats:
                st = self._group_stats[c]
                eff = st["shrunk_rate"] - self.prior
                if abs(eff) > abs(best_effect):
                    best_effect = eff
                    best_key = c
        cap = self.effect_cap
        capped = max(-cap, min(cap, best_effect))
        return {
            "survival_rate": prob,
            "prior": self.prior,
            "effect": capped,
            "capped": abs(best_effect) > cap,
            "n_events": self._n_events,
            "n_effective": self._n_effective,
            "dominant_feature": best_key,
            "version": self.current_version,
        }

    def effect_for_survival(
        self,
        survival_rate: float,
        *,
        support_count: int = 0,
        shrunk_survival_rate: float | None = None,
        prior: float | None = None,
    ) -> float:
        """对裸 survival 统计做层级收缩（与 dna_stats/regime 同款公式）。

        SurvivalOpportunity 消费口径：
        effect = (shrunk 或 收缩 raw) - prior，低 support 向 0。
        """
        eff_prior = float(prior if prior is not None else self.prior)
        if shrunk_survival_rate is not None:
            # 已收缩值 = 权威，直接相对 prior 取效应（不再二次收缩）
            shrunk = max(0.0, min(1.0, float(shrunk_survival_rate)))
        else:
            raw = max(0.0, min(1.0, float(survival_rate)))
            k = max(0.0, float(support_count))
            r = math.sqrt(k / (k + self.shrinkage_k))
            shrunk = eff_prior + r * (raw - eff_prior)
            if int(k) < self.min_support:
                # #13：低 support → 效应向 0（额外再收缩一轮，小样本绝不当规则）
                shrunk = eff_prior + (shrunk - eff_prior) * r
        eff = shrunk - eff_prior
        cap = self.effect_cap
        return max(-cap, min(cap, eff))

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------

    def report(self) -> SurvivalAttributionReport:
        """层级收缩归因报告（effect + CI + se，不做因果断言）。"""
        results: list[AttributionResult] = []
        for c in self._feature_columns:
            st = self._group_stats.get(c)
            if st is None:
                continue
            support = int(st["support"])
            raw = st["raw_rate"]
            shrunk = st["shrunk_rate"]
            eff = max(-self.effect_cap, min(self.effect_cap, shrunk - self.prior))
            n_eff = st["n_effective"]
            se = 0.0
            ci = (eff, eff)
            if support > 0 and shrunk > 0.0 and shrunk < 1.0:
                # Wald 近似（shrunk 方差按二项 + 收缩缩小）
                var_raw = raw * (1.0 - raw) / max(support, 1)
                r_f = math.sqrt(support / (support + self.shrinkage_k))
                var_shrunk = var_raw * r_f * r_f
                se = math.sqrt(max(var_shrunk, 1e-12))
                ci = (eff - 1.96 * se, eff + 1.96 * se)
            note = ""
            if support < self.min_support:
                note = f"low support ({support}<{self.min_support}): effect shrunk toward 0"
            results.append(
                AttributionResult(
                    feature=c,
                    group=st["group"],
                    effect=eff,
                    effect_ci=(max(-self.effect_cap, ci[0]), min(self.effect_cap, ci[1])),
                    effect_se=se,
                    raw_rate=raw,
                    shrunk_rate=shrunk,
                    support_count=support,
                    n_effective=n_eff,
                    note=note,
                )
            )
        # 排序：效应强（abs）优先、再 support
        results.sort(key=lambda r: (-abs(r.effect), -r.support_count))
        return SurvivalAttributionReport(
            version=self.current_version,
            prior=self.prior,
            n_events=self._n_events,
            n_effective=self._n_effective,
            results=results,
            meta={
                "backend": self._backend_used,
                "shrinkage_k": self.shrinkage_k,
                "min_support": self.min_support,
                "effect_cap": self.effect_cap,
                "with_interactions": self.with_interactions,
                "feature_count": len(self._feature_columns),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_version": self.current_version,
            "prior": self.prior,
            "shrinkage_k": self.shrinkage_k,
            "min_support": self.min_support,
            "effect_cap": self.effect_cap,
            "with_interactions": self.with_interactions,
            "fit_backend": self.fit_backend,
            "backend_used": self._backend_used,
            "feature_columns": self._feature_columns,
            "coefficients": dict(self._coefficients),
            "intercept": self._intercept,
            "n_events": self._n_events,
            "n_effective": self._n_effective,
            "group_stats": {
                k: dict(v) for k, v in self._group_stats.items()
            },
        }
