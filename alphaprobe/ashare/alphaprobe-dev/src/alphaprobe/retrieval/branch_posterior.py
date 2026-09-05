"""Beta 分支成功后验（任务书 §19-§20）+ T10 hierarchical 层扩展。

Success 定义：child 过 L3（search_valid 段）AND（ΔFactorFitness > delta_fitness
OR ΔPoolUtility > delta_pool）。Beta(alpha0=1, beta0=1) 先验，后验
P_success = (alpha0 + s) / (alpha0 + beta0 + s + f)。

维护每 node 的 n_attempt / n_success / n_elite / mean_delta_fitness /
mean_novelty_gain / mean_pool_utility_gain。

A6（V3.1）：三个 running mean 各自带有效样本计数
（n_delta_fitness / n_novelty_gain / n_delta_pool）——对应 metric 为 None 的
attempt 只推进 n_attempt，不推进该 metric 的计数，也不稀释该 mean（均值只用
自己的有效观测数当分母）。

T10（plan Task 10 / Part F2）hierarchical 扩展层：``ContextualRetriever`` 的
三层 Beta/Bernoulli 统计（``LayerStats`` 单层行 / ``HierarchicalStats`` 三层
收缩视图）与幅度模型（``GainModel`` / ``CostModel``）定义在本文件——
``contextual_retriever.py`` 从本文件 import 并 re-export，调用方只 import
contextual_retriever 即可。本文件不 import retrieval 其它模块，保持零循环。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    "BetaBranchPosterior",
    "DEFAULT_ALPHA0",
    "DEFAULT_BETA0",
    "DEFAULT_DELTA_FITNESS",
    "DEFAULT_DELTA_POOL",
    # T10 hierarchical 层（contextual_retriever re-export）
    "GainModel",
    "CostModel",
    "LayerStats",
    "HierarchicalStats",
    "NEUTRAL_SUCCESS",
    "DEFAULT_PRIOR_ALPHA",
    "DEFAULT_PRIOR_BETA",
    "DEFAULT_EXPECTED_COST",
    "_beta_sample",
]

DEFAULT_ALPHA0 = 1.0
DEFAULT_BETA0 = 1.0
#: child 需过 L3 且 ΔFactorFitness 超过该值才记为 success（§19 阈值，可配）
DEFAULT_DELTA_FITNESS = 0.005
#: child 需 ΔPoolUtility 超过该值才记为 success（§19 阈值，可配）
DEFAULT_DELTA_POOL = 0.01

# ---------------------------------------------------------------------------
# T10 hierarchical 层常量
# ---------------------------------------------------------------------------

#: 无观测时的中性成功率（Beta(1,1) 后验 = 0.5）。Part G #13：缺证据绝不偏 0/偏 1。
NEUTRAL_SUCCESS = 0.5

#: 每层 Beta 先验（同 BetaBranchPosterior 的 alpha0/beta0=1）。
DEFAULT_PRIOR_ALPHA = 1.0
DEFAULT_PRIOR_BETA = 1.0

#: 无 cost 证据时的默认期望 cost（单位归一；0~1）。action cost 注入缺失时中性值。
DEFAULT_EXPECTED_COST = 0.5

#: Thompson 采样分位数 guard：Beta 采样极端接近 0/1 时钳位，避免 0 概率抹掉 EV。
_BETA_CLAMP_EPS = 1e-6

#: schema 维度缺省 key（无 schema 上下文的 global action 行）。
NO_SCHEMA = ""

EPS = 1e-12


@dataclass
class BetaBranchPosterior:
    """node 维度（或 node×action）的分支后验统计。

    构造后未观测任何 attempt 时 posterior_success = 0.5（Beta(1,1) 先验）。
    ``observe_attempt`` 需要外部传入 is_l3_pass + delta_fitness + delta_pool，
    是否达标（success）由本类用阈值判定——保持确定性、可单测。
    """

    alpha0: float = DEFAULT_ALPHA0
    beta0: float = DEFAULT_BETA0
    delta_fitness: float = DEFAULT_DELTA_FITNESS
    delta_pool: float = DEFAULT_DELTA_POOL

    n_attempt: int = 0
    n_success: int = 0
    n_elite: int = 0
    mean_delta_fitness: float = 0.0
    mean_novelty_gain: float = 0.0
    mean_pool_utility_gain: float = 0.0
    # A6：per-metric 有效观测计数（分母与对应 mean 严格一致；None 观测不推进）
    n_delta_fitness: int = 0
    n_novelty_gain: int = 0
    n_delta_pool: int = 0

    _latest: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------

    def observe_attempt(
        self,
        *,
        is_l3_pass: bool,
        delta_fitness: float | None = None,
        delta_pool: float | None = None,
        novelty_gain: float | None = None,
        elite: bool = False,
    ) -> bool:
        """记录一次 attempt，返回本次是否记为 success。

        Success = is_l3_pass AND (ΔFitness > delta_fitness OR ΔPoolUtility > delta_pool)。
        """
        gained = (
            (delta_fitness is not None and delta_fitness > self.delta_fitness)
            or (delta_pool is not None and delta_pool > self.delta_pool)
        )
        success = bool(is_l3_pass and gained)
        self.n_attempt += 1
        if success:
            self.n_success += 1
        if elite:
            self.n_elite += 1
        # A6：每个 mean 只用各自的**有效观测**推进（分母 = 该 metric 非 None 次数，
        # 不是 n_attempt）。metric 为 None → 不推进对应计数、不稀释对应均值。
        if delta_fitness is not None:
            self.n_delta_fitness += 1
            self.mean_delta_fitness = _running_mean(
                self.mean_delta_fitness, float(delta_fitness), self.n_delta_fitness
            )
        if novelty_gain is not None:
            self.n_novelty_gain += 1
            self.mean_novelty_gain = _running_mean(
                self.mean_novelty_gain, float(novelty_gain), self.n_novelty_gain
            )
        if delta_pool is not None:
            self.n_delta_pool += 1
            self.mean_pool_utility_gain = _running_mean(
                self.mean_pool_utility_gain, float(delta_pool), self.n_delta_pool
            )
        self._latest = {
            "is_l3_pass": bool(is_l3_pass),
            "delta_fitness": delta_fitness,
            "delta_pool": delta_pool,
            "novelty_gain": novelty_gain,
            "elite": bool(elite),
            "success": success,
        }
        return success

    def mark_elite(self) -> None:
        self.n_elite += 1

    # ------------------------------------------------------------------
    # 后验
    # ------------------------------------------------------------------

    @property
    def s(self) -> float:
        """Beta 后验成功计数（含先验 alpha0）。"""
        return self.alpha0 + self.n_success

    @property
    def f(self) -> float:
        """Beta 后验失败计数（含先验 beta0）。"""
        return self.beta0 + (self.n_attempt - self.n_success)

    @property
    def posterior_success(self) -> float:
        """P_success = (alpha0 + s) / (alpha0 + beta0 + s + f)。"""
        return self.s / (self.s + self.f)

    @property
    def saturation(self) -> float:
        """attempts ≥ 10 后才启用（与 memory store 的 exploration_state 一致）。"""
        if self.n_attempt < 10:
            return 0.0
        return 1.0 - self.n_success / max(self.n_attempt, 1)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "alpha0": self.alpha0,
            "beta0": self.beta0,
            "delta_fitness": self.delta_fitness,
            "delta_pool": self.delta_pool,
            "n_attempt": self.n_attempt,
            "n_success": self.n_success,
            "n_elite": self.n_elite,
            "mean_delta_fitness": self.mean_delta_fitness,
            "mean_novelty_gain": self.mean_novelty_gain,
            "mean_pool_utility_gain": self.mean_pool_utility_gain,
            # A6：per-metric 有效样本计数（序列化保真，往返后分母不丢）
            "n_delta_fitness": self.n_delta_fitness,
            "n_novelty_gain": self.n_novelty_gain,
            "n_delta_pool": self.n_delta_pool,
            "posterior_success": self.posterior_success,
            "saturation": self.saturation,
        }
        if self._latest:
            d["latest"] = dict(self._latest)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BetaBranchPosterior":
        d = dict(d or {})
        latest = d.pop("latest", None)
        kwargs = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        # 历史 dict（V3.1 之前）没有 per-metric 计数 → 由 n_attempt 时代的
        # 均值反推有效样本数：只要 mean 非 0/有观测，令计数 = n_attempt。
        # （向后兼容：旧运行数据 reload 后均值语义不变。）
        obj = cls(**kwargs)
        if obj.n_delta_fitness <= 0 and obj.n_attempt > 0 and obj.mean_delta_fitness != 0.0:
            obj.n_delta_fitness = obj.n_attempt
        if obj.n_novelty_gain <= 0 and obj.n_attempt > 0 and obj.mean_novelty_gain != 0.0:
            obj.n_novelty_gain = obj.n_attempt
        if obj.n_delta_pool <= 0 and obj.n_attempt > 0 and obj.mean_pool_utility_gain != 0.0:
            obj.n_delta_pool = obj.n_attempt
        if isinstance(latest, dict):
            obj._latest = dict(latest)
        return obj


def _running_mean(current: float, new_value: float, n: int) -> float:
    if n <= 0:
        return current
    return current + (new_value - current) / n


# ---------------------------------------------------------------------------
# T10 hierarchical 层：采样 / 幅度模型 / 三层 Beta 统计
# （contextual_retriever.py import 并 re-export；调用方不直接 import 本文件）
# ---------------------------------------------------------------------------


def _beta_sample(
    a: float, b: float, rng: random.Random | None = None
) -> float:
    """Beta(a,b) 采样（a,b>0）。退化钳位到 (eps, 1-eps) 保数值稳定。

    每次调用独立采样；同 rng 同序列。
    """
    rng = rng if rng is not None else random  # 模块级 fallback（默认独立）
    a = max(float(a), EPS)
    b = max(float(b), EPS)
    x = rng.betavariate(a, b)
    if not math.isfinite(x):
        x = a / (a + b)
    return min(max(x, _BETA_CLAMP_EPS), 1.0 - _BETA_CLAMP_EPS)


def _median(values: Sequence[float] | None) -> float | None:
    vs = [float(v) for v in (values or []) if math.isfinite(float(v))]
    if not vs:
        return None
    vs.sort()
    n = len(vs)
    if n % 2 == 1:
        return vs[n // 2]
    return (vs[n // 2 - 1] + vs[n // 2]) / 2.0


def _trimmed_mean(values: Sequence[float], *, trim: float) -> float | None:
    vs = [float(v) for v in values if math.isfinite(float(v))]
    if not vs:
        return None
    vs.sort()
    k = max(0, int(round(len(vs) * trim)))
    if 2 * k >= len(vs):
        return sum(vs) / len(vs)
    seg = vs[k : len(vs) - k]
    return sum(seg) / len(seg) if seg else None


def _clip01(v: float | None, *, neutral: float = NEUTRAL_SUCCESS) -> float:
    if v is None:
        return float(neutral)
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return float(neutral)
    if not math.isfinite(fv):
        return float(neutral)
    return min(max(fv, 0.0), 1.0)


@dataclass
class GainModel:
    """Gain 的幅度统计（plan Task 10 最低要求）。

    贝叶斯 magnitude model 留接口：替换本类（duck-typing 实现
    ``observe(...)`` / ``expected_gain()`` / ``to_dict()`` / ``from_dict()``）
    即可注入更强模型。

    统计：
    - ``median_delta_fitness`` / ``robust_mean_delta_fitness``：ΔFitness 稳健中心；
    - ``median_delta_pool``：ΔPoolUtility 稳健中心；
    - ``median_novelty_gain``：novelty gain 稳健中心；
    - ``n_delta_fitness`` / ``n_delta_pool`` / ``n_novelty``：per-metric 有效样本数
      （A6 语义：metric 为 None 的观测不推进对应计数、不稀释均值）。
    """

    # running 数据（全量保留以便换窗口算稳健中心；观测规模受预算约束很小）
    _delta_fitness: list[float] = field(default_factory=list)
    _delta_pool: list[float] = field(default_factory=list)
    _novelty: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self._delta_fitness)

    @property
    def n_delta_fitness(self) -> int:
        return len(self._delta_fitness)

    @property
    def n_delta_pool(self) -> int:
        return len(self._delta_pool)

    @property
    def n_novelty(self) -> int:
        return len(self._novelty)

    def observe(
        self,
        *,
        delta_fitness: float | None = None,
        delta_pool: float | None = None,
        novelty_gain: float | None = None,
    ) -> None:
        if delta_fitness is not None and math.isfinite(float(delta_fitness)):
            self._delta_fitness.append(float(delta_fitness))
        if delta_pool is not None and math.isfinite(float(delta_pool)):
            self._delta_pool.append(float(delta_pool))
        if novelty_gain is not None and math.isfinite(float(novelty_gain)):
            self._novelty.append(float(novelty_gain))

    @property
    def median_delta_fitness(self) -> float | None:
        return _median(self._delta_fitness)

    @property
    def median_delta_pool(self) -> float | None:
        return _median(self._delta_pool)

    @property
    def median_novelty_gain(self) -> float | None:
        return _median(self._novelty)

    @property
    def robust_mean_delta_fitness(self) -> float | None:
        """截尾均值（trim 20%）——对异常 child 稳健。"""
        return _trimmed_mean(self._delta_fitness, trim=0.2)

    def expected_gain(self) -> float:
        """综合期望增益（0~1）。

        半程尺度：ΔFitness 0.06 视为「大增益」（维度打满 1.0），0.03 为中等
        （0.75）——中等与高增益在 [0,1] 上真正拉开差距，不早饱和。
        """
        df = self.median_delta_fitness
        if df is None:
            return NEUTRAL_SUCCESS
        df_half = _clip01(
            0.5 + 0.5 * min(max(float(df) / 0.06, 0.0), 1.0),
            neutral=0.5,
        )
        nov = self.median_novelty_gain
        if nov is not None:
            nov_half = _clip01(
                0.5 + 0.5 * min(max(float(nov) / 0.5, 0.0), 1.0),
                neutral=0.5,
            )
            base = 0.7 * df_half + 0.3 * nov_half
        else:
            base = df_half
        return min(max(base, 0.0), 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "delta_fitness": list(self._delta_fitness),
            "delta_pool": list(self._delta_pool),
            "novelty": list(self._novelty),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GainModel":
        m = cls()
        for k, dst in (
            ("delta_fitness", m._delta_fitness),
            ("delta_pool", m._delta_pool),
            ("novelty", m._novelty),
        ):
            raw = (d or {}).get(k) or []
            dst.extend(float(v) for v in raw if math.isfinite(float(v)))
        return m


@dataclass
class CostModel:
    """ExpectedCost 的估计（0~1 归一）。

    直接 track 观测到的相对 cost（调用方在 ``observe_attempt(cost=...)`` 提供；
    未提供 → 中性 ``DEFAULT_EXPECTED_COST``）。cost 0 = 免费（EV 不受惩罚）。
    """

    _costs: list[float] = field(default_factory=list)
    default_cost: float = DEFAULT_EXPECTED_COST

    @property
    def n(self) -> int:
        return len(self._costs)

    def observe(self, cost: float | None) -> None:
        if cost is not None and math.isfinite(float(cost)) and float(cost) >= 0.0:
            self._costs.append(float(cost))

    def expected_cost(self) -> float:
        if not self._costs:
            return float(self.default_cost)
        return min(max(sum(self._costs) / len(self._costs), 0.0), 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {"costs": list(self._costs), "default_cost": float(self.default_cost)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CostModel":
        m = cls()
        raw = (d or {}).get("costs") or []
        m._costs = [float(v) for v in raw if math.isfinite(float(v)) and float(v) >= 0.0]
        m.default_cost = float((d or {}).get("default_cost", DEFAULT_EXPECTED_COST))
        return m


@dataclass
class LayerStats:
    """一层（global action / schema×action / parent×action×schema）的 Beta 行。

    s/f 含先验计数（alpha0/beta0=1）：后验成功 mean = (alpha0+n_success) /
    (alpha0+beta0+n_attempt)。``n_attempt`` 是**成功或失败都已观测**的尝试数
    （success 判定用 is_l3_pass，与 branch_posterior 对齐）。
    """

    alpha0: float = DEFAULT_PRIOR_ALPHA
    beta0: float = DEFAULT_PRIOR_BETA
    n_attempt: int = 0
    n_success: int = 0

    @property
    def posterior_mean(self) -> float:
        """(a0+s)/(a0+b0+n)；无观测 → 0.5（#13）。"""
        return (self.alpha0 + self.n_success) / (
            self.alpha0 + self.beta0 + self.n_attempt
        )

    def sample(self, rng: random.Random) -> float:
        return _beta_sample(
            self.alpha0 + self.n_success,
            self.beta0 + (self.n_attempt - self.n_success),
            rng,
        )

    @property
    def n(self) -> int:
        return self.n_attempt

    def observe(self, *, success: bool) -> None:
        self.n_attempt += 1
        if success:
            self.n_success += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha0": self.alpha0,
            "beta0": self.beta0,
            "n_attempt": self.n_attempt,
            "n_success": self.n_success,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LayerStats":
        d = dict(d or {})
        return cls(
            alpha0=float(d.get("alpha0", DEFAULT_PRIOR_ALPHA)),
            beta0=float(d.get("beta0", DEFAULT_PRIOR_BETA)),
            n_attempt=int(d.get("n_attempt", 0)),
            n_success=int(d.get("n_success", 0)),
        )


@dataclass
class HierarchicalStats:
    """一个 (action) 键下跨 schema 汇总的层级视图。

    组合三层 posterior 的收缩结果：

    - ``global``：global action 层（schema 无关）的 Beta 行；
    - ``schema``：dict[schema] -> LayerStats（schema×action 层）；
    - ``parent``：dict[schema] -> dict[parent] -> LayerStats（parent×action×schema 层）；
    - 每层附带 GainModel / CostModel 统计（per-key 聚合）。

    收缩：加权后验均值混合（weight = sqrt(n/(n+k))，k=shrinkage_strength 默认
    16——同 fitness/confidence 的 reliability 语义；同观测下 parent 权重随 n
    单调增，n=0 → 完全用上层）。
    """

    action: str
    alpha0: float = DEFAULT_PRIOR_ALPHA
    beta0: float = DEFAULT_PRIOR_BETA
    shrinkage_strength: float = 16.0
    global_layer: LayerStats = field(default_factory=LayerStats)
    _schema: dict[str, LayerStats] = field(default_factory=dict)
    _parent: dict[str, dict[str, LayerStats]] = field(default_factory=dict)
    _gain: GainModel = field(default_factory=GainModel)
    _cost: CostModel = field(default_factory=CostModel)
    #: global 行的（可选）gain/cost 观测（parent/schema 行为空时回落）
    _global_gain: GainModel = field(default_factory=GainModel)
    _global_cost: CostModel = field(default_factory=CostModel)

    # -- 观测 --------------------------------------------------------------

    def observe(
        self,
        *,
        parent_id: str | None,
        schema_id: str | None,
        success: bool,
        delta_fitness: float | None,
        delta_pool: float | None,
        novelty_gain: float | None,
        cost: float | None,
        gain: bool,
    ) -> None:
        """记录一次 attempt 到对应层。parent/schema 为 None → 只进 global/上层。

        global action 层只收 schema 无关观测（保持同一 action 的 schema 差异）：
        带 schema 的观测只进 schema×action / parent×action×schema 行。
        """
        schema = str(schema_id) if schema_id is not None else NO_SCHEMA
        parent = str(parent_id) if parent_id is not None else None
        if not schema:
            self.global_layer.observe(success=success)
            self._global_gain.observe(
                delta_fitness=delta_fitness if gain else None,
                delta_pool=delta_pool if gain else None,
                novelty_gain=novelty_gain if gain else None,
            )
            if cost is not None and math.isfinite(float(cost)) and float(cost) >= 0.0:
                self._global_cost.observe(cost)
        if schema:
            sl = self._schema.setdefault(
                schema, LayerStats(alpha0=self.alpha0, beta0=self.beta0)
            )
            sl.observe(success=success)
        if schema and parent:
            pl = self._parent.setdefault(schema, {}).setdefault(
                parent, LayerStats(alpha0=self.alpha0, beta0=self.beta0)
            )
            pl.observe(success=success)
            # gain/cost 只挂 parent×action×schema 行（最具体层）
            self._gain.observe(
                delta_fitness=delta_fitness if gain else None,
                delta_pool=delta_pool if gain else None,
                novelty_gain=novelty_gain if gain else None,
            )
            if cost is not None and math.isfinite(float(cost)) and float(cost) >= 0.0:
                self._cost.observe(cost)

    # -- 收缩 --------------------------------------------------------------

    def _weight(self, n: int) -> float:
        """reliability-style 权重：n=0 → 0；n 大 → 接近 1。"""
        if n <= 0:
            return 0.0
        return math.sqrt(n / (n + float(self.shrinkage_strength)))

    def _shrunk(
        self,
        layer: LayerStats | None,
        upper: float,
        *,
        hierarchical: bool,
    ) -> float:
        """单层收缩：own mean × w + upper × (1-w)。

        - 无该层行（layer=None）→ 完全用上层。
        - ``hierarchical=False``（ablation）→ 只用自身行（自身无观测 → 上层）。
        """
        if layer is None:
            return float(upper)
        own = layer.posterior_mean
        if not hierarchical:
            return own
        w = self._weight(layer.n)
        return w * own + (1.0 - w) * upper

    def mean_success(
        self,
        *,
        parent_id: str | None,
        schema_id: str | None,
        hierarchical: bool = True,
    ) -> float:
        """逐层收缩的后验成功均值。parent=None → 只走 schema/global。"""
        schema = str(schema_id) if schema_id is not None else NO_SCHEMA
        parent = str(parent_id) if parent_id is not None else None
        g = self.global_layer.posterior_mean if hierarchical else NEUTRAL_SUCCESS
        # 无任何观测（含 global）→ 中性 0.5
        if self.global_layer.n <= 0 and not self._schema and not self._parent:
            return NEUTRAL_SUCCESS
        if not hierarchical:
            # 非层级：parent → schema（如果有行）→ global 直接取各自 posterior
            if parent and schema in self._parent and parent in self._parent[schema]:
                return self._parent[schema][parent].posterior_mean
            if schema in self._schema:
                return self._schema[schema].posterior_mean
            if self.global_layer.n > 0:
                return self.global_layer.posterior_mean
            return NEUTRAL_SUCCESS
        # schema 层（向 global 收缩）
        s_layer = self._schema.get(schema) if schema else None
        s_mean = self._shrunk(s_layer, g, hierarchical=True)
        # parent 层（向 s_mean 收缩）
        p_layer = None
        if parent and schema in self._parent:
            p_layer = self._parent[schema].get(parent)
        return self._shrunk(p_layer, s_mean, hierarchical=True)

    def sample_success(
        self,
        *,
        parent_id: str | None,
        schema_id: str | None,
        hierarchical: bool,
        rng: random.Random,
    ) -> float:
        """Thompson：逐层 Beta 采样后按同样权重收缩（audit 之外的成功概率）。"""
        schema = str(schema_id) if schema_id is not None else NO_SCHEMA
        parent = str(parent_id) if parent_id is not None else None
        # 先采样三层（无行 → 退化用上层采样值）。
        g_draw = self.global_layer.sample(rng) if self.global_layer.n > 0 else NEUTRAL_SUCCESS
        s_layer = self._schema.get(schema) if schema else None
        s_draw = s_layer.sample(rng) if s_layer is not None else g_draw
        p_layer = None
        if parent and schema in self._parent:
            p_layer = self._parent[schema].get(parent)
        p_draw = p_layer.sample(rng) if p_layer is not None else s_draw
        if not hierarchical:
            if p_layer is not None:
                return p_draw
            if s_layer is not None:
                return s_draw
            return g_draw
        # 逐层加权收缩（与 mean 同权重；上层基底用其*采样值*，Thompson 语义）。
        w_s = self._weight(s_layer.n) if s_layer is not None else 0.0
        s_eff = w_s * s_draw + (1.0 - w_s) * g_draw
        w_p = self._weight(p_layer.n) if p_layer is not None else 0.0
        p_eff = w_p * p_draw + (1.0 - w_p) * s_eff
        return min(max(p_eff, _BETA_CLAMP_EPS), 1.0 - _BETA_CLAMP_EPS)

    def gain_of(self, *, parent_id: str | None, schema_id: str | None) -> float:
        """期望增益：parent×action×schema 的 gain 优先，回落 schema/global。"""
        schema = str(schema_id) if schema_id is not None else NO_SCHEMA
        parent = str(parent_id) if parent_id is not None else None
        if parent and schema in self._parent and parent in self._parent[schema]:
            if self._gain.n > 0:
                return self._gain.expected_gain()
        if schema in self._schema:
            if self._gain.n > 0:
                return self._gain.expected_gain()
        if self._global_gain.n > 0:
            return self._global_gain.expected_gain()
        return NEUTRAL_SUCCESS

    def cost_of(self, *, parent_id: str | None, schema_id: str | None) -> float:
        schema = str(schema_id) if schema_id is not None else NO_SCHEMA
        parent = str(parent_id) if parent_id is not None else None
        if parent and schema in self._parent and parent in self._parent[schema]:
            if self._cost.n > 0:
                return self._cost.expected_cost()
        if schema in self._schema:
            if self._cost.n > 0:
                return self._cost.expected_cost()
        if self._global_cost.n > 0:
            return self._global_cost.expected_cost()
        return self._cost.expected_cost()  # default_cost when empty

    # -- 序列化 -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "alpha0": self.alpha0,
            "beta0": self.beta0,
            "shrinkage_strength": self.shrinkage_strength,
            "global": self.global_layer.to_dict(),
            "schema": {k: v.to_dict() for k, v in self._schema.items()},
            "parent": {
                s: {p: v.to_dict() for p, v in ps.items()}
                for s, ps in self._parent.items()
            },
            "gain": self._gain.to_dict(),
            "cost": self._cost.to_dict(),
            "global_gain": self._global_gain.to_dict(),
            "global_cost": self._global_cost.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "HierarchicalStats":
        d = dict(d or {})
        obj = cls(
            action=str(d.get("action", "")),
            alpha0=float(d.get("alpha0", DEFAULT_PRIOR_ALPHA)),
            beta0=float(d.get("beta0", DEFAULT_PRIOR_BETA)),
            shrinkage_strength=float(d.get("shrinkage_strength", 16.0)),
        )
        obj.global_layer = LayerStats.from_dict(d.get("global") or {})
        obj._schema = {
            str(k): LayerStats.from_dict(v) for k, v in (d.get("schema") or {}).items()
        }
        obj._parent = {
            str(s): {str(p): LayerStats.from_dict(v) for p, v in ps.items()}
            for s, ps in (d.get("parent") or {}).items()
        }
        obj._gain = GainModel.from_dict(d.get("gain") or {})
        obj._cost = CostModel.from_dict(d.get("cost") or {})
        obj._global_gain = GainModel.from_dict(d.get("global_gain") or {})
        obj._global_cost = CostModel.from_dict(d.get("global_cost") or {})
        return obj

    # 公开只读统计（diagnostics）
    def schema_stats_of(self, schema_id: str | None) -> LayerStats | None:
        if schema_id is None:
            return None
        return self._schema.get(str(schema_id))

    def parent_stats_of(
        self, parent_id: str | None, schema_id: str | None
    ) -> LayerStats | None:
        if parent_id is None or schema_id is None:
            return None
        return self._parent.get(str(schema_id), {}).get(str(parent_id))

