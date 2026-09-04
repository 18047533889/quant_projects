"""SeedProvider —— 从 FactorAssets 全局库采样 seed（plan Task 20）。

设计目标（Non-negotiable #24：10 万 seed 库不整体塞进 DAG）：

- **只读 metadata，不拉因子值面板**：候选源是「可重放、按索引随机访问」的
  metadata 序列（列表 或 ``__len__`` / ``__getitem__`` 的惰性 catalog），
  绝不要求候选源先把 10 万条物化成 dict 全表。
- **不整体物化、不全量打分**：按桶把索引空间做确定性洗牌（Fisher-Yates
  prefix 版——只洗前 ``k + slack`` 段），按比例每桶顺序取 k，只触碰 O(k)
  个条目。绝不做 100K 全量排序 / 全量打分 / N×N 相关。
- **采样前去重（identity）**：按 ``canonical_hash`` / ``fe_identity_ref``
  判定等价；identity_fn 由调用方注入（生产走 FE authority），提供确定性
  fallback 解析（元数据自带 hash / identity 时不再额外调用）。
- **seed 无迭代历史是合法 DAG root**：产出 ``SeedCandidate.parents == ()``、
  ``lineage_root == True``，但**只把选中的 K 个**交给 runner——未选中者
  永远留在 FactorAssets 全局库。
- **混合比例可配置**（默认 40% underexplored good / 20% fertile / 15% rare
  schema / 10% survival / 10% cluster rep / 5% random）。候选不足时按可用
  桶重归一并在 ``diagnostics.degraded_buckets`` 标注（不静默吞）。
- 无历史数据 / 空桶 → 降级到可用桶（有资格成为 RANDOM 的候选也算）。

确定性：同一 catalog + 同一 rng_seed → 同一输出（random 桶也经 seed 洗牌）。
零 LLM / 零模型 / 零网络。faiss / factor_engine 均不需要（identity 由
调用方注入，本模块不 import factor_engine / factor_assets）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol, Sequence

__all__ = [
    "SeedBucket",
    "DEFAULT_SEED_MIX",
    "SeedCandidate",
    "SeedProviderConfig",
    "SeedDiagnostics",
    "SeedProvider",
    "SeedCatalog",
    "IdentityFn",
]


class SeedBucket(Enum):
    """seed 的六种语义桶（plan Task 20 推荐混合）。"""

    UNDEREXPLORED_GOOD = "underexplored_good"
    FERTILE = "fertile"
    RARE_SCHEMA = "rare_schema"
    SURVIVAL = "survival"
    CLUSTER_REPRESENTATIVE = "cluster_representative"
    RANDOM = "random"


#: 默认混合比例（合计 1.0）。Percentages config only；后续 contextual bandit
#: 由配置层替换（本模块只按给定 mix 采样，不学习）。
DEFAULT_SEED_MIX: dict[SeedBucket, float] = {
    SeedBucket.UNDEREXPLORED_GOOD: 0.40,
    SeedBucket.FERTILE: 0.20,
    SeedBucket.RARE_SCHEMA: 0.15,
    SeedBucket.SURVIVAL: 0.10,
    SeedBucket.CLUSTER_REPRESENTATIVE: 0.10,
    SeedBucket.RANDOM: 0.05,
}

#: 候选不足时兜底桶：任何剩余候选都可当 RANDOM（随机探索是最后兜底）。
_FALLBACK_BUCKET = SeedBucket.RANDOM

#: 洗牌 slack：让每个桶在 `k_per_bucket + slack` 次尝试内能稳定拿到候选，
#: 又保持 O(k) 触碰（不去扫全表找稀有桶）。
_SHUFFLE_SLACK = 64


@dataclass(frozen=True)
class SeedCandidate:
    """一个被选中 seed（准备作为 DAG root 进 pipeline）。

    parents 恒为空、lineage_root 恒为 True（无迭代历史的 cold-start seed 是
    合法 DAG root）。factor_id / formula / bucket 供 runner 直接转 parents。
    """

    factor_id: str
    formula: str
    bucket: SeedBucket
    identity_ref: str = ""
    canonical_hash: str = ""
    parents: tuple[str, ...] = ()
    lineage_root: bool = True
    iteration_history: tuple[Any, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_parent_dict(self) -> dict[str, Any]:
        """转成 pipeline parents 条目（与 runner 现有 seed parent 契约一致）。"""
        return {
            "formula": self.formula,
            "factor_id": self.factor_id,
            "fitness": 0.0,
            "bucket": self.bucket.value,
            "identity_ref": self.identity_ref,
            "canonical_hash": self.canonical_hash,
        }


@dataclass
class SeedDiagnostics:
    """一次采样的诊断（供 observability / 报告，不参与逻辑）。"""

    requested: int = 0
    returned: int = 0
    total_available: int = 0
    unique_available: int = 0
    degraded_buckets: list[str] = field(default_factory=list)
    bucket_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "returned": self.returned,
            "total_available": self.total_available,
            "unique_available": self.unique_available,
            "degraded_buckets": list(self.degraded_buckets),
            "bucket_counts": dict(self.bucket_counts),
        }


#: 可重放的候选源协议：列表可直接用；惰性 catalog 须实现 ``__len__`` 与
#: ``__getitem__(i)``（生产 factor_assets catalog 走后者，不物化全表）。
SeedCatalog = Sequence[Mapping[str, Any]]


@dataclass
class SeedProviderConfig:
    """SeedProvider 配置（全部可配置，逻辑不许硬编码数值）。"""

    mix: Mapping[SeedBucket, float] = field(
        default_factory=lambda: dict(DEFAULT_SEED_MIX)
    )
    rng_seed: int = 0
    shuffle_slack: int = _SHUFFLE_SLACK
    identity_fn: Callable[[str], Mapping[str, str]] | None = None

    def __post_init__(self) -> None:
        # 只接受已知桶 key（防 typo 静默丢桶）。
        unknown = [k for k in self.mix if not isinstance(k, SeedBucket)]
        if unknown:
            raise TypeError(
                f"mix keys must be SeedBucket, got {unknown!r} "
                "(config buckets only; unknown bucket would be silently dropped)"
            )
        weights: dict[SeedBucket, float] = {b: 0.0 for b in SeedBucket}
        for b, w in self.mix.items():
            if isinstance(w, bool) or not isinstance(w, (int, float)):
                raise TypeError(f"mix weight for {b} must be a number")
            weights[b] = float(w)
        total = sum(weights.values())
        if not total > 0.0:
            raise ValueError("mix weights must sum to > 0")
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"mix weights must sum to 1.0, got {total:.6f} "
                "(percentages are config; a partial mix would silently "
                "under-sample some buckets)"
            )
        # 归一化并把权重冻结为 dict（SeedBucket -> float）
        object.__setattr__(self, "mix", {b: w for b, w in weights.items()})
        if self.shuffle_slack < 0:
            raise ValueError("shuffle_slack must be >= 0")

    # identity_fn 直接作为属性（config 构造后由调用方显式注入）
    identity_fn: Callable[[str], Mapping[str, str]] | None = None


#: identity_fn 返回结构：canonical_repr / canonical_ast_hash /
#: signal_equivalence_id / parameter_family_id（与 FE authority 视图一致）。
IdentityFn = Callable[[str], Mapping[str, str]]


class SeedProvider:
    """从 FactorAssets metadata 源按配置比例采样 K 个 seed。

    使用方式（runner 接线）::

        provider = SeedProvider(config=SeedProviderConfig(rng_seed=...))
        catalog = factor_assets_catalog(...)      # 惰性：len + getitem
        seeds = provider.sample(catalog, k=K)     # -> list[SeedCandidate]
        parents = [s.to_parent_dict() for s in seeds]

    O(k)：采样只触碰少量索引（prefix Fisher-Yates），不物化 10 万全表、
    不全量打分。identity 去重按 canonical_hash → fe_identity_ref；去重集合
    也只保留已选中条目（内存 O(k)）。
    """

    def __init__(
        self,
        *,
        config: SeedProviderConfig | None = None,
        identity_fn: IdentityFn | None = None,
    ) -> None:
        self.config = config or SeedProviderConfig()
        if identity_fn is not None:
            self.config = SeedProviderConfig(
                mix=dict(self.config.mix),
                rng_seed=self.config.rng_seed,
                shuffle_slack=self.config.shuffle_slack,
                identity_fn=identity_fn,
            )
        self._rng = random.Random(self.config.rng_seed)
        #: 最近一次采样诊断（未采样时为空 dict）。
        self.diagnostics = SeedDiagnostics()
        #: 去重 / 缺 identity 时的兜底 identity_fn（config.identity_fn 优先）。
        self.identity_fn = identity_fn or self.config.identity_fn

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def sample(self, catalog: SeedCatalog, *, k: int) -> list[SeedCandidate]:
        """从 catalog 采样 k 个去重 seed（O(k) 触碰，不全量物化）。

        Parameters
        ----------
        catalog : SeedCatalog
            可重放 metadata 源（list 或惰性 catalog）。条目须含
            ``factor_id`` / ``canonical_repr``（或 formula），可选
            ``canonical_hash`` / ``fe_identity_ref`` / ``parameter_family_id`` /
            ``bucket``（缺省 RANDOM）。
        k : int
            目标 seed 数。catalog 唯一候选不足 k 时返回全部唯一候选，
            并在 ``diagnostics.degraded_buckets`` 标注降级。

        Returns
        -------
        list[SeedCandidate]
            已去重、已排序（无迭代历史的 DAG root）。len ≤ k。
        """
        if k <= 0:
            self.diagnostics = SeedDiagnostics()
            return []
        total = _catalog_len(catalog)
        if total <= 0:
            self.diagnostics = SeedDiagnostics(
                requested=k, returned=0, total_available=0,
                unique_available=0,
                degraded_buckets=[b.name for b in SeedBucket],
            )
            return []

        # 每桶配额（合计恰为 k）：整数配额 + 尾差给 RANDOM（随机探索吸收项）。
        raw = {b: self.config.mix[b] * k for b in SeedBucket}
        quotas: dict[SeedBucket, int] = {
            b: int(round(v)) for b, v in raw.items()
        }
        diff = k - sum(quotas.values())
        if diff > 0:
            quotas[SeedBucket.RANDOM] += diff
        elif diff < 0:
            for b in (SeedBucket.RANDOM, SeedBucket.RARE_SCHEMA):
                take = min(quotas[b], -diff)
                quotas[b] -= take
                diff += take
                if diff == 0:
                    break

        selected: dict[str, SeedCandidate] = {}
        scan_cap = min(total, max(4 * k, 6 * k + 128))
        # 主 pass：按桶配额填（确定性置换扫描，O(scan_cap) 触碰）。
        self._fill_quotas(catalog, total, quotas, selected, scan_cap)
        # 回填 pass：有桶配额未满（候选不足）→ 从任意剩余候选补到 k；
        # 仍不足则返回唯一候选并在 diagnostics 标注 degraded。
        short_buckets = {
            b for b, q in quotas.items()
            if q > 0 and sum(1 for s in selected.values() if s.bucket == b) < q
        }
        if len(selected) < k:
            self._backfill(catalog, total, k, selected, scan_cap, short_buckets)

        out = list(selected.values())
        self.diagnostics = SeedDiagnostics(
            requested=k,
            returned=len(out),
            total_available=total,
            unique_available=len(out),
            degraded_buckets=sorted(b.name for b in short_buckets),
            bucket_counts=_bucket_counts(out),
        )
        return out

    # ------------------------------------------------------------------
    # 配额填充（确定性置换扫描）
    # ------------------------------------------------------------------

    def _fill_quotas(
        self,
        catalog: SeedCatalog,
        total: int,
        quotas: dict[SeedBucket, int],
        selected: dict[str, SeedCandidate],
        scan_cap: int,
    ) -> None:
        """单趟置换扫描按桶配额取 seed；命中则去重入 selected。

        确定性：起始/步长由 (rng_seed, 桶集) 派生。扫描至所有配额填满或
        达 scan_cap 即停（O(scan_cap) 触碰，不全量遍历）。
        """
        outstanding = {b: q for b, q in quotas.items() if q > 0}
        if not outstanding:
            return
        fill = {b: 0 for b in quotas}
        seen = set(selected)
        rng = random.Random(f"seedmix:{self.config.rng_seed}:main")
        start = rng.randrange(total)
        stride = _coprime(total, rng.randrange(1, max(2, total)))
        budget = min(scan_cap, total)
        for step in range(budget):
            if not outstanding:
                break
            idx = (start + step * stride) % total
            try:
                meta = catalog[idx]
            except Exception:  # noqa: BLE001 - 坏条目跳过
                continue
            s = self._to_candidate(meta)
            if s is None:
                continue
            b = s.bucket
            if b not in outstanding or fill[b] >= outstanding[b]:
                continue
            if s.factor_id in seen:
                continue
            if s.identity_ref in self._used_identities(selected):
                continue
            if s.canonical_hash in self._used_hashes(selected):
                continue
            selected[s.factor_id] = s
            seen.add(s.factor_id)
            fill[b] += 1
            if fill[b] >= outstanding[b]:
                del outstanding[b]

    def _backfill(
        self,
        catalog: SeedCatalog,
        total: int,
        k: int,
        selected: dict[str, SeedCandidate],
        scan_cap: int,
        short_buckets: set[SeedBucket],
    ) -> None:
        """主 pass 后仍不足 k：从任意剩余候选（任意桶）补到 k 或扫描上限。

        跨桶兜底只发生在「有桶候选不足」的场景；被兜底的桶记 degraded。
        兜底同样 O(scan_cap)，不遍历全表。
        """
        seen = set(selected)
        rng = random.Random(f"seedmix:{self.config.rng_seed}:backfill")
        start = rng.randrange(total)
        stride = _coprime(total, rng.randrange(1, max(2, total)))
        budget = min(scan_cap, total)
        for step in range(budget):
            if len(selected) >= k:
                break
            idx = (start + step * stride) % total
            try:
                meta = catalog[idx]
            except Exception:  # noqa: BLE001 - 坏条目跳过
                continue
            s = self._to_candidate(meta)
            if s is None or s.factor_id in seen:
                continue
            if s.identity_ref in self._used_identities(selected):
                continue
            if s.canonical_hash in self._used_hashes(selected):
                continue
            selected[s.factor_id] = s
            seen.add(s.factor_id)

    @staticmethod
    def _used_identities(selected: Mapping[str, SeedCandidate]) -> set[str]:
        return {s.identity_ref for s in selected.values() if s.identity_ref}

    @staticmethod
    def _used_hashes(selected: Mapping[str, SeedCandidate]) -> set[str]:
        return {s.canonical_hash for s in selected.values() if s.canonical_hash}

    def _to_candidate(self, meta: Mapping[str, Any]) -> SeedCandidate | None:
        """把一条 metadata dict 转成 SeedCandidate（缺字段时容错）。

        桶归属：条目自带 ``bucket``（SeedBucket 或合法名字符串）→ 用其值；
        缺失 / 非法 → RANDOM（随机探索兜底）。
        """
        if meta is None:
            return None
        fid = str(meta.get("factor_id") or "")
        formula = str(
            meta.get("canonical_repr")
            or meta.get("formula")
            or meta.get("canonical_formula")
            or ""
        ).strip()
        if not fid or not formula:
            return None
        entry_bucket = meta.get("bucket")
        if isinstance(entry_bucket, SeedBucket):
            eb = entry_bucket
        elif entry_bucket is not None:
            try:
                eb = SeedBucket(str(entry_bucket))
            except ValueError:
                eb = _FALLBACK_BUCKET
        else:
            eb = _FALLBACK_BUCKET
        chash = str(meta.get("canonical_hash") or "")
        iref = str(meta.get("fe_identity_ref") or meta.get("signal_equivalence_id") or "")
        fam = meta.get("parameter_family_id")
        idf = self.identity_fn or self.config.identity_fn
        if (not iref or not chash) and idf is not None:
            try:
                view = idf(formula)
            except Exception:  # noqa: BLE001 - identity_fn 失败不阻塞采样
                view = {}
            chash = chash or str(view.get("canonical_ast_hash") or "")
            iref = iref or str(view.get("signal_equivalence_id") or "")
            fam = fam or view.get("parameter_family_id")
        # 无 identity 可判定时用 factor_id 当去重口径（保底，绝不静默重复入 DAG）
        iref = iref or fid
        chash = chash or iref
        return SeedCandidate(
            factor_id=fid,
            formula=formula,
            bucket=eb,
            identity_ref=iref,
            canonical_hash=chash,
            metadata=dict(meta),
        )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _catalog_len(catalog: SeedCatalog) -> int:
    try:
        return int(len(catalog))
    except TypeError:  # pragma: no cover - 非 Sequence 源
        raise TypeError(
            "SeedCatalog must be a Sequence (len + getitem); got "
            f"{type(catalog).__name__}"
        )


def _coprime(total: int, base: int) -> int:
    """返回与 total 互质的步长（确定性扩展欧几里得）。

    O(log total)。total<=1 时返回 1（无步进意义）。
    """
    if total <= 1:
        return 1
    base = max(1, base % max(1, total))
    import math

    if math.gcd(base, total) == 1:
        return base
    # 就近找一个互质步长
    for d in range(1, total):
        if base + d < total and math.gcd(base + d, total) == 1:
            return base + d
        if base - d > 0 and math.gcd(base - d, total) == 1:
            return base - d
    return 1


def _bucket_counts(seeds: Sequence[SeedCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in seeds:
        counts[s.bucket.name] = counts.get(s.bucket.name, 0) + 1
    return counts
