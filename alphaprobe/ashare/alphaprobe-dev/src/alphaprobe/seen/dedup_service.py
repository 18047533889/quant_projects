"""DedupService 高层 API（§22/§31 判定语义）。

全实现：lookup_identity / reserve / register_evaluation / register_alias /
mark_exported / nearest_by_fingerprint / ingest_seed / stats。

判定语义（严格）：
- canonical_ast_hash 相等 → EXACT_DUPLICATE（不读行情）
- signal_equivalence_id 相等 → SIGN_EQUIVALENT_DUPLICATE（不读行情）
- family_id 相等 → 不 reject，返回 family 统计供 saturation
- rank corr >= 0.995 → RANK_EQUIVALENT_DUPLICATE（confirm 阶段，correlation 计算在 C 阶段）
- 0.90 <= corr < 0.995 → HIGHLY_CORRELATED（绝不 hard reject）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alphaprobe.seen.fingerprint import (
    FINGERPRINT_VERSION,
    HIGHLY_CORRELATED_THRESHOLD,
    RANK_EXACT_THRESHOLD,
    hamming_to_similarity,
)
from alphaprobe.seen.nearest import NearestIndex, NearestNeighbor
from alphaprobe.seen.registry import RegistryService
from alphaprobe.seen.reservation import IdentityLookupResult, ReservationResult, ReservationService
from alphaprobe.seen.store import SeenStore


@dataclass(frozen=True)
class SeedIngestResult:
    """§35 seed 导入结果。"""

    created: bool
    factor_id: str | None = None
    factor_pk: int | None = None
    verdict: str = "NEW"
    existing_factor_id: str | None = None
    existing_factor_pk: int | None = None


# 配置对象（§36，不许散落写死）
@dataclass
class DedupConfig:
    """去重配置（§36）。集中一处，不许散落写死。"""
    rank_exact_threshold: float = RANK_EXACT_THRESHOLD  # 0.995
    highly_correlated_threshold: float = HIGHLY_CORRELATED_THRESHOLD  # 0.90
    max_hamming_for_rank_equivalent: int = 18  # 256-bit ≈ 7% hamming
    num_bands: int = 16  # LSH banding（§34）
    bits_per_band: int = 16
    family_saturation_soft_limit: int = 30  # 族成员数建议上限
    family_saturation_hard_limit: int = 50  # 族成员数硬上限


class DedupService:
    """§22/§31 Dedup 高层 API。编排 ReservationService + RegistryService + NearestIndex。"""

    def __init__(
        self,
        store: SeenStore,
        config: DedupConfig | None = None,
    ) -> None:
        self._store = store
        self._config = config or DedupConfig()
        self._reservation = ReservationService(store)
        self._registry = RegistryService(store)
        self._nearest = NearestIndex(store, num_bands=self._config.num_bands)

    # -- 只读判定 ---------------------------------------------------------------

    def lookup_identity(
        self,
        identity: Any,
        *,
        identity_version: str | None = None,
    ) -> IdentityLookupResult:
        """§22/§31：给定 identity 返回判定结果。"""
        return self._reservation.lookup_identity(
            identity, identity_version=identity_version
        )

    # -- 写路径 ---------------------------------------------------------------

    def reserve(
        self,
        identity: Any,
        *,
        source_system: str = "alphaprobe",
        run_id: str = "",
        worker_id: str = "",
        identity_version: str | None = None,
    ) -> ReservationResult:
        """Atomic reservation（§11.6/§59）。"""
        return self._reservation.reserve(
            identity,
            source_system=source_system,
            run_id=run_id,
            worker_id=worker_id,
            identity_version=identity_version,
        )

    def register_evaluation(
        self,
        factor_id: str,
        *,
        metrics: dict[str, Any] | None = None,
        fingerprint: bytes | None = None,
        subtree_hashes: list[str] | None = None,
        passed_hard_gates: bool = False,
        fingerprint_version: str | None = None,
        sample_date_set_id: str | None = None,
    ) -> None:
        """status → EVALUATED，更新 family / subtree 统计。

        Parameters
        ----------
        factor_id : str
            已入库的因子 id。
        metrics : dict, optional
            评估 metric bundle（含 search_fitness 等）。
        fingerprint : bytes, optional
            rank fingerprint。
        subtree_hashes : list[str], optional
            子树哈希列表。
        passed_hard_gates : bool
            是否通过硬门限。
        fingerprint_version : str, optional
            fingerprint 版本。
        sample_date_set_id : str, optional
            采样日期集 id。
        """
        factor = self._store.get_factor_by_id(factor_id)
        if factor is None:
            raise ValueError(f"factor_id not found: {factor_id}")
        factor_pk = factor["factor_pk"]

        # 更新 status
        self._store.update_status(factor_pk, "EVALUATED")
        # 计数 EXACT_DUPLICATE / SIGN_EQUIVALENT_DUPLICATE 判定累计已在
        # reservation 层记录 rediscovery（stats），此处只记录 EVALUATED
        self._store.incr_meta(f"verdict:EVALUATED:{factor.get('identity_version','1')}", by=1)

        # fingerprint
        ver = fingerprint_version or FINGERPRINT_VERSION
        if fingerprint is not None:
            self._nearest.add(factor_pk, fingerprint, fingerprint_version=ver,
                              sample_date_set_id=sample_date_set_id)

        # subtree stats
        if subtree_hashes:
            import time
            now = time.time()
            success = bool(metrics and metrics.get("search_fitness", 0) > 0) if metrics else True
            elite = bool(metrics and metrics.get("is_elite", False)) if metrics else False
            for sh in subtree_hashes:
                self._store.upsert_subtree_stat(sh, None, factor_pk, success, elite, now)

        # family 更新
        family_id = factor.get("parameter_family_id")
        if family_id and metrics:
            search_fitness = metrics.get("search_fitness")
            self._store.upsert_family(
                family_id=family_id,
                family_template=None,
                factor_pk=factor_pk,
                factor_id=factor_id,
                search_fitness=search_fitness,
                outcome="evaluated" if passed_hard_gates else "rejected",
            )
    def register_alias(
        self,
        identity: Any,
        canonical_factor_id: str,
        *,
        source_system: str = "alphaprobe",
    ) -> bool:
        """跨 Miner 同 hash 只加 alias。"""
        return self._registry.register_alias(
            identity, canonical_factor_id, source_system=source_system
        )

    def mark_exported(self, factor_id: str) -> None:
        """status → EXPORTED。"""
        factor_pk = self._store.factor_pk_by_id(factor_id)
        if factor_pk is not None:
            self._store.update_status(factor_pk, "EXPORTED")

    # -- ANN ---------------------------------------------------------------

    def nearest_by_fingerprint(
        self,
        fp: bytes,
        k: int = 20,
        *,
        fingerprint_version: str = FINGERPRINT_VERSION,
        max_hamming: int | None = None,
    ) -> list[NearestNeighbor]:
        """LSH banding ANN top-k。"""
        return self._nearest.nearest(
            fp, k=k, fingerprint_version=fingerprint_version, max_hamming=max_hamming
        )

    # -- 相关性判定（供 C 阶段 evaluator 调用） -----------------------------

    def confirm_signal_similarity(
        self,
        candidate_fp: bytes,
        neighbors: list[NearestNeighbor],
        *,
        correlation_values: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """输入预计算的 corr 值，返回判定结果列表。

        Parameters
        ----------
        candidate_fp : bytes
            候选因子 fingerprint。
        neighbors : list[NearestNeighbor]
            ANN 邻居列表。
        correlation_values : list[float], optional
            预计算的 rank correlation 值（与 neighbors 一一对应）。
            如果为 None，用 hamming 距离估计。

        Returns
        -------
        list[dict]
            [{"factor_id": str, "correlation": float, "verdict": str}, ...]
            verdict: RANK_EQUIVALENT, HIGHLY_CORRELATED, LOW_CORRELATION
        """
        results: list[dict[str, Any]] = []
        for i, neighbor in enumerate(neighbors):
            if correlation_values is not None and i < len(correlation_values):
                corr = correlation_values[i]
            else:
                corr = hamming_to_similarity(neighbor.hamming)

            if corr >= self._config.rank_exact_threshold:
                verdict = "RANK_EQUIVALENT_DUPLICATE"
            elif corr >= self._config.highly_correlated_threshold:
                verdict = "HIGHLY_CORRELATED"
            else:
                verdict = "LOW_CORRELATION"

            results.append({
                "factor_id": neighbor.factor_id,
                "correlation": corr,
                "verdict": verdict,
            })
        return results

    # -- Seed ingestion ------------------------------------------------------

    def ingest_seed(
        self,
        identity: Any,
        *,
        source_system: str = "seed_library",
        run_id: str = "",
        fingerprint: bytes | None = None,
        identity_version: str | None = None,
    ) -> SeedIngestResult:
        result = self._registry.ingest_seed(
            identity,
            source_system=source_system,
            run_id=run_id,
            fingerprint=fingerprint,
            identity_version=identity_version,
        )
        if fingerprint and result.get("factor_pk"):
            self._nearest.add(
                result["factor_pk"], fingerprint,
                fingerprint_version=identity_version or FINGERPRINT_VERSION,
            )
        return SeedIngestResult(
            created=result["created"],
            factor_id=result["factor_id"],
            factor_pk=result["factor_pk"],
            verdict=result["verdict"],
            existing_factor_id=result.get("existing_factor_id"),
        )

    # -- 统计 ---------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """各 verdict 计数、库大小、per-version。"""
        result: dict[str, Any] = {}
        result["total_factors"] = self._store.count_factors()
        result["total_aliases"] = self._store.count_aliases()
        result["total_families"] = self._store.count_families()
        result["total_subtrees"] = self._store.count_subtrees()
        result["total_actions"] = self._store.count_actions()
        result["parse_failures"] = self._store.count_parse_failures()

        meta_rows = self._store.q(
            "SELECT key, value FROM seen_meta WHERE key LIKE 'verdict:%'"
        )
        verdicts: dict[str, int] = {}
        for key, val in meta_rows:
            verdicts[str(key)] = int(val)
        result["verdicts"] = verdicts

        version_rows = self._store.q(
            "SELECT key, value FROM seen_meta WHERE key LIKE 'version:%:count'"
        )
        versions: dict[str, int] = {}
        for key, val in version_rows:
            parts = str(key).split(":")
            ver = parts[1] if len(parts) >= 2 else "unknown"
            versions[ver] = int(val)
        result["per_version"] = versions

        result["fingerprint_versions"] = self._store.fingerprint_versions()
        return result

    # -- 构建（类方法工厂）-------------------------------------------------------

    @classmethod
    def build_seen_index(
        cls,
        db_path: str | None = None,
        config: DedupConfig | None = None,
    ) -> "DedupService":
        """工厂方法：创建 store + DedupService。"""
        store = SeenStore(db_path)
        return cls(store, config=config)