"""ANN Top-K（LSH band 检索 → 候选精确 hamming 排序，不做全库扫）。

§32-§35：256-bit SimHash → 4 band × 64-bit LSH banding；
查询取同 band 候选并集 → 精确 hamming top-k。
1m 级也不做全表扫。
"""

from __future__ import annotations

from dataclasses import dataclass

from alphaprobe.seen import fingerprint as fpmod
from alphaprobe.seen.store import SeenStore


@dataclass(frozen=True)
class NearestNeighbor:
    """§32 邻居结果。"""

    factor_id: str
    factor_pk: int
    hamming: int
    fingerprint_version: str
    correlation_estimate: float | None = None


class NearestIndex:
    """LSH banding ANN Top-K 索引（SQLite-backed）。

    用法：
    1. 入库时用 ``lsh_bands(fp)`` 计算 bands，存 store.seen_fingerprints。
    2. 查询时 ``nearest(fp, k)`` 从同 band 候选取并集 → 精确 hamming top-k。
    """

    def __init__(self, store: SeenStore, *, num_bands: int | None = None) -> None:
        self._store = store
        self._num_bands = num_bands or fpmod.NUM_BANDS

    # -- 入库 ---------------------------------------------------------------

    def add(
        self,
        factor_pk: int,
        fp: bytes | None,
        *,
        fingerprint_version: str = fpmod.FINGERPRINT_VERSION,
        sample_date_set_id: str | None = None,
    ) -> None:
        """写入 fingerprint + bands（band 每行建索引，便于查询）。"""
        if fp is None:
            return
        bands = fpmod.lsh_bands(fp)
        self._store.upsert_fingerprint(
            factor_pk, fp, fingerprint_version, sample_date_set_id, bands
        )

    # -- 查询 ---------------------------------------------------------------

    def nearest(
        self,
        fp: bytes | None,
        k: int = 20,
        *,
        fingerprint_version: str = fpmod.FINGERPRINT_VERSION,
        max_hamming: int | None = None,
    ) -> list[NearestNeighbor]:
        """LSH band 检索 → 候选精确 hamming 排序 → top-k。

        不做全库扫：只查与查询指纹同 band 的行（带索引）。
        """
        if fp is None:
            return []
        scored = fpmod.candidate_union_scan(
            self._store, fp, k=k,
            fingerprint_version=fingerprint_version,
            num_bands=self._num_bands,
            max_hamming=max_hamming,
        )

        # 取 factor_id
        out: list[NearestNeighbor] = []
        for pk, hamming, _fp_b in scored[:k]:
            factor = self._store.get_factor_by_pk(pk)
            if factor is None:
                continue
            corr = fpmod.hamming_to_similarity(hamming)
            out.append(
                NearestNeighbor(
                    factor_id=factor["factor_id"],
                    factor_pk=pk,
                    hamming=hamming,
                    fingerprint_version=fingerprint_version,
                    correlation_estimate=corr,
                )
            )
        return out

    def nearest_by_id(
        self,
        factor_id: str,
        k: int = 20,
        *,
        fingerprint_version: str = fpmod.FINGERPRINT_VERSION,
        max_hamming: int | None = None,
    ) -> list[NearestNeighbor]:
        """用已入库因子的 fingerprint 查询最近邻。"""
        factor = self._store.get_factor_by_id(factor_id)
        if factor is None:
            return []
        row = self._store.q1(
            "SELECT fp FROM seen_fingerprints WHERE factor_pk=? AND fingerprint_version=? "
            "LIMIT 1",
            (factor["factor_pk"], fingerprint_version),
        )
        if row is None:
            return []
        return self.nearest(bytes(row[0]), k=k, fingerprint_version=fingerprint_version,
                            max_hamming=max_hamming)
