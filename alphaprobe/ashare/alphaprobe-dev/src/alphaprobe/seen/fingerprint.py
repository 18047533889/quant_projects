"""Rank fingerprint 构建（复用 dedup.rank_fingerprint）+ LSH banding。

§32-§35: 256-bit SimHash + 4 band × 64-bit LSH banding。
fingerprint 元数据必须带 fingerprint_version / sample_date_set_id（§34），
版本不同不比较。

LSH banding：256-bit → 4 band × 64-bit；每 band 值建索引行；
查询取同 band 候选并集 → 精确 hamming top-k。
不做全表扫。
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from alphaprobe.dedup import fingerprint_hamming, rank_fingerprint


# 配置
FINGERPRINT_VERSION = "v1"
NUM_BANDS = 16
BITS_PER_BAND = 16  # 256 / 16 = 16
RANK_EXACT_THRESHOLD = 0.995  # §36
HIGHLY_CORRELATED_THRESHOLD = 0.90  # §36
MAX_HAMMING_FOR_RANK_EQUIVALENT = 18  # 256-bit: ~7% hamming ≈ 0.995 rank corr


def build_rank_fingerprint(
    cross_section_ranks_by_date: "list[dict[str, float]] | None",
    *,
    fingerprint_version: str = FINGERPRINT_VERSION,
) -> bytes | None:
    """构建 rank fingerprint（复用 dedup.rank_fingerprint）。

    Parameters
    ----------
    cross_section_ranks_by_date : list[dict[str, float]] | None
        每个采样日的 {stock: rank}（0~1）。
    fingerprint_version : str
        fingerprint 版本（§34），默认 v1。

    Returns
    -------
    bytes | None
        256-bit SimHash fingerprint，或 None（缺数据时）。
    """
    return rank_fingerprint(cross_section_ranks_by_date)


def lsh_bands(fp: bytes) -> list[int]:
    """256-bit SimHash → 16 × 16-bit band 整数。

    小 band（16-bit）保证高召回：轻微扰动下仍有多个 band 命中，
    查询时取同 band 候选并集 → 精确 hamming top-k（§32-§35）。

    Parameters
    ----------
    fp : bytes
        32 字节（256-bit）fingerprint。

    Returns
    -------
    list[int]
        16 个 16-bit 整数，每个 band 一个。
    """
    if len(fp) != 32:
        raise ValueError(f"expected 32 bytes, got {len(fp)}")
    bands: list[int] = []
    for i in range(NUM_BANDS):
        chunk = fp[i * 2 : (i + 1) * 2]
        v = int.from_bytes(chunk, "big")
        bands.append(v)
    return bands


def lsh_candidates(
    fp: bytes,
    bands: list[int],
    *,
    fingerprint_version: str = FINGERPRINT_VERSION,
) -> dict[str, list[tuple[int, bytes]]]:
    """构建 LSH 候选查询参数。

    返回 dict, key=band_index, value=list of (band_value, fingerprint_version)。

    调用方用这些参数去 store 查 ``seen_fingerprints`` 同 band 的所有
    factor_pk，然后做精确 hamming 排序。
    """
    result: dict[str, list[tuple[int, bytes]]] = {}
    for bi, bv in enumerate(bands):
        result[str(bi)] = [(bv, fp)]
    return result


def nearest_by_hamming(
    candidates: Sequence[tuple[int, bytes]],  # (factor_pk, fp_bytes)
    query_fp: bytes,
    k: int = 20,
    max_hamming: int = MAX_HAMMING_FOR_RANK_EQUIVALENT * 2,
) -> list[tuple[int, int, bytes]]:
    """从候选集中精确 hamming 排序取 top-k。

    Parameters
    ----------
    candidates : Sequence[tuple[int, bytes]]
        (factor_pk, fp_bytes) 候选列表。
    query_fp : bytes
        查询 fingerprint。
    k : int
        返回 top-k。
    max_hamming : int
        最大 hamming 距离阈值。

    Returns
    -------
    list[tuple[int, int, bytes]]
        [(factor_pk, hamming_distance, fp_bytes), ...] 按 hamming 升序。
    """
    scored: list[tuple[int, int, bytes]] = []
    for factor_pk, fp in candidates:
        d = fingerprint_hamming(query_fp, fp)
        if d <= max_hamming:
            scored.append((factor_pk, d, fp))
    scored.sort(key=lambda t: t[1])
    return scored[:k]


def candidate_union_scan(
    store: Any,
    query_fp: bytes,
    k: int = 20,
    *,
    fingerprint_version: str = FINGERPRINT_VERSION,
    num_bands: int | None = None,
    max_hamming: int | None = None,
) -> list[tuple[int, int, bytes]]:
    """LSH band 候选并集 → 精确 hamming 排序 top-k。

    Parameters
    ----------
    store : SeenStore
        存储。
    query_fp : bytes
        查询 fingerprint（32 字节）。
    k : int
        top-k。
    fingerprint_version : str
        fingerprint 版本（§34）。
    num_bands : int, optional
        band 数（默认 NUM_BANDS=16）。
    max_hamming : int, optional
        最大 hamming 阈值（默认 MAX_HAMMING_FOR_RANK_EQUIVALENT*2=36）。

    Returns
    -------
    list[tuple[int, int, bytes]]
        [(factor_pk, hamming, fp_bytes), ...] 按 hamming 升序。
    """
    nb = num_bands or NUM_BANDS
    max_h = max_hamming if max_hamming is not None else MAX_HAMMING_FOR_RANK_EQUIVALENT * 2
    bands = lsh_bands(query_fp)
    if len(bands) != nb:
        raise ValueError(f"expected {nb} bands, got {len(bands)}")

    # 同 band 候选并集（band_value 存 TEXT 避免 64-bit 溢出）
    candidate_pks: dict[int, bytes] = {}
    for bi in range(nb):
        rows = store.q(
            "SELECT factor_pk, fp FROM seen_fingerprints"
            " WHERE fingerprint_version=? AND band_index=? AND band_value=?",
            (fingerprint_version, bi, str(bands[bi])),
        )
        for pk, fp_bytes in rows:
            candidate_pks[int(pk)] = bytes(fp_bytes)

    if not candidate_pks:
        return []
    candidates = [(pk, fp_b) for pk, fp_b in candidate_pks.items()]
    return nearest_by_hamming(candidates, query_fp, k=k, max_hamming=max_h)


def hamming_to_similarity(hamming: int, total_bits: int = 256) -> float:
    """Hamming 距离 → rank correlation 相似度估计。

    对 256-bit SimHash，hamming=0 → 1.0, hamming=128 → 0.0。
    """
    return 1.0 - hamming / total_bits


def similarity_to_hamming(sim: float, total_bits: int = 256) -> int:
    """相似度 → 等价 hamming 阈值（向下取整）。"""
    return int((1.0 - sim) * total_bits)