"""Measured, bounded memory envelope for the verified F32 coverage benchmark."""

MANIFEST_SHA256 = "b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864"
EVIDENCE_FILE = "real_cos_f32_coverage_raw_identity_cache_ab_20261001.json"
PARENT_PEAK_RSS_KIB = 9_953_452
WORKER_PEAK_RSS_KIB = 14_436_108
ARRAY_UPPER_BOUND_BYTES = 3000 * 5500 * 32 * 8


def measured_coverage_peak_bytes(manifest_sha256, metric_ids):
    """Keep 8x the array bound and 20% above observed parent+worker RSS.

    The preflight caller still enforces the working-set cap, 8 GiB available
    RAM headroom, COS transfer limits and GPU admission. Other requests must
    retain their broader measured/extrapolated envelope.
    """
    if (type(manifest_sha256) is not str or type(metric_ids) is not tuple
            or manifest_sha256 != MANIFEST_SHA256 or metric_ids != ("coverage",)):
        return None
    combined = (PARENT_PEAK_RSS_KIB + WORKER_PEAK_RSS_KIB) * 1024
    empirical = (combined * 6 + 4) // 5
    gib = 1024**3
    empirical = ((empirical + gib - 1) // gib) * gib
    return max(ARRAY_UPPER_BOUND_BYTES * 8, empirical)
