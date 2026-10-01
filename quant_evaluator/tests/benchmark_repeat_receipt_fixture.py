"""Small, complete repeat-receipt fixtures for metric benchmark CLI tests."""


def fake_repeat_receipts(backend, metrics, *, config_hash="same", reason=None):
    metrics = tuple(metrics)
    metric_receipts = {}
    for index, metric in enumerate(metrics, start=1):
        metric_receipts[metric] = {
            "artifact_sha256": f"{index:064x}",
            "value_shape": [1], "value_count": 1,
            "finite_mask_sha256": f"{index + 100:064x}",
            "finite_mask_shape": [1], "finite_count": 1,
            "valid_mask_sha256": f"{index + 200:064x}",
            "valid_mask_shape": [1], "valid_count": 1,
            "counts_sha256": f"{index + 300:064x}",
            "counts_shape": [1], "count_sum": 32,
            "observation_counts": [32],
            "observation_counts_sha256": f"{index + 400:064x}",
            "observation_counts_shape": [1],
            "parity_scope": "full_metric_artifact_exact_sha256",
        }
    receipts = []
    for repeat in (1, 2):
        receipt = {
            "repeat": repeat, "elapsed_s": 0.1, "backend_used": backend,
            "auto_backend_reason": reason, "auto_backend_profile": None,
            "metric_backends": {metric: backend for metric in metrics},
            "config_hash": config_hash,
            "execution_receipt": {"backend_used": backend,
                                  "auto_backend_reason": reason,
                                  "config_hash": config_hash},
            "artifact_sha256": f"{500:064x}",
            "artifact_parity_scope": "full_metric_artifacts_exact_sha256",
            "metrics": metric_receipts,
        }
        if metrics == ("coverage",):
            receipt["value_snapshots"] = {"coverage": [0.75]}
        receipts.append(receipt)
    return receipts
