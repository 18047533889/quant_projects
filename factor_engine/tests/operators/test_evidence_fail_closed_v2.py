from __future__ import annotations


def test_pandas_production_status_does_not_fall_back_to_lifecycle_label(monkeypatch):
    """A production target without valid evidence must remain non-routable."""
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry
    import backend.evidence_provenance as provenance
    import backend.operator_capability as capability

    load_all()
    canonical = "KAMA"
    catalog = OperatorRegistry._catalog[canonical]
    meta = catalog.setdefault("backend_meta", {}).setdefault("pandas_numpy", {})
    original_certified = meta.get("production_certified")
    original_source = meta.get("certification_source")
    try:
        catalog["status"] = "production"
        meta["production_certified"] = False
        meta["certification_source"] = None
        monkeypatch.setattr(provenance, "evidence_artifact_valid", lambda *a, **k: False)
        assert capability._pandas_status(canonical) == "implemented"
        assert capability.production_eligible_backends(canonical) == ()
    finally:
        meta["production_certified"] = original_certified
        meta["certification_source"] = original_source


def test_stateful_full_replay_operator_does_not_advertise_streaming():
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    for canonical in ("KAMA", "Supertrend", "SupertrendDirection", "PSAR"):
        catalog = OperatorRegistry._catalog[canonical]
        checkpoint = catalog["checkpoint_contract"]
        pandas_meta = catalog["backend_meta"]["pandas_numpy"]
        assert checkpoint["segmented_execution_supported"] is False
        assert pandas_meta["supports_streaming"] is False
        assert catalog["full_history_replay_required"] is True
        assert catalog["incremental_strategy"] == "full_replay"
