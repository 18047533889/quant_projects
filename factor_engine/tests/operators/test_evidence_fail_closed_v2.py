from __future__ import annotations


def test_segmented_execution_set_matches_runtime_restore_implementations():
    """Every advertised segmented operator must have a real checkpoint-restore
    branch in ``stateful_runtime.execute_stateful_segment`` and advertise
    streaming in the runtime catalog; the runtime is the single gate."""
    import numpy as np
    import pandas as pd

    from cleaned_operators import load_all
    from cleaned_operators.production_hardening import SEGMENTED_EXECUTION_CANONICALS
    from cleaned_operators.registry import OperatorRegistry
    from stateful_contract import StatefulCheckpointRegistry
    from stateful_runtime import execute_stateful_segment

    load_all()
    assert SEGMENTED_EXECUTION_CANONICALS, "segmented set must not be empty"

    # Minimal one-instrument inputs that exercise each runtime branch.  Values
    # are deliberately short: the restore path must accept them without raising
    # "stateful runtime not implemented".
    timestamps = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    high = np.array([1.0, 1.1, 1.2])
    low = np.array([0.9, 0.9, 0.95])
    close = np.array([1.0, 1.05, 1.1])
    smokes = {
        "ts_ema": {"x": close},
        "ts_ewm_std": {"x": close},
        "ts_ewm_var": {"x": close},
        "ts_ewm_cov": {"x": close, "y": high},
        "ts_ewm_corr": {"x": close, "y": high},
        "RSI_WILDER": {"x": close},
        "ATR_WILDER": {"high": high, "low": low, "close": close},
        "ADX": {"high": high, "low": low, "close": close},
        "MACD_line": {"x": close},
        "MACD_signal": {"x": close},
        "MACD_hist": {"x": close},
    }
    assert set(smokes) == set(SEGMENTED_EXECUTION_CANONICALS), (
        "SEGMENTED_EXECUTION_CANONICALS must stay in sync with stateful_runtime"
    )
    for canonical in sorted(SEGMENTED_EXECUTION_CANONICALS):
        spec = StatefulCheckpointRegistry.get(canonical)
        assert spec is not None, f"{canonical} segmented but not registered"
        assert spec.segmented_execution_supported, f"{canonical} contract not segmented"
        # Restore implementation must actually exist.
        execute_stateful_segment(
            canonical, smokes[canonical], timestamps=timestamps, instrument="A",
            input_identity={"dataset": "smoke"}, starts_at_dataset_origin=True,
        )
        catalog = OperatorRegistry.catalog()[canonical]
        assert catalog["segmented_execution_supported"] is True, canonical
        pandas_meta = catalog["backend_meta"]["pandas_numpy"]
        assert pandas_meta["supports_streaming"] is True, canonical
        assert catalog["checkpoint_contract"]["segmented_execution_supported"] is True, canonical


def test_stateful_operators_without_runtime_fail_closed():
    """``trade_when`` is registered for checkpoint management but has no restore
    implementation, so it must not advertise segmented execution and must stay
    full-replay / non-streaming."""
    from cleaned_operators import load_all
    from cleaned_operators.production_hardening import STATEFUL_CHECKPOINTS
    from cleaned_operators.registry import OperatorRegistry
    from stateful_contract import StatefulCheckpointRegistry

    load_all()
    spec = StatefulCheckpointRegistry.get("trade_when")
    assert spec is not None
    assert spec.segmented_execution_supported is False
    assert "trade_when" in STATEFUL_CHECKPOINTS
    catalog = OperatorRegistry.catalog()["trade_when"]
    assert catalog["segmented_execution_supported"] is False
    assert catalog["backend_meta"]["pandas_numpy"]["supports_streaming"] is False
    assert catalog["full_history_replay_required"] is True
    assert catalog["incremental_strategy"] == "full_replay"


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
        monkeypatch.setattr(
            provenance, "evidence_artifact_valid", lambda *args, **kwargs: False
        )
        assert capability._pandas_status(canonical) == "implemented"
        assert capability.production_eligible_backends(canonical) == ()
    finally:
        meta["production_certified"] = original_certified
        meta["certification_source"] = original_source


def test_every_unrestored_stateful_operator_is_full_replay_and_non_streaming():
    from cleaned_operators import load_all
    from cleaned_operators.production_hardening import (
        SEGMENTED_EXECUTION_CANONICALS,
        STATEFUL_CHECKPOINTS,
    )
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    unrestored = set(STATEFUL_CHECKPOINTS).difference(
        SEGMENTED_EXECUTION_CANONICALS
    )
    assert unrestored
    for canonical in sorted(unrestored):
        catalog = OperatorRegistry._catalog[canonical]
        checkpoint = catalog["checkpoint_contract"]
        pandas_meta = catalog["backend_meta"]["pandas_numpy"]
        assert checkpoint["segmented_execution_supported"] is False, canonical
        assert pandas_meta["supports_streaming"] is False, canonical
        assert catalog["full_history_replay_required"] is True, canonical
        assert catalog["incremental_strategy"] == "full_replay", canonical
