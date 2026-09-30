"""Source assurance mode integration without CUDA or wall-clock races."""
from dataclasses import replace

import os
import pytest

from quant_evaluator.runtime import backend_calibration as c
from quant_evaluator.runtime.source_identity import ProcessSourceIdentity
from quant_evaluator.tests.test_backend_calibration_edges import harness


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    path = tmp_path / "kernel.py"
    path.write_text("VALUE = 1\n")
    identity = ProcessSourceIdentity(tmp_path)
    monkeypatch.setattr(c, "_SOURCE_IDENTITY", identity)
    monkeypatch.setattr(c, "_source_fingerprint",
                        lambda: identity.identify(strict_full_content=True).digest)
    return path, identity


def test_modes_share_identity_without_false_source_drift(harness, source_tree):
    run, calls, _, _, _, _, _, policy, cache = harness
    # Reapply the real identity seam after the harness's legacy source stub.
    guarded = replace(policy, source_check_mode="stat_guarded")
    first = run(current_policy=guarded)
    assert first.metadata["source_check"]["files_hashed"] == 1
    assert run(current_policy=policy).metadata["status"] == "calibrated"
    repeat = run(current_policy=guarded)
    assert repeat.metadata["status"] == "cache_hit"
    assert repeat.metadata["source_check"]["files_hashed"] == 0
    assert repeat.metadata["source_check"]["bytes_hashed"] == 0
    assert not c._PROCESS_SOURCE_DRIFTED
    assert len(cache) == 2 and len(calls) == 9


def test_guarded_source_drift_permanently_disables_route_reuse(harness, source_tree):
    run, calls, _, _, _, _, _, policy, cache = harness
    path, _ = source_tree
    guarded = replace(policy, source_check_mode="stat_guarded")
    run(current_policy=guarded)
    path.write_text("VALUE = 2\n")
    # Same-size writes within one filesystem timestamp tick may retain all
    # stat guards. This test requires a visible edit, not strict assurance.
    stamp = path.stat().st_mtime_ns + 1_000_000_000
    os.utime(path, ns=(stamp, stamp))
    assert run(current_policy=guarded).metadata["process_source_drifted"]
    path.write_text("VALUE = 1\n")
    assert run(current_policy=guarded).metadata["status"] == "calibrated"
    assert len(cache) == 1 and len(calls) == 12


def test_strict_override_does_not_mutate_identity_default(source_tree):
    _, identity = source_tree
    a = identity.identify(strict_full_content=True)
    b = identity.identify()
    assert a.digest == b.digest
    assert a.full_content_checked and b.files_hashed == 0
    assert not identity.strict_full_content


@pytest.mark.parametrize("mode", [None, "unknown", 1])
def test_invalid_assurance_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="source_check_mode"):
        c.CalibrationPolicy(source_check_mode=mode)
