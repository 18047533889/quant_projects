"""Evidence files are sampled once per overlay, never cached across overlays."""
import json

def test_overlay_reuses_one_read_but_refreshes_and_fails_closed(tmp_path, monkeypatch):
    from factor_engine.backend import factor_operator_evidence as evidence
    from factor_engine.backend import evidence_delta, evidence_provenance
    from factor_engine.cleaned_operators import production_hardening
    from factor_engine.cleaned_operators import production_certification_overlay as overlay
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    path = tmp_path / "verified.json"
    monkeypatch.setattr(evidence, "VERIFIED_PATH", path)
    monkeypatch.setattr(evidence_delta, "install_evidence_delta", lambda: None)
    monkeypatch.setattr(evidence_provenance, "evidence_artifact_valid", lambda: False)
    monkeypatch.setattr(production_hardening, "factor_production_targets",
                        lambda: {"probe_a", "probe_b"})
    monkeypatch.setattr(evidence, "pandas_reference_production_safe", lambda name: True)
    catalog = {"probe_a": {}, "probe_b": {}}
    monkeypatch.setattr(OperatorRegistry, "_catalog", catalog)
    calls = []
    read = evidence.load_factor_operator_evidence
    def counted_read():
        calls.append(1)
        return read()
    monkeypatch.setattr(evidence, "load_factor_operator_evidence", counted_read)
    def write(flag):
        path.write_text(json.dumps({"operators": {
            name: {"runtime_execution_verified": flag, "semantic_golden_verified": False,
                   "source_contract_verified": False} for name in catalog}}))
    write(True)
    overlay.apply_evidence_certification_overlay()
    assert all(c["runtime_execution_verified"] for c in catalog.values())
    assert all(not c["semantic_golden_verified"] and not c["source_contract_verified"]
               for c in catalog.values())
    assert len(calls) == 1
    write(False)
    overlay.apply_evidence_certification_overlay()
    assert len(calls) == 2
    assert all(not c["runtime_execution_verified"] for c in catalog.values())
    path.write_text("{invalid")
    overlay.apply_evidence_certification_overlay()
    assert len(calls) == 3
    assert all(not c["runtime_execution_verified"] for c in catalog.values())
    write(True)
    monkeypatch.setattr(evidence, "pandas_reference_production_safe", lambda name: False)
    overlay.apply_evidence_certification_overlay()
    assert len(calls) == 4
    assert all(not c["runtime_execution_verified"] for c in catalog.values())
