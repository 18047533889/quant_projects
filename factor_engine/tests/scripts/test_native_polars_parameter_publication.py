import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import pytest

from factor_engine.runtime.parameter_domain_store import ParameterDomainCertificationStore, FE_ROOT
from factor_engine.scripts import native_polars_parameter_publication as publication
from factor_engine.scripts.native_polars_parameter_points import certify_selected_point
from factor_engine.tests.scripts.test_native_polars_parameter_points import _fixture


@pytest.fixture
def output():
    base = FE_ROOT.parent / "evidence" / "factor_engine"
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base, prefix="fe-parameter-publication-test-") as directory:
        yield Path(directory) / "store.json"


def _certified():
    native, reference = _fixture()
    store = ParameterDomainCertificationStore()
    key = certify_selected_point("ts_mean", [native], [reference], {"window": 2},
        context=SimpleNamespace(data_source=None), store=store)
    return store, key


def test_published_points_bind_current_worktree_and_load_without_hash_relabel(output):
    store, key = _certified()
    generation = publication.capture_generation(["ts_mean"])
    publication.publish_store(store, output, generation)
    payload = json.loads(output.read_text())
    assert payload["_meta"]["source_tree"] == "working_tree"
    assert "scripts/native_polars_parameter_points.py" in payload["_meta"]["source_input_hashes"]
    loaded = ParameterDomainCertificationStore()
    assert loaded.load_json(output) == 1
    from factor_engine.runtime.parameter_domain_store import _verify_working_tree_inputs
    _verify_working_tree_inputs(loaded)
    assert loaded.exact_call_is_certified("ts_mean", dict(key.parameter_point),
        backend=key.backend, semantic_version=key.semantic_version,
        execution_variant=key.execution_variant, source_context=key.source_context,
        dtype=key.dtype, grain=key.grain)
    assert list(output.parent.iterdir()) == [output]


def test_source_drift_does_not_publish(output, monkeypatch):
    store, _ = _certified()
    generation = publication.capture_generation(["ts_mean"])
    altered = copy.deepcopy(generation)
    altered["head"] = "changed"
    monkeypatch.setattr(publication, "capture_generation", lambda *_: altered)
    with pytest.raises(RuntimeError, match="source generation changed"):
        publication.publish_store(store, output, generation)
    assert not output.exists()


def test_existing_evidence_cannot_be_overwritten(output):
    store, _ = _certified()
    generation = publication.capture_generation(["ts_mean"])
    output.write_text("existing-evidence")
    with pytest.raises(FileExistsError):
        publication.publish_store(store, output, generation)
    assert output.read_text() == "existing-evidence"


def test_empty_store_cannot_publish_and_temporary_file_is_cleaned(output):
    _fixture()
    generation = publication.capture_generation(["ts_mean"])
    with pytest.raises(ValueError, match="freshly executed"):
        publication.publish_store(ParameterDomainCertificationStore(), output, generation)
    assert list(output.parent.iterdir()) == []


def test_outside_project_evidence_cannot_publish(tmp_path):
    with pytest.raises(ValueError, match="inside project evidence"):
        publication.publish_store(ParameterDomainCertificationStore(), tmp_path / "outside.json", {})
