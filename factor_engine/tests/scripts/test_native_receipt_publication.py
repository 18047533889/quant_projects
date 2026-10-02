import json
import pytest
from factor_engine.scripts import native_receipt_publication as publication


def test_success_replaces_receipt_without_temporary_files(tmp_path):
    path = tmp_path / "receipts.json"
    path.write_text("old")
    publication.publish_receipts(path, {"receipts": {"passed": True}})
    assert json.loads(path.read_text())["receipts"]["passed"]
    assert list(tmp_path.iterdir()) == [path]


def test_replace_failure_preserves_receipt_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "receipts.json"
    path.write_text("verified-old")
    def fail(*args):
        raise OSError("publication failed")
    monkeypatch.setattr(publication.os, "replace", fail)
    with pytest.raises(OSError, match="publication failed"):
        publication.publish_receipts(path, {"receipts": {}})
    assert path.read_text() == "verified-old"
    assert list(tmp_path.iterdir()) == [path]


def test_serialization_failure_does_not_create_directory(tmp_path):
    path = tmp_path / "new-dir" / "receipts.json"
    with pytest.raises(TypeError):
        publication.publish_receipts(path, {"invalid": object()})
    assert not path.parent.exists()
