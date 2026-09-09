import hashlib
from scripts.audit_legacy_branch_content import current_path_candidates, reconcile_inventory


def test_renamed_dataaccess_is_only_a_mapping_not_automatic_closure(tmp_path):
    assert current_path_candidates("AutoFactorEvaluation-RECONSTRUCT/dataaccess/a.py")[-1] == "data_access/a.py"
    path = tmp_path / "data_access"
    path.mkdir()
    file = path / "a.py"
    file.write_text("x = 2\n")
    inv = {"branches": [{"ref": "kept", "tip": "old",
        "changed": [{"path": "dataaccess/a.py", "status": "M", "blob": "different"}]}]}
    rows = reconcile_inventory(inv, tmp_path, {"kept"})[0]["changed"]
    assert rows[0]["classification"] == "NEEDS_SEMANTIC_REVIEW"
    assert rows[0]["current_path"] == "data_access/a.py"
    data = file.read_bytes()
    inv["branches"][0]["changed"][0]["blob"] = hashlib.sha1(
        b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    result = reconcile_inventory(inv, tmp_path, {"kept"})[0]
    assert result["changed"][0]["classification"] == "CURRENT_BYTES_IDENTICAL"
    assert result["semantic_closure"] == "NOT_VERIFIED"


def test_unmapped_runtime_is_preserved_for_review(tmp_path):
    inv = {"branches": [{"ref": "kept", "tip": "old",
        "changed": [{"path": "missing/runtime.py", "status": "A", "blob": "old"}]}]}
    result = reconcile_inventory(inv, tmp_path, {"kept"})
    assert result[0]["changed"][0]["classification"] == "NEEDS_SEMANTIC_REVIEW"
    assert reconcile_inventory(inv, tmp_path, set()) == []
