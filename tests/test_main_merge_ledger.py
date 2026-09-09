from scripts.ledger_provenance import observe, reconcile


def report(history, status, spec="spec", tree="tree", manual=False, **kwargs):
    observe(history, "M01", dict(status=status, source_tree=tree, **kwargs),
        source="input.json", source_bytes=status.encode(), spec_hash=spec, manual=manual)


def test_old_pass_new_fail_preserves_both():
    history = {}
    report(history, "PASS")
    report(history, "FAIL")
    result = reconcile(history["M01"], current_tree="tree")
    assert result["status"] == "NEEDS_RECONCILIATION"
    assert len(result["source_records"]) == 2


def test_same_id_different_spec_or_tree_is_not_overwritten():
    history = {}
    report(history, "PASS")
    report(history, "PASS", spec="other")
    report(history, "PASS", tree="other")
    assert reconcile(history["M01"], current_tree="tree")["status"] == "NEEDS_RECONCILIATION"
    assert len(history["M01"]) == 3


def test_test_name_is_not_execution_and_manual_override_is_not_verified():
    history = {}
    report(history, "FIXED_LOCAL", manual=True, tests=["existing_test.py"])
    result = reconcile(history["M01"], current_tree="tree")
    assert result["status"] == "MANUAL_ASSERTION"
    assert result["current_execution_status"] == "NOT_RUN"


def test_implementation_change_invalidates_old_claim():
    history = {}
    report(history, "PASS", implementation_hashes={"metric.py": "old"})
    result = reconcile(history["M01"], current_tree="tree", implementation_hashes={"metric.py": "new"})
    assert result["status"] == "NEEDS_RECONCILIATION"
