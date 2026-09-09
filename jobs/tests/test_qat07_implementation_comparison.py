from jobs.qa.run_qat07_import_matrix import compare_package_sources, parse_probe_output, reconcile_collected_tests


def test_collection_receipt_reconciles_parameterized_cases_to_definitions():
    source = "def test_bridge(): pass\nclass TestAPI:\n    def test_mask(self): pass\n"
    receipt = reconcile_collected_tests(source, [
        "test_file.py::test_bridge[int]", "test_file.py::test_bridge[str]",
        "test_file.py::TestAPI::test_mask",
    ])
    assert receipt["status"] == "PASS"
    assert receipt["definition_count"] == 2
    assert receipt["collected_case_count"] == 3


def test_collection_receipt_rejects_overwritten_missing_and_unexpected_tests():
    duplicated = "def test_bridge(): pass\ndef test_bridge(): pass\n"
    assert reconcile_collected_tests(duplicated, ["f.py::test_bridge"])["status"] == "BLOCKED"
    source = "def test_bridge(): pass\ndef test_missing(): pass\n"
    receipt = reconcile_collected_tests(source, ["f.py::test_bridge", "f.py::test_extra"])
    assert receipt["status"] == "BLOCKED"
    assert receipt["missing"] == ["test_missing"]
    assert receipt["unexpected"] == ["test_extra"]
    assert reconcile_collected_tests("", [])["status"] == "BLOCKED"


def test_source_difference_is_not_certified_by_successful_import(tmp_path):
    source, wheel = tmp_path / 'source', tmp_path / 'wheel'
    source.mkdir()
    wheel.mkdir()
    (source / '__init__.py').write_text('result = 1\n')
    (wheel / '__init__.py').write_text('result = 2\n')
    receipt = compare_package_sources(wheel, source)
    assert receipt['status'] == 'BLOCKED'
    assert receipt['differences'][0]['reason'] == 'different_implementation'
    (wheel / '__init__.py').write_text('result = 1\n')
    assert compare_package_sources(wheel, source)['status'] == 'PASS'


def test_no_sources_or_missing_source_cannot_pass(tmp_path):
    assert compare_package_sources(tmp_path / 'absent', tmp_path)['status'] == 'BLOCKED'
    wheel = tmp_path / 'wheel'
    wheel.mkdir()
    (wheel / 'extra.py').write_text('result = 1\n')
    assert compare_package_sources(wheel, tmp_path / 'absent')['differences'][0]['reason'] == 'missing_checkout_source'


def test_probe_parser_requires_structured_final_receipt():
    assert parse_probe_output('warning\n{"lag_guard_observed": false}') == {'lag_guard_observed': False}
    assert parse_probe_output('"real_fe_loaded": true\nnot-json') == {}
