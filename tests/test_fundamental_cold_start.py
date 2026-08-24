# -*- coding: utf-8
"""Cold-start source contract tests; no alternate factor library fallback."""
from factor_engine.fundamental_cold_start import EXPECTED_FILENAME, load_cold_start


def test_missing_cold_start_is_explicitly_unavailable(tmp_path):
    rows, report = load_cold_start(tmp_path / EXPECTED_FILENAME)
    assert rows == []
    assert report.status == "unavailable"
    assert report.error == "missing_cold_start_source" or report.error.startswith("invalid_")


def test_wrong_filename_is_rejected(tmp_path):
    wrong = tmp_path / "fundamental_factors_6208.csv"
    wrong.write_text("factor_id,dsl\nX,add(a,b)\n", encoding="utf-8")
    try:
        load_cold_start(wrong)
    except ValueError as exc:
        assert EXPECTED_FILENAME in str(exc)
    else:
        raise AssertionError("alternate factor library must not be accepted")
