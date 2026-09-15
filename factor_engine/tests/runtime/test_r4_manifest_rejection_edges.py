"""Rejected input edges must not abort independent, otherwise valid items."""
import hashlib
import time

import pytest

from factor_engine.runtime.finite_manifest import FiniteFactorManifest
from factor_engine.runtime.resume_validation import _manifest_fingerprint, ResumeIdentityError
from factor_engine.runtime.default_engine import iter_parsed_factor_definitions
from factor_engine.runtime.default_execution_policy import resolve_default_policy
from factor_engine.tests.runtime.test_v6_bounded_pipeline import FakeEngine, FakeFactor
from factor_engine.runtime.bounded_pipeline import execute_run_many_durable


@pytest.mark.parametrize("duplicate", [False, True])
def test_capped_pickle_rejection_has_a_valid_manifest_identity(tmp_path, duplicate):
    limit = 128
    huge = FakeFactor("huge")
    huge.padding = b"x" * 1024
    factors = [huge, FakeFactor("huge" if duplicate else "ok")]
    with FiniteFactorManifest.ingest(
        factors, tmp_path / "manifest.sqlite3", max_definition_bytes=limit,
    ) as manifest:
        rows = list(manifest.records())
        assert rows[0].definition_bytes == limit + 1
        assert rows[0].definition_digest == ""
        assert not rows[0].valid
        count, digest = _manifest_fingerprint(
            manifest._db, max_definition_bytes=limit, max_factors=10,
            deadline=time.monotonic() + 10,
        )
        assert count == 2 and len(digest) == 64


def test_oversized_pickle_rejection_does_not_abort_good_peer(tmp_path):
    policy = resolve_default_policy()
    huge = FakeFactor("huge")
    huge.padding = b"x" * (policy.max_definition_bytes + 1)
    receipt = execute_run_many_durable(
        FakeEngine(), [huge, FakeFactor("ok")],
        policy=policy, artifact_root=tmp_path,
    )
    assert receipt["status"] == "COMPLETED_WITH_ERRORS"
    assert receipt["counts"] == {"REJECTED": 1, "SUCCEEDED": 1}
    outcomes = {item["name"]: item for item in receipt["outcomes"]}
    assert outcomes["huge"]["error_code"] == "DEFINITION_TOO_LARGE"
    assert outcomes["huge"].get("artifact") is None


@pytest.mark.parametrize("field,value", [
    ("definition_bytes", 130), ("valid", 1),
    ("error_code", "INVALID_FACTOR_DEFINITION"),
])
def test_payloadless_oversize_metadata_cannot_claim_other_shapes(tmp_path, field, value):
    huge = FakeFactor("huge")
    huge.padding = b"x" * 1024
    with FiniteFactorManifest.ingest(
        [huge], tmp_path / "manifest.sqlite3", max_definition_bytes=128,
    ) as manifest:
        manifest._db.execute(f"UPDATE factors SET {field}=?", (value,))
        with pytest.raises(ResumeIdentityError):
            _manifest_fingerprint(
                manifest._db, max_definition_bytes=128, max_factors=10,
                deadline=time.monotonic() + 10,
            )


@pytest.mark.parametrize("formula", ["close + '\\ud800'".replace("\\ud800", chr(0xd800)),
                                     chr(0xdfff), "close + " + repr(chr(0xd800)),
                                     "close + " + "9" * 400, "None + None",
                                     "close + (True / False)"])
def test_invalid_unicode_formula_is_rejected_without_losing_peer(formula):
    from factor_engine.runtime.finite_manifest import RejectedFactorDefinition

    items = list(iter_parsed_factor_definitions([
        {"name": "bad", "formula": formula}, {"name": "ok", "formula": "close + open"},
    ]))
    assert isinstance(items[0], RejectedFactorDefinition)
    assert items[0].error_code == "INVALID_DSL"
    raw = formula.encode("utf-8", errors="surrogatepass")
    assert items[0].definition_digest == hashlib.sha256(raw).hexdigest()
    assert items[0].definition_bytes == len(raw)
    assert items[1].name == "ok"


def test_parser_uses_structured_error_for_invalid_unicode():
    from factor_engine.api.dsl_parser import parse_factor, DSLParseError
    with pytest.raises(DSLParseError):
        parse_factor(chr(0xd800), name="bad", surface="all")

def test_integer_literal_over_float_range_has_a_typed_magnitude_rejection():
    from factor_engine.api.dsl_parser import parse_factor, DSLParseError
    with pytest.raises(DSLParseError, match="max_literal_magnitude"):
        parse_factor("close + " + "9" * 400, name="bad", surface="all")
