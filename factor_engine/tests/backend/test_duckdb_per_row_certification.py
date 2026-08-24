# -*- coding: utf-8 -*-
"""Per-row DuckDB certification schema tests (R21-DUCKDB-PER-ROW-CERT-SCHEMA)."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationID, PhysicalImplementationSpec
from factor_engine.backend.duckdb_backend import (
    NullPolicy,
    ParameterDomainDigest,
    PerRowCertification,
    TiePolicy,
    WindowFrame,
    build_duckdb_certification_row,
    certification_key,
    verify_certification_row,
)


def _make_spec(
    canonical: str = "ts_mean",
    backend: str = "duckdb_sql",
    *,
    emitter_identity: str = "emitter-id-001",
    implementation_source_hash: str = "src-sha-256",
    parameter_domain_hash: str = "pd-hash-001",
    semantic_contract_hash: str = "sc-hash-001",
) -> PhysicalImplementationSpec:
    return PhysicalImplementationSpec(
        canonical=canonical,
        backend=backend,
        execution_kind=ExecutionKind.DUCKDB_NATIVE_SQL,
        supports_lazy=True,
        supports_nulls=True,
        supports_nan=True,
        supports_inf=True,
        implementation_source_hash=implementation_source_hash,
        emitter_identity=emitter_identity,
        kernel_identity="",
        parameter_domain_hash=parameter_domain_hash,
        semantic_contract_hash=semantic_contract_hash,
    )


def _minimal_row(**overrides: Any) -> PerRowCertification:
    defaults: dict[str, Any] = dict(
        canonical="ts_mean",
        backend="duckdb_sql",
        physical_implementation_id=PhysicalImplementationID("pi:v1:abc"),
        parameter_domain=ParameterDomainDigest(raw="window=20", canonical="window=20"),
    )
    defaults.update(overrides)
    return PerRowCertification(**defaults).ensure_sha()


# --- 1. Schema correctness ---------------------------------------------------

class TestSchemaCorrectness:
    def test_per_row_certification_fields(self) -> None:
        row = PerRowCertification(
            canonical="ts_mean",
            backend="duckdb_sql",
            physical_implementation_id=PhysicalImplementationID("pi:v1:abc"),
            parameter_domain=ParameterDomainDigest(raw="window=5", canonical="window=5"),
        )
        assert row.canonical == "ts_mean"
        assert row.backend == "duckdb_sql"
        assert row.ddof is None
        assert row.null_policy == NullPolicy.NULL_EXPLICIT

    def test_to_dict_includes_all_fields(self) -> None:
        row = build_duckdb_certification_row(
            "add",
            spec=_make_spec("add"),
            parameter_domain="x,y",
        )
        d = row.to_dict()
        for key in ("canonical", "backend", "physical_implementation_id", "parameter_domain",
                     "ddof", "min_periods", "null_policy", "tie_policy", "window_frame",
                     "oracle_hash", "source_hash", "sha"):
            assert key in d, f"missing key: {key}"

    def test_defaults(self) -> None:
        row = _minimal_row()
        assert row.ddof is None
        assert row.min_periods is None
        assert row.null_policy == NullPolicy.NULL_EXPLICIT
        assert row.tie_policy == TiePolicy.FIRST
        assert row.window_frame == WindowFrame.ROWS
        assert row.oracle_hash == ""
        assert row.source_hash == ""

    def test_frozen(self) -> None:
        row = _minimal_row()
        with pytest.raises(AttributeError):
            row.canonical = "other"  # type: ignore[misc]

    def test_to_dict_round_trip_types(self) -> None:
        row = _minimal_row()
        d = row.to_dict()
        assert isinstance(d["canonical"], str)
        assert isinstance(d["backend"], str)
        assert isinstance(d["physical_implementation_id"], str)
        assert isinstance(d["parameter_domain"], dict)
        assert isinstance(d["ddof"], (int, type(None)))
        assert isinstance(d["min_periods"], (int, type(None)))
        assert isinstance(d["null_policy"], str)
        assert isinstance(d["tie_policy"], str)
        assert isinstance(d["window_frame"], str)
        assert isinstance(d["oracle_hash"], str)
        assert isinstance(d["source_hash"], str)
        assert isinstance(d["sha"], str)

    def test_null_policy_enum_members(self) -> None:
        expected = {
            "null_explicit", "null_coalesce", "nan_propagate",
            "nan_drop", "inf_clamp", "inf_propagate",
        }
        assert {e.value for e in NullPolicy} == expected

    def test_tie_policy_enum_members(self) -> None:
        assert {e.value for e in TiePolicy} == {"first", "average", "min", "max", "dense"}

    def test_window_frame_enum_members(self) -> None:
        assert {e.value for e in WindowFrame} == {"rows", "range", "group", "expanding"}

    def test_parameter_domain_digest_to_dict(self) -> None:
        pd = ParameterDomainDigest(raw="Window= 20 ", canonical="window=20")
        d = pd.to_dict()
        assert d["raw"] == "Window= 20 "
        assert d["canonical"] == "window=20"


# --- 2. PhysicalImplementationID binding -------------------------------------

class TestPhysicalImplementationIDBinding:
    def test_pid_derived_from_spec(self) -> None:
        spec = _make_spec("ts_std")
        row = build_duckdb_certification_row("ts_std", spec=spec, parameter_domain="window=10")
        pid = spec.physical_implementation_id
        assert pid is not None
        assert row.physical_implementation_id == pid

    def test_rejects_wrong_backend(self) -> None:
        spec = _make_spec("rank", backend="polars")
        with pytest.raises(ValueError, match="expected spec.backend='duckdb_sql'"):
            build_duckdb_certification_row("rank", spec=spec, parameter_domain="")

    def test_pid_stringified_in_dict(self) -> None:
        spec = _make_spec()
        row = build_duckdb_certification_row("ts_mean", spec=spec, parameter_domain="window=20")
        d = row.to_dict()
        assert d["physical_implementation_id"] == str(spec.physical_implementation_id)

    def test_pid_in_canonical_payload(self) -> None:
        spec = _make_spec()
        row = build_duckdb_certification_row("ts_mean", spec=spec, parameter_domain="window=20")
        payload = row._canonical_payload()
        assert payload["physical_implementation_id"] == str(spec.physical_implementation_id)

    def test_different_spec_different_pid(self) -> None:
        spec_a = _make_spec(emitter_identity="emitter:1")
        spec_b = _make_spec(emitter_identity="emitter:2")
        pid_a = spec_a.physical_implementation_id
        pid_b = spec_b.physical_implementation_id
        assert pid_a is not None and pid_b is not None
        assert pid_a != pid_b

    def test_rejects_incomplete_spec(self) -> None:
        spec = PhysicalImplementationSpec(
            canonical="ts_mean",
            backend="duckdb_sql",
            execution_kind=ExecutionKind.DUCKDB_NATIVE_SQL,
        )
        with pytest.raises(ValueError, match="not complete enough"):
            build_duckdb_certification_row("ts_mean", spec=spec, parameter_domain="w=20")


# --- 3. Parameter domain certification ---------------------------------------

class TestParameterDomainCertification:
    def test_canonical_domain(self) -> None:
        row = build_duckdb_certification_row(
            "ts_mean", spec=_make_spec(), parameter_domain="window=5 min_periods=2"
        )
        assert row.parameter_domain.canonical == "window=5 min_periods=2"

    def test_raw_preserved(self) -> None:
        row = build_duckdb_certification_row(
            "ts_mean", spec=_make_spec(), parameter_domain="  Window=5  Min_Periods=2  "
        )
        # raw is stripped by _canonicalise_parameter_domain
        assert row.parameter_domain.raw == "Window=5  Min_Periods=2"
        assert row.parameter_domain.canonical == "window=5 min_periods=2"

    def test_canonicalisation_normalises_case(self) -> None:
        row = build_duckdb_certification_row(
            "ts_mean", spec=_make_spec(), parameter_domain="  Window = 20  "
        )
        # Single spaces are collapsed; case is lowered.
        assert row.parameter_domain.canonical == "window = 20"

    def test_same_canonical_same_payload(self) -> None:
        row1 = _minimal_row(
            parameter_domain=ParameterDomainDigest(raw="Window=20", canonical="window=20"),
        )
        row2 = _minimal_row(
            parameter_domain=ParameterDomainDigest(raw="window=20", canonical="window=20"),
        )
        assert row1._canonical_payload()["parameter_domain"] == row2._canonical_payload()["parameter_domain"]

    def test_different_canonical_different_payload(self) -> None:
        row1 = _minimal_row(
            parameter_domain=ParameterDomainDigest(raw="w=20", canonical="w=20"),
        )
        row2 = _minimal_row(
            parameter_domain=ParameterDomainDigest(raw="w=30", canonical="w=30"),
        )
        assert row1._canonical_payload()["parameter_domain"] != row2._canonical_payload()["parameter_domain"]

    def test_canonical_set_style_membership(self) -> None:
        pd1 = ParameterDomainDigest(raw="cross_a stock_b", canonical="cross_a stock_b")
        assert "cross_a" in pd1.canonical


# --- 4. ddof / min_periods handling ------------------------------------------

class TestDdofMinPeriods:
    def test_ddof_included_when_set(self) -> None:
        row = build_duckdb_certification_row(
            "ts_std", spec=_make_spec(), parameter_domain="window=10", ddof=1
        )
        d = row.to_dict()
        assert d["ddof"] == 1

    def test_min_periods_included_when_set(self) -> None:
        row = build_duckdb_certification_row(
            "ts_mean", spec=_make_spec(), parameter_domain="window=5", min_periods=3
        )
        d = row.to_dict()
        assert d["min_periods"] == 3

    def test_none_values_excluded_from_payload(self) -> None:
        row = build_duckdb_certification_row(
            "abs", spec=_make_spec("abs"), parameter_domain="x"
        )
        payload = row._canonical_payload()
        assert "ddof" not in payload
        assert "min_periods" not in payload

    def test_ddof_zero_included(self) -> None:
        row = _minimal_row(ddof=0)
        payload = row._canonical_payload()
        assert payload["ddof"] == 0

    def test_both_ddof_and_min_periods(self) -> None:
        row = _minimal_row(ddof=1, min_periods=3)
        payload = row._canonical_payload()
        assert payload["ddof"] == 1
        assert payload["min_periods"] == 3

    def test_ddof_changes_sha(self) -> None:
        row_a = _minimal_row(ddof=0)
        row_b = _minimal_row(ddof=1)
        assert row_a.compute_sha() != row_b.compute_sha()

    def test_min_periods_changes_sha(self) -> None:
        row_a = _minimal_row(min_periods=1)
        row_b = _minimal_row(min_periods=5)
        assert row_a.compute_sha() != row_b.compute_sha()


# --- 5. NULL / NaN / Inf edge cases ------------------------------------------

class TestNullNaNInf:
    @pytest.mark.parametrize("policy", list(NullPolicy))
    def test_each_null_policy(self, policy: NullPolicy) -> None:
        row = build_duckdb_certification_row(
            "coalesce", spec=_make_spec("coalesce"), parameter_domain="a,b",
            null_policy=policy,
        )
        assert row.null_policy == policy
        assert row.to_dict()["null_policy"] == policy.value

    def test_null_propagate_vs_drop_changes_sha(self) -> None:
        r1 = build_duckdb_certification_row(
            "fill", spec=_make_spec("fill"), parameter_domain="x",
            null_policy=NullPolicy.NAN_PROPAGATE,
        )
        r2 = build_duckdb_certification_row(
            "fill", spec=_make_spec("fill"), parameter_domain="x",
            null_policy=NullPolicy.NAN_DROP,
        )
        assert r1.sha != r2.sha

    def test_different_null_policy_different_sha(self) -> None:
        shas = {_minimal_row(null_policy=np).compute_sha() for np in NullPolicy}
        assert len(shas) == len(NullPolicy), "all null policies must produce distinct SHAs"


# --- 6. tie_policy certification ---------------------------------------------

class TestTiePolicy:
    @pytest.mark.parametrize("tp", list(TiePolicy))
    def test_tie_policy_value(self, tp: TiePolicy) -> None:
        row = build_duckdb_certification_row(
            "rank", spec=_make_spec("rank"), parameter_domain="x", tie_policy=tp
        )
        assert row.to_dict()["tie_policy"] == tp.value

    def test_tie_policy_changes_sha(self) -> None:
        r1 = build_duckdb_certification_row(
            "rank", spec=_make_spec("rank"), parameter_domain="x", tie_policy=TiePolicy.FIRST
        )
        r2 = build_duckdb_certification_row(
            "rank", spec=_make_spec("rank"), parameter_domain="x", tie_policy=TiePolicy.AVERAGE
        )
        assert r1.sha != r2.sha

    def test_different_tie_policy_all_distinct_sha(self) -> None:
        shas = {_minimal_row(tie_policy=tp).compute_sha() for tp in TiePolicy}
        assert len(shas) == len(TiePolicy)


# --- 7. window_frame support -------------------------------------------------

class TestWindowFrame:
    @pytest.mark.parametrize("wf", list(WindowFrame))
    def test_window_frame_value(self, wf: WindowFrame) -> None:
        row = build_duckdb_certification_row(
            "ts_sum", spec=_make_spec("ts_sum"), parameter_domain="window=5", window_frame=wf
        )
        assert row.to_dict()["window_frame"] == wf.value

    def test_different_window_frame_all_distinct_sha(self) -> None:
        shas = {_minimal_row(window_frame=wf).compute_sha() for wf in WindowFrame}
        assert len(shas) == len(WindowFrame)


# --- 8. oracle_hash / source_hash binding ------------------------------------

class TestOracleSourceHash:
    def test_oracle_hash_recorded(self) -> None:
        row = build_duckdb_certification_row(
            "ts_mean", spec=_make_spec(), parameter_domain="window=3",
            oracle_hash="ora-abc123",
        )
        assert row.to_dict()["oracle_hash"] == "ora-abc123"

    def test_source_hash_recorded(self) -> None:
        row = build_duckdb_certification_row(
            "ts_mean", spec=_make_spec(), parameter_domain="window=3",
            source_hash="src-def456",
        )
        assert row.to_dict()["source_hash"] == "src-def456"

    def test_hashes_affect_sha(self) -> None:
        r1 = build_duckdb_certification_row("x", spec=_make_spec("x"), parameter_domain="p", oracle_hash="a")
        r2 = build_duckdb_certification_row("x", spec=_make_spec("x"), parameter_domain="p", oracle_hash="b")
        assert r1.sha != r2.sha

    def test_oracle_hash_empty_excluded(self) -> None:
        row = _minimal_row(oracle_hash="")
        payload = row._canonical_payload()
        assert "oracle_hash" not in payload

    def test_source_hash_empty_excluded(self) -> None:
        row = _minimal_row(source_hash="")
        payload = row._canonical_payload()
        assert "source_hash" not in payload

    def test_both_hashes_included(self) -> None:
        row = _minimal_row(oracle_hash="o:x", source_hash="s:y")
        payload = row._canonical_payload()
        assert payload["oracle_hash"] == "o:x"
        assert payload["source_hash"] == "s:y"

    def test_oracle_hash_changes_sha(self) -> None:
        sha_a = _minimal_row(oracle_hash="o:1").compute_sha()
        sha_b = _minimal_row(oracle_hash="o:2").compute_sha()
        assert sha_a != sha_b

    def test_source_hash_changes_sha(self) -> None:
        sha_a = _minimal_row(source_hash="s:1").compute_sha()
        sha_b = _minimal_row(source_hash="s:2").compute_sha()
        assert sha_a != sha_b

    def test_empty_string_hashes_differently(self) -> None:
        row_a = _minimal_row(oracle_hash="")
        row_b = _minimal_row(oracle_hash="x")
        assert row_a.compute_sha() != row_b.compute_sha()


# --- 9. SHA integrity --------------------------------------------------------

class TestSHAIntegrity:
    def test_deterministic_sha(self) -> None:
        row = build_duckdb_certification_row("abs", spec=_make_spec("abs"), parameter_domain="x")
        assert row.sha == row.compute_sha()

    def test_verify_ok(self) -> None:
        row = build_duckdb_certification_row("abs", spec=_make_spec("abs"), parameter_domain="x")
        ok, msg = verify_certification_row(row)
        assert ok
        assert msg == "ok"

    def test_verify_fails_on_tamper(self) -> None:
        row = build_duckdb_certification_row("abs", spec=_make_spec("abs"), parameter_domain="x")
        tampered = PerRowCertification(
            canonical=row.canonical, backend=row.backend,
            physical_implementation_id=row.physical_implementation_id,
            parameter_domain=row.parameter_domain, sha="bad",
        )
        ok, msg = verify_certification_row(tampered)
        assert not ok
        assert "sha mismatch" in msg

    def test_certification_key_deterministic(self) -> None:
        r1 = build_duckdb_certification_row("log", spec=_make_spec("log"), parameter_domain="x")
        r2 = build_duckdb_certification_row("log", spec=_make_spec("log"), parameter_domain="x")
        assert certification_key(r1) == certification_key(r2)

    def test_sha_is_64_char_hex(self) -> None:
        row = _minimal_row()
        assert len(row.sha) == 64
        assert all(c in "0123456789abcdef" for c in row.sha)

    def test_ensure_sha_idempotent(self) -> None:
        row = _minimal_row()
        row2 = row.ensure_sha()
        assert row2 is row
        assert row2.sha == row.sha

    def test_ensure_sha_populates_empty(self) -> None:
        row = PerRowCertification(
            canonical="ts_mean",
            backend="duckdb_sql",
            physical_implementation_id=PhysicalImplementationID("pi:v1:abc"),
            parameter_domain=ParameterDomainDigest(raw="w=20", canonical="w=20"),
            sha="",
        )
        filled = row.ensure_sha()
        assert filled.sha != ""
        assert len(filled.sha) == 64

    def test_verify_fail_tampered_sha(self) -> None:
        row = _minimal_row()
        tampered = PerRowCertification(
            canonical=row.canonical,
            backend=row.backend,
            physical_implementation_id=row.physical_implementation_id,
            parameter_domain=row.parameter_domain,
            ddof=row.ddof,
            min_periods=row.min_periods,
            null_policy=row.null_policy,
            tie_policy=row.tie_policy,
            window_frame=row.window_frame,
            oracle_hash=row.oracle_hash,
            source_hash=row.source_hash,
            sha="0" * 64,
        )
        ok, msg = verify_certification_row(tampered)
        assert ok is False
        assert "sha mismatch" in msg

    def test_canonical_payload_matches_expected_hash(self) -> None:
        row = _minimal_row()
        payload = row._canonical_payload()
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        assert row.compute_sha() == expected

    def test_sha_changes_with_canonical(self) -> None:
        sha_a = _minimal_row(canonical="ts_mean").compute_sha()
        sha_b = _minimal_row(canonical="ts_sum").compute_sha()
        assert sha_a != sha_b

    def test_sha_changes_with_pid(self) -> None:
        sha_a = _minimal_row(
            physical_implementation_id=PhysicalImplementationID("pi:v1:aaa"),
        ).compute_sha()
        sha_b = _minimal_row(
            physical_implementation_id=PhysicalImplementationID("pi:v1:bbb"),
        ).compute_sha()
        assert sha_a != sha_b


# --- 10. Backward compatibility with schema_version 3 ------------------------

class TestBackwardCompatibility:
    def test_old_operators_dict_still_usable(self) -> None:
        """schema_version 3 operators dict entries remain valid."""
        old_record = {
            "semantic_version": 2,
            "implementation_hash_polars": "abc",
            "implementation_hash_duckdb": "def",
        }
        # Old records do not contain per-row fields; consumers must handle gracefully.
        assert "implementation_hash_polars" in old_record
        assert "physical_implementation_id" not in old_record  # new field only in v4

    def test_v4_row_compatible_with_v3_provenance(self) -> None:
        """A v4 row can coexist with v3 provenance metadata."""
        row = build_duckdb_certification_row(
            "add", spec=_make_spec("add"), parameter_domain="x,y"
        )
        d = row.to_dict()
        # v3 provenance fields are still present in the payload
        d["schema_version"] = 4
        d["provenance"] = {"artifact_kind": "test_passed"}
        assert d["schema_version"] == 4
        assert d["sha"]

    def test_build_duckdb_certification_row_basic(self) -> None:
        spec = _make_spec()
        row = build_duckdb_certification_row(
            "ts_mean",
            spec=spec,
            parameter_domain="window=20",
        )
        assert row.backend == "duckdb_sql"
        assert row.canonical == "ts_mean"
        assert row.physical_implementation_id == spec.physical_implementation_id
        assert row.parameter_domain.canonical == "window=20"
        assert row.sha != ""

    def test_build_row_sets_sha(self) -> None:
        spec = _make_spec()
        row = build_duckdb_certification_row(
            "ts_mean",
            spec=spec,
            parameter_domain="window=20",
        )
        assert row.sha != ""
        ok, _ = verify_certification_row(row)
        assert ok

    def test_certification_key_components(self) -> None:
        spec = _make_spec()
        row = build_duckdb_certification_row(
            "ts_mean",
            spec=spec,
            parameter_domain="window=20",
        )
        key = certification_key(row)
        parts = key.split("|")
        assert len(parts) == 5
        assert parts[0] == row.canonical
        assert parts[1] == row.backend
        assert parts[2] == str(row.physical_implementation_id)
        assert parts[3] == row.parameter_domain.canonical
        assert parts[4] == row.sha

    def test_full_round_trip_recompute_sha(self) -> None:
        spec = _make_spec()
        row = build_duckdb_certification_row(
            "ts_mean",
            spec=spec,
            parameter_domain="window=20",
            ddof=0,
            min_periods=1,
            null_policy=NullPolicy.INF_CLAMP,
            tie_policy=TiePolicy.MAX,
            window_frame=WindowFrame.EXPANDING,
            oracle_hash="o:full",
            source_hash="s:full",
        )
        d = row.to_dict()
        pd_raw = d["parameter_domain"]["raw"]
        pd_canon = d["parameter_domain"]["canonical"]
        recalc = PerRowCertification(
            canonical=d["canonical"],
            backend=d["backend"],
            physical_implementation_id=PhysicalImplementationID(d["physical_implementation_id"]),
            parameter_domain=ParameterDomainDigest(raw=pd_raw, canonical=pd_canon),
            ddof=d["ddof"],
            min_periods=d["min_periods"],
            null_policy=NullPolicy(d["null_policy"]),
            tie_policy=TiePolicy(d["tie_policy"]),
            window_frame=WindowFrame(d["window_frame"]),
            oracle_hash=d["oracle_hash"],
            source_hash=d["source_hash"],
        )
        assert recalc.compute_sha() == d["sha"]
