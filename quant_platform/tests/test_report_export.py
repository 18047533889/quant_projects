"""QRP-P10 — ReportExportArtifact tests.

Cover (task requirements):
(a) content_hash derived-only — tampering with ``content`` after export is
    detected by ``from_dict`` (stale/mismatched hash raises ValueError);
(b) round-trip compatibility — ``report_to_dict`` / ``from_dict`` round-trip,
    including legacy dicts *without* a ``content_hash`` key (hash recomputed);
(c) JSON-unserealisable content is rejected fail-closed at export;
(d) registry registration — ``ReportExporter.to_registry`` registers a
    REPORT_EXPORT ArtifactRef (plus unknown-type rejection);
(e) producer validation — only the three mandated producers are accepted.

Plus: refs requirement, exported_at awareness, health kind without refs,
serializability walks nested structures, explicit ``exported_at`` round-trip,
live adapter shape for ``from_backtest_stats``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_platform.app.contracts import ARTIFACT_TYPE_REPORT_EXPORT, ArtifactRef
from quant_platform.app.export import (
    REPORT_KINDS,
    REPORT_PRODUCERS,
    ReportExportArtifact,
    ReportExporter,
    ReportKind,
    ReportProducer,
    from_dict,
    report_to_dict,
    serializable_content,
)
from quant_platform.app.storage.registry import (
    ArtifactRegistry,
    UnknownArtifactTypeError,
)

EXPORTER = ReportExporter()


def _export(content=None, **kw) -> ReportExportArtifact:
    kind = kw.pop("kind", "backtest")
    return EXPORTER.export(
        refs={"backtest": "bt-1"},
        content=content or {"sharpe": 1.5, "tags": ["a", "b"]},
        kind=kind,
        **kw,
    )


def _snapshot() -> dict:
    art = _export()
    data = report_to_dict(art)
    return data


# ---- (a) content_hash derived-only ----------------------------------------


def test_content_hash_is_derived_derived_only():
    art = _export()
    assert len(art.content_hash) == 64
    assert set(art.content_hash) <= set("0123456789abcdef")
    # the snapshot carries the SAME artifact's derived hash
    data = report_to_dict(art)
    assert data["content_hash"] == art.content_hash
    # a caller-supplied content_hash can never reach the artifact
    # (the constructor has no such parameter — only artifact_type is constant)
    with pytest.raises(TypeError):
        ReportExportArtifact(
            report_kind="backtest",
            backtest_ref="x",
            content={},
            content_hash="x" * 64,
        )


def test_tampered_content_after_export_detected():
    data = _snapshot()
    data["content"] = {**data["content"], "sharpe": 99.0}  # content was tampered
    with pytest.raises(ValueError, match="stale content_hash"):
        from_dict(data)


def test_hash_changes_with_content():
    a = _export({"sharpe": 1.0})
    b = _export({"sharpe": 2.0})
    assert a.content_hash != b.content_hash


# ---- (b) round-trip compatibility -----------------------------------------


def test_round_trip_report_to_dict_from_dict():
    art = _export()
    data = report_to_dict(art)
    assert data["artifact_type"] == ARTIFACT_TYPE_REPORT_EXPORT  # "REPORT_EXPORT"
    assert data["content"] == dict(art.content)
    restored = from_dict(data)
    assert restored == art
    assert restored.content == dict(art.content)
    assert restored.content_hash == art.content_hash


def test_round_trip_explicit_exported_at_preserved():
    art = _export(exported_at="2026-08-28T02:00:00+00:00")
    restored = from_dict(report_to_dict(art))
    assert restored.exported_at.startswith("2026-08-28T02:00:00")
    assert restored == art


def test_legacy_dict_without_content_hash_recomputes():
    data = _snapshot()
    data.pop("content_hash")  # legacy export dicts have no hash key
    restored = from_dict(data)
    assert restored.content_hash  # recomputed, derived-only
    assert restored.content == data["content"]


# ---- (c) JSON-unserializable content rejected -------------------------------


def test_non_serializable_content_rejected_at_export():
    class _Weird:
        pass

    with pytest.raises(TypeError, match="JSON-serializable"):
        _export(content={"bad": _Weird()})


def test_nested_non_serializable_content_rejected():
    with pytest.raises(TypeError, match="JSON-serializable"):
        _export(content={"a": {"b": [object()]}})


def test_serializable_content_walks_nested_and_copies():
    original = {"nested": {"k": [1, 2.5, "x", True, None]}}
    out = serializable_content(original)
    assert out == original
    out["nested"]["k"].append("mutated")
    assert original["nested"]["k"] == [1, 2.5, "x", True, None]  # defensive copy


def test_non_mapping_content_rejected():
    with pytest.raises(TypeError, match="mapping"):
        _export(content=[1, 2, 3])


def test_nan_content_rejected_by_bytes_encoding():
    # JSON can't represent NaN in the *registry bytes* so reject it fail-closed.
    with pytest.raises(TypeError, match="finite float|JSON-serializable"):
        _export(content={"m": float("nan")})


# ---- (d) registry registration ---------------------------------------------


def test_to_registry_registers_report_export_artifact():
    reg = ArtifactRegistry()
    try:
        art = _export()
        ref = EXPORTER.to_registry(art, reg)
        assert isinstance(ref, ArtifactRef)
        assert ref.artifact_type == ARTIFACT_TYPE_REPORT_EXPORT
        assert ref.content_hash == art.content_hash
        assert ref.producer_type == "platform"
        assert ref.producer_source_ref == "bt-1"
        assert ref.media_type == "application/json"
        assert ref.size_bytes == len(__import__("json").dumps(
            dict(art.content), sort_keys=True
        ).encode("utf-8"))
        assert reg.contains(art.content_hash)
        resolved = reg.resolve(art.content_hash)
        assert resolved is not None
        assert resolved.artifact_type == ARTIFACT_TYPE_REPORT_EXPORT
    finally:
        reg.close()


def test_to_registry_idempotent_reresister():
    reg = ArtifactRegistry()
    try:
        art = _export()
        first = EXPORTER.to_registry(art, reg)
        second = EXPORTER.to_registry(art, reg)
        assert second == first
        assert len(reg.list_versions(first.artifact_id)) == 1
        art2 = _export(content={"sharpe": 2.0})
        third = EXPORTER.to_registry(art2, reg)
        assert third.artifact_id == first.artifact_id
        assert len(reg.list_versions(first.artifact_id)) == 2
    finally:
        reg.close()


def test_to_registry_health_kind_without_refs():
    reg = ArtifactRegistry()
    try:
        art = EXPORTER.export(
            kind=ReportKind.HEALTH.value,
            content={"healthy": True},
        )
        ref = EXPORTER.to_registry(art, reg)
        assert ref.artifact_type == ARTIFACT_TYPE_REPORT_EXPORT
        assert len(reg.list_versions(ref.artifact_id)) == 1
    finally:
        reg.close()


def test_registry_rejects_wrong_artifact_type():
    reg = ArtifactRegistry()
    try:
        bad = object.__new__(ArtifactRef)
        for name, fld in ArtifactRef.__dataclass_fields__.items():
            object.__setattr__(bad, name, fld.default)
        for name, val in {
            "artifact_id": "x",
            "artifact_type": "NOT_A_REAL_TYPE",
            "schema_version": "1.0",
            "content_hash": "a" * 64,
            "storage_uri": "cos://x/y",
            "size_bytes": 1,
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "producer_type": "test",
            "producer_version": "0.1",
        }.items():
            object.__setattr__(bad, name, val)
        with pytest.raises(UnknownArtifactTypeError):
            reg.register(bad)
        assert reg.contains("a" * 64) is False
    finally:
        reg.close()


# ---- (e) producer validation ------------------------------------------------


@pytest.mark.parametrize(
    "producer",
    [p.value for p in ReportProducer],
)
def test_allowed_producers_ok(producer):
    art = _export(producer=producer)
    assert art.producer == producer


def test_unknown_producer_rejected():
    with pytest.raises(ValueError, match="unknown producer"):
        _export(producer="excel_producer")


def test_exported_producers_constants_match():
    assert REPORT_PRODUCERS == {
        "vectorbt_qs",
        "research_platform",
        "platform",
    }
    assert set(REPORT_KINDS) == {"backtest", "campaign", "health"}


# ---- refs requirement + exported_at awareness -------------------------------


def test_backtest_kind_requires_at_least_one_ref():
    with pytest.raises(ValueError, match="requires a backtest_ref"):
        EXPORTER.export(content={"x": 1}, kind="backtest")


def test_health_kind_does_not_require_refs():
    art = EXPORTER.export(kind="health", content={"ok": True})
    assert art.report_kind == "health"
    assert art.backtest_ref is None
    assert art.campaign_ref is None


def test_campaign_kind_accepts_campaign_ref():
    art = EXPORTER.export(
        refs={"campaign": "camp-21"}, content={}, kind="campaign",
        producer=ReportProducer.RESEARCH_PLATFORM.value,
    )
    assert art.campaign_ref == "camp-21"
    assert art.report_kind == "campaign"


def test_unknown_kind_rejected():
    with pytest.raises(ValueError, match="unknown report_kind"):
        _export(kind="accounting")


def test_naive_exported_at_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        _export(exported_at="2026-08-28T02:00:00")


def test_default_exported_at_is_aware():
    art = _export()
    dt = datetime.fromisoformat(art.exported_at)
    assert dt.tzinfo is not None
    assert dt.utcoffset() is not None


# ---- from_backtest_stats adapter --------------------------------------------


def test_from_backtest_stats_builds_backtest_artifact():
    stats = {"Sharpe Ratio": 1.7, "hits": 8}
    art = EXPORTER.from_backtest_stats(stats, backtest_ref="bt-42")
    assert art.report_kind == "backtest"
    assert art.backtest_ref == "bt-42"
    assert art.content["Sharpe Ratio"] == 1.7
    assert art.producer == "platform"
    restored = from_dict(report_to_dict(art))
    assert restored == art


def test_from_backtest_stats_requires_ref():
    with pytest.raises(ValueError, match="requires at least backtest_ref"):
        EXPORTER.from_backtest_stats({"x": 1})