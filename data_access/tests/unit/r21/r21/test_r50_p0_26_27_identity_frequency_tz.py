"""R50 P0-26/27: DataReadIdentity multi-field strict + timezone round-trip.

Regression tests for two correctness bugs fixed in R50:

P0-26 (multi-field strict):
    ``ResolvedFieldIdentity._tuple()``/``content_canonical`` previously dropped
    ``frequency``. Two DataReadIdentity instances whose fields differ ONLY in
    frequency (daily vs minute) compared equal and produced identical digests —
    a minute-bar read collapsed to the daily-bar identity (wrong cache reuse /
    wrong semantic resolution). Now ``frequency`` is a frozen dataclass field,
    is compiled from SemanticField.frequency in ``_field_identity``, and is
    part of ``content_canonical``.

P0-27 (timezone round-trip):
    ``DataReadIdentity.to_dict()`` -> ``from_dict()`` serialized
    ``provenance_notes`` as a list; ``from_dict`` re-fed the list into the
    frozen dataclass unchanged, so the round-tripped instance compared
    ``() != []`` and lost equality. ``__post_init__`` now tuple-izes
    ``provenance_notes`` like every other sequence field, so
    ``from_dict(to_dict(x)) == x`` holds and the digest is stable across the
    parquet/JSON string round-trip.
"""
from __future__ import annotations

from datetime import datetime, timezone

from data_access.read.data_read_identity import DataReadIdentity
from data_access.read.semantic_catalog import SemanticField


def _field(
    name: str,
    *,
    frequency: str | None = "daily",
    grain: str | None = "instrument",
    source_unit: str | None = "yuan",
    canonical_unit: str | None = "yuan",
    availability: str = "same_day",
) -> SemanticField:
    """SemanticField placeholder with the R50-relevant semantic dimensions."""
    return SemanticField(
        logical_name=name,
        dataset="bars",
        physical_name=name,
        market="ashare",
        frequency=frequency,
        grain=grain,
        source_unit=source_unit,
        canonical_unit=canonical_unit,
        availability=availability,
        temporal_model="panel",
        pit_fidelity="effective_only",
    )


# ---------------------------------------------------------------------------
# P0-26: frequency is a semantic identity dimension.
# ---------------------------------------------------------------------------

def test_frequency_difference_breaks_identity_equality() -> None:
    """Two reads differing ONLY in field.frequency (daily vs minute) must NOT
    be equal and must NOT share a digest."""
    daily = DataReadIdentity(
        dataset="bars", revision="r1", fields=(_field("vwap", frequency="daily"),)
    )
    minute = DataReadIdentity(
        dataset="bars", revision="r1", fields=(_field("vwap", frequency="minute"),)
    )
    assert daily != minute
    assert daily.digest != minute.digest
    # The field identity itself must disagree on frequency.
    assert daily.fields[0].frequency == "daily"
    assert minute.fields[0].frequency == "minute"
    assert daily.fields[0] != minute.fields[0]
    assert daily.fields[0].content_canonical != minute.fields[0].content_canonical


def test_frequency_present_in_serialized_field_identity() -> None:
    """frequency survives ResolvedFieldIdentity.to_dict()/from_dict()."""
    rid = DataReadIdentity(
        dataset="bars", revision="r1", fields=(_field("vwap", frequency="minute"),)
    )
    d = rid.to_dict()
    assert d["fields"][0]["frequency"] == "minute"
    rt = DataReadIdentity.from_dict(d)
    assert rt.fields[0].frequency == "minute"
    assert rt.fields[0] == rid.fields[0]


def test_identical_fields_still_collapse_to_same_identity() -> None:
    """Identical semantic fields -> identical identity (no false breakage)."""
    a = DataReadIdentity(
        dataset="bars", revision="r1", fields=(_field("vwap", frequency="daily"),)
    )
    b = DataReadIdentity(
        dataset="bars", revision="r1", fields=(_field("vwap", frequency="daily"),)
    )
    assert a == b
    assert a.digest == b.digest


def test_grain_and_unit_remain_identity_dimensions() -> None:
    """grain (instrument vs snapshot) and unit (level vs return) still break
    identity — regression guard for DA-P0-01."""
    g_a = DataReadIdentity(
        dataset="bars", revision="r1", fields=(_field("vwap", grain="instrument"),)
    )
    g_b = DataReadIdentity(
        dataset="bars", revision="r1", fields=(_field("vwap", grain="snapshot"),)
    )
    assert g_a != g_b
    assert g_a.digest != g_b.digest

    u_a = DataReadIdentity(
        dataset="bars",
        revision="r1",
        fields=(_field("vwap", source_unit="yuan", canonical_unit="yuan"),),
    )
    u_b = DataReadIdentity(
        dataset="bars",
        revision="r1",
        fields=(_field("vwap", source_unit="return", canonical_unit="return"),),
    )
    assert u_a != u_b
    assert u_a.digest != u_b.digest


def test_market_bar_catalog_daily_vs_minute_distinct() -> None:
    """End-to-end through the semantic catalog: daily vwap vs minute_vwap are
    distinct reads (dataset+physical+frequency differ)."""
    from data_access.read.semantic_catalog import get_semantic_catalog

    catalog = get_semantic_catalog()
    daily_field = catalog.resolve_one("vwap", dataset="ashare_stock_daily_adj")
    minute_field = catalog.resolve_one("minute_vwap", dataset="ashare_stock_minute_adj")
    assert daily_field is not None and minute_field is not None
    daily = DataReadIdentity(
        dataset="ashare_stock_daily_adj", revision="r1", fields=(daily_field,)
    )
    minute = DataReadIdentity(
        dataset="ashare_stock_minute_adj", revision="r1", fields=(minute_field,)
    )
    assert daily != minute
    assert daily.digest != minute.digest


# ---------------------------------------------------------------------------
# P0-27: timezone / naive-aware round-trip.
# ---------------------------------------------------------------------------

def test_aware_datetime_round_trip_keeps_equality() -> None:
    """A read with timezone-aware time_range survives to_dict/from_dict
    equality (previously provenance_notes () vs [] broke ==)."""
    aware = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc)
    rid = DataReadIdentity(
        dataset="bars",
        revision="r1",
        time_range=(aware, aware),
        provenance_notes=("revision: manifest",),
    )
    rt = DataReadIdentity.from_dict(rid.to_dict())
    assert rt == rid
    assert rt.digest == rid.digest
    assert rt.time_range == ("2024-01-02T15:00:00+00:00", "2024-01-02T15:00:00+00:00")


def test_naive_datetime_round_trip_keeps_equality() -> None:
    """Naive time_range (the common date-only daily read) also round-trips."""
    naive = datetime(2024, 1, 2, 15, 0)
    rid = DataReadIdentity(dataset="bars", revision="r1", time_range=(naive, naive))
    rt = DataReadIdentity.from_dict(rid.to_dict())
    assert rt == rid
    assert rt.time_range == ("2024-01-02T15:00:00", "2024-01-02T15:00:00")


def test_naive_vs_aware_same_wall_clock_differ_in_identity() -> None:
    """A naive endpoint and an aware endpoint that render to different ISO
    strings must produce different identities (honest tz representation)."""
    naive = datetime(2024, 1, 2, 15, 0)
    aware = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc)
    a = DataReadIdentity(dataset="bars", revision="r1", time_range=(naive, naive))
    b = DataReadIdentity(dataset="bars", revision="r1", time_range=(aware, aware))
    assert a != b
    assert a.digest != b.digest


def test_provenance_notes_tupleized_on_construction() -> None:
    """list provenance_notes is tuple-ized so == and digest stay stable."""
    rid = DataReadIdentity(
        dataset="bars",
        revision="r1",
        provenance_notes=["note-a", "note-b"],
    )
    assert rid.provenance_notes == ("note-a", "note-b")
