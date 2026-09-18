from datetime import date, datetime, timedelta, timezone

import pytest

from data_access.r30._shared import stable_digest_full
from data_access.r30.calendar_snapshot import canonical
from data_access.r30.universe_snapshot import UniverseSnapshot


BASE = dict(
    universe_id="CSI300",
    market="A",
    members=("000002.SZ", "000001.SZ"),
    membership_policy_version="membership-v1",
    tradability_policy_version="tradability-v1",
    source_snapshot="source-v1",
)


def test_interval_is_normalised_serialised_and_changes_identity():
    snapshot = UniverseSnapshot.build(
        **BASE,
        effective_start=datetime(2024, 1, 1, 8, tzinfo=timezone(timedelta(hours=8))),
        effective_end=date(2024, 12, 31),
    )
    assert snapshot.effective_start == "2024-01-01T00:00:00Z"
    assert snapshot.effective_end == "2024-12-31"
    assert snapshot.to_dict()["effective_start"] == snapshot.effective_start
    assert snapshot.snapshot_id != UniverseSnapshot.build(**BASE).snapshot_id


@pytest.mark.parametrize(
    ("start", "end", "error"),
    [
        ("2024-01-01", None, ValueError),
        (None, "2024-12-31", ValueError),
        ("2024/01/01", "2024-12-31", ValueError),
        ("2024-02-30", "2024-12-31", ValueError),
        ("2024-01-01T00:00:00", "2024-12-31", ValueError),
        ("2025-01-01", "2024-12-31", ValueError),
        (1, 2, TypeError),
    ],
)
def test_interval_validation_is_strict(start, end, error):
    with pytest.raises(error):
        UniverseSnapshot.build(**BASE, effective_start=start, effective_end=end)


def test_legacy_snapshot_digest_is_unchanged():
    snapshot = UniverseSnapshot.build(**BASE)
    expected = stable_digest_full(
        canonical("CSI300"),
        canonical("A"),
        canonical(("000001.SZ", "000002.SZ")),
        canonical("membership-v1"),
        canonical("tradability-v1"),
        canonical("source-v1"),
    )
    assert snapshot.snapshot_id == expected
    assert snapshot.effective_start is None
    assert snapshot.effective_end is None


def test_from_store_never_infers_interval_from_request_time_range():
    class Store:
        market = "A"

        def _resolve_universe_instruments(self, universe_id, time_range, instruments):
            return ("000001.SZ",)

    snapshot = UniverseSnapshot.from_store(
        Store(), "CSI300", time_range=("2024-01-01", "2024-12-31")
    )
    assert snapshot.effective_start is None
    assert snapshot.effective_end is None


def test_from_store_accepts_only_explicit_trusted_interval_arguments():
    snapshot = UniverseSnapshot.from_store(
        object(),
        "CSI300",
        members=("000001.SZ",),
        market="A",
        effective_start="2024-01-01",
        effective_end="2024-12-31",
    )
    assert (snapshot.effective_start, snapshot.effective_end) == (
        "2024-01-01",
        "2024-12-31",
    )
