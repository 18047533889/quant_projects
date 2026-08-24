# -*- coding: utf-8 -*-
"""R24-081..083 + R24-239: the composite manifest records EXACTLY the epoch
verified for a read.  A source advancing between the verified read and the
manifest record (data=A, manifest=B) is refused — never cached as "B verified".
"""
from __future__ import annotations

import pandas as pd
import pytest

from factor_engine.storage.composite_source import (
    CompositeDataSource,
    CompositeSnapshotVerificationError,
)
from factor_engine.storage.datasource import DataSource
from tests.storage.test_composite_source import _build_series


class _AdvancingSnapshotSource(DataSource):
    """A panel source whose snapshot token advances on the N-th manifest read.

    ``advance_after`` selects which ``_child_snapshot_manifest`` observation
    flips the token A→B, so a test can place the drift deterministically between
    ``revalidate`` (verify) and ``_record_snapshot_manifest`` (record).
    """

    def __init__(self, data: dict, *, advance_after: int = 100) -> None:
        self.data = data
        self._reads = 0
        self._advance_after = advance_after
        self._token = "A"

    def load_column(self, name: str):
        return self.data[name]

    def snapshot_token(self) -> str | None:
        self._reads += 1
        if self._reads >= self._advance_after:
            self._token = "B"
        return self._token

    def temporal_contract(self):
        # R24-087: a plain panel with a token is safe for generic asof.
        from factor_engine.storage.datasource import TemporalContract

        return TemporalContract(
            temporal_sensitivity="none",
            snapshot_capability="token",
            join_capability="generic_asof",
        )


class _EpochEncodeSource(_AdvancingSnapshotSource):
    """A source whose DATA depends on the current snapshot token, so a test can
    prove data-epoch == manifest-epoch after the load."""

    def __init__(self, data_a: dict, data_b: dict, *, advance_after: int = 100) -> None:
        super().__init__(data_a, advance_after=advance_after)
        self._data_b = data_b

    def load_column(self, name: str):
        return self._data_b[name] if self._token == "B" else self.data[name]


def test_drift_between_verify_and_manifest_record_ends_coherent() -> None:
    # R24-081..083: A read → A→B exactly between the read's verification and
    # the manifest record.  The outcome must NEVER be data=A + manifest=B:
    # either a retry re-reads under B (coherent B/B) or a mismatch raises.
    price = _EpochEncodeSource(
        {"close": _build_series([("2024-01-02", "AAA", 100.0)])},
        {"close": _build_series([("2024-01-02", "AAA", 100.0)])},
        advance_after=100,
    )
    fund = _EpochEncodeSource(
        {"pe": _build_series([("2024-01-01", "AAA", 10.0)])},
        {"pe": _build_series([("2024-01-01", "AAA", 99.0)])},
        # Manifest is observed: 1) invalidate-if-changed, 2) barrier begin,
        # 3) barrier revalidate, 4) _record_snapshot_manifest.  Flipping on the
        # 4th observation puts the drift between revalidate(3, sees A) and
        # record(4, sees B) — the exact R24-081 race.
        advance_after=4,
    )
    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fund},
        aliases={"pe": "fundamental.pe"},
        joins={"fundamental": {"method": "asof_backward"}},
        production=True,
    )
    try:
        pe = source.load_column("pe")
    except CompositeSnapshotVerificationError:
        # fail-closed: retry later; the mismatched epoch was never recorded.
        return
    # The load completed (after retry under B): DATA epoch must equal the
    # recorded MANIFEST epoch.
    read_value = float(pe.loc[(pd.Timestamp("2024-01-02"), "AAA")])
    manifest_epoch = dict(source._snapshot_manifest or ())["fundamental"].version
    if manifest_epoch == "B":
        assert read_value == pytest.approx(99.0), "data=A but manifest=B"
    else:
        assert read_value == pytest.approx(10.0)


def test_stable_source_records_verified_epoch() -> None:
    price = _AdvancingSnapshotSource(
        {"close": _build_series([("2024-01-02", "AAA", 100.0)])}, advance_after=100
    )
    fund = _AdvancingSnapshotSource(
        {"pe": _build_series([("2024-01-01", "AAA", 10.0)])}, advance_after=100
    )
    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fund},
        aliases={"pe": "fundamental.pe"},
        joins={"fundamental": {"method": "asof_backward"}},
    )
    pe = source.load_column("pe")
    assert pe.loc[(pd.Timestamp("2024-01-02"), "AAA")] == pytest.approx(10.0)
    # The recorded manifest reflects the epoch that was actually read (A).
    manifest = dict(source._snapshot_manifest or ())
    assert manifest["fundamental"].version == "A"
