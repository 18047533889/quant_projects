# -*- coding: utf-8 -*-
"""R24-079/080: a composite child with NO snapshot token is UNVERIFIABLE —
``version=None, verified=True`` must never happen.  ``None == None`` is never
treated as "no update"."""
from __future__ import annotations

import pandas as pd
import pytest

from storage.composite_source import (
    CompositeDataSource,
    CompositeSnapshotVerificationError,
)
from storage.datasource import DataSource
from tests.storage.test_composite_source import _build_series


class _NoTokenSource(DataSource):
    """A source that exposes no snapshot token at all (unverifiable)."""

    def __init__(self, data: dict) -> None:
        self.data = data

    def load_column(self, name: str):
        return self.data[name]


class _BrokenTokenSource(DataSource):
    """A source whose token read raises (verified=False)."""

    def __init__(self, data: dict) -> None:
        self.data = data

    def load_column(self, name: str):
        return self.data[name]

    def snapshot_token(self) -> str:
        raise RuntimeError("token read failed")


def test_child_without_token_is_unverifiable_not_verified() -> None:
    price = _NoTokenSource({"close": _build_series([("2024-01-02", "AAA", 100.0)])})
    fund = _NoTokenSource({"pe": _build_series([("2024-01-01", "AAA", 10.0)])})
    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fund},
        aliases={"pe": "fundamental.pe"},
        joins={"fundamental": {"method": "asof_backward"}},
        production=True,
    )
    manifest = source._child_snapshot_manifest(refresh=False)
    states = dict(manifest)
    # R24-079: no token → UNVERIFIABLE (production).  Never verified=True.
    assert states["fundamental"].version is None
    assert states["fundamental"].verified is False


def test_verified_true_requires_a_real_token() -> None:
    # R24-079: version=None + verified=True must never occur in production.
    price = _NoTokenSource({"close": _build_series([("2024-01-02", "AAA", 100.0)])})
    fund = _NoTokenSource({"pe": _build_series([("2024-01-01", "AAA", 10.0)])})
    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fund},
        joins={"fundamental": {"method": "asof_backward"}},
        production=True,
    )
    for _name, state in source._child_snapshot_manifest(refresh=False):
        if state.version is None:
            assert state.verified is False


def test_token_read_failure_is_never_confirmed_none() -> None:
    price = _NoTokenSource({"close": _build_series([("2024-01-02", "AAA", 100.0)])})
    fund = _BrokenTokenSource({"pe": _build_series([("2024-01-01", "AAA", 10.0)])})
    source = CompositeDataSource(
        anchor_source="price",
        anchor_column="close",
        sources={"price": price, "fundamental": fund},
        joins={"fundamental": {"method": "asof_backward"}},
        production=True,
    )
    # A broken token is unverifiable — never a "confirmed None == no update".
    states = dict(source._child_snapshot_manifest(refresh=False))
    assert states["fundamental"].verified is False


def test_unverified_manifest_is_always_changed() -> None:
    # P0-29/R24-080: an unverified observation forces invalidation.
    from storage.composite_source import SnapshotState

    old = (("f", SnapshotState(version="A", verified=True)),)
    new_unverified = (("f", SnapshotState(version="A", verified=False)),)
    assert CompositeDataSource._snapshot_manifest_changed(old, new_unverified)
    same = (("f", SnapshotState(version="A", verified=True)),)
    assert not CompositeDataSource._snapshot_manifest_changed(old, same)
