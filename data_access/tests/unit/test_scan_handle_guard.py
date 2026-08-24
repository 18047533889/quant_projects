from __future__ import annotations

from datetime import datetime, timezone

import pytest

from data_access.core.exceptions import ValidationError
from data_access.read.query_budget import QueryBudget
from data_access.read.read_contract import DataSnapshot, ReadLineage
from data_access.read.scan_handle import ScanHandle


class _FakeLazyFrame:
    def fetch(self, *_args, **_kwargs):
        raise AssertionError("fetch should be blocked before reaching the LazyFrame")

    def sink_parquet(self, *_args, **_kwargs):
        raise AssertionError("sink_parquet should be blocked before reaching the LazyFrame")


def _handle() -> ScanHandle:
    snapshot = DataSnapshot(
        snapshot_id="snap",
        dataset="demo",
        registry_hash="registry",
        schema_hash="schema",
        file_manifest_hash="manifest",
        files=(),
        created_at=datetime.now(timezone.utc),
    )
    return ScanHandle(
        _lf=_FakeLazyFrame(),
        snapshot=snapshot,
        budget=QueryBudget(),
        lineage=ReadLineage(dataset="demo", columns=("value",)),
    )


def test_production_blocks_materialization_bypasses(monkeypatch) -> None:
    monkeypatch.setenv("QUANT_PRODUCTION_MODE", "1")
    handle = _handle()

    with pytest.raises(ValidationError, match="受控 collect"):
        handle.fetch(10)
    with pytest.raises(ValidationError, match="受控 collect"):
        handle.sink_parquet("out.parquet")
    with pytest.raises(ValidationError, match="裸 LazyFrame"):
        handle.lazyframe()


def test_development_can_access_underlying_non_collect_methods(monkeypatch) -> None:
    monkeypatch.delenv("QUANT_PRODUCTION_MODE", raising=False)
    handle = _handle()

    assert handle.lazyframe().__class__ is _FakeLazyFrame
