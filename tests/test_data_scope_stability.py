from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from storage.data_scope import compute_data_scope


@dataclass
class _Source:
    dataset: str = "factor_lake"
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    params: dict | None = None
    fields: dict | None = None
    instrument_filter: list[str] | None = None
    timestamp_unit: str | None = None
    normalize_timestamp: bool | None = None
    read_auto: bool = False
    lazy_scan: bool = False
    data_snapshot_id: str | None = None
    root: Path | None = None


def test_scope_is_stable_for_mapping_order() -> None:
    left = _Source(
        params={"factor_id": "mom", "version": 2},
        fields={"close": "Close", "open": "Open"},
        instrument_filter=["MSFT", "AAPL"],
    )
    right = _Source(
        params={"version": 2, "factor_id": "mom"},
        fields={"open": "Open", "close": "Close"},
        instrument_filter=["AAPL", "MSFT"],
    )

    assert compute_data_scope(left) == compute_data_scope(right)


def test_scope_changes_with_execution_relevant_read_options() -> None:
    base = _Source(params={"factor_id": "mom"})
    variants = [
        _Source(params={"factor_id": "value"}),
        _Source(params={"factor_id": "mom"}, timestamp_unit="ns"),
        _Source(params={"factor_id": "mom"}, normalize_timestamp=True),
        _Source(params={"factor_id": "mom"}, read_auto=True),
        _Source(params={"factor_id": "mom"}, lazy_scan=True),
        _Source(params={"factor_id": "mom"}, data_snapshot_id="snapshot-v2"),
    ]

    base_scope = compute_data_scope(base)
    assert all(compute_data_scope(item) != base_scope for item in variants)


def test_path_and_dataclass_values_are_serialized_stably() -> None:
    a = _Source(root=Path("~/quant/data"), params={"factor_id": "mom"})
    b = _Source(root=Path("~/quant/data"), params={"factor_id": "mom"})

    assert compute_data_scope(a) == compute_data_scope(b)
