"""Safety gate for synthetic source overrides on the managed default path."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_access.core.exceptions import ValidationError


def _write_registry(path: Path, data_root: Path) -> None:
    path.write_text(
        f"""ashare_stock_daily_adj:
  kind: static
  access_mode: published
  layout: plain
  hive_partitioning: false
  union_by_name: true
  root: {data_root}
  glob: "*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    AdjClose: double
""",
        encoding="utf-8",
    )


def test_production_rejects_synthetic_read_root_before_any_cos_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local fixture cannot impersonate the approved canonical source."""
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    data_root = tmp_path / "synthetic-data"
    data_root.mkdir()
    pd.DataFrame(
        {
            "TradeDate": [pd.Timestamp("2024-01-02").date()],
            "Symbol": ["000001.SZ"],
            "AdjClose": [10.0],
        }
    ).to_parquet(data_root / "panel.parquet", index=False)
    registry_path = tmp_path / "datasets.yaml"
    _write_registry(registry_path, data_root)
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(registry_path))

    def forbid_cos(*_args, **_kwargs):
        raise AssertionError("strict local-override rejection must precede COS access")

    monkeypatch.setattr(
        "data_access.cos.remote.prepare_cos_remote_paths", forbid_cos
    )
    monkeypatch.setattr(
        "data_access.cos.mirror.ensure_local_mirror_for_dataset", forbid_cos
    )
    from data_access import get_store, reset_store

    reset_store()
    store = get_store()
    with pytest.raises(
        ValidationError,
        match=(
            "production/严格读模式禁止 read_root 覆盖数据集 "
            "'ashare_stock_daily_adj' 根路径"
        ),
    ):
        store.prepare_read(
            "ashare_stock_daily_adj",
            columns=("AdjClose",),
            params={"read_root": str(data_root)},
            time_range=("2024-01-02", "2024-01-02"),
            instrument_filter=("000001.SZ",),
            run_mode="production",
            snapshot_policy="fail_if_changed",
        )
    reset_store()
