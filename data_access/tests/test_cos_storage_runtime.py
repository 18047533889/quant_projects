from __future__ import annotations

from dataclasses import replace

from data_access.cos import mirror, remote
from data_access.cos_storage_runtime import install_cos_storage_runtime


def test_period_event_and_sparse_layouts_do_not_enumerate_decision_days():
    install_cos_storage_runtime()
    assert mirror.mirror_spec_for_dataset("us_stock_balance").layout == "period_files"
    assert mirror.mirror_spec_for_dataset("us_stock_dividend").layout == "event_files"
    assert mirror.mirror_spec_for_dataset("us_stock_valuation_daily").layout == "sparse_files"
    paths = remote.build_remote_paths(
        "us_stock_balance", time_range=("2024-01-01", "2024-12-31")
    )
    assert len(paths) == 1
    assert paths[0].endswith("/**/*.parquet")
    assert "2024-01-01.parquet" not in paths[0]


def test_daily_httpfs_uses_month_glob_not_weekend_object_names():
    install_cos_storage_runtime()
    paths = remote.build_remote_paths(
        "ashare_stock_daily", time_range=("2024-01-01", "2024-01-31")
    )
    assert len(paths) == 1
    assert paths[0].endswith("/2024-01-*.parquet")


def test_auto_recursive_layout_requires_complete_marker(tmp_path):
    install_cos_storage_runtime()
    name = "us_stock_balance"
    original = mirror.DATASET_MIRROR_REGISTRY[name]
    patched = replace(original, local_root=tmp_path)
    mirror.DATASET_MIRROR_REGISTRY[name] = patched
    try:
        table_dir = tmp_path / str(patched.table)
        table_dir.mkdir(parents=True)
        (table_dir / "2024-03-31.parquet").write_bytes(b"parquet-placeholder")
        assert not remote.local_mirror_complete_for_range(
            name, time_range=("2024-01-01", "2024-12-31")
        )
        (table_dir / ".data_access_complete.json").write_text("{}", encoding="utf-8")
        assert remote.local_mirror_complete_for_range(
            name, time_range=("2024-01-01", "2024-12-31")
        )
    finally:
        mirror.DATASET_MIRROR_REGISTRY[name] = original
