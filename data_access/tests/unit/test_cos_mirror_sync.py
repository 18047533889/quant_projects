"""cos_mirror：本地镜像同步逻辑（mock COS CLI，不访问网络）。"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from data_access.cos.mirror import (
    MirrorSpec,
    ensure_local_mirror,
    skip_cos_mirror,
    sync_dataset,
)


@pytest.fixture
def mirror_env(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_ACCESS_SKIP_COS_MIRROR", raising=False)
    spec = MirrorSpec(
        cos_prefix="cos://test/prefix",
        local_root=tmp_path / "local",
        table="StockDailyBar",
        layout="daily_parquet",
    )
    registry = {"test_ashare_daily": spec}
    monkeypatch.setattr(
        "data_access.cos.mirror.DATASET_MIRROR_REGISTRY",
        registry,
    )
    monkeypatch.setattr(
        "data_access.cos.mirror.mirror_spec_for_dataset",
        lambda name: registry.get(name),
    )
    return spec


def test_skip_cos_mirror_env(monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    assert skip_cos_mirror() is True


def test_sync_daily_file_invokes_cos_cli(mirror_env, tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        dest = Path(cmd[-1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"PAR1")

    with patch("data_access.cos.mirror.subprocess.run", side_effect=fake_run):
        sync_dataset(
            "test_ashare_daily",
            time_range=("2024-01-02", "2024-01-02"),
        )

    assert len(calls) == 1
    assert calls[0][0] == "clean-cos-ro"
    assert calls[0][1] == "cp"
    assert calls[0][2].endswith("StockDailyBar/2024-01-02.parquet")
    assert (mirror_env.local_root / "StockDailyBar" / "2024-01-02.parquet").exists()


def test_missing_dense_object_is_error(mirror_env):
    """R25 P0-015：dense 行情缺失 partition 必须 fail-closed（不再是 debug 跳过）。

    旧行为「404 → debug → return」会让 StockDailyBar 缺交易日文件静默漏读——
    D1 密集行情是数据缺口，必须报 MissingRequiredPartition。
    """
    import subprocess

    from data_access.core.exceptions import MissingRequiredPartition

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            cmd,
            stderr="cos object not found: test",
        )

    with patch("data_access.cos.mirror.subprocess.run", side_effect=fake_run):
        with pytest.raises(MissingRequiredPartition):
            sync_dataset(
                "test_ashare_daily",
                time_range=("2024-01-07", "2024-01-07"),
            )


def test_missing_sparse_object_empty_ok(tmp_path):
    """R25 P0-015：稀疏事件表缺失 partition → empty_ok 静默跳过（不 hard fail）。"""
    import subprocess

    from data_access.cos.mirror import _sync_cos_file

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            cmd,
            stderr="cos object not found: test",
        )

    dest = tmp_path / "events" / "2024-01-02.parquet"
    with patch("data_access.cos.mirror.subprocess.run", side_effect=fake_run):
        _sync_cos_file(
            "cos://test/events/2024-01-02.parquet",
            dest,
            missing_semantics="empty_ok",
            dataset_name="test_sparse_event",
        )
    assert not dest.exists()  # 缺失被静默跳过（empty_ok）


def test_ensure_local_mirror_raises_without_local_data_or_time_range(mirror_env):
    from data_access.core.exceptions import ValidationError

    with pytest.raises(ValidationError, match="未指定 time_range"):
        ensure_local_mirror("test_ashare_daily", time_range=None)


def test_ensure_local_mirror_pulls_missing_range(mirror_env):
    calls: list[str] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[2])
        dest = Path(cmd[-1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"PAR1")

    # 先放一天，再请求两天，应只拉缺失的那天
    existing = mirror_env.local_root / "StockDailyBar" / "2024-01-02.parquet"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"PAR1")

    with patch("data_access.cos.mirror.subprocess.run", side_effect=fake_run):
        ensure_local_mirror(
            "test_ashare_daily",
            time_range=("2024-01-02", "2024-01-03"),
        )

    assert len(calls) == 1
    assert calls[0].endswith("2024-01-03.parquet")


def test_hive_date_sync_layout(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_ACCESS_SKIP_COS_MIRROR", raising=False)
    spec = MirrorSpec(
        cos_prefix="cos://test/adj_factor",
        local_root=tmp_path / "adj_factor",
        layout="hive_date",
        file_name="data.parquet",
    )
    registry = {"test_adj": spec}
    monkeypatch.setattr("data_access.cos.mirror.DATASET_MIRROR_REGISTRY", registry)
    monkeypatch.setattr(
        "data_access.cos.mirror.mirror_spec_for_dataset",
        lambda name: registry.get(name),
    )

    calls: list[str] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[2])
        dest = Path(cmd[-1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"PAR1")

    with patch("data_access.cos.mirror.subprocess.run", side_effect=fake_run):
        sync_dataset("test_adj", time_range=("2024-01-02", "2024-01-02"))

    assert calls[0].endswith("date=2024-01-02/data.parquet")
    assert (tmp_path / "adj_factor" / "date=2024-01-02" / "data.parquet").exists()


def test_iter_years_for_hive_year(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_ACCESS_SKIP_COS_MIRROR", raising=False)
    spec = MirrorSpec(
        cos_prefix="cos://test/universe",
        local_root=tmp_path / "universe",
        layout="hive_year",
        file_name="data.parquet",
    )
    registry = {"test_uni": spec}
    monkeypatch.setattr("data_access.cos.mirror.DATASET_MIRROR_REGISTRY", registry)
    monkeypatch.setattr(
        "data_access.cos.mirror.mirror_spec_for_dataset",
        lambda name: registry.get(name),
    )

    calls: list[str] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[2])
        dest = Path(cmd[-1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"PAR1")

    with patch("data_access.cos.mirror.subprocess.run", side_effect=fake_run):
        sync_dataset("test_uni", time_range=("2023-12-31", "2024-01-02"))

    assert sorted(calls) == [
        "cos://test/universe/year=2023/data.parquet",
        "cos://test/universe/year=2024/data.parquet",
    ]
