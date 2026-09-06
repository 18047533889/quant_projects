# -*- coding: utf-8 -*-
"""staging 行级删除与 Kline prefetch 测试。"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest

pd = pytest.importorskip("pandas")


def test_real_staging_materialization_binds_and_publishes_exact_identity(
    tmp_path, monkeypatch
):
    """Real local staging bytes carry the receipt consumed by publication."""
    from data_access import get_store, reset_store
    from data_access.registry.loader import DatasetRegistry
    from data_access.store import DataAccessStore, get_shared_engine
    from factor_engine.storage.materialize.lake_publish import publish_factor_lake
    from factor_engine.storage.materializer import ParquetMaterializer

    monkeypatch.setenv(
        "QUANTSOCIETY_WORKSPACE_DATA_ROOT", str(tmp_path / "data_access_workspace")
    )
    monkeypatch.setenv("RUN_NAMESPACE", "real_staging_receipt")
    monkeypatch.setenv("QUANT_RUN_NAMESPACE", "real_staging_receipt")
    reset_store()
    try:
        configured = get_store()
        staging_ds = dataclasses.replace(
            configured.get_dataset("factor_lake_staging"),
            root_template=str(
                tmp_path
                / "data_access_workspace"
                / "staging"
                / "${RUN_NAMESPACE}"
                / "factor_lake"
                / "factors"
                / "{factor_id}"
            ),
            authorized_root=tmp_path / "data_access_workspace" / "staging",
        )
        published_ds = dataclasses.replace(
            configured.get_dataset("factor_lake"),
            root_template=str(
                tmp_path
                / "data_access_workspace"
                / "published"
                / "factor_lake"
                / "factors"
                / "{factor_id}"
            ),
            authorized_root=tmp_path / "data_access_workspace" / "published",
        )
        store = DataAccessStore(
            registry=DatasetRegistry(
                {"factor_lake_staging": staging_ds, "factor_lake": published_ds}
            ),
            engine=get_shared_engine(),
        )
        monkeypatch.setattr("data_access.get_store", lambda: store)
        idx = pd.MultiIndex.from_product(
            [pd.to_datetime(["2026-08-07"]), ["A"]],
            names=["timestamp", "instrument"],
        )
        materializer = ParquetMaterializer(lake_root=tmp_path / "lake")
        summary = materializer.materialize(
            factor_id="receipt_factor",
            result=pd.Series([1.0], index=idx),
            ast_hash="receipt-hash",
            frequency="1d",
            write_target="staging",
            defer_watermark=True,
            run_lineage={"run_id": "receipt-run"},
            value_dtype="float64",
        )
        identity = summary["staging"]["identity"]
        assert identity["run_id"] == "receipt-run"
        assert identity["inventory"]
        identity_path = (
                store.resolve_dataset_path(
                    "factor_lake_staging", factor_id="receipt_factor"
                )
            / ".fe_staging_identity.json"
        )
        assert identity_path.is_file()
        assert json.loads(identity_path.read_text()) == identity

        published = publish_factor_lake(
            factor_id="receipt_factor",
            lake_root=tmp_path / "lake",
            approve=True,
            sync_from_local=False,
            reconcile=False,
            expected_staging_generation=identity["generation_id"],
            expected_manifest_digest=identity["manifest_digest"],
            expected_run_id=identity["run_id"],
            frequency="1d",
        )
        assert published["factor_id"] == "receipt_factor"
        assert published["approved"] is True
        assert published["generation_id"] == identity["generation_id"]
        assert published["manifest_digest"] == identity["manifest_digest"]
        assert published["run_id"] == identity["run_id"]
        assert published["watermark"] is None
        assert materializer.catalog.get_watermark("receipt_factor") is None
        published_root = store.resolve_dataset_path(
            "factor_lake", factor_id="receipt_factor"
        )
        frame = pd.concat(
            [pd.read_parquet(path) for path in published_root.rglob("*.parquet")],
            ignore_index=True,
        )
        assert frame[["datetime", "asset", "value"]].to_dict("records") == [
            {"datetime": pd.Timestamp("2026-08-07"), "asset": "A", "value": 1.0}
        ]
    finally:
        reset_store()


def test_delete_staging_rows_after_watermark(tmp_path, monkeypatch):
    staging_root = tmp_path / "staging" / "ns" / "factor_lake" / "factors" / "f1"
    part = staging_root / "year=2024"
    part.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "asset": ["A", "A", "A"],
            "value": [1.0, 2.0, 3.0],
            "year": [2024, 2024, 2024],
        }
    )
    df.to_parquet(part / "data.parquet", index=False)

    class FakeStore:
        def resolve_dataset_path(self, dataset, **params):
            assert dataset == "factor_lake_staging"
            return staging_root

        def delete_rows(self, dataset, **kwargs):
            from factor_engine.storage.staging_loader import _delete_staging_rows_local

            return _delete_staging_rows_local(factor_id=kwargs.get("factor_id", "f1"), **{
                k: v for k, v in kwargs.items() if k in ("start", "end", "after")
            })

    monkeypatch.setattr("data_access.get_store", lambda: FakeStore())

    from factor_engine.storage.staging_loader import delete_staging_rows

    result = delete_staging_rows("f1", after="2024-01-02")
    assert result["rows_deleted"] == 2
    kept = pd.read_parquet(part / "data.parquet")
    assert len(kept) == 1
    assert kept.iloc[0]["datetime"] == pd.Timestamp("2024-01-02")


def test_kline_prefetch_columns_batch(tmp_path):
    pytest.importorskip("pyarrow")
    root = tmp_path / "kline"
    root.mkdir()
    for day, val in [("2024-01-02", 1.0), ("2024-01-03", 2.0)]:
        pd.DataFrame(
            {
                "ticker": ["A"],
                "window_start": [pd.Timestamp(day)],
                "close": [val],
                "open": [val + 0.1],
            }
        ).to_parquet(root / f"{day}.parquet", index=False)

    from factor_engine.storage.kline_parquet_source import KlineParquetSource

    src = KlineParquetSource(
        root=str(root),
        fields={"close": "close", "open": "open"},
        normalize_timestamp=True,
    )
    src.prefetch_columns(["close", "open"])
    assert "close" in src._column_cache
    assert "open" in src._column_cache
    assert len(src._column_cache["close"]) == 2
