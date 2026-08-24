"""Phase 5：git commit、registry data_source、prefetch 测试。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from factor_engine.runtime.lineage import resolve_git_commit_hash
from factor_engine.storage.catalog import FactorCatalog
from factor_engine.storage.data_access_source import DataAccessSource


def test_resolve_git_commit_hash_in_quant_projects_repo():
    from workspace_paths import quant_projects_root

    commit = resolve_git_commit_hash(repo_root=str(quant_projects_root()))
    assert commit is None or (len(commit) >= 7 and commit.isalnum())


def test_catalog_register_persists_data_source_json():
    with tempfile.TemporaryDirectory() as tmp:
        cat = FactorCatalog(Path(tmp) / "catalog.sqlite")
        ds_cfg = {"type": "data_access", "dataset": "ashare_stock_daily"}
        cat.register(
            "f1",
            author="tester",
            frequency="1d",
            ast_hash="abc",
            data_source_config=ds_cfg,
        )
        info = cat.get_factor_info("f1")
        assert info is not None
        stored = json.loads(info["data_source_json"])
        assert stored["dataset"] == "ashare_stock_daily"
        cat.close()


def test_data_access_source_has_prefetch_columns():
    source = DataAccessSource(dataset="ashare_stock_daily", fields={"close": "Close"})
    assert callable(getattr(source, "prefetch_columns", None))
