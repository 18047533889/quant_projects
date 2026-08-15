"""Explicit repository selection without changing lifecycle authority exports."""
from __future__ import annotations
from pathlib import Path
from factor_assets.errors import CapabilityError
from factor_assets.registry.repository import AssetRepository
from factor_assets.registry.sqlite_repository import SQLiteLifecycleRepository

def create_repository(*, production: bool = False, db_path: str | Path | None = None):
    if production:
        if db_path is None or str(db_path) == ":memory:":
            raise CapabilityError("production requires an explicit SQLite db_path")
        path = Path(db_path)
        if not path.exists():
            raise CapabilityError("production db_path must already exist")
        if not path.is_file():
            raise CapabilityError("production db_path must be a file")
        return SQLiteLifecycleRepository(path)
    if db_path is not None:
        return SQLiteLifecycleRepository(db_path)
    return AssetRepository()
