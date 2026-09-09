"""Explicit repository selection without changing lifecycle authority exports."""
from __future__ import annotations
from pathlib import Path
from factor_assets.errors import CapabilityError
from factor_assets.registry.repository import AssetRepository
from factor_assets.registry.sqlite_repository import SQLiteLifecycleRepository

def create_repository(*, production: bool = False, db_path: str | Path | None = None,
                      transactional_outbox=None):
    if production:
        if db_path is None or str(db_path) == ":memory:":
            raise CapabilityError("production requires an explicit SQLite db_path")
        path = Path(db_path)
        if not path.exists():
            raise CapabilityError("production db_path must already exist")
        if not path.is_file():
            raise CapabilityError("production db_path must be a file")
        if transactional_outbox is None:
            from quant_platform.app.db.sqlite_backend import SqliteDb
            from factor_assets.adapters.platform_outbox import PlatformLifecycleOutboxAdapter
            platform_db=SqliteDb(str(path), create=True); platform_db.close()
            transactional_outbox=PlatformLifecycleOutboxAdapter()
        return SQLiteLifecycleRepository(path, transactional_outbox=transactional_outbox)
    if db_path is not None:
        return SQLiteLifecycleRepository(db_path, transactional_outbox=transactional_outbox)
    return AssetRepository()

def create_platform_lifecycle(*, db_path: str | Path, authorization_resolver):
    """Production composition: trusted authorization plus platform Outbox."""
    if authorization_resolver is None:
        raise CapabilityError("production lifecycle requires authorization_resolver")
    from quant_platform.app.db.sqlite_backend import SqliteDb
    from factor_assets.adapters.platform_outbox import PlatformLifecycleOutboxAdapter
    from factor_assets.registry.lifecycle import LifecycleOrchestrator
    path=Path(db_path)
    platform_db=SqliteDb(str(path), create=True)
    platform_db.close()
    repository=SQLiteLifecycleRepository(path, transactional_outbox=PlatformLifecycleOutboxAdapter())
    return LifecycleOrchestrator(repository, authorization_resolver), repository
