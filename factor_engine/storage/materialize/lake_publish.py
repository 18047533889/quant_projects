"""因子湖 staging → published 发布（带人工审批门）。"""

from __future__ import annotations

import os
import hashlib
import json
import shutil
import sqlite3
import tempfile
import uuid
from pathlib import Path
from typing import Any

from factor_engine.storage.exceptions import FactorNotFoundError

_STAGING_IDENTITY = ".fe_staging_identity.json"


def _require_local_publication_storage(dataset_name: str) -> Any:
    """Return the configured store only when the dataset is authoritatively local."""
    from data_access import get_store
    from data_access.core.storage import StorageBackend, resolve_storage_for_dataset

    store = get_store()
    dataset = store.get_dataset(dataset_name)
    storage = resolve_storage_for_dataset(dataset)
    try:
        backend = StorageBackend(storage.type)
    except ValueError as exc:
        raise ValueError(
            f"unsupported publication storage backend {storage.type!r}"
        ) from exc
    if backend is not StorageBackend.LOCAL:
        raise ValueError(
            f"{backend.value} publication lacks expected-inventory/CAS wiring; rejected"
        )
    return store


def _factor_inventory(root: Path) -> list[dict[str, Any]]:
    """Content-addressed inventory; unreadable parquet fails closed."""
    import pyarrow.parquet as pq

    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.parquet")):
        if path.name.startswith(".") or path.is_symlink():
            continue
        metadata = pq.read_metadata(path)
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        entries.append({
            "path": path.relative_to(root).as_posix(),
            "rows": int(metadata.num_rows),
            "bytes": int(path.stat().st_size),
            "sha256": digest,
        })
    if not entries:
        raise ValueError(f"staging has no readable parquet inventory: {root}")
    return entries


def _inventory_digest(entries: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def produce_coverage_receipt(
    *,
    factor_id: str,
    materialization_receipt: dict[str, Any],
    expected_keys: Any,
    universe_snapshot: str,
    calendar_id: str,
    market_timezone: str,
    frequency: str,
    coverage_intervals: list[dict[str, Any]] | None = None,
    timestamp_column: str = "datetime",
    instrument_column: str = "asset",
) -> dict[str, Any]:
    """Stream-verify the staged key axis and bind proof into a receipt.

    ``expected_keys`` is a single-pass iterable of ``(timestamp, instrument)``.
    Only key columns are read from parquet; SQLite provides disk-backed uniqueness
    and exact missing/extra checks without materializing either axis as a Python set.
    """
    import pandas as pd
    import pyarrow.parquet as pq
    from data_access.write.mutation_lock import mutation_lock
    from factor_engine.runtime.materialize_batch import WriteReceipt, WriteState
    from factor_engine.security.factor_id import validate_factor_id

    factor_id = validate_factor_id(factor_id)
    if not all(isinstance(value, str) and value for value in (
        universe_snapshot, calendar_id, market_timezone, frequency,
        timestamp_column, instrument_column
    )):
        raise ValueError("coverage context fields must be non-empty strings")
    receipt = WriteReceipt.from_dict(materialization_receipt, expected_items=[factor_id])
    item = receipt.items[factor_id]
    if item.state is not WriteState.COMMITTED:
        raise ValueError("coverage proof requires a COMMITTED receipt")
    staging_dir = _resolve_staging_factor_dir(factor_id)

    def encoded_key(timestamp: Any, instrument: Any) -> tuple[int, str]:
        ts = pd.Timestamp(timestamp)
        if pd.isna(ts):
            raise ValueError("coverage keys require non-null timestamps")
        if ts.tzinfo is None:
            ts = ts.tz_localize(market_timezone)
        ts_key = int(ts.tz_convert("UTC").value)
        if instrument is None or bool(pd.isna(instrument)):
            raise ValueError("coverage keys require non-null instruments")
        if isinstance(instrument, str):
            inst_key = json.dumps(["str", instrument], ensure_ascii=False)
        elif isinstance(instrument, bool):
            inst_key = json.dumps(["bool", instrument])
        elif isinstance(instrument, int):
            inst_key = json.dumps(["int", instrument])
        else:
            raise ValueError("coverage instrument keys must be str/bool/int")
        return ts_key, inst_key

    with mutation_lock(staging_dir):
        inventory = _factor_inventory(staging_dir)
        digest = _inventory_digest(inventory)
        if item.inventory != inventory or item.inventory_digest != digest:
            raise ValueError("staged bytes do not match materialization receipt inventory")
        with tempfile.TemporaryDirectory(prefix="fe-coverage-") as tmp:
            db = sqlite3.connect(str(Path(tmp) / "keys.sqlite"))
            try:
                db.execute("CREATE TABLE expected (ts INTEGER NOT NULL, inst TEXT NOT NULL, PRIMARY KEY(ts, inst))")
                db.execute("CREATE TABLE actual (ts INTEGER NOT NULL, inst TEXT NOT NULL, PRIMARY KEY(ts, inst))")
                expected_count = 0
                for raw in expected_keys:
                    if not isinstance(raw, (tuple, list)) or len(raw) != 2:
                        raise ValueError("expected_keys entries must be timestamp/instrument pairs")
                    key = encoded_key(raw[0], raw[1])
                    try:
                        db.execute("INSERT INTO expected VALUES (?, ?)", key)
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(f"duplicate expected coverage key: {key}") from exc
                    expected_count += 1
                actual_count = 0
                for entry in inventory:
                    parquet = pq.ParquetFile(staging_dir / entry["path"])
                    for batch in parquet.iter_batches(
                        columns=[timestamp_column, instrument_column], batch_size=65536
                    ):
                        values = batch.to_pydict()
                        for timestamp, instrument in zip(
                            values[timestamp_column], values[instrument_column]
                        ):
                            key = encoded_key(timestamp, instrument)
                            try:
                                db.execute("INSERT INTO actual VALUES (?, ?)", key)
                            except sqlite3.IntegrityError as exc:
                                raise ValueError(f"duplicate actual coverage key: {key}") from exc
                            actual_count += 1
                missing = db.execute(
                    "SELECT ts, inst FROM expected EXCEPT SELECT ts, inst FROM actual LIMIT 1"
                ).fetchone()
                extra = db.execute(
                    "SELECT ts, inst FROM actual EXCEPT SELECT ts, inst FROM expected LIMIT 1"
                ).fetchone()
                if missing or extra or actual_count != expected_count:
                    raise ValueError(
                        "actual coverage axis differs from expected axis: "
                        f"expected={expected_count} actual={actual_count} "
                        f"missing_sample={missing} extra_sample={extra}"
                    )
                axis_hasher = hashlib.sha256()
                for ts_key, inst_key in db.execute(
                    "SELECT ts, inst FROM actual ORDER BY ts, inst"
                ):
                    axis_hasher.update(str(ts_key).encode())
                    axis_hasher.update(b"\0")
                    axis_hasher.update(inst_key.encode())
                    axis_hasher.update(b"\n")
                axis_digest = axis_hasher.hexdigest()
                actual_min_ns, actual_max_ns = db.execute(
                    "SELECT MIN(ts), MAX(ts) FROM actual"
                ).fetchone()
                raw_intervals = list(coverage_intervals or [])
                if not raw_intervals or actual_min_ns is None:
                    raise ValueError("coverage proof requires non-empty actual keys and intervals")
                normalized_intervals = []
                previous_end = None
                covered_count = 0
                for interval in sorted(raw_intervals, key=lambda value: pd.Timestamp(value["start"])):
                    start = pd.Timestamp(interval["start"])
                    end = pd.Timestamp(interval["end"])
                    if start.tzinfo is None or end.tzinfo is None:
                        raise ValueError("coverage interval timestamps must be timezone-aware")
                    start_ns = int(start.tz_convert("UTC").value)
                    end_ns = int(end.tz_convert("UTC").value)
                    if start_ns > end_ns or (
                        previous_end is not None and start_ns <= previous_end
                    ):
                        raise ValueError("coverage intervals must be ordered and non-overlapping")
                    interval_count = int(db.execute(
                        "SELECT COUNT(*) FROM actual WHERE ts >= ? AND ts <= ?",
                        (start_ns, end_ns),
                    ).fetchone()[0])
                    if (
                        interval.get("expected_rows") != interval_count
                        or interval.get("observed_rows") != interval_count
                    ):
                        raise ValueError("coverage interval counts do not match actual keys")
                    normalized_intervals.append({
                        "start": pd.Timestamp(start_ns, unit="ns", tz="UTC").isoformat(),
                        "end": pd.Timestamp(end_ns, unit="ns", tz="UTC").isoformat(),
                        "expected_rows": interval_count,
                        "observed_rows": interval_count,
                    })
                    previous_end = end_ns
                    covered_count += interval_count
                if (
                    normalized_intervals[0]["start"]
                    != pd.Timestamp(actual_min_ns, unit="ns", tz="UTC").isoformat()
                    or normalized_intervals[-1]["end"]
                    != pd.Timestamp(actual_max_ns, unit="ns", tz="UTC").isoformat()
                    or covered_count != actual_count
                ):
                    raise ValueError("coverage intervals do not exactly bound actual key timestamps")
            finally:
                db.close()
    item.coverage_proof = {
        "proof_version": "sqlite-axis-v1",
        "actual_axis_digest": axis_digest,
        "expected_axis_digest": axis_digest,
        "expected_key_count": expected_count,
        "actual_key_count": actual_count,
        "universe_snapshot": universe_snapshot,
        "calendar_id": calendar_id,
        "market_timezone": market_timezone,
        "frequency": frequency,
        "inventory_digest": digest,
        "run_id": item.run_id,
        "actual_start": normalized_intervals[0]["start"],
        "actual_end": normalized_intervals[-1]["end"],
        "coverage_intervals": normalized_intervals,
    }
    receipt.validate()
    return receipt.to_dict()


def _write_staging_identity_locked(
    *,
    factor_id: str,
    materialization_receipt: dict[str, Any],
    coverage_intervals: list[dict[str, Any]] | None = None,
    coverage_complete: bool = False,
    frequency: str | None = None,
) -> dict[str, Any]:
    """Write identity while the caller holds the staging mutation lock."""
    staging_dir = _resolve_staging_factor_dir(factor_id)
    from factor_engine.runtime.materialize_batch import WriteReceipt, WriteState

    receipt = WriteReceipt.from_dict(
        materialization_receipt, expected_items=[factor_id]
    )
    item = receipt.items[factor_id]
    if item.state is not WriteState.COMMITTED:
        raise ValueError("staging identity requires a COMMITTED materialization receipt")
    inventory = _factor_inventory(staging_dir)
    if item.inventory != inventory or item.inventory_digest != _inventory_digest(inventory):
        raise ValueError(
            "staged bytes do not match the inventory bound into materialization receipt"
        )
    payload = {
        "factor_id": factor_id,
        "generation_id": str(receipt.generation_id),
        "run_id": str(item.run_id),
        "materialization_manifest_digest": receipt.manifest_digest,
        "inventory": inventory,
        "manifest_digest": _inventory_digest(inventory),
        "coverage_intervals": list(coverage_intervals or []),
        "coverage_complete": bool(coverage_complete),
        "frequency": frequency,
        "coverage_proof": item.coverage_proof,
    }
    if coverage_complete:
        if not frequency:
            raise ValueError("coverage-complete staging identity requires frequency")
        _validated_coverage(
            payload, frequency,
            observed_rows=sum(int(item["rows"]) for item in inventory),
            universe_snapshot=(item.coverage_proof or {}).get("universe_snapshot"),
        )
    from data_access.core.atomic import atomic_write_text

    atomic_write_text(
        staging_dir / _STAGING_IDENTITY,
        json.dumps(payload, sort_keys=True, indent=2),
    )
    return payload


def write_staging_identity(
    *,
    factor_id: str,
    materialization_receipt: dict[str, Any],
    coverage_intervals: list[dict[str, Any]] | None = None,
    coverage_complete: bool = False,
    frequency: str | None = None,
) -> dict[str, Any]:
    """Bind staged bytes to the exact run/generation before publication."""
    _require_local_publication_storage("factor_lake_staging")
    staging_dir = _resolve_staging_factor_dir(factor_id)
    from data_access.write.mutation_lock import mutation_lock

    with mutation_lock(staging_dir):
        return _write_staging_identity_locked(
            factor_id=factor_id,
            materialization_receipt=materialization_receipt,
            coverage_intervals=coverage_intervals,
            coverage_complete=coverage_complete,
            frequency=frequency,
        )


def _read_verified_staging_identity(factor_id: str) -> dict[str, Any]:
    staging_dir = _resolve_staging_factor_dir(factor_id)
    path = staging_dir / _STAGING_IDENTITY
    if not path.is_file():
        raise ValueError(f"staging identity manifest missing for factor={factor_id!r}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = _factor_inventory(staging_dir)
    if payload.get("inventory") != actual or payload.get("manifest_digest") != _inventory_digest(actual):
        raise ValueError("staging identity manifest does not match current parquet bytes")
    return payload


class PublishNotApprovedError(PermissionError):
    """因子湖发布未获审批时抛出。
    
    参数:
        无
    """


class PublishInDoubtError(RuntimeError):
    """Storage may be committed; caller must reconcile and must not replay."""

    def __init__(self, message: str, *, generation_id: str, manifest_digest: str):
        super().__init__(message)
        self.generation_id = generation_id
        self.manifest_digest = manifest_digest


def is_publish_approved(*, approve: bool = False) -> bool:
    """检查发布是否已获审批。
    
    参数:
        approve: 是否显式审批发布（可选）
    
    返回:
        bool
    """
    if approve:
        return True
    return os.environ.get("QUANT_PUBLISH_APPROVED", "").lower() in (
        "1",
        "true",
        "yes",
    )


def require_publish_approval(*, approve: bool = False) -> None:
    """要求发布审批，未通过则抛错。
    
    参数:
        approve: 是否显式审批发布（可选）
    
    返回:
        无
    """
    if not is_publish_approved(approve=approve):
        raise PublishNotApprovedError(
            "因子湖发布需要显式审批：传入 approve=True 或设置 QUANT_PUBLISH_APPROVED=1"
        )


def _resolve_staging_factor_dir(factor_id: str) -> Path:
    """解析 staging 中因子目录路径。
    
    参数:
        factor_id: 因子唯一标识
    
    返回:
        Path
    """
    from data_access import get_store

    store = get_store()
    return store.resolve_dataset_path("factor_lake_staging", factor_id=factor_id)


def _count_parquet_rows(root: Path) -> int:
    """统计目录下 Parquet 文件总行数。
    
    参数:
        root: 根目录路径
    
    返回:
        int
    """
    import pyarrow.parquet as pq

    total = 0
    if not root.exists():
        return 0
    for pq_file in root.rglob("*.parquet"):
        total += pq.read_metadata(pq_file).num_rows
    return total


def sync_local_factor_to_staging(
    *,
    factor_id: str,
    lake_root: str | Path,
    materialization_receipt: dict[str, Any] | None = None,
    coverage_intervals: list[dict[str, Any]] | None = None,
    coverage_complete: bool = False,
    frequency: str | None = None,
) -> dict[str, Any]:
    """将本地因子湖目录原子复制到 staging。
    
    参数:
        factor_id: 因子唯一标识（可选）
        lake_root: 因子湖根目录（可选）
    
    返回:
        dict[str, Any]
    """
    from factor_engine.security.factor_id import factor_dir_for, validate_factor_id

    factor_id = validate_factor_id(factor_id)
    _require_local_publication_storage("factor_lake_staging")
    source = factor_dir_for(lake_root, factor_id)
    if not source.exists():
        raise FileNotFoundError(f"本地因子目录不存在: {source}")

    staging_dir = _resolve_staging_factor_dir(factor_id)
    staging_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = staging_dir.parent / f".{factor_id}.staging_tmp_{uuid.uuid4().hex[:8]}"
    backup_dir = staging_dir.parent / f".{factor_id}.staging_bak_{uuid.uuid4().hex[:8]}"
    from data_access.write.mutation_lock import mutation_lock
    receipt = None
    if materialization_receipt is not None:
        from factor_engine.runtime.materialize_batch import WriteReceipt, WriteState
        receipt = WriteReceipt.from_dict(
            materialization_receipt, expected_items=[factor_id]
        )
        if receipt.items[factor_id].state is not WriteState.COMMITTED:
            raise ValueError("staging sync requires a COMMITTED materialization receipt")

    # The lock file lives inside its dataset root. Never rename staging_dir itself:
    # doing so would move the locked inode and permit a competing writer to create
    # and acquire a new lock in the replacement root.
    with mutation_lock(source):
        if receipt is not None:
            item = receipt.items[factor_id]
            source_inventory = _factor_inventory(source)
            if (
                item.inventory != source_inventory
                or item.inventory_digest != _inventory_digest(source_inventory)
            ):
                raise ValueError(
                    "local source bytes do not match materialization receipt inventory"
                )
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        shutil.copytree(
            source, tmp_dir,
            ignore=shutil.ignore_patterns(".data-access.mutation.lock", _STAGING_IDENTITY),
        )
        with mutation_lock(staging_dir):
            try:
                backup_dir.mkdir(parents=True)
                for child in list(staging_dir.iterdir()):
                    if child.name == ".data-access.mutation.lock":
                        continue
                    os.replace(child, backup_dir / child.name)
                for child in list(tmp_dir.iterdir()):
                    os.replace(child, staging_dir / child.name)
                if receipt is not None:
                    staged_inventory = _factor_inventory(staging_dir)
                    item = receipt.items[factor_id]
                    if (
                        item.inventory != staged_inventory
                        or item.inventory_digest != _inventory_digest(staged_inventory)
                    ):
                        raise ValueError(
                            "copied staging bytes do not match materialization receipt"
                        )
                    identity = _write_staging_identity_locked(
                        factor_id=factor_id,
                        materialization_receipt=materialization_receipt,
                        coverage_intervals=coverage_intervals,
                        coverage_complete=coverage_complete,
                        frequency=frequency,
                    )
                rows = _count_parquet_rows(staging_dir)
                shutil.rmtree(backup_dir, ignore_errors=True)
            except Exception:
                for child in list(staging_dir.iterdir()):
                    if child.name == ".data-access.mutation.lock":
                        continue
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
                if backup_dir.exists():
                    for child in list(backup_dir.iterdir()):
                        os.replace(child, staging_dir / child.name)
                raise
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                shutil.rmtree(backup_dir, ignore_errors=True)

    result = {
        "factor_id": factor_id,
        "source_dir": str(source),
        "staging_dir": str(staging_dir),
        "rows": rows,
    }
    if materialization_receipt is not None:
        result["identity"] = identity
    return result


def _validated_coverage(
    identity: dict[str, Any], frequency: str, *, observed_rows: int | None = None,
    universe_snapshot: str | None = None,
) -> list[dict[str, Any]]:
    """Validate timezone-aware, ordered, gap-free declared coverage."""
    import pandas as pd
    from pandas.tseries.frequencies import to_offset

    if not identity.get("coverage_complete"):
        return []
    proof = identity.get("coverage_proof")
    required = {
        "actual_axis_digest", "expected_axis_digest", "universe_snapshot",
        "calendar_id", "market_timezone", "frequency", "inventory_digest", "run_id",
    }
    if not isinstance(proof, dict) or not required.issubset(proof):
        raise ValueError("coverage_complete requires typed axis/universe/calendar proof")
    if not all(isinstance(proof[key], str) and proof[key] for key in required):
        raise ValueError("coverage proof fields must be non-empty strings")
    if proof["actual_axis_digest"] != proof["expected_axis_digest"]:
        raise ValueError("actual axis does not match expected axis")
    if proof["frequency"] != frequency:
        raise ValueError("coverage proof frequency mismatch")
    if proof["inventory_digest"] != identity.get("manifest_digest"):
        raise ValueError("coverage proof is not bound to staged inventory")
    if proof["run_id"] != identity.get("run_id"):
        raise ValueError("coverage proof is not bound to materialization run")
    if proof.get("proof_version") != "sqlite-axis-v1":
        raise ValueError("unsupported coverage proof producer/version")
    if proof.get("actual_key_count") != proof.get("expected_key_count"):
        raise ValueError("coverage proof key counts differ")
    if universe_snapshot is not None and proof["universe_snapshot"] != universe_snapshot:
        raise ValueError("coverage proof universe snapshot mismatch")
    raw = list(identity.get("coverage_intervals") or [])
    if not raw:
        raise ValueError("coverage_complete requires non-empty coverage_intervals")
    try:
        offset = to_offset(frequency)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid published frequency {frequency!r}") from exc
    parsed: list[tuple[pd.Timestamp, pd.Timestamp, int, int]] = []
    for interval in raw:
        if not isinstance(interval, dict) or "start" not in interval or "end" not in interval:
            raise ValueError("invalid coverage interval")
        start = pd.Timestamp(interval["start"])
        end = pd.Timestamp(interval["end"])
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("coverage interval timestamps must be timezone-aware")
        start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
        if start > end:
            raise ValueError("coverage interval start exceeds end")
        expected = interval.get("expected_rows")
        observed = interval.get("observed_rows")
        if (
            isinstance(expected, bool) or isinstance(observed, bool)
            or not isinstance(expected, int) or not isinstance(observed, int)
            or expected <= 0 or observed != expected
        ):
            raise ValueError(
                "coverage_complete requires positive equal expected_rows/observed_rows"
            )
        parsed.append((start, end, expected, observed))
    parsed.sort()
    merged: list[list[Any]] = []
    for start, end, expected, observed in parsed:
        if not merged:
            merged.append([start, end, expected, observed])
            continue
        if start > merged[-1][1] + offset:
            raise ValueError(
                f"coverage_complete contains a gap after {merged[-1][1].isoformat()}"
            )
        if end > merged[-1][1]:
            merged[-1][1] = end
        merged[-1][2] += expected
        merged[-1][3] += observed
    if observed_rows is not None and sum(item[3] for item in merged) != observed_rows:
        raise ValueError("coverage observed_rows does not match staged parquet inventory")
    normalized = [
        {
            "start": start.isoformat(), "end": end.isoformat(),
            "expected_rows": expected, "observed_rows": observed,
        }
        for start, end, expected, observed in merged
    ]
    if proof.get("coverage_intervals") != normalized:
        raise ValueError("declared coverage intervals do not match produced axis proof")
    if not normalized or proof.get("actual_start") != normalized[0]["start"] or proof.get("actual_end") != normalized[-1]["end"]:
        raise ValueError("coverage proof endpoints do not match declared intervals")
    return normalized


def staging_has_data(factor_id: str) -> bool:
    """检查 staging 是否已有因子数据。

    参数:
        factor_id: 因子唯一标识

    返回:
        bool
    """
    staging_dir = _resolve_staging_factor_dir(factor_id)
    return _count_parquet_rows(staging_dir) > 0


def advance_published_watermark(
    *,
    factor_id: str,
    catalog: Any,
    store: Any,
    publish_identity: dict[str, Any],
    frequency: str,
    universe_snapshot: str | None = None,
) -> dict[str, Any] | None:
    """publish 成功后推进权威水位线（staging-only 时 defer 的水位线在此 commit）。

    #收官轮 P0（Integration）：staging-only 物化不再在 materialize() 里推进权威
    水位线（只算到 STAGED）；只有 ``publish_from_staging`` 成功晋升后这里才把
    FactorCatalog 的正式水位线推进到 PUBLISHED。发布失败（抛异常）时不调用 → 水位线
    保持原样，增量调度器不会误以为「已正式提交」而 skip。

    返回更新后的 watermark dict；published 目录不可读/无数据时返回 None（水位线不动）。
    """
    if not publish_identity.get("coverage_complete"):
        raise ValueError("published watermark requires manifest coverage_complete evidence")
    intervals = _validated_coverage(
        publish_identity, frequency, universe_snapshot=universe_snapshot
    )
    lake_dir = store.resolve_dataset_path("factor_lake", factor_id=factor_id)
    if not lake_dir.is_dir():
        return None
    published_inventory = _factor_inventory(lake_dir)
    if _inventory_digest(published_inventory) != publish_identity.get("manifest_digest"):
        raise ValueError("published bytes do not match committed staging manifest")
    starts = [str(interval["start"]) for interval in intervals]
    ends = [str(interval["end"]) for interval in intervals]
    start_date = min(starts)
    end_date = max(ends)
    rows = sum(int(item["rows"]) for item in published_inventory)
    update_published = getattr(catalog, "update_published_watermark", None)
    if not callable(update_published):
        raise ValueError(
            "catalog lacks generation-aware update_published_watermark; "
            "legacy date-only watermark cannot record publication evidence"
        )
    result = update_published(
        factor_id=factor_id,
        start_date=start_date,
        end_date=end_date,
        row_count=rows,
        generation_id=publish_identity["generation_id"],
        frequency=frequency,
        committed_tip=end_date,
        coverage_intervals=intervals,
        universe_snapshot=universe_snapshot,
    )
    return result


def publish_factor_lake(
    *,
    factor_id: str,
    lake_root: str | Path | None = None,
    approve: bool = False,
    sync_from_local: bool = True,
    reconcile: bool = True,
    data_source_config: dict[str, Any] | None = None,
    expected_staging_generation: str | None = None,
    expected_manifest_digest: str | None = None,
    expected_run_id: str | None = None,
    frequency: str | None = None,
    universe_snapshot: str | None = None,
    materialization_receipt: dict[str, Any] | None = None,
    coverage_intervals: list[dict[str, Any]] | None = None,
    coverage_complete: bool = False,
    expected_axis_keys: Any = None,
    calendar_id: str | None = None,
    market_timezone: str | None = None,
) -> dict[str, Any]:
    """审批后将 staging 晋升为 published 因子湖。
    
    参数:
        factor_id: 因子唯一标识（可选）
        lake_root: 因子湖根目录（可选）
        approve: 是否显式审批发布（可选）
        sync_from_local: 见函数签名（可选）
        reconcile: 见函数签名（可选）
        data_source_config: 见函数签名（可选）
    
    返回:
        dict[str, Any]
    """
    require_publish_approval(approve=approve)
    if not expected_staging_generation or not expected_manifest_digest or not expected_run_id:
        raise ValueError(
            "publish requires expected_staging_generation, expected_manifest_digest, "
            "and expected_run_id"
        )
    if not frequency:
        raise ValueError("publish requires an explicit frequency before storage mutation")
    if coverage_complete and not universe_snapshot:
        raise ValueError("coverage-complete publish requires universe_snapshot")
    if coverage_complete and (
        materialization_receipt is None or expected_axis_keys is None
        or not calendar_id or not market_timezone
    ):
        raise ValueError(
            "coverage-complete publish requires materialization_receipt, "
            "expected_axis_keys, calendar_id, and market_timezone"
        )

    # ``resolve_dataset_path`` is not a capability check: URI-backed datasets
    # are also represented as Path objects by DataAccess. Resolve the declared
    # storage backend before catalog creation, sync, or any other side effect.
    store = _require_local_publication_storage("factor_lake")
    for dataset_name in ("factor_lake", "factor_lake_staging"):
        _require_local_publication_storage(dataset_name)
    from factor_engine.runtime.snapshot_reconcile import reconcile_data_snapshot
    from factor_engine.storage.catalog import FactorCatalog
    from factor_engine.util.workspace_paths import default_factor_lake_root

    root = Path(lake_root or default_factor_lake_root())
    catalog = FactorCatalog(root / "_catalog.sqlite")
    if catalog.get_factor_info(factor_id) is None:
        raise FactorNotFoundError(f"因子 '{factor_id}' 未在 catalog 注册")
    if not callable(getattr(catalog, "update_published_watermark", None)):
        raise ValueError("catalog lacks generation-aware published watermark capability")

    if reconcile:
        report = reconcile_data_snapshot(
            factor_id=factor_id,
            lake_root=root,
            data_source_config=data_source_config,
            catalog=catalog,
        )
        if not report["ok"]:
            raise ValueError(
                "data_snapshot_id 对账失败: "
                + ", ".join(report["mismatches"])
            )

    sync_summary = None
    if sync_from_local:
        sync_summary = sync_local_factor_to_staging(
            factor_id=factor_id, lake_root=root,
            materialization_receipt=materialization_receipt,
            coverage_intervals=None,
            coverage_complete=False,
            frequency=frequency,
        )

    if coverage_complete:
        materialization_receipt = produce_coverage_receipt(
            factor_id=factor_id,
            materialization_receipt=materialization_receipt,
            expected_keys=expected_axis_keys,
            universe_snapshot=universe_snapshot,
            calendar_id=calendar_id,
            market_timezone=market_timezone,
            frequency=frequency,
            coverage_intervals=coverage_intervals,
        )
        identity = write_staging_identity(
            factor_id=factor_id,
            materialization_receipt=materialization_receipt,
            coverage_intervals=coverage_intervals,
            coverage_complete=True,
            frequency=frequency,
        )

    if not callable(getattr(store, "publish_from_staging", None)):
        raise ValueError("store lacks atomic publish capability")
    try:
        target_dir = store.resolve_dataset_path("factor_lake", factor_id=factor_id)
    except Exception as exc:
        raise ValueError(
            "remote publish capability cannot accept expected inventory/CAS; rejected"
        ) from exc
    if not isinstance(target_dir, Path):
        raise ValueError("non-local publish without expected inventory/CAS is unsupported")
    staging_dir = _resolve_staging_factor_dir(factor_id)
    from data_access.write.mutation_lock import mutation_lock

    with mutation_lock(staging_dir):
        identity = _read_verified_staging_identity(factor_id)
        if identity.get("generation_id") != expected_staging_generation:
            raise ValueError("stale staging generation does not match publish request")
        if identity.get("run_id") != expected_run_id:
            raise ValueError("staging run_id does not match publish request")
        if identity.get("manifest_digest") != expected_manifest_digest:
            raise ValueError("staging manifest digest does not match publish request")
        if identity.get("frequency") != frequency:
            raise ValueError("staging frequency does not match publish request")
        if identity.get("coverage_complete") and not universe_snapshot:
            raise ValueError("coverage-complete publish requires universe_snapshot")
        _validated_coverage(
            identity, frequency,
            observed_rows=sum(int(item["rows"]) for item in identity["inventory"]),
            universe_snapshot=universe_snapshot,
        )
        # Hold the same dataset-root mutation lock used by DataAccess writers
        # across expected-inventory verification and the atomic local publish.
        try:
            publish_result = store.publish_from_staging(
                "factor_lake_staging",
                "factor_lake",
                factor_id=factor_id,
            )
        except Exception as exc:
            # DataAccess exposes this exact post-commit/audit-loss condition.
            # It is never safe to turn it into an item-by-item replay.
            if type(exc).__name__ == "CommittedButAuditFailed":
                raise PublishInDoubtError(
                    f"publish commit outcome requires reconciliation: {exc}",
                    generation_id=identity["generation_id"],
                    manifest_digest=identity["manifest_digest"],
                ) from exc
            raise
    watermark = None
    if identity.get("coverage_complete"):
        if not frequency:
            raise ValueError("coverage-complete publish requires frequency")
        try:
            watermark = advance_published_watermark(
                factor_id=factor_id, catalog=catalog, store=store,
                publish_identity=identity, frequency=frequency,
                universe_snapshot=universe_snapshot,
            )
        except Exception as exc:
            raise PublishInDoubtError(
                f"data published but watermark reconciliation failed: {exc}",
                generation_id=identity["generation_id"],
                manifest_digest=identity["manifest_digest"],
            ) from exc
    return {
        "factor_id": factor_id,
        "approved": True,
        "run_id": identity["run_id"],
        "generation_id": identity["generation_id"],
        "manifest_digest": identity["manifest_digest"],
        "sync": sync_summary,
        "publish": publish_result,
        "watermark": watermark,
    }
