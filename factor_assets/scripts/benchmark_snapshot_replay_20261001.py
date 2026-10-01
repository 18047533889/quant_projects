#!/usr/bin/env python3
"""Bounded A/B benchmark for FactorAssets point-in-time snapshot replay.

Run from the project checkout, for example:
  python factor_assets/scripts/benchmark_snapshot_replay_20261001.py
  python factor_assets/scripts/benchmark_snapshot_replay_20261001.py --assets 1000 --events-per-asset 50

The reference path reproduces the pre-one-pass per-asset replay. Results are
receipts about the checked source revision; they do not claim future evidence.
No data files are read or written unless --receipt is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import socket
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from factor_assets.contracts.asset import AssetMetadata, FactorAsset
from factor_assets.contracts.lifecycle import LifecycleState, StateEvent
from factor_assets.contracts.lineage import LineageRef
from factor_assets.registry.snapshots import (
    SnapshotManager,
    SnapshotQuery,
    SnapshotResult,
)
import factor_assets.registry.snapshots as snapshots_module

DEFAULT_ASSET_SCALES = (100, 1000)
MAX_ASSETS = 1000
MAX_EVENTS = 50_000
MAX_EVENTS_PER_ASSET = 50
MAX_REPETITIONS = 21
MEMORY_FLOOR_BYTES = 512 * 1024 * 1024
DISK_FLOOR_BYTES = 1024 * 1024 * 1024
RECEIPT_TIME = "2000-01-01T00:00:00+00:00"


def _legacy_snapshot(
    assets: list[FactorAsset], events: list[StateEvent], query: SnapshotQuery
) -> SnapshotResult:
    """Test/benchmark-only reproduction of the pre-optimization snapshot path."""
    normalize = snapshots_module._normalize_ts
    as_of = normalize(query.as_of_timestamp)
    states: dict[str, LifecycleState] = {}
    for event in sorted(events, key=lambda item: normalize(item.timestamp)):
        if normalize(event.timestamp) > as_of:
            break
        states[event.factor_id] = event.to_state

    matching: list[FactorAsset] = []
    for asset in assets:
        if normalize(asset.registered_at) > as_of:
            continue
        state = states.get(asset.factor_id, LifecycleState.REGISTERED)
        if query.factor_id and asset.factor_id != query.factor_id:
            continue
        if query.lifecycle_state and state != query.lifecycle_state:
            continue
        if query.campaign_id and asset.lineage.campaign_id != query.campaign_id:
            continue
        if query.tags and not set(query.tags).issubset(set(asset.tags)):
            continue

        # Intentionally retain the old per-asset sort and scan as the baseline.
        first_evaluated = approved = production_ready = None
        historical_state = LifecycleState.REGISTERED
        for event in sorted(events, key=lambda item: normalize(item.timestamp)):
            if event.factor_id != asset.factor_id:
                continue
            if normalize(event.timestamp) > as_of:
                break
            historical_state = event.to_state
            if event.to_state is LifecycleState.EVALUATED and first_evaluated is None:
                first_evaluated = event.timestamp
            elif event.to_state is LifecycleState.APPROVED and approved is None:
                approved = event.timestamp
            elif event.to_state is LifecycleState.PRODUCTION_READY and production_ready is None:
                production_ready = event.timestamp
        from dataclasses import replace

        matching.append(replace(
            asset,
            lifecycle_state=historical_state,
            first_evaluated_at=first_evaluated,
            approved_at=approved,
            production_ready_at=production_ready,
        ))
    return SnapshotResult(query, tuple(matching), len(matching), RECEIPT_TIME)


def _available_memory_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _source_hashes() -> dict[str, str]:
    return {
        "snapshots.py": _sha256(ROOT / "factor_assets/registry/snapshots.py"),
        "contract_asset.py": _sha256(ROOT / "factor_assets/contracts/asset.py"),
        "contract_lifecycle.py": _sha256(ROOT / "factor_assets/contracts/lifecycle.py"),
        "contract_lineage.py": _sha256(ROOT / "factor_assets/contracts/lineage.py"),
        "benchmark_script": _sha256(Path(__file__).resolve()),
    }


def _ensure_receipt_available(path: Path) -> None:
    if os.path.lexists(path):
        raise SystemExit(f"refusing receipt destination that already exists: {path}")


def _ensure_sources_unchanged(initial_head: str, initial_hashes: dict[str, str]) -> None:
    if _head() != initial_head or _source_hashes() != initial_hashes:
        raise SystemExit("source or HEAD changed during timing; refusing receipt")


def _write_receipt(path: Path, serialized: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def _build_case(asset_count: int, per_asset: int):
    assets = []
    events = []
    states = (
        LifecycleState.EVALUATED,
        LifecycleState.APPROVED,
        LifecycleState.PRODUCTION_READY,
        LifecycleState.DEPRECATED,
        LifecycleState.RETIRED,
    )
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(asset_count):
        factor_id = f"FA{index:04d}"
        metadata = AssetMetadata(
            factor_id=factor_id, canonical_repr=f"factor_{index}",
            canonical_hash=f"sha256:{index:064x}", frequency="daily",
            domains=("price",), timing="daily",
        )
        assets.append(FactorAsset(
            metadata=metadata,
            lineage=LineageRef(factor_id=factor_id, parents=()),
            lifecycle_state=LifecycleState.REGISTERED,
            registered_at=epoch.isoformat(),
        ))
        from_state = LifecycleState.REGISTERED
        for step in range(per_asset):
            to_state = states[step % len(states)]
            events.append(StateEvent(
                factor_id=factor_id, from_state=from_state, to_state=to_state,
                # Same step across factors creates stable-sort timestamp ties.
                timestamp=(epoch + timedelta(days=1, minutes=step)).isoformat(),
                evidence_refs=(),
            ))
            from_state = to_state
    as_of = (epoch + timedelta(days=1, minutes=max(0, per_asset * 4 // 5))).isoformat()
    return assets, events, SnapshotQuery(as_of_timestamp=as_of)


def _content(result: SnapshotResult):
    return result.query, result.total_count, result.assets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets", type=int, nargs="+", default=list(DEFAULT_ASSET_SCALES),
        help="one or more asset scales (default: 100 1000; maximum 1000)",
    )
    parser.add_argument("--events-per-asset", type=int, default=12)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--receipt", type=Path,
                        help="optional JSON receipt path; existing files are refused")
    args = parser.parse_args()
    if any(not 1 <= count <= MAX_ASSETS for count in args.assets):
        parser.error(f"each --assets value must be in [1, {MAX_ASSETS}]")
    if not 0 <= args.events_per_asset <= MAX_EVENTS_PER_ASSET:
        parser.error(f"--events-per-asset must be in [0, {MAX_EVENTS_PER_ASSET}]")
    if max(args.assets) * args.events_per_asset > MAX_EVENTS:
        parser.error(f"asset/event product exceeds {MAX_EVENTS} events")
    if not 1 <= args.repetitions <= MAX_REPETITIONS:
        parser.error(f"--repetitions must be in [1, {MAX_REPETITIONS}]")
    return args


def _benchmark_scale(asset_count: int, per_asset: int, repetitions: int) -> dict:
    assets, events, query = _build_case(asset_count, per_asset)
    manager = SnapshotManager()
    # Warm in the reverse order from the first measured repetition.
    actual = manager.create_snapshot(assets, events, query)
    expected = _legacy_snapshot(assets, events, query)
    if _content(actual) != _content(expected):
        raise AssertionError("optimized snapshot differs from legacy replay")

    optimized_times = []
    legacy_times = []
    trials = []
    for repetition in range(repetitions):
        order = (("legacy", _legacy_snapshot), ("one_pass", manager.create_snapshot))
        if repetition % 2:
            order = tuple(reversed(order))
        trial = {"repetition": repetition + 1, "route_order": []}
        for route, call in order:
            trial["route_order"].append(route)
            started = perf_counter()
            result = call(assets, events, query)
            elapsed = perf_counter() - started
            if _content(result) != _content(expected):
                raise AssertionError(f"{route} output changed during timing")
            (legacy_times if route == "legacy" else optimized_times).append(elapsed)
            trial[f"{route}_seconds"] = elapsed
        trial["outputs_equal_including_order_and_timestamps"] = True
        trials.append(trial)
    legacy_median = median(legacy_times)
    optimized_median = median(optimized_times)
    return {
        "assets": asset_count,
        "events": len(events),
        "events_per_asset": per_asset,
        "outputs_equal_including_order_and_timestamps": True,
        "repetitions_per_route": repetitions,
        "legacy_median_seconds": legacy_median,
        "one_pass_median_seconds": optimized_median,
        "speedup": legacy_median / optimized_median if optimized_median else None,
        "trials": trials,
    }


def main() -> None:
    args = _parse_args()
    if args.receipt:
        _ensure_receipt_available(args.receipt)
    started_at = datetime.now(timezone.utc).isoformat()
    disk = shutil.disk_usage(ROOT)
    memory_available = _available_memory_bytes()
    if disk.free < DISK_FLOOR_BYTES:
        raise SystemExit(f"refusing run: free disk below {DISK_FLOOR_BYTES} bytes")
    if memory_available < MEMORY_FLOOR_BYTES:
        raise SystemExit(f"refusing run: available memory below {MEMORY_FLOOR_BYTES} bytes")

    initial_head = _head()
    initial_hashes = _source_hashes()
    results = [
        _benchmark_scale(count, args.events_per_asset, args.repetitions)
        for count in args.assets
    ]
    _ensure_sources_unchanged(initial_head, initial_hashes)

    completed_at = datetime.now(timezone.utc).isoformat()
    raw_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = raw_peak_rss if sys.platform == "darwin" else raw_peak_rss * 1024
    receipt = {
        "kind": "bounded_factor_assets_snapshot_replay_ab",
        "head_before_and_after": initial_head,
        "source_sha256": initial_hashes,
        "fixture_generator": "deterministic_step_timestamp_v1",
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "results": results,
        "guards": {
            "max_assets": MAX_ASSETS, "max_events": MAX_EVENTS,
            "max_events_per_asset": MAX_EVENTS_PER_ASSET,
            "max_repetitions": MAX_REPETITIONS,
            "available_memory_bytes_at_start": memory_available,
            "disk_free_bytes_at_start": disk.free,
            "peak_rss_bytes_at_end": peak_rss_bytes,
        },
    }
    serialized = json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    if args.receipt:
        _write_receipt(args.receipt, serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
