"""Bounded shared-axis preparation for isolated real-COS width workers."""
from __future__ import annotations

from pathlib import Path

from quant_evaluator.scripts import benchmark_real_cos_factor_tiles as tiles


def prepare_axis_index(args, temporary_directory):
    """Validate/reuse an index or build one once for all child workers."""
    manifest_sha = tiles.MANIFEST_SHA256
    records = tiles.select_source_records(
        tiles.read_manifest(manifest_sha), args.factors,
        args.max_object_mib, args.max_total_mib)
    if args.axis_index and args.axis_index.exists():
        index_path = args.axis_index
        common_dates, common_assets, source_rows = tiles.read_axis_index(
            index_path, manifest_sha, records)
    else:
        common_dates, common_assets, source_rows = tiles.intersect_axes(
            tiles.iter_frames(records, manifest_sha, args.max_object_mib,
                              reuse_manifest=True), len(records))
        index = tiles.build_axis_index(
            manifest_sha, records, common_dates, common_assets, source_rows)
        index_path = Path(temporary_directory) / "source-axis-index.json"
        tiles.write_axis_index_atomic(index_path, index)
        common_dates, common_assets, source_rows = tiles.read_axis_index(
            index_path, manifest_sha, records)
    dates, assets, labels = tiles.load_labels(
        common_dates, common_assets, args.days, args.assets)
    shape = (len(dates), len(assets), len(records))
    if not all(value > 0 for value in shape):
        raise ValueError("prepared source shape is empty")
    del labels
    return index_path, shape


def width_memory_admission(*, shape, widths, source_budget_bytes,
                           max_prefetch_memory_bytes, prefetch_workers,
                           prefetch_enabled, extra_assembly_bytes_per_cell):
    """Return per-width estimates and the first applicable admission failure."""
    from quant_evaluator.runtime.source_memory_budget import estimate_source_memory

    time_size, asset_size, factor_count = shape
    estimates = {}
    for width in widths:
        item = estimate_source_memory(
            time_size=time_size, asset_size=asset_size,
            factor_count=factor_count, max_tile_size=width,
            dtype_itemsize=8, prefetch_workers=prefetch_workers,
            prefetch_enabled=prefetch_enabled,
            extra_assembly_bytes_per_cell=extra_assembly_bytes_per_cell)
        rejection = ("prefetch_memory_budget" if
                     item.prefetch_bytes > max_prefetch_memory_bytes else
                     "source_memory_budget" if item.total_bytes > source_budget_bytes else None)
        estimates[str(width)] = {
            "tile_factors": item.tile_factors,
            "estimated_assembly_bytes": item.assembly_bytes,
            "estimated_prefetch_bytes": item.prefetch_bytes,
            "estimated_peak_source_bytes": item.total_bytes,
            "admitted": rejection is None,
            "rejection_category": rejection,
        }
    return estimates
