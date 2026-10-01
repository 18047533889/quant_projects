"""Pure, conservative GPU working-set estimates for supported DAGs."""

_COVERAGE_REDUCTION_RESERVE_BYTES = 64 * 1024**2


def _require_nonnegative_int(name, value):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def estimate_coverage_working_set_bytes(
    time_count: int,
    asset_count: int,
    factor_tile: int,
    factor_dtype_bytes: int,
    label_dtype_bytes: int,
    *,
    force_fp64: bool = False,
) -> int:
    """Bound the exact coverage-only GPU DAG, including transient masks.

    Includes staged factors and labels, three full-size boolean masks to cover
    the finite-mask expression and allocator overlap, reduction outputs, 1.5x
    headroom, and a fixed 64 MiB reduction/pool reserve. Unrelated live pool
    bytes are added separately by the device-session caller.
    """
    for name, value in (("time_count", time_count),
                        ("asset_count", asset_count),
                        ("factor_tile", factor_tile)):
        _require_nonnegative_int(name, value)
    for name, value in (("factor_dtype_bytes", factor_dtype_bytes),
                        ("label_dtype_bytes", label_dtype_bytes)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(force_fp64) is not bool:
        raise ValueError("force_fp64 must be a bool")

    factor_size = max(factor_dtype_bytes, 8) if force_fp64 else factor_dtype_bytes
    label_size = max(label_dtype_bytes, 8)
    panel_cells = time_count * asset_count
    tile_cells = panel_cells * factor_tile
    factor_bytes = tile_cells * factor_size
    label_bytes = panel_cells * label_size
    mask_bytes = 3 * tile_cells
    reduction_bytes = 2 * time_count * factor_tile * 8
    reduction_bytes += factor_tile * 16 + panel_cells
    base_bytes = factor_bytes + label_bytes + mask_bytes + reduction_bytes
    return (3 * base_bytes + 1) // 2 + _COVERAGE_REDUCTION_RESERVE_BYTES
