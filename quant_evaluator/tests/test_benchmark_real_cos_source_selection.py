import pytest

from quant_evaluator.scripts import benchmark_real_cos_factor_tiles as tiles


def _manifest(count=61, size=12):
    result = {}
    for index in range(count):
        name, digest = f"factor{index:03d}", f"{index:064x}"
        result[name] = {
            "verified": True, "status": "materialized_not_evaluated",
            "bytes": size, "sha256": digest,
            "uri": f"{tiles.POOL}/{digest}/{name}.parquet",
        }
    return result


def test_exact_f8_source_selection_preserves_large_tile_cli_bounds():
    manifest = _manifest()
    selected = tiles.select_source_records(manifest, 8, 1, 1)
    assert len(selected) == 8
    assert [row[0] for row in selected] == [f"factor{i:03d}" for i in range(8)]
    with pytest.raises(ValueError, match="count 33..64"):
        tiles.select_records(manifest, 8, 1, 1)
    assert tiles.select_source_records(manifest, 61, 1, 1) == tiles.select_records(manifest, 61, 1, 1)


@pytest.mark.parametrize("count", [True, 7, 9, 32, 65])
def test_source_selector_does_not_widen_other_small_profiles(count):
    with pytest.raises(ValueError):
        tiles.select_source_records(_manifest(), count, 1, 1)


def test_f8_source_selection_keeps_manifest_binding_and_byte_caps():
    with pytest.raises(ValueError, match="bounded verified"):
        tiles.select_source_records(_manifest(7), 8, 1, 1)
    with pytest.raises(ValueError, match="bounded verified"):
        tiles.select_source_records(_manifest(8, size=200_000), 8, 1, 1)
    manifest = _manifest(8)
    manifest["factor000"]["uri"] = "unbound"
    with pytest.raises(ValueError, match="manifest bound"):
        tiles.select_source_records(manifest, 8, 1, 1)
