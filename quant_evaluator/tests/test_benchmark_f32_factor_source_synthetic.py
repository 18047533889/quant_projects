from quant_evaluator.scripts import benchmark_f32_factor_source_synthetic as bench
import numpy as np


def test_preflight_requires_both_ram_and_gpu_headroom():
    gpu = {"status": "ok", "devices": [{"name": "NVIDIA L20", "passes": True}]}
    assert bench.preflight(bench.MIN_RAM, gpu)["pass"] is True
    assert bench.preflight(bench.MIN_RAM - 1, gpu)["pass"] is False
    assert bench.preflight(bench.MIN_RAM, {"status": "unavailable"})["pass"] is False
    assert bench.preflight(bench.MIN_RAM, {"status": "ok", "devices": [
        {"name": "NVIDIA L20", "passes": True}, {"name": "NVIDIA L20", "passes": True}
    ]})["pass"] is False
    assert bench.preflight(bench.MIN_RAM, {"status": "ok", "devices": [
        {"name": "Other GPU", "passes": True}
    ]})["pass"] is False


def test_benchmark_is_synthetic_and_tile_bounded():
    assert (bench.T, bench.N, bench.F, bench.TILE) == (2586, 5461, 32, 2)
    assert bench.METRICS == ("rank_ic", "rank_ic_series")
    source = bench.SyntheticSource(seed=7)
    assert source.time_axis.values.dtype == np.int64
    assert source.asset_axis.values.dtype == np.int64
    gate = bench.preflight(bench.MIN_RAM, {"status": "ok", "devices": [
        {"name": "NVIDIA L20", "passes": True}
    ]})
    assert gate["factor_tensor_materialized"] is False
    assert gate["disk_panel_bytes"] == 0


def test_parity_includes_observation_counts():
    import numpy as np
    good = {"metrics": {"rank_ic": np.array([0.1]), "rank_ic_series": np.array([[0.1]])},
            "observation_counts": {"rank_ic": np.array([20]),
                                  "rank_ic_series": np.array([20])}}
    assert bench.compare(good, good)["rank_ic"]["observation_counts"]["equal"]
    changed = {"metrics": good["metrics"],
               "observation_counts": {**good["observation_counts"], "rank_ic": np.array([19])}}
    try:
        bench.compare(good, changed)
    except AssertionError:
        pass
    else:
        raise AssertionError("observation count mismatch must fail parity")
