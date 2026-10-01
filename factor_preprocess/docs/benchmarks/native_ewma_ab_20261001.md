# Native EWMA and research fallback benchmark

Run on 2026-10-01 at `qs-compute-gpu-hk-01`, the FE public EWMA route had lower median latency than `_ewma_fp_research` in every measured case. The 2-million-row run showed 2.13×–2.70× speedups. The 12.5-million-row run showed 2.58×–3.23× speedups. The results describe this host and software stack.

## Method

The benchmark compared the public `factor_preprocess.transforms.rolling.ewma` route with the explicit pandas research implementation, `_ewma_fp_research`. Both routes received the same long-form frame and `min_periods=1`. Each case used half-life 3.7 or 20, asset-major or time-major row order, and deterministic synthetic values. Missing-value cases used no NaNs, 1% sparse NaNs, all-NaN input, or a contiguous 20% gap. The large run used the first two patterns.

The small run covered 1,000 dates × 2,000 assets (2,000,000 rows) in 16 cases. The large run covered 2,500 dates × 5,000 assets (12,500,000 rows) in eight additional cases. Asset and date columns used int32; values used float64. The generator used seed 20261001 with a recorded case seed for each condition. The benchmark used synthetic data; it did not read COS data.

Each route ran once as a full-size warm-up. The benchmark then timed two pairs in opposite order, FE then FP and FP then FE. It checked full-output parity after each pair at `rtol=1e-12` and `atol=1e-12`. All 24 cases passed both parity checks. The 16 small cases also passed an independent pandas `Series.shift(1).ewm(adjust=False)` oracle on four assets per case. Maximum absolute FE/FP difference was `2.89e-15` in the small run and `3.34e-15` in the large run. Some results were numerically equal within tolerance but not bit-exact.

## Large-run timings

Each entry reports the median of the two measured calls for that route. The speedup column is FP median divided by FE median.

| Row order | Missing values | Half-life | FE route (s) | FP research (s) | FP / FE |
|---|---:|---:|---:|---:|---:|
| Asset-major | None | 3.7 | 1.399 | 3.850 | 2.75× |
| Asset-major | None | 20 | 1.451 | 3.891 | 2.68× |
| Asset-major | 1% sparse | 3.7 | 1.485 | 3.837 | 2.58× |
| Asset-major | 1% sparse | 20 | 1.394 | 3.820 | 2.74× |
| Time-major | None | 3.7 | 1.682 | 5.429 | 3.23× |
| Time-major | None | 20 | 1.756 | 5.670 | 3.23× |
| Time-major | 1% sparse | 3.7 | 1.778 | 5.557 | 3.13× |
| Time-major | 1% sparse | 20 | 1.687 | 5.390 | 3.20× |

The small-run speedup range was 2.13×–2.70× across its 16 cases. The receipts contain every warm-up and timed duration.

## Limits and reproduction

The runner refused to allocate an input frame above 512 MiB. It required at least 8 GiB of available RAM before small cases and 16 GiB before large cases. The measured pandas frame used 32,000,132 bytes at 2 million rows and 200,000,132 bytes at 12.5 million rows. These figures measure input-frame size, not process peak RSS; Polars plans, intermediate arrays, and outputs add memory during execution. The host was shared, so other processes could affect latency and available-memory readings. These measurements do not predict latency on other machines or under other workloads.

Run from the quant_projects root with fresh receipt paths. `--include-large` runs the small cases first, then the large cases.

```sh
python factor_preprocess/scripts/benchmark_native_ewma.py \
  --only-small --max-input-mib 512 \
  --min-available-gib-small 8 --min-available-gib-large 16 \
  --seed 20261001 --receipt /tmp/ewma-small-run.json

python factor_preprocess/scripts/benchmark_native_ewma.py \
  --include-large --max-input-mib 512 \
  --min-available-gib-small 8 --min-available-gib-large 16 \
  --seed 20261001 --receipt /tmp/ewma-large-run.json
```

The run used Python 3.12.3, NumPy 2.2.6, pandas 2.3.3, and Polars 1.42.1 on Linux `qs-compute-gpu-hk-01`.

## Evidence identity

Both receipts recorded status `complete`, HEAD `8c6d114c3887c6127308d58288bf4304b5888663`, and matching source hashes before and after the run.

| Receipt | Cases | SHA-256 |
|---|---:|---|
| [`native_ewma_small_20261001.json`](native_ewma_small_20261001.json) | 16 | `84181ecd68c35258b1dc45205548c87dd0e1dd591e4fa1c6b63ed933f5e47b0e` |
| [`native_ewma_large_20261001.json`](native_ewma_large_20261001.json) | 24 | `f2a745000ae1d6ccc09d6cfa48869f539af6f575d7a22cd682fe4e464a6edd96` |
