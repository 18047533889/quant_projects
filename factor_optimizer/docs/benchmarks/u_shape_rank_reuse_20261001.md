# U-shape TRAIN rank reuse benchmark (2026-10-01)

## Result

Across four synthetic panel sizes, cached rank reuse was slower at 76,800 observed TRAIN rows and faster in all three tested sizes from 500,000 to 1,500,000 TRAIN rows. The smallest tested size with a speedup was 500,000 rows; the exact crossover between 76,800 and 500,000 was not measured. Reuse should remain gated rather than unconditional.

| Full panel (dates × assets) | Observed TRAIN frame rows | Cached median | Uncached median | Cached elapsed change | Median peak RSS cached / uncached |
|---|---:|---:|---:|---:|---:|
| 200 × 640 (128k cells) | 76,800 | 1.444 s | 1.267 s | +14.0% slower | 363.6 / 321.0 MiB (+42.6 MiB) |
| 167 × 5,000 (835k cells) | 500,000 | 4.206 s | 4.617 s | 8.9% faster | 587.1 / 537.9 MiB (+49.2 MiB) |
| 334 × 5,000 (1.67m cells) | 1,000,000 | 7.715 s | 9.468 s | 18.5% faster | 914.3 / 861.8 MiB (+52.5 MiB) |
| 500 × 5,000 (2.5m cells) | 1,500,000 | 11.085 s | 14.669 s | 24.4% faster | 1,135.4 / 1,090.3 MiB (+45.1 MiB) |

Each row is the median of three serial cached/uncached pairs. Order alternated cached/uncached, uncached/cached, cached/uncached. All 24 persisted runs produced identical candidate-evidence digests within each panel and every pair passed digest parity. The digest for each panel was:

- 200 × 640: `f72610a2837a5ebaf02e0da6c995a316043813facfb1da602df3bc9df048ee52`
- 167 × 5,000: `7c3ab5efd44cf8d21878cc98523de70175d26ceeddd42f7b3484e19c8b8f1c17`
- 334 × 5,000: `f673946be7211ac034607953a1aef6957e1244e39e2e12ea4ca962af7469849e`
- 500 × 5,000: `ce9a746c1963ed9ae05853ddc4e09b1fe3452d47ca0db80722d07946b2411fc1`

Every cached run recorded 13 actual cache hits, one miss, and one rank calculation, with zero bypasses or evictions. Retained rank payloads were 614,400; 4,000,000; 8,000,000; and 12,000,000 bytes for the four sizes. Uncached runs recorded zero cache hits and rank calls. Thus the existing 128 MiB cache cap was not approached: the largest retained payload was 12 MB. The associated process peak RSS still increased by about 43–53 MiB at the tested sizes.

## Observed TRAIN rows and evidence

TRAIN row counts come from frames actually observed at the reuse path, not just the split formula. In cached runs, all 14 frame-fingerprint observations and the FE-rank input had the same observed row count for that panel. In uncached runs, the first 14 direct rank-shape FP executions all had the same row count; later validation/full materialization frames were recorded separately. For every run, observed rows matched the split-derived count: 76,800; 500,000; 1,000,000; and 1,500,000. The matrix runner rejects records if actual frame lengths disagree with the expected count or split.

The durable evidence consists of 24 compact JSON records under [`u_shape_rank_reuse_20261001_runs`](u_shape_rank_reuse_20261001_runs/); the largest record is 3,316 bytes. Each record stores its exact panel and optimizer seeds, full optimization configuration, input shape and observed frame rows, elapsed time, peak RSS, candidate digest, cache counters, source hashes/revision, Python/NumPy versions, and host load/memory snapshot. The panel seed was 20261001; the optimizer seed was 73. The panel was generated with NumPy `default_rng`/PCG64 standard-normal values and squared-value labels; no project market data was read.

The source identity in all records is project HEAD `e336523a4400884ee77791536dbfd14de7d27497`, benchmark CLI SHA-256 `86df6c12072d8c4aeeb9cfafc000101075adfc26943fb1aa24a7aa65de19eda1`, and matrix runner SHA-256 `4cf0de078a5641a7636f5d89a24e08fc4d0b75aa0930c366a5490b9a0d41fc48`. Each JSON includes the hashes of the relevant implementation and optimizer files and marks the relevant working tree state.

## Method and limits

The serial matrix runner [`run_u_shape_rank_reuse_matrix.py`](../../scripts/run_u_shape_rank_reuse_matrix.py) executed one mode per process across 200×640, 167×5,000, 334×5,000, and 500×5,000 input panels. It refuses to overwrite an existing non-empty run directory, writes one record per invocation, verifies actual TRAIN-frame lengths and evidence parity, and emits medians without creating another artifact. The harness rejects panels above 3,000,000 cells.

The recorded environment was Python 3.12.3, NumPy 2.2.6, and 32 CPUs. Available memory was about 66.4 GiB across the run; 1-minute host load snapshots ranged from 3.36 to 6.18. Load is host-wide and cannot be attributed to this benchmark. Maximum process peak RSS was 1,135.75 MiB. These are synthetic, single-host research measurements, not production performance or admission evidence.

## Candidate-count sensitivity and default admission

A second serial matrix measured U-only, inverted-U-only, and combined searches at 500,000 and 1,000,000 actual TRAIN-frame rows. Each profile has three alternating cached/uncached pairs. All 18 pairs produced identical candidate-evidence digests, winners, TRAIN gain, and validation lower bounds. The 36 compact records reside in [`u_shape_rank_reuse_candidate_sensitivity_20261001_runs`](u_shape_rank_reuse_candidate_sensitivity_20261001_runs/).

| Distinct eligible plans | TRAIN rows | Cached median | Uncached median | Cached elapsed change | Extra median peak RSS |
|---:|---:|---:|---:|---:|---:|
| 8 | 500,000 | 3.409091 s | 3.346966 s | +1.856% | 46.74 MiB |
| 8 | 1,000,000 | 5.965029 s | 6.709277 s | -11.093% | 74.41 MiB |
| 6 | 500,000 | 2.479573 s | 2.146877 s | +15.497% | 43.65 MiB |
| 6 | 1,000,000 | 4.137386 s | 4.372840 s | -5.384% | 69.53 MiB |
| 14 | 500,000 | 4.212285 s | 4.700871 s | -10.394% | 49.13 MiB |
| 14 | 1,000,000 | 7.722141 s | 9.560958 s | -19.233% | 45.22 MiB |

The records bind HEAD `c46d1cb1a1a743a131acdaa539f5eaa57f2a571b` plus hashes of the uncommitted cache implementation and harness at measurement time. HEAD alone does not reconstruct those measurements. These synthetic panels do not certify speed on market factors or other hardware.

The research optimizer now admits its invocation-local rank cache when `(actual_train_rows >= 500_000 and distinct_eligible_plans >= 14)` or `(actual_train_rows >= 1_000_000 and distinct_eligible_plans >= 6)`. It counts valid, unwrapped U/inverted-U rank-shape plan identities after candidate-budget admission; orientation duplicates count once. It measures the actual post-baseline TRAIN search frame, including any required prefix history. Smaller searches use the ordinary FE-backed execution path without cache hashing. The 128 MiB rank payload limit still applies, and validation/final materialization do not use this cache.

The helper module `factor_optimizer.shape_rank_reuse` owns eligibility, distinct counting and admission. `research_batch` uses those helpers before its TRAIN loop. A disabled cache bypasses fingerprinting and ranking even for an empty frame. The cache retains an immutable rank feature for one invocation and verifies the input/context fingerprint before reuse.

The benchmark harness forces cache admission in `cached` mode and forces rejection in `uncached` mode to measure both alternatives, including profiles that the default gate rejects. New records mark `production_gate_bypassed`; the harness restores its patches on exit. The optimizer itself uses the gate above. A/B parity proves the tested output comparisons, not a universal fastest-backend guarantee.
