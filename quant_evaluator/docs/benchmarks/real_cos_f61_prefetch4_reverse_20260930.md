# Real COS F61: four-worker prefetch, reverse-order repeat

Evidence: `real_cos_f61_prefetch4_reverse_pearson_ab_20260930.json`. This
repeats the four-worker COS-prefetch experiment with CPU before strict CUDA,
then auto. Comparators are the two-worker consuming run
`real_cos_f61_consuming_source_pearson_ab_20260930.json` and the first
four-worker run `real_cos_f61_prefetch4_pearson_ab_20260930.json`.

All three reports are complete and use the same 2586 × 5461 × 61 float64
workload, four metrics, manifest, factor IDs, source request identity and
snapshot. Each report has four actual metric entries under
`comparison.metrics` and `auto_comparison.metrics`; none are absent. Across all
three reports, the per-backend value hashes are identical:

| Metric | CPU value SHA256 | CUDA value SHA256 |
| --- | --- | --- |
| `pearson_ic` | `645c9886a22996b48c440716ea3d3e0c183d421d7deaf2bfe51ff5a186e94a88` | `c419b6074992c6a5ef2042298b15d6d3d86f66f251c7ee4e98db8b6959b84747` |
| `pearson_ic_series` | `8897d79222d0a81ca40a94b7e280774592a0d6a2e31f79c6c06871738ffeffb4` | `1c3694a7658e9e423e3ceb92ffb7343467d0c5ea87b883ece0d643904f889c8e` |
| `pearson_ic_std` | `d35371018387cde67db964b541dfc72f82df248777e9d9d1545f336c20c89f95` | `523668fbc1ffceaf9fe43a3e4793365a1a72b63aa3f7d4744127b968beaa3191` |
| `pearson_ic_ir` | `f7d18b62dc035b6b5bb65781242f1809cd963366911e695856075598dcf044bd` | `cba4a2f7579d8ef6257fd5f316298d3fb03abca0b3e365579d6a24576f8bdad5` |

CPU and CUDA hashes differ from each other, but each backend's hashes are stable
across the three runs. CPU/CUDA comparisons pass for all 157,929 elements
(157,746 series values and 183 scalar values); finite masks and observation
counts match for every metric. There are 157,283 finite values. Maximum
absolute difference is 8.3267e-16. The auto comparison also passes for all four
metrics; auto selected CUDA via
`bounded_f61_pearson_chain_gpu_tile16`, completed four tiles, and used a
four-object prefetch window. Auto runs do not publish separate value-hash
fields in this JSON schema.

| Run | CPU seconds / source-read seconds | CUDA seconds / source-read seconds | Auto seconds / source-read seconds |
| --- | ---: | ---: | ---: |
| 2 workers (CPU → CUDA → auto) | 134.4443 / 88.5440 | 90.4782 / 87.7460 | 89.8809 / 87.9541 |
| 4 workers, first (CUDA → CPU → auto) | 96.1931 / 51.0144 | 60.6541 / 57.7930 | 56.4398 / 54.6353 |
| 4 workers, reverse (CPU → CUDA → auto) | 118.9734 / 73.6898 | 57.3536 / 54.5605 | 53.8077 / 51.8387 |

The reverse run confirms the same result hashes in the opposite CPU/CUDA order.
Its CPU time and source-read time are slower than the first four-worker run;
CUDA and auto times are close, with shorter source-read time. These are three
ordered measurements, not randomized repeated trials, so cache and IO
variation remain plausible contributors. Source-read time is reported
separately from whole-request time because source IO dominates much of these
runs.

The logical peak source estimate is 11,467,641,280 bytes under the 12 GiB
(12,884,901,888-byte) source cap. Highest recorded process peak RSS across the
reverse run's CPU, CUDA and auto phases is 13,818,896 KiB (~13.18 GiB); this is
an observed process peak, not the sum of phase peaks or the logical source cap.
Preflight passed before CPU/CUDA and before auto. Disk was 98% used with 45 GB
available when this note was prepared; no benchmark was rerun for this review.
