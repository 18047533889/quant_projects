# GPU average ranks without per-chunk host reads

We replaced group-ID/bincount rank aggregation with a device run-bound kernel
in `kernels/gpu/sorted_rank_runs.py`. The caller keeps the stable sort and
finite-value mask. A singleton value takes its 1-based sorted position;
a tied value uses binary searches for its run boundaries. The mean rank is
`(first_zero_based + upper_exclusive + 1) / 2`. Invalid positions remain NaN.
Distinct finite-level counts still come from the same sorted run starts.
This removes `int(gid.max())`, its host synchronization and group-bin arrays.
The existing conservative chunk budget remains in force.

We ran the full COS F32 public benchmark on 2586 dates and 5461 stocks,
requesting `rank_ic_series`. The script interleaves CPU, strict CUDA, auto,
auto, strict CUDA, CPU; each child performs a cold and warm call.

| Route | Warm call seconds, two children |
| --- | --- |
| CPU | 66.583379, 67.303929 |
| Strict CUDA | 17.673466, 17.478603 |
| Default auto | 17.632300, 17.393965 |

The script reported parity pass and one semantic config hash. Both auto
children used CUDA via the certified F32 singleton route, without calibration.
Reported GPU peak was 15,760,462,848 bytes. The adjacent
`real_cos_f32_rank_runs_stdout_20261001.json` records terminal summaries;
this run did not save the complete artifact/comparison report. We do not use
this abbreviated evidence as a new portable route certificate.

The earlier 2026-09-29 archive for the same manifest, shape and semantic
config hash reports strict-CUDA warm calls of 33.817993 and 33.419384 seconds.
Those are separate-date observations. CPU times also changed, so the archive
does not isolate the causal speedup under identical machine load.
A small old/new rank-kernel experiment showed exact output equality, but its
old implementation exceeded the 1 GiB pool reservation budget. That JSON
records the breach and cache contamination; it is exploratory evidence only.
The bounded reproduction helper defaults to eight rows and separate pools.

A subsequent eight-row, 5461-column experiment enforced a 1 GiB limit on
each independent CuPy pool. It completed six F32/F64 cases, with exact output
equality, one warmup and three timed calls per implementation:

| dtype / distribution | New GPU event median ms | Old GPU event median ms |
| --- | ---: | ---: |
| F32 unique | 0.399360 | 0.949248 |
| F32 17 tied levels | 0.387072 | 0.899040 |
| F32 constant | 0.396288 | 0.895904 |
| F64 unique | 0.386048 | 0.897024 |
| F64 17 tied levels | 0.387136 | 0.886784 |
| F64 constant | 0.379904 | 0.904256 |

Pool reservation after calls was 2.19–2.54 MB for the new implementation and
3.24–8.48 MB for the old one. These allocator statistics are not live peaks;
the microbenchmark also does not establish full-pipeline speedup.
