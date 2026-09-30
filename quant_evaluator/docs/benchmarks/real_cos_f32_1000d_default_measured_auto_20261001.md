# Default auto adoption: real COS F32, 1000 days

We tested 32 existing COS factors over 1000 dates and 5461 stocks, using
float64 values and Pearson IC, its series, standard deviation and IR.
The manifest digest is
`b2cf8709e68d0d2b3168fcf3a4ccbb207b4be0f1be510e77df01b9ddb42e6864`.
The adjacent JSON records source objects and execution receipts.

| Measurement | Seconds |
| --- | ---: |
| Source loading | 106.665844 |
| Calibration, including setup and six complete calls | 46.085238 |
| CPU steady-state median, two repetitions | 9.209069 |
| Strict CUDA steady-state median, two repetitions | 5.424864 |
| First calibration input fingerprint | 0.867599 |
| First calibration setup | 1.094613 |
| Explicit measured-auto cache hit | 6.309649 |
| Ordinary auto, with exact candidate verification | 6.316926 |

We alternated CPU/CUDA order, ran one warmup pair, and checked parity after
each of the three pairs. Counts, finite masks, artifacts and metric values
matched at `rtol=1e-10`, `atol=1e-12`. The saved calibrated-to-cache and
calibrated-to-ordinary comparisons also report no mismatch.

The ordinary-auto receipt identifies `measured_auto_exact_candidate`, policy
`measured_auto_registry_exact_v1`, profile `exact_content_process_local`, and
CUDA for each requested metric. The script required this receipt through
`--require-measured-route`; output parity alone would not pass the run.
The candidate used two repetitions and a 360-second calibration budget,
which confirms default adoption can use an accepted non-default measurement
policy. QE still recomputes the exact key with that original policy.

This experiment establishes adoption for this exact Pearson request. The
ordinary-auto time includes identity verification; the raw CUDA median does
not. Initial calibration adds replay cost, and new contents need new evidence.
These numbers do not certify the fastest route for other metrics or device
loads. Runtime peak process RSS was 6,843,880 KiB; this excludes GPU memory
and child-process peaks. The JSON does not inventory runtime packages or
source revision. The runtime source receipt describes disk files, not loaded
Python code; source drift or hot reload requires a fresh process.
