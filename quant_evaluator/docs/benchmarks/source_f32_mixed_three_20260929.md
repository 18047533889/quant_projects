# F32 factor-source mixed-three routing evidence (2026-09-29)

This certifies one exact `evaluate_factor_source_batch` request:
`rank_ic`, `quantile_spread`, and `factor_turnover_rate`, on a deterministic
on-demand synthetic float64 source with shape 2586 days × 5461 assets × 32
factors and 2-factor tiles. No COS objects or disk panel were read or written.

Machine-readable records:

- `synthetic_f32_source_mixed_three_aba_20260929.json`: CPU/CUDA/CPU.
- `synthetic_f32_source_mixed_three_auto_cuda_20260929.json`: strict CUDA/auto.

The source generates each factor from its own seeded random stream. All runs
use seed 442, identical axes and labels, and read exactly 16 tiles. A tile is
225,954,336 bytes; the complete factor tensor is never materialized. Before
the CPU/CUDA/CPU run, host available memory was about 45.4 GiB and L20 free
memory about 44.3 GiB. The first run's process RSS high-water was 0.97 GiB;
the full A/B/A sequence reached 1.73 GiB. CUDA peak reserved memory was
2,991,313,408 bytes.

| Request | Time (s) | Result |
| --- | ---: | --- |
| CPU | 164.579 | Reference |
| CUDA strict | 11.390 | All three outputs and observation counts match CPU |
| CPU repeat | 163.697 | Bit-identical to first CPU result |
| CUDA strict repeat | 11.478 | Strict CUDA comparison run |
| Auto | 10.967 | Routed to CUDA; matches strict CUDA exactly |

All outputs have 32 factor values. CPU/CUDA maximum absolute differences were
1.57e-18 for rank IC, 3.79e-18 for quantile spread, and 4.11e-15 for turnover;
all passed rtol=1e-8, atol=1e-10. Each metric's 32 observation counts matched
between CPU and CUDA. CPU repeated values were identical. Auto and strict CUDA
were exactly equal for all values and observation counts. All runs processed
16 source tiles. Auto reported
`bounded_f32_mixed_three_gpu` and used CUDA.

The source API auto gate is exact: float64 factor and label values, shape
2586×5461×32, effective source tile width 2, the three metric IDs above,
default precision policy, one NVIDIA L20, and at least 14 GiB both actually
free and available under the configured VRAM fraction. Other shapes, tile
widths, dtypes, metric sets, precision policies, hardware, or insufficient
free VRAM use CPU under `auto`, with the rejection reason in metadata.
Explicit `cuda_strict` requests retain their own validation and fail closed.

These times include regeneration of the synthetic factor tiles and do not
measure COS I/O or a real factor-source adapter. The result certifies the
three columnar metric outputs and observation counts for this request, not
the richer `evaluate()` artifacts, other workloads, or production factor
publication. The source snapshot ID is caller supplied; this benchmark does
not independently verify a COS manifest or factor provenance.
