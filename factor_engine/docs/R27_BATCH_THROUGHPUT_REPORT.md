# R27 Batch Throughput Report

- HEAD: 956e6cd61d52cdead98bf85a372c7b4629a4cdb9 dirty=True
- scheduler probe: {'ok': True, 'wall_s': 0.4, 'results': ['a', 'b', 'c'], 'done': 4, 'broker_stage': 'NORMAL', 'scheduler_explanations': 4}
- hard gates: 31/31 passed

## Lightweight throughput (synthetic 20-day 2-instrument panel)

- factors: 20
- serial_wall_s: 0.183
- run_many_parallel_wall_s: 0.862
- fast_compute_wall_s: 0.997
- fast_factors_per_min: 1203.4
- fast_results: 20
- fast_done: 34
- fast_broker_stage: NORMAL

## Hard gates (R27-230..260)

- **R27_ADAPTIVE_SHARD_SIZE=True** — adaptive shard size present
- **R27_AS_COMPLETED_STREAM_MATERIALIZE=True** — as_completed streaming present
- **R27_AUTO_SHARD_SEMANTIC_SAFE=True** — semantic shard legality present
- **R27_BOUNDED_WRITE_BACKPRESSURE=True** — bounded write queue + backpressure present
- **R27_CACHE_BENEFIT_DENSITY=True** — benefit-density style reuse scoring present
- **R27_COLUMN_OVERLAP_NOT_FALSE_DEPENDENCY=True** — column overlap no longer blocks parallelism
- **R27_COMPUTE_WRITE_OVERLAP=True** — compute/write pipeline overlap present
- **R27_CPU_TOKEN_ADMISSION=True** — cpu token allocator present
- **R27_DATAACCESS_SCAN_COST_INTEGRATED=True** — DataAccess ScanCost public API bridge present
- **R27_DISK_FULL_ZERO=True** — spill free-disk guard prevents disk-full
- **R27_DOUBLE_CACHE_BUDGET_CLOSED=True** — global resource broker budgets cache layers
- **R27_DYNAMIC_MEMORY_HEADROOM=True** — live memory headroom present
- **R27_EXTERNAL_WORKLOAD_AWARE=True** — external workload aware pressure stage present
- **R27_GLOBAL_PREFETCH_UNION_REMOVED=True** — wave-based prepare (no full-batch union) present
- **R27_HYBRID_THREAD_PROCESS_EXECUTOR=True** — hybrid executor present
- **R27_IO_TOKEN_ADMISSION=True** — io token allocator present
- **R27_LARGE_BATCH_THROUGHPUT_BENCH_PASS=True** — batch throughput benchmark present
- **R27_MEMORY_TOKEN_ADMISSION=True** — memory token admission (ResourceBroker) present
- **R27_NATIVE_MULTI_ROOT_FUSION=True** — native fusion grouping present
- **R27_NESTED_PARALLELISM_CLOSED=True** — nested parallelism env closure present
- **R27_OOM_ZERO=True** — scheduler probe ran without OOM
- **R27_OPERATOR_COST_CALIBRATED=True** — operator cost calibrated peak present
- **R27_PIT_SEMANTICS_PRESERVED=True** — fast path reuses same compile→execute→materialize chain
- **R27_PROCESS_FAMILY_MEMORY_ACCOUNTED=True** — process family RSS accounted
- **R27_READ_WAVE_MEMORY_BOUNDED=True** — read wave planner present
- **R27_SHARD_FULLRUN_EQUIVALENCE=True** — shard equivalence test present
- **R27_SHARED_NODES_PARALLELIZED=True** — shared nodes parallel materialization present
- **R27_SPILL_TOKEN_ADMISSION=True** — spill token admission present
- **R27_SPILL_VS_RECOMPUTE_DECISION=True** — spill-vs-recompute calibration + cheap-operator contract present
- **R27_THREAD_PROCESS_SERIAL_EQUIVALENCE=True** — fast-vs-serial equivalence test present
- **R27_TRUE_DAG_NODE_SCHEDULER=True** — PhysicalFactorDAG + AdaptiveBatchScheduler present
