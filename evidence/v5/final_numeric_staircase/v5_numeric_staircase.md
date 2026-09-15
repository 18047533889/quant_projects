# V5 QE numerical staircase

Run (UTC): `2026-09-07T16:52:52Z`

> Scope warning: this is a small-panel bounded-memory validation with T=4 and N=20. It is not a production-throughput benchmark and must not be extrapolated to production panel sizes.

Every shard used the public `EvaluationRequest`/`evaluate` path through `QEJobHandler`, with atomic result-sink writes, then released per-shard objects.

| Factors | Shards | factor×instance done | Failed | Failure rate | Seconds | factor×instance/s | Peak RSS MiB | Peak delta MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 4 | 2,000 | 0 | 0.000000% | 1.762 | 1134.8 | 163.5 | 3.5 |
| 10,000 | 40 | 20,000 | 0 | 0.000000% | 21.094 | 948.1 | 164.2 | 0.8 |
| 100,000 | 391 | 200,000 | 0 | 0.000000% | 180.430 | 1108.5 | 170.5 | 1.8 |

## Configuration

```json
{
  "N": 20,
  "T": 4,
  "batch_max": 256,
  "instances": [
    {
      "cost_profile": "gross",
      "horizon": 1,
      "leg": "unspecified",
      "metric_id": "rank_ic",
      "metric_version": "3.0.0",
      "output_mode": "FULL_DIAGNOSTIC",
      "parameters": {
        "min_assets": 10,
        "min_periods": 1
      },
      "portfolio_profile": "default",
      "price_convention": "vwap_to_vwap",
      "quantile_builder_parameters": {},
      "scenario_id": "scale",
      "usage_profile": "research"
    },
    {
      "cost_profile": "gross",
      "horizon": 1,
      "leg": "unspecified",
      "metric_id": "coverage",
      "metric_version": "0.1.0",
      "output_mode": "FULL_DIAGNOSTIC",
      "parameters": {
        "min_assets": 10
      },
      "portfolio_profile": "default",
      "price_convention": "vwap_to_vwap",
      "quantile_builder_parameters": {},
      "scenario_id": "scale",
      "usage_profile": "research"
    }
  ],
  "path": "public EvaluationRequest/evaluate via QEJobHandler atomic sink",
  "production_throughput_claim": false,
  "totals": [
    1000,
    10000,
    100000
  ]
}
```

## Hardware

```json
{
  "hostname": "qs-compute-gpu-hk-01",
  "logical_cpus": 32,
  "memory": {
    "MemTotal": "96796196 kB",
    "SwapTotal": "2035708 kB"
  },
  "numpy": "2.2.6",
  "platform": "Linux-6.8.0-124-generic-x86_64-with-glibc2.39",
  "processor": "AMD EPYC 9K84 96-Core Processor",
  "python": "3.12.3",
  "thread_env": {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "POLARS_MAX_THREADS": "2"
  }
}
```
