# R50 bounded shared-DAG regression

Added a 257-factor synthetic run_many sink regression on six observations.
The test forces a 32-factor wave boundary to exercise nine waves, uses two worker slots, and verifies each output immediately in the sink rather than retaining all output panels.

Assertions:
- all 257 distinct named outputs match independent rolling-mean-plus-offset values;
- nine waves complete;
- the shared ts_mean kernel is called exactly once;
- transient cross-wave cache and associated resource leases are released.

Full cross-wave suite: 16 passed, evidence/r50-257-shared-factor-root.log.
Sampled family RSS peak 807854080 bytes, elapsed watchdog time 60.36s (includes imports/setup and all 16 tests, NOT kernel throughput). Sampled guard is not a hard memory cap.

This is a synthetic cache/wave regression, not a 110k-real-factor run, production acceptance, GPU benchmark, or proof of optimal performance. No production artifacts written.
