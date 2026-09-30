# Temporal axis fingerprint regression (2026-10-01)

A real DataAccess/COS F32 materialized batch loaded successfully with shape
2586 × 5461 × 32 (125.146 seconds source loading). Before either backend ran,
calibration failed on its datetime64 time coordinates:

```text
ValueError: cannot include dtype 'M' in a buffer
```

The direct `memoryview(contiguous)` operation cannot expose NumPy temporal
dtype buffers. The fingerprint now takes a zero-copy uint8 view of each
bounded contiguous chunk before hashing. The original dtype and full logical
shape remain in the digest prefix, so temporal units/endian differences are
not erased. Chunking, object-scalar canonicalization and input immutability
are unchanged.

Regression tests compare exact logical C bytes for datetime64[D]/[ns], NaT
scalars, timedelta64[h], strided/empty date arrays and non-native-endian
integers. A public CPU/CUDA calibrated request with explicit date and asset
coordinates passes parity and produces a fresh cache-hit bundle. The combined
calibration/public/fork suite passed 60 tests. This does not by itself certify
the complete real F32 calibration; that experiment must be rerun separately.
