# Backend capability matrix

Generate the current matrix from the final runtime registry:

```bash
python factor_engine/scripts/sync_backend_docs.py
```

The generated report separates backend registration, native non-bridge implementation, parity verification, and production-safe routing. Registration alone is not production certification. CI verifies that repeated generation is deterministic.
