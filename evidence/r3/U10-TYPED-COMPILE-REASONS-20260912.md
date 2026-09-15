# U10 typed per-factor compile failures — 2026-09-12

Scope: `factor_engine/runtime/bounded_pipeline.py` compile-worker protocol and its main/refill consumers only. Existing purpose, artifact-plan/receipt, resume identity, reader and copy-lifecycle changes were preserved.

The compile worker now returns an exact per-factor typed envelope:

```text
{code, error_type, message, retryable}
```

`code` prefers the exception's explicit `reason_code`, but accepts only 1–64 uppercase ASCII letters/digits/underscore beginning with an uppercase letter. The worker protocol also bounds `error_type` to 128 characters and `message` to 4,096 characters. Invalid or absent codes become `INVALID_FACTOR_COMPILE`; they cannot inject retry flags or protocol fields.

This preserves concrete deterministic reasons including `KNOWN_BAD_IMPLEMENTATION`, `DATA_SOURCE_MISSING`, and `UNKNOWN_FIELD`. `CatalogUnavailable`/registry failures have no such source-missing code and therefore are not misclassified as missing data. Only the existing `ResourceAdmissionError` and `TransientIOError` taxonomy classes mark a compile envelope retryable; all attempts remain subject to the persisted finite run budget.

Main-wave and refill consumers terminalize each factor separately and preserve the typed code. Valid peers continue. Speculative prefetch validates the same envelope but does not own terminal state; a failed speculative wave is reclaimed by the ordinary path exactly once, preventing duplicate terminal writes and attempt inflation.

Verification:

```text
.venv/bin/python_p3_12 -m pytest -q \
  factor_engine/tests/runtime/test_v7_pipeline_integrity.py \
  factor_engine/tests/runtime/test_v8_continuous_refill_integrity.py
23 passed, 5 warnings in 21.60s
```

The warnings are existing multiprocessing fork deprecation warnings. Focused tests prove exact typed preservation for a good factor, known-bad binding and missing source; malformed code/envelopes are rejected; refill compile rejection spends no execution attempt and later valid work completes.
