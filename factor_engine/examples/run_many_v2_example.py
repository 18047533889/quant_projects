"""Default durable batch call after an administrator approves the data profile.

Set FACTOR_ENGINE_V2_PROFILE to that existing profile. It must explicitly bind
the date range, universe, canonical HFQ source and isolated output directory.
This example neither invents that authorization nor publishes factors.

Usage from an existing factor-definition program::

    if __name__ == "__main__":
        receipt = materialize_all(all_factors)

No backend, concurrency, batch-size or memory tuning is required from callers.
The v2 implementation remains subject to its current evidence/admission gates.
The script main guard is required by spawn; it is not a performance option.
"""

from factor_engine import get_engine


def materialize_all(factors):
    with get_engine() as engine:
        return engine.run_many(factors)
