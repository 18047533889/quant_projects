"""Independent pointwise oracle and bounded synthetic PIT scaling probe."""
import hashlib
import json
import time
import resource
from pathlib import Path

import numpy as np
import pandas as pd
from factor_engine.pit_contract import select_visible_row_bundles, AvailabilityPrecision


def execute(decisions, events):
    return select_visible_row_bundles(
        decisions, events, available_policy="same_day",
        precision=AvailabilityPrecision.TIMESTAMP_SECOND,
    )


def oracle_probe():
    rng = np.random.default_rng(90206)
    n = 600
    events = pd.DataFrame({
        "instrument": rng.choice(["A", "B", "C"], n),
        "period_end": pd.Timestamp("2000-01-01", tz="UTC") + pd.to_timedelta(rng.integers(0, 500, n), unit="D"),
        "available_at": pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(rng.integers(0, 100, n), unit="h"),
        "revision_id": np.arange(n),
        "value": rng.normal(size=n),
    }).sample(frac=1, random_state=7)
    decisions = pd.DataFrame({
        "instrument": rng.choice(["A", "B", "C", "missing"], 400),
        "decision_timestamp": pd.Timestamp("2025-01-01", tz="UTC") + pd.to_timedelta(rng.integers(-5, 110, 400), unit="h"),
    })
    actual = execute(decisions, events)
    expected = []
    for row in decisions.itertuples(index=False):
        visible = events[(events.instrument == row.instrument) & (events.available_at <= row.decision_timestamp)]
        if visible.empty:
            expected.append(np.nan)
        else:
            latest = visible.loc[visible.period_end == visible.period_end.max()]
            expected.append(latest.sort_values(["available_at", "revision_id"], kind="stable").iloc[-1].value)
    np.testing.assert_allclose(pd.to_numeric(actual.value, errors="coerce").to_numpy(dtype=float), expected, rtol=0, atol=0, equal_nan=True)
    # Poison events strictly after every decision must not alter any output.
    poison = events.iloc[:20].copy()
    poison["available_at"] = pd.Timestamp("2030-01-01", tz="UTC")
    poison["revision_id"] += n
    poisoned = execute(decisions, pd.concat([events, poison], ignore_index=True))
    pd.testing.assert_frame_equal(actual, poisoned)
    return {"decisions": len(decisions), "events": n, "exact_oracle": "PASS", "future_poison": "PASS"}


def scale(n):
    events = pd.DataFrame({
        "instrument": ["A"] * n,
        "period_end": pd.date_range("2000-01-01", periods=n, tz="UTC"),
        "available_at": pd.date_range("2025-01-01", periods=n, freq="s", tz="UTC"),
        "revision_id": np.arange(n), "value": np.arange(n, dtype=float),
    })
    decisions = pd.DataFrame({
        "instrument": ["A"] * (n * 4),
        "decision_timestamp": pd.date_range("2025-01-01", periods=n * 4, freq="s", tz="UTC"),
    })
    start = time.perf_counter()
    result = execute(decisions, events)
    seconds = time.perf_counter() - start
    np.testing.assert_array_equal(result.value.to_numpy(), np.minimum(np.arange(n * 4), n - 1))
    return {"events": n, "decisions": n * 4, "seconds": seconds, "exact_oracle": "PASS"}


if __name__ == "__main__":
    import factor_engine.pit_contract as module
    result = {"oracle": oracle_probe(), "scales": [scale(1000), scale(4000)],
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "source_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
              "real_market_data": "NOT_RUN"}
    print(json.dumps(result, indent=2))
