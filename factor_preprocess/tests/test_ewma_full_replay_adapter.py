import json
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from factor_preprocess.adapters.ewma_full_replay import (
    FrozenEwmaReplaySpec,
    apply_frozen_ewma_full_replay,
)
import factor_preprocess.adapters.ewma_full_replay as replay_module
from factor_preprocess.transforms import ewma


def _frame(n=24, *, assets=("A", "B")):
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    rows = []
    for j, asset in enumerate(assets):
        for i, date in enumerate(dates):
            value = float(i + 10 * j)
            if i in (4, 11):
                value = np.nan
            rows.append((asset, date, value))
    return pd.DataFrame(rows, columns=["asset_id", "date", "value"])


def _spec(**changes):
    values = dict(
        state_key="factor:price:ewma", source_identity="snapshot:v1",
        feature_identity="feature:v1", security_identity="permanent-id:v1",
        time_identity="calendar:utc:v1", halflife=4.0, min_periods=3,
        max_history_rows=1000,
    )
    values.update(changes)
    return FrozenEwmaReplaySpec(**values)


def _run_chunks(frame, sizes, path):
    outputs = []
    dates = list(frame["date"].drop_duplicates())
    pos = 0
    for index, size in enumerate(sizes):
        chosen = dates[pos:pos + size]
        if not chosen:
            break
        segment = frame[frame["date"].isin(chosen)].copy()
        out, receipt = apply_frozen_ewma_full_replay(
            segment, spec=_spec(), checkpoint_path=path, bootstrap=index == 0,
        )
        assert receipt["execution_mode"] == "FULL_REPLAY_ONLY"
        outputs.append(pd.Series(out.to_numpy(), index=segment.index))
        pos += size
    return pd.concat(outputs).sort_index()


def test_public_full_daily_random_chunk_and_restart_oracle(tmp_path):
    frame = _frame()
    expected = ewma(frame, halflife=4.0, min_periods=3)
    daily = _run_chunks(frame, [1] * 24, tmp_path / "daily.json")
    random_chunks = _run_chunks(frame, [3, 7, 2, 5, 7], tmp_path / "chunks.json")
    # Each call re-opens the durable JSON checkpoint, exercising restart state.
    np.testing.assert_allclose(daily, expected, equal_nan=True)
    np.testing.assert_allclose(random_chunks, expected, equal_nan=True)


def test_future_poison_and_new_asset_do_not_change_history(tmp_path):
    frame = _frame()
    prefix = frame[frame["date"] < pd.Timestamp("2024-01-15", tz="UTC")]
    expected = ewma(prefix, halflife=4.0, min_periods=3)
    poisoned = frame.copy()
    poisoned.loc[poisoned["date"] >= pd.Timestamp("2024-01-15", tz="UTC"), "value"] = 1e12
    extra = _frame(24, assets=("NEW",))
    poisoned = pd.concat([poisoned, extra], ignore_index=True)
    actual = ewma(poisoned, halflife=4.0, min_periods=3)
    lookup = pd.Series(actual.to_numpy(), index=pd.MultiIndex.from_frame(poisoned[["asset_id", "date"]]))
    keys = pd.MultiIndex.from_frame(prefix[["asset_id", "date"]])
    np.testing.assert_allclose(lookup.reindex(keys), expected, equal_nan=True)

    path = tmp_path / "new-asset.json"
    first_dates = list(frame["date"].drop_duplicates())[:12]
    later_dates = list(frame["date"].drop_duplicates())[12:]
    first = frame[frame["date"].isin(first_dates)]
    later = frame[frame["date"].isin(later_dates)]
    apply_frozen_ewma_full_replay(first, spec=_spec(), checkpoint_path=path, bootstrap=True)
    combined_later = pd.concat([later, extra[extra["date"].isin(later_dates)]], ignore_index=True)
    out, _ = apply_frozen_ewma_full_replay(combined_later, spec=_spec(), checkpoint_path=path)
    expected_existing = ewma(frame, halflife=4.0, min_periods=3).loc[later.index]
    actual_existing = out.loc[combined_later["asset_id"].isin(["A", "B"])]
    np.testing.assert_allclose(actual_existing, expected_existing, equal_nan=True)


@pytest.mark.parametrize("field", [
    "state_key", "source_identity", "feature_identity", "security_identity",
    "time_identity", "halflife", "min_periods", "max_history_rows",
])
def test_resume_rejects_changed_frozen_identity(tmp_path, field):
    frame = _frame(4, assets=("A",))
    path = tmp_path / "state.json"
    apply_frozen_ewma_full_replay(frame.iloc[:2], spec=_spec(), checkpoint_path=path, bootstrap=True)
    changed = 8.0 if field == "halflife" else (7 if field in {"min_periods", "max_history_rows"} else f"changed:{field}")
    with pytest.raises(ValueError, match="identity mismatch"):
        apply_frozen_ewma_full_replay(frame.iloc[2:], spec=_spec(**{field: changed}), checkpoint_path=path)


def test_non_append_revision_integrity_and_budget_fail_closed(tmp_path):
    frame = _frame(5, assets=("A",))
    path = tmp_path / "state.json"
    apply_frozen_ewma_full_replay(frame.iloc[:3], spec=_spec(), checkpoint_path=path, bootstrap=True)
    with pytest.raises(ValueError, match="non-append revision"):
        apply_frozen_ewma_full_replay(frame.iloc[2:4], spec=_spec(), checkpoint_path=path)
    payload = json.loads(path.read_text())
    payload["rows"][0][2] = 999.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="integrity mismatch"):
        apply_frozen_ewma_full_replay(frame.iloc[3:4], spec=_spec(), checkpoint_path=path)

    budget_path = tmp_path / "budget.json"
    with pytest.raises(ValueError, match="budget exceeded"):
        apply_frozen_ewma_full_replay(
            frame, spec=_spec(max_history_rows=4), checkpoint_path=budget_path, bootstrap=True,
        )
    assert not budget_path.exists()


@pytest.mark.parametrize(
    "change",
    [
        {"halflife": True}, {"halflife": float("inf")}, {"halflife": float("nan")},
        {"min_periods": True}, {"min_periods": 1.5},
        {"max_history_rows": False}, {"max_history_rows": 2.5},
        {"source_identity": None}, {"feature_identity": 7},
    ],
)
def test_spec_is_strict(change):
    with pytest.raises(ValueError):
        _spec(**change)


def test_explicit_utc_and_kernel_implementation_identity_are_frozen(tmp_path, monkeypatch):
    naive = _frame(2, assets=("A",))
    naive["date"] = naive["date"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="UTC timezone-aware"):
        apply_frozen_ewma_full_replay(
            naive, spec=_spec(), checkpoint_path=tmp_path / "naive.json", bootstrap=True,
        )
    non_utc = _frame(2, assets=("A",))
    non_utc["date"] = non_utc["date"].dt.tz_convert("Asia/Hong_Kong")
    with pytest.raises(ValueError, match="timezone must be UTC"):
        apply_frozen_ewma_full_replay(
            non_utc, spec=_spec(), checkpoint_path=tmp_path / "hk.json", bootstrap=True,
        )

    frame = _frame(4, assets=("A",))
    path = tmp_path / "kernel.json"
    apply_frozen_ewma_full_replay(frame.iloc[:2], spec=_spec(), checkpoint_path=path, bootstrap=True)
    monkeypatch.setattr(replay_module, "KERNEL_IMPLEMENTATION_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="identity mismatch"):
        apply_frozen_ewma_full_replay(frame.iloc[2:], spec=_spec(), checkpoint_path=path)


def test_competing_subprocess_writers_cannot_lose_history(tmp_path):
    frame = _frame(4, assets=("A",))
    path = tmp_path / "race.json"
    apply_frozen_ewma_full_replay(frame.iloc[:2], spec=_spec(), checkpoint_path=path, bootstrap=True)
    code = textwrap.dedent("""
        import sys
        import pandas as pd
        from factor_preprocess.adapters.ewma_full_replay import FrozenEwmaReplaySpec, apply_frozen_ewma_full_replay
        spec = FrozenEwmaReplaySpec(state_key='factor:price:ewma', source_identity='snapshot:v1', feature_identity='feature:v1', security_identity='permanent-id:v1', time_identity='calendar:utc:v1', halflife=4.0, min_periods=3, max_history_rows=1000)
        frame = pd.DataFrame([('A', pd.Timestamp('2024-01-03', tz='UTC'), 2.0), ('A', pd.Timestamp('2024-01-04', tz='UTC'), 3.0)], columns=['asset_id','date','value'])
        try:
            apply_frozen_ewma_full_replay(frame, spec=spec, checkpoint_path=sys.argv[1])
        except Exception as exc:
            print(type(exc).__name__ + ':' + str(exc))
            raise SystemExit(3)
    """)
    processes = [subprocess.Popen([sys.executable, "-c", code, str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    results = [process.communicate(timeout=20) + (process.returncode,) for process in processes]
    assert sorted(result[2] for result in results) == [0, 3], results
    failure = next(result for result in results if result[2] == 3)
    assert "concurrent writer" in failure[0] or "non-append revision" in failure[0]
    payload = json.loads(path.read_text())
    assert len(payload["rows"]) == 4
