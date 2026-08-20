"""LabelBundle causal timing chain validation tests (QE-LABEL-TIMING).

Covers the enforced chain:
    decision_time <= signal_available_time <= execution_time <= label_start_time < label_end_time
"""

import numpy as np
import pandas as pd
import pytest

from quant_evaluator.contracts.label_bundle import LabelBundle


def _consistent_ints():
    return dict(
        target_id="forward_return_1d",
        values=np.array([0.1, 0.2, 0.3]),
        horizon=1,
        decision_time=(0, 1, 2),
        signal_available_time=(1, 2, 3),
        execution_time=(2, 3, 4),
        label_start_time=(3, 4, 5),
        label_end_time=(4, 5, 6),
    )


class TestCausalChainViolations:
    def test_decision_after_signal_available_raises(self):
        """(a) decision_time > signal_available_time must raise."""
        kwargs = _consistent_ints()
        kwargs["decision_time"] = (2, 3, 4)  # > signal_available_time (1, 2, 3)
        with pytest.raises(ValueError, match="decision_time must be <= signal_available_time"):
            LabelBundle(**kwargs)

    def test_execution_before_signal_available_raises(self):
        """(b) execution_time < signal_available_time must raise."""
        kwargs = _consistent_ints()
        kwargs["execution_time"] = (0, 1, 2)  # < signal_available_time (1, 2, 3)
        with pytest.raises(ValueError, match="signal_available_time must be <= execution_time"):
            LabelBundle(**kwargs)

    def test_execution_after_label_start_raises(self):
        """(c) execution_time > label_start_time must raise."""
        kwargs = _consistent_ints()
        kwargs["execution_time"] = (4, 5, 6)  # > label_start_time (3, 4, 5)
        with pytest.raises(ValueError, match="execution_time must be <= label_start_time"):
            LabelBundle(**kwargs)

    def test_signal_after_label_start_raises_without_execution(self):
        """Without execution_time, signal_available_time must still be <= label_start."""
        kwargs = _consistent_ints()
        kwargs.pop("execution_time")
        kwargs["signal_available_time"] = (4, 5, 6)  # > label_start_time (3, 4, 5)
        with pytest.raises(ValueError, match="execution_time must be <= label_start_time"):
            LabelBundle(**kwargs)

    def test_label_start_ge_label_end_raises(self):
        """(d) label_start_time >= label_end_time must raise (equal and greater)."""
        kwargs = _consistent_ints()
        kwargs["label_end_time"] = (3, 5, 6)  # equal at position 0
        with pytest.raises(ValueError, match="label_start_time must be < label_end_time"):
            LabelBundle(**kwargs)

        kwargs = _consistent_ints()
        kwargs["label_end_time"] = (2, 5, 6)  # strictly before at position 0
        with pytest.raises(ValueError, match="label_start_time must be < label_end_time"):
            LabelBundle(**kwargs)

    def test_decision_after_label_start_raises_with_defaults(self):
        """Default signal_available_time = decision_time must still precede label window."""
        kwargs = _consistent_ints()
        kwargs.pop("signal_available_time")
        kwargs.pop("execution_time")
        kwargs["label_start_time"] = (0, 0, 5)  # position 1: decision 1 > label_start 0
        with pytest.raises(ValueError, match="execution_time must be <= label_start_time"):
            LabelBundle(**kwargs)

    def test_signal_available_length_mismatch_raises(self):
        kwargs = _consistent_ints()
        kwargs["signal_available_time"] = (1, 2)
        with pytest.raises(ValueError, match="signal_available_time length"):
            LabelBundle(**kwargs)

    def test_unorderable_timing_raises(self):
        kwargs = _consistent_ints()
        kwargs["signal_available_time"] = (1, "not-orderable", 3)
        with pytest.raises(ValueError, match="must be orderable"):
            LabelBundle(**kwargs)


class TestCausalChainValid:
    def test_fully_consistent_bundle_passes(self):
        """(e) Fully consistent bundle (all timing fields supplied) passes."""
        bundle = LabelBundle(**_consistent_ints())
        assert bundle.num_observations() == 3
        assert bundle.signal_available_time == (1, 2, 3)

    def test_consistent_with_defaults(self):
        """(e) Consistent bundle using only required fields passes."""
        bundle = LabelBundle(
            target_id="ret_1d",
            values=np.array([0.1, 0.2, 0.3]),
            horizon=1,
            decision_time=(0, 1, 2),
            label_start_time=(0, 1, 2),
            label_end_time=(1, 2, 3),
        )
        assert bundle.signal_available_time == ()

    def test_numpy_datetime64_tuples_work(self):
        """(f) numpy datetime64 timing tuples are orderable and accepted."""
        base = np.datetime64("2026-01-05T10:00:00", "s")
        kwargs = _consistent_ints()
        kwargs["decision_time"] = tuple(base + np.timedelta64(i, "s") for i in range(3))
        kwargs["signal_available_time"] = tuple(
            base + np.timedelta64(i + 1, "s") for i in range(3)
        )
        kwargs["execution_time"] = tuple(base + np.timedelta64(i + 2, "s") for i in range(3))
        kwargs["label_start_time"] = tuple(base + np.timedelta64(i + 3, "s") for i in range(3))
        kwargs["label_end_time"] = tuple(base + np.timedelta64(i + 4, "s") for i in range(3))
        bundle = LabelBundle(**kwargs)
        assert bundle.num_observations() == 3

        # And a violation still raises with datetime64.
        bad = _consistent_ints()
        bad["decision_time"] = tuple(base + np.timedelta64(i + 5, "s") for i in range(3))
        bad["signal_available_time"] = tuple(base + np.timedelta64(i + 1, "s") for i in range(3))
        bad["execution_time"] = tuple(base + np.timedelta64(i + 2, "s") for i in range(3))
        bad["label_start_time"] = tuple(base + np.timedelta64(i + 3, "s") for i in range(3))
        bad["label_end_time"] = tuple(base + np.timedelta64(i + 4, "s") for i in range(3))
        with pytest.raises(ValueError, match="decision_time must be <= signal_available_time"):
            LabelBundle(**bad)

    def test_pd_timestamps_work(self):
        """pd.Timestamp timings are also orderable."""
        base = pd.Timestamp("2026-01-05 10:00:00")
        bundle = LabelBundle(
            target_id="ret_1d",
            values=np.array([0.1, 0.2]),
            horizon=1,
            decision_time=(base, base + pd.Timedelta(seconds=1)),
            signal_available_time=(base, base + pd.Timedelta(seconds=1)),
            label_start_time=(base + pd.Timedelta(seconds=2), base + pd.Timedelta(seconds=3)),
            label_end_time=(base + pd.Timedelta(seconds=3), base + pd.Timedelta(seconds=4)),
        )
        assert bundle.num_observations() == 2

    def test_single_observation_missing_optionals_passes(self):
        """(g) One observation, no optional timing fields — backward compatible."""
        bundle = LabelBundle(
            target_id="ret_1d",
            values=np.array([0.05]),
            horizon=5,
            decision_time=(10,),
            label_start_time=(10,),
            label_end_time=(11,),
        )
        assert bundle.num_observations() == 1
        assert bundle.execution_time == ()
        assert bundle.signal_available_time == ()
