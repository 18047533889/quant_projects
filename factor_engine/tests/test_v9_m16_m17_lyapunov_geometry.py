import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True, scope="module")
def _load_registry():
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded

    ensure_cleaned_loaded()


def _oracle(series, window, tau, dim, horizon, min_anchors, physical):
    eps = 1e-12
    result = np.full(len(series), np.nan)
    span = (dim - 1) * tau
    theiler = dim * tau
    for end in range(window - 1, len(series)):
        chunk = np.asarray(series[end - window + 1 : end + 1], dtype=float)
        if not np.isfinite(chunk[-1]):
            continue
        vectors, times = [], []
        if physical:
            for anchor in range(span, window):
                positions = [anchor - offset * tau for offset in range(dim - 1, -1, -1)]
                values = chunk[positions]
                if np.all(np.isfinite(values)):
                    vectors.append(values)
                    times.append(anchor)
        else:
            finite_times = np.flatnonzero(np.isfinite(chunk))
            values = chunk[finite_times]
            for r in range(max(0, len(values) - span)):
                vectors.append([values[r + q * tau] for q in range(dim)])
                times.append(int(finite_times[r + span]))
        if len(vectors) < 2:
            continue
        X = np.asarray(vectors, dtype=float)
        times = np.asarray(times, dtype=int)
        med = np.median(X, axis=0)
        mad = 1.4826 * np.median(np.abs(X - med), axis=0)
        scale = np.where(mad > eps, mad, np.std(X, axis=0))
        if np.any(~np.isfinite(scale)) or np.any(scale <= eps):
            continue
        Z = (X - med) / scale
        if physical:
            lookup = {int(time): i for i, time in enumerate(times)}
            eligible = [
                i for i, time in enumerate(times)
                if all(int(time) + k in lookup for k in range(horizon + 1))
            ]
        else:
            lookup = None
            eligible = list(range(max(0, len(Z) - horizon)))
        curves = []
        for i in eligible:
            best_j, best_distance = None, np.inf
            for j in eligible:
                if abs(int(times[j]) - int(times[i])) <= theiler:
                    continue
                distance = float(np.sqrt(np.sum((Z[j] - Z[i]) ** 2)))
                if eps < distance < best_distance:
                    best_j, best_distance = j, distance
            if best_j is None:
                continue
            curve = [0.0]
            for k in range(1, horizon + 1):
                if physical:
                    ip = lookup[int(times[i]) + k]
                    jp = lookup[int(times[best_j]) + k]
                else:
                    ip, jp = i + k, best_j + k
                distance = float(np.sqrt(np.sum((Z[ip] - Z[jp]) ** 2)))
                if not np.isfinite(distance):
                    break
                curve.append(np.log(distance + eps) - np.log(best_distance + eps))
            if len(curve) == horizon + 1:
                curves.append(curve)
        if len(curves) < min_anchors:
            continue
        mean_curve = np.mean(np.asarray(curves), axis=0)
        ks = np.arange(horizon + 1, dtype=float)
        result[end] = np.sum((ks - ks.mean()) * (mean_curve - mean_curve.mean())) / np.sum(
            (ks - ks.mean()) ** 2
        )
    return result


def test_physical_embedding_uses_true_bar_axis_and_anchor_labels():
    from factor_engine.cleaned_operators.local_lyapunov import _embedding_geometry

    series = np.array([0.0, 1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0])
    physical, physical_times = _embedding_geometry(series, 1, 3, physical_time=True)
    compressed, compressed_times = _embedding_geometry(series, 1, 3, physical_time=False)
    np.testing.assert_array_equal(physical, [[3, 4, 5], [4, 5, 6], [5, 6, 7]])
    np.testing.assert_array_equal(physical_times, [5, 6, 7])
    np.testing.assert_array_equal(compressed[0], [0, 1, 3])
    np.testing.assert_array_equal(compressed_times, [3, 4, 5, 6, 7])


@pytest.mark.parametrize("physical", [False, True])
@pytest.mark.parametrize("horizon", [1, 4, 7])
def test_kernel_matches_independent_double_loop_oracle(physical, horizon):
    from factor_engine.cleaned_operators.local_lyapunov import _lyapunov_series

    series = np.random.default_rng(1617).normal(size=70).cumsum()
    series[[20, 21, 44]] = np.nan
    actual = _lyapunov_series(series, 32, 2, 3, horizon, 2, physical)
    expected = _oracle(series, 32, 2, 3, horizon, 2, physical)
    np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-12, atol=1e-12)


def test_horizon_smaller_than_embedding_span_emits():
    from factor_engine.cleaned_operators.local_lyapunov import _lyapunov_series

    series = np.random.default_rng(42).normal(size=80)
    result = _lyapunov_series(series, 32, 2, 3, 1, 3, False)
    assert np.isfinite(result).sum() == 49


def test_incomplete_nearest_trajectory_does_not_hide_complete_neighbor():
    from factor_engine.cleaned_operators.local_lyapunov import _lyapunov_series

    series = np.random.default_rng(917).normal(size=64).cumsum()
    series[[29, 46]] = np.nan
    actual = _lyapunov_series(series, 36, 1, 3, 4, 1, True)
    expected = _oracle(series, 36, 1, 3, 4, 1, True)
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert np.isfinite(actual).any()


def test_duplicate_initial_neighbor_does_not_hide_positive_neighbor(monkeypatch):
    from factor_engine.cleaned_operators import local_lyapunov as module

    # The first candidate is an exact duplicate of the anchor, while the next
    # legal complete trajectory has positive separation.  Candidate filtering
    # must happen before argmin.
    vectors = np.array([
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
        [0.0, 0.0], [2.0, 1.0], [4.0, 1.0],
        [0.0, 2.0], [3.0, 2.0], [6.0, 2.0],
    ])
    times = np.arange(len(vectors)) * 3
    monkeypatch.setattr(
        module, "_embedding_geometry",
        lambda *_args, **_kwargs: (vectors.copy(), times.copy()),
    )
    result = module._lyapunov_series(
        np.arange(9.0), 9, 1, 2, 2, 1, physical_time=False
    )
    assert result[-1] == pytest.approx(0.3877113332503152)


def test_physical_converging_successors_use_log_epsilon_without_crash(monkeypatch):
    from factor_engine.cleaned_operators import local_lyapunov as module

    # Two initially distinct complete trajectories merge at k=1.  Zero future
    # distance is a valid convergence observation under the existing
    # log(distance + epsilon) regularisation, not structural incompleteness.
    vectors = np.array([
        [0.0, 0.0], [1.0, 1.0], [2.0, 2.0],
        [4.0, 0.0], [1.0, 1.0], [2.0, 2.0],
    ])
    times = np.array([0, 1, 2, 4, 5, 6])
    monkeypatch.setattr(
        module, "_embedding_geometry",
        lambda *_args, **_kwargs: (vectors.copy(), times.copy()),
    )
    result = module._lyapunov_series(
        np.arange(7.0), 7, 1, 2, 2, 1, physical_time=True
    )
    assert np.isfinite(result[-1])
    assert result[-1] < 0.0


@pytest.mark.parametrize("physical", [False, True])
def test_public_prefix_and_window_warmup(physical):
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    series = np.random.default_rng(1716).normal(size=90).cumsum()
    frame = pd.DataFrame({"A": series})
    op = OperatorRegistry.get("ts_local_lyapunov_exponent", "pandas_numpy")
    kwargs = dict(
        window=32, tau=2, embedding_dim=3, horizon=1,
        min_anchors=3, physical_time=physical,
    )
    full = op.calculate(frame, **kwargs)
    prefix = op.calculate(frame.iloc[:60], **kwargs)
    pd.testing.assert_frame_equal(prefix, full.iloc[:60])
    split = 55
    warmup = kwargs["window"] - 1
    chunk = op.calculate(frame.iloc[split - warmup :], **kwargs)
    pd.testing.assert_frame_equal(chunk.iloc[warmup:], full.iloc[split:])


def test_joint_parameter_domain_rejects_before_kernel(monkeypatch):
    import factor_engine.cleaned_operators.local_lyapunov as module
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    def unexpected(*_args, **_kwargs):
        raise AssertionError("kernel was reached")

    monkeypatch.setattr(module, "_lyapunov_series", unexpected)
    op = OperatorRegistry.get("ts_local_lyapunov_exponent", "pandas_numpy")
    frame = pd.DataFrame({"A": np.arange(30.0)})
    with pytest.raises(ValueError, match="two complete trajectories"):
        op.calculate(
            frame, window=12, tau=2, embedding_dim=3,
            horizon=1, min_anchors=1, physical_time=False,
        )


def test_time_policies_have_distinct_bound_identity():
    from factor_engine.cleaned_operators.base import bind_operator_call
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get("ts_local_lyapunov_exponent", "pandas_numpy")
    frame = pd.DataFrame({"A": np.arange(40.0)})
    common = dict(window=32, tau=2, embedding_dim=3, horizon=1, min_anchors=1)
    compressed = bind_operator_call(op, (frame,), {**common, "physical_time": False})
    physical = bind_operator_call(op, (frame,), {**common, "physical_time": True})
    assert compressed.bound.normalized_values != physical.bound.normalized_values
