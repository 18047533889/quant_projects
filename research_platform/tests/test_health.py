from research_platform.health import FactorHealthMonitor, FactorHealthState
from research_platform.artifacts import EvaluationBundle
from datetime import datetime, timezone, timedelta

TZ = timezone.utc
NOW = datetime.now(TZ)


def _bundle(factor_id, ic):
    return EvaluationBundle(
        artifact_id=f"{factor_id}-eval", artifact_type="evaluation_bundle",
        factor_id=factor_id, ic=ic,
    )


def test_initial_healthy():
    m = FactorHealthMonitor()
    st = m.update("f1", _bundle("f1", 0.05), NOW)
    assert st == FactorHealthState.HEALTHY


def test_stale():
    m = FactorHealthMonitor(stale_after_days=10)
    st = m.update("f1", _bundle("f1", 0.05), NOW - timedelta(days=30))
    assert st == FactorHealthState.STALE


def test_decaying():
    m = FactorHealthMonitor()
    base = NOW - timedelta(days=60)
    for i in range(30):
        m.update("f1", _bundle("f1", 0.10), base + timedelta(days=i))
    for i in range(10):
        m.update("f1", _bundle("f1", 0.01), base + timedelta(days=30 + i))
    assert m.state("f1") == FactorHealthState.DECAYING


def test_failed():
    m = FactorHealthMonitor()
    base = NOW - timedelta(days=60)
    for i in range(30):
        m.update("f1", _bundle("f1", 0.10), base + timedelta(days=i))
    for i in range(10):
        m.update("f1", _bundle("f1", -0.05), base + timedelta(days=30 + i))
    assert m.state("f1") == FactorHealthState.FAILED


def test_revived_after_decaying():
    m = FactorHealthMonitor()
    base = NOW - timedelta(days=10)
    # healthy history, then a strong recovery point with previous state FAILED
    for i in range(5):
        m.update("f1", _bundle("f1", 0.10), base + timedelta(days=i))
    # force a bad previous state to test the revive transition deterministically
    m._last_state["f1"] = FactorHealthState.FAILED
    st = m.update("f1", _bundle("f1", 0.10), base + timedelta(days=6))
    assert st == FactorHealthState.REVIVED
