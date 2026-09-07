import pytest

from factor_engine.runtime.persistent_run_state import PersistentRunState


def test_commit_intent_survives_reopen_and_cannot_rebind(tmp_path):
    path = tmp_path / "state.sqlite"
    state = PersistentRunState(path)
    state.register(0, "factor")
    assert state.get_commit_intent(0) is None
    state.record_commit_intent(0, "a" * 32)
    state.close()
    state = PersistentRunState(path)
    assert state.get_commit_intent(0) == {"generation": "a" * 32, "commit_state": "INTENT"}
    state.record_commit_intent(0, "a" * 32)
    with pytest.raises(RuntimeError, match="different generation"):
        state.record_commit_intent(0, "b" * 32)
    with pytest.raises(ValueError, match="generation differs"):
        state.terminal(0, "SUCCEEDED", commit_state="VERIFIED", artifact={
            "committed": True, "verified": True, "generation": "b" * 32})
    state.terminal(0, "SUCCEEDED", commit_state="VERIFIED", artifact={
        "committed": True, "verified": True, "generation": "a" * 32})
    assert state.get_commit_intent(0)["commit_state"] == "VERIFIED"
    with pytest.raises(RuntimeError, match="terminal"):
        state.record_commit_intent(0, "a" * 32)
    state.close()


@pytest.mark.parametrize("generation", ["../bad", "", "A" * 32, True])
def test_bad_commit_intent_never_mutates_state(tmp_path, generation):
    state = PersistentRunState(tmp_path / "state.sqlite")
    state.register(0, "factor")
    with pytest.raises(ValueError):
        state.record_commit_intent(0, generation)
    assert state.get_commit_intent(0) is None
    state.close()


@pytest.mark.parametrize("value", [True, 1.0, 2.5, 0, 4, "3"])
def test_attempt_policy_is_strict_integer(tmp_path, value):
    with pytest.raises(ValueError):
        PersistentRunState(tmp_path / "state.sqlite", max_attempts=value)


def test_reopening_cannot_increase_persisted_attempt_cap(tmp_path):
    path = tmp_path / "state.sqlite"
    state = PersistentRunState(path, max_attempts=2)
    state.register(0, "factor")
    assert state.consume_attempt(0, "execution") == 1
    state.close()
    with pytest.raises(ValueError, match="persisted retry policy"):
        PersistentRunState(path, max_attempts=3)
    state = PersistentRunState(path, max_attempts=2)
    assert state.consume_attempt(0, "execution") == 2
    assert state.consume_attempt(0, "execution") is None
    state.close()


def test_ordinal_cannot_be_silently_rebound(tmp_path):
    state = PersistentRunState(tmp_path / "state.sqlite")
    state.register(0, "first")
    state.register(0, "first")
    with pytest.raises(ValueError, match="different factor"):
        state.register(0, "second")
    assert state.outcomes_page()[0].name == "first"
    state.close()


def test_success_without_commit_proof_rejected(tmp_path):
    state = PersistentRunState(tmp_path / "state.sqlite")
    state.register(0, "factor")
    with pytest.raises(ValueError, match="verified committed"):
        state.terminal(0, "SUCCEEDED")
    assert state.outcomes_page()[0].state == "ACCEPTED"
    state.close()


@pytest.mark.parametrize("terminal", ["SUCCEEDED", "REUSED"])
@pytest.mark.parametrize("artifact,commit_state", [
    ({"committed": False, "verified": True}, "VERIFIED"),
    ({"committed": True, "verified": False}, "VERIFIED"),
    ({"committed": 1, "verified": True}, "VERIFIED"),
    ({"committed": True, "verified": True}, "UNKNOWN"),
])
def test_incomplete_success_proof_never_mutates_ordinal(tmp_path, terminal, artifact, commit_state):
    state = PersistentRunState(tmp_path / "state.sqlite")
    state.register(0, "factor")
    with pytest.raises(ValueError, match="verified committed"):
        state.terminal(0, terminal, artifact=artifact, commit_state=commit_state)
    assert state.outcomes_page()[0].state == "ACCEPTED"
    state.close()


def test_reopening_cannot_silently_change_to_lower_retry_policy(tmp_path):
    path = tmp_path / "state.sqlite"
    state = PersistentRunState(path, max_attempts=3)
    state.close()
    with pytest.raises(ValueError, match="persisted retry policy"):
        PersistentRunState(path, max_attempts=2)


def test_paged_outcomes_are_exact_and_include_failure_reason(tmp_path):
    state = PersistentRunState(tmp_path / "state.sqlite")
    for ordinal in range(25):
        state.register(ordinal, f"factor-{ordinal}")
        state.terminal(ordinal, "FAILED", error_code="INVALID_INPUT", detail=f"detail-{ordinal}")
    pages = [state.outcomes_page(start=start, limit=7) for start in (0, 7, 14, 21)]
    assert [record.ordinal for page in pages for record in page] == list(range(25))
    assert pages[2][0].error_detail == "detail-14"
    assert pages[2][0].commit_state == "NOT_STARTED"
    with pytest.raises(ValueError):
        state.outcomes_page(limit=100000)
    state.close()
