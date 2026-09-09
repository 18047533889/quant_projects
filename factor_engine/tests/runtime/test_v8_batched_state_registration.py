import sqlite3

import pytest

from factor_engine.runtime.persistent_run_state import PersistentRunState


def test_thousand_records_need_two_bounded_commits_not_thousand(tmp_path):
    state = PersistentRunState(tmp_path / "state.sqlite3")
    statements = []
    state._db.set_trace_callback(statements.append)
    try:
        state.register_many((ordinal, f"factor-{ordinal}") for ordinal in range(1000))
        assert state.counts() == {"ACCEPTED": 1000}
        assert sum(sql == "COMMIT" for sql in statements) == 2
        assert state._db.execute("select sum(attempts) from outcomes").fetchone() == (0,)
    finally:
        state.close()


def test_existing_terminal_and_attempts_are_not_reset(tmp_path):
    state = PersistentRunState(tmp_path / "state.sqlite3")
    try:
        state.register(0, "alpha")
        state.consume_attempt(0, "compute")
        state.terminal(0, "FAILED", error_code="original-failure")
        state.register_many([(0, "alpha"), (1, "beta")])
        assert state._db.execute(
            "select state,attempts,error_code from outcomes where ordinal=0"
        ).fetchone() == ("FAILED", 1, "original-failure")
    finally:
        state.close()


def test_input_is_streamed_and_page_commits_before_consuming_next_page(tmp_path):
    path = tmp_path / "state.sqlite3"
    state = PersistentRunState(path)
    observations = []

    def records():
        for ordinal in range(5):
            if ordinal in (2, 4):
                with sqlite3.connect(path) as observer:
                    observations.append(observer.execute("select count(*) from outcomes").fetchone()[0])
            yield ordinal, f"factor-{ordinal}"

    try:
        state.register_many(records(), batch_size=2)
        assert observations == [2, 4]
        assert state.counts() == {"ACCEPTED": 5}
    finally:
        state.close()


def test_conflict_rolls_back_only_current_page_and_survives_reopen(tmp_path):
    path = tmp_path / "state.sqlite3"
    state = PersistentRunState(path)
    try:
        with pytest.raises(ValueError, match="different factor name"):
            state.register_many([(0, "a"), (1, "b"), (2, "c"), (0, "wrong")], batch_size=2)
        assert state.counts() == {"ACCEPTED": 2}
    finally:
        state.close()
    with sqlite3.connect(path) as db:
        assert db.execute("select ordinal,name from outcomes order by ordinal").fetchall() == [(0, "a"), (1, "b")]


@pytest.mark.parametrize("batch_size", [True, 0, 513, 1.5])
def test_invalid_page_bound_does_not_consume_input(tmp_path, batch_size):
    state = PersistentRunState(tmp_path / "state.sqlite3")
    def records():
        pytest.fail("invalid batch size must not consume input")
        yield 0, "unused"
    try:
        with pytest.raises(ValueError, match="batch size"):
            state.register_many(records(), batch_size=batch_size)
        assert state.counts() == {}
    finally:
        state.close()


@pytest.mark.parametrize("fault", ["iterator", "sqlite"])
def test_mid_page_failure_rolls_back_page_without_poisoning_connection(tmp_path, fault):
    state = PersistentRunState(tmp_path / "state.sqlite3")
    if fault == "sqlite":
        state._db.execute(
            "CREATE TRIGGER injected BEFORE INSERT ON outcomes WHEN NEW.ordinal=3 "
            "BEGIN SELECT RAISE(ABORT, 'injected SQLite failure'); END"
        )

    def records():
        for ordinal in range(5):
            if fault == "iterator" and ordinal == 3:
                raise RuntimeError("injected iterator failure")
            yield ordinal, f"factor-{ordinal}"

    try:
        error = RuntimeError if fault == "iterator" else sqlite3.IntegrityError
        with pytest.raises(error, match="injected"):
            state.register_many(records(), batch_size=2)
        assert state.counts() == {"ACCEPTED": 2}
        state.register(99, "connection-still-usable")
        assert state.counts() == {"ACCEPTED": 3}
    finally:
        state.close()
