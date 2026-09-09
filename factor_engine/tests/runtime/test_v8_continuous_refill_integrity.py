import sqlite3
import os
import json
import threading
import time

import pytest

from factor_engine.runtime.bounded_pipeline import execute_run_many_durable, WorkerProtocolError
from factor_engine.runtime.default_execution_policy import resolve_default_policy
from factor_engine.runtime.persistent_run_state import PersistentRunState
from factor_engine.tests.runtime.test_v7_direct_artifact_pipeline import (
    Broker, ContinuousRefillEngine, Factor, build_continuous_refill_engine,
)


def _assert_event_pids_exited(path):
    pids = {int(line.split(":")[3]) for line in path.read_text().splitlines()
            if len(line.split(":")) > 3}
    for pid in pids:
        deadline = time.monotonic() + 2
        while True:
            try: os.kill(pid, 0)
            except ProcessLookupError: break
            assert time.monotonic() < deadline
            time.sleep(.01)


class InvalidBEngine(ContinuousRefillEngine):
    def run_many_parallel(self, factors, *, result_policy, sink=None, **kwargs):
        if factors[0].name == "b":
            # A wrong descriptor identity is equivalent to an invalid ordinal/generation receipt.
            sink("c", self._value("b"))
            return {"results": {}}
        return super().run_many_parallel(
            factors, result_policy=result_policy, sink=sink, **kwargs
        )


def build_invalid_b_engine(config):
    config = {**config, "marker": __import__("pathlib").Path(config["marker"])}
    return InvalidBEngine(config)


class InvalidDefinitionFactor(Factor):
    def __init__(self, name, fail_pickle=False):
        super().__init__(name)
        self.fail_pickle = fail_pickle

    def __reduce__(self):
        if self.fail_pickle:
            raise TypeError("deliberately unpicklable factor definition")
        # Cross the supervised-ingestion spawn boundary, then fail the child's
        # independent manifest-definition serialization.
        return type(self), (self.name, True)


class InvalidAEngine(ContinuousRefillEngine):
    def run_many_parallel(self, factors, *, result_policy, sink=None, **kwargs):
        if factors[0].name == "a":
            sink("c", self._value("a"))
            return {"results": {}}
        return super().run_many_parallel(
            factors, result_policy=result_policy, sink=sink, **kwargs
        )


def build_invalid_a_engine(config):
    config = {**config, "marker": __import__("pathlib").Path(config["marker"])}
    return InvalidAEngine(config)


class CompileRejectCEngine(ContinuousRefillEngine):
    def compile(self, factor):
        if factor.name == "c":
            raise ValueError("deliberate compile rejection")
        return super().compile(factor)


def build_compile_reject_c_engine(config):
    config = {**config, "marker": __import__("pathlib").Path(config["marker"])}
    return CompileRejectCEngine(config)


def test_invalid_b_never_starts_c_or_spends_c_attempt(tmp_path):
    captured = []
    def run():
      try:
        execute_run_many_durable(
            None, [Factor(x) for x in "abc"],
            policy=resolve_default_policy({"initial_lookahead_factors": 1}),
            artifact_root=tmp_path / "artifacts", run_kwargs={"broker": Broker()},
            engine_factory=build_invalid_b_engine,
            engine_factory_config={"marker": str(tmp_path), "modes": str(tmp_path / "events"), "hold": "a"},
        )
      except BaseException as exc:
        captured.append(exc)
    thread = threading.Thread(target=run); thread.start()
    try:
        deadline = time.monotonic() + 3
        while not captured and time.monotonic() < deadline:
            time.sleep(.01)
        detected_while_a_blocked = bool(captured)
    finally:
        (tmp_path / "release-a").write_text("release")
        thread.join(10)
    assert not thread.is_alive(), "coordinator did not exit after releasing blocked factor A"
    _assert_event_pids_exited(tmp_path / "events")
    assert detected_while_a_blocked
    assert isinstance(captured[0], WorkerProtocolError)
    lines = (tmp_path / "events").read_text().splitlines()
    assert not any(line.startswith("c:start:") for line in lines)
    state = next((tmp_path / "artifacts").glob("*/state.sqlite3"))
    with sqlite3.connect(state) as db:
        assert db.execute("select attempts from outcomes where name='c'").fetchone()[0] == 0


def test_a_blocked_while_b_c_d_e_refill_without_handle_chain(tmp_path, monkeypatch):
    box = []
    captured = []
    persisted = tmp_path / "persisted"
    original_terminal = PersistentRunState.terminal

    def audited_terminal(self, ordinal, *args, **kwargs):
        result = original_terminal(self, ordinal, *args, **kwargs)
        with persisted.open("a") as stream:
            stream.write(f"{ordinal}:{time.monotonic()}\n")
        return result

    monkeypatch.setattr(PersistentRunState, "terminal", audited_terminal)
    def run():
        try:
            box.append(execute_run_many_durable(
                None, [Factor(x) for x in "abcde"],
                policy=resolve_default_policy({"initial_lookahead_factors": 1}),
                artifact_root=tmp_path / "artifacts", run_kwargs={"broker": Broker()},
                engine_factory=build_continuous_refill_engine,
                engine_factory_config={"marker": str(tmp_path), "modes": str(tmp_path / "events"), "hold": "a"},
            ))
        except BaseException as exc:
            captured.append(exc)
    thread = threading.Thread(target=run); thread.start()
    lines = []
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            lines = (tmp_path / "events").read_text().splitlines() if (tmp_path / "events").exists() else []
            if any(line.startswith("e:done:") for line in lines): break
            if captured: break
            time.sleep(.01)
        c_done = any(line.startswith("c:done:") for line in lines)
        d_done = any(line.startswith("d:done:") for line in lines)
        e_done = any(line.startswith("e:done:") for line in lines)
        a_done = any(line.startswith("a:done:") for line in lines)
    finally:
        (tmp_path / "release-a").write_text("release")
        thread.join(10)
    assert not thread.is_alive(), "coordinator did not exit after releasing blocked factor A"
    assert not captured, f"coordinator failed: {captured[0]!r}" if captured else ""
    _assert_event_pids_exited(tmp_path / "events")
    assert c_done and d_done and e_done and not a_done
    assert box[0]["status"] == "SUCCEEDED"
    assert box[0]["execution_batches"] == 5
    parsed = [line.split(":") for line in (tmp_path / "events").read_text().splitlines()]
    starts = {item[0]: float(item[2]) for item in parsed if item[1] == "start"}
    dones = {item[0]: float(item[2]) for item in parsed if item[1] == "done"}
    durable = {int(line.split(":")[0]): float(line.split(":")[1])
               for line in persisted.read_text().splitlines()}
    assert durable[1] < starts["c"]
    assert durable[2] < starts["d"]
    assert durable[3] < starts["e"]
    transitions = sorted(
        [(starts[name], 1) for name in starts] + [(dones[name], -1) for name in dones],
        key=lambda item: (item[0], item[1]),
    )
    active = peak = 0
    for _when, delta in transitions:
        active += delta; peak = max(peak, active)
    assert peak == 2
    assert len({int(item[3]) for item in parsed if item[1] == "start"}) == 5
    with sqlite3.connect(box[0]["state_path"]) as db:
        assert db.execute("select attempts from outcomes order by ordinal").fetchall() == [(1,)] * 5


def test_invalid_refill_manifest_entry_is_terminalized_before_d_refills(tmp_path):
    box, captured = [], []

    def run():
        try:
            box.append(execute_run_many_durable(
                None, [Factor("a"), Factor("b"), InvalidDefinitionFactor("c"), Factor("d")],
                policy=resolve_default_policy({"initial_lookahead_factors": 1}),
                artifact_root=tmp_path / "artifacts", run_kwargs={"broker": Broker()},
                engine_factory=build_continuous_refill_engine,
                engine_factory_config={"marker": str(tmp_path), "modes": str(tmp_path / "events"), "hold": "a"},
            ))
        except BaseException as exc:
            captured.append(exc)

    thread = threading.Thread(target=run); thread.start()
    lines = []
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            lines = (tmp_path / "events").read_text().splitlines() if (tmp_path / "events").exists() else []
            if any(line.startswith("d:done:") for line in lines) or captured:
                break
            time.sleep(.01)
        assert not captured
        assert any(line.startswith("d:done:") for line in lines)
        assert not any(line.startswith("a:done:") for line in lines)
    finally:
        (tmp_path / "release-a").write_text("release")
        thread.join(10)
    assert not thread.is_alive()
    assert not captured
    _assert_event_pids_exited(tmp_path / "events")
    state_path = next((tmp_path / "artifacts").glob("*/state.sqlite3"))
    with sqlite3.connect(state_path) as db:
        assert db.execute(
            "select state,attempts,error_code from outcomes where ordinal=2"
        ).fetchone() == ("REJECTED", 0, "INVALID_FACTOR_DEFINITION")
        assert db.execute(
            "select state,attempts from outcomes where ordinal=3"
        ).fetchone() == ("SUCCEEDED", 1)


def test_simultaneous_done_valid_new_slot_cannot_refill_before_old_fatal(
    tmp_path, monkeypatch,
):
    import factor_engine.runtime.bounded_pipeline as pipeline

    original_await_any = pipeline._await_any_direct_slot
    first = True

    def await_both_once(slots, token):
        nonlocal first
        if first and len(slots) == 2:
            first = False
            deadline = time.monotonic() + 5
            while not all(slot.handle.done for slot in slots):
                assert time.monotonic() < deadline, "both initial slots did not complete"
                time.sleep(.01)
        return original_await_any(slots, token)

    monkeypatch.setattr(pipeline, "_await_any_direct_slot", await_both_once)
    with pytest.raises(WorkerProtocolError):
        execute_run_many_durable(
            None, [Factor(x) for x in "abc"],
            policy=resolve_default_policy({"initial_lookahead_factors": 1}),
            artifact_root=tmp_path / "artifacts", run_kwargs={"broker": Broker()},
            engine_factory=build_invalid_a_engine,
            engine_factory_config={"marker": str(tmp_path), "modes": str(tmp_path / "events"), "hold": "none"},
        )
    lines = (tmp_path / "events").read_text().splitlines()
    assert not any(line.startswith("c:start:") for line in lines)
    state_path = next((tmp_path / "artifacts").glob("*/state.sqlite3"))
    with sqlite3.connect(state_path) as db:
        assert db.execute("select attempts from outcomes where ordinal=2").fetchone()[0] == 0
    _assert_event_pids_exited(tmp_path / "events")


def test_refill_compile_rejection_spends_no_attempt_or_execution_batch(tmp_path):
    receipt = execute_run_many_durable(
        None, [Factor(name) for name in "abcd"],
        policy=resolve_default_policy({"initial_lookahead_factors": 1}),
        artifact_root=tmp_path / "artifacts", run_kwargs={"broker": Broker()},
        engine_factory=build_compile_reject_c_engine,
        engine_factory_config={
            "marker": str(tmp_path), "modes": str(tmp_path / "events"), "hold": "none",
        },
    )
    assert receipt["execution_batches"] == 3
    parsed = (tmp_path / "events").read_text().splitlines()
    assert not any(line.startswith("c:start:") for line in parsed)
    assert any(line.startswith("d:done:") for line in parsed)
    with sqlite3.connect(receipt["state_path"]) as db:
        assert db.execute(
            "select state,attempts,error_code from outcomes where ordinal=2"
        ).fetchone() == ("REJECTED", 0, "INVALID_FACTOR_COMPILE")
        assert db.execute(
            "select state,attempts from outcomes where ordinal=3"
        ).fetchone() == ("SUCCEEDED", 1)
    _assert_event_pids_exited(tmp_path / "events")


def test_done_ack_persists_before_racing_cancel_and_c_is_not_admitted(tmp_path, monkeypatch):
    import factor_engine.runtime.bounded_pipeline as pipeline
    from factor_engine.runtime.exceptions import Cancellation, CancellationToken

    token = CancellationToken()
    original_await_any = pipeline._await_any_direct_slot
    injected = False

    def cancel_after_new_slot_done(slots, active_token):
        nonlocal injected
        if not injected and len(slots) == 2:
            injected = True
            newer = max(slots, key=lambda slot: slot.cursor)
            deadline = time.monotonic() + 5
            while not newer.handle.done:
                assert time.monotonic() < deadline, "newer slot did not complete"
                time.sleep(.01)
            token.cancel()
        return original_await_any(slots, active_token)

    monkeypatch.setattr(pipeline, "_await_any_direct_slot", cancel_after_new_slot_done)
    with pytest.raises(Cancellation) as caught:
        execute_run_many_durable(
            None, [Factor(name) for name in "abc"],
            policy=resolve_default_policy({
                "initial_lookahead_factors": 1,
                "cooperative_cancel_grace_seconds": .01,
                "worker_exit_observation_seconds": 1,
            }),
            artifact_root=tmp_path / "artifacts", run_kwargs={"broker": Broker()},
            engine_factory=build_continuous_refill_engine,
            engine_factory_config={
                "marker": str(tmp_path), "modes": str(tmp_path / "events"), "hold": "a",
            },
            cancellation_token=token,
        )
    receipt = json.loads(__import__("pathlib").Path(caught.value.receipt_path).read_text())
    lines = (tmp_path / "events").read_text().splitlines()
    assert not any(line.startswith("c:start:") for line in lines)
    with sqlite3.connect(receipt["state_path"]) as db:
        assert db.execute(
            "select state,commit_state from outcomes where ordinal=1"
        ).fetchone() == ("SUCCEEDED", "VERIFIED")
        assert db.execute(
            "select attempts from outcomes where ordinal=2"
        ).fetchone() == (0,)
    _assert_event_pids_exited(tmp_path / "events")
