"""Reusable process worker with generation fencing and finite termination."""
from __future__ import annotations

import multiprocessing as mp
import pickle
import math
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable


class WorkerTimedOut(TimeoutError):
    pass


class WorkerStaleResponse(RuntimeError):
    """Fenced response rejected after process/transport retirement was proven."""


class WorkerTransportFailed(RuntimeError):
    """Parent IPC failed and both process and transport exit were observed.

    Distinct from a user function raising EOFError in a still-live worker.
    A caller with a durable commit intent may now reconcile without rewriting.
    """


class WorkerQuarantined(RuntimeError):
    def __init__(self, message: str, *, pid: int | None = None) -> None:
        super().__init__(message)
        self.pid = pid
        self.cleanup_pending = True


def _worker_loop(connection: Any, generation: str, inherited_function: Any = None) -> None:
    while True:
        message = connection.recv()
        if message is None:
            return
        request_id, payload = message
        if request_id == "__close__":
            try:
                close = getattr(inherited_function, "close", None)
                if callable(close):
                    close()
                response = (request_id, generation, True, pickle.dumps(None))
            except BaseException as exc:
                response = (request_id, generation, False,
                            pickle.dumps(RuntimeError(f"worker cleanup failed: {type(exc).__name__}")))
            connection.send(response)
            connection.close()
            return
        try:
            decoded = pickle.loads(payload)
            if inherited_function is None:
                function, args, kwargs = decoded
            else:
                args, kwargs = decoded
                function = inherited_function
            # The received message and decoded envelope are transport-only.
            # Drop both before user code runs so the serialized request does
            # not overlap the decoded arguments and the function workspace.
            payload = decoded = message = None
            value = function(*args, **kwargs)
            response = (request_id, generation, True, pickle.dumps(value))
        except BaseException as exc:
            try:
                failure = pickle.dumps(exc)
            except BaseException:
                failure = pickle.dumps(RuntimeError(
                    f"worker exception could not be serialized: {type(exc).__name__}"))
            response = (request_id, generation, False, failure)
        # Drop live engine/result objects before acknowledging completion. The
        # serialized response is the only remaining payload while parent owns
        # any associated lease.
        function = args = kwargs = value = decoded = None
        connection.send(response)
        del response, payload, message


@dataclass(frozen=True)
class WorkerResult:
    value: Any
    generation: str


class SupervisedReusableWorker:
    """One long-lived isolated worker; never pretends timeout released memory.

    A supplied resource lease is released only after a normal response or after
    the operating system proves the worker exited. If exit cannot be observed,
    the worker is quarantined and the lease remains owned by the caller.
    """

    def __init__(self, *, cancel_grace_seconds: float = 5.0,
                 exit_observation_seconds: float = 10.0,
                 context: str = "spawn", function: Callable[..., Any] | None = None) -> None:
        self.cancel_grace_seconds = float(cancel_grace_seconds)
        self.exit_observation_seconds = float(exit_observation_seconds)
        self._context = mp.get_context(context)
        self._process: mp.Process | None = None
        self._connection: Any = None
        self.generation = ""
        self.quarantined = False
        self._function = function
        self._transport: threading.Thread | None = None
        self._call_lock = threading.Lock()
        for value in (self.cancel_grace_seconds, self.exit_observation_seconds):
            if not math.isfinite(value) or value < 0:
                raise ValueError("worker grace periods must be finite and nonnegative")

    def start(self) -> None:
        if self.quarantined:
            raise WorkerQuarantined("quarantined worker cannot be reused", pid=self._process.pid)
        if self._process is not None and self._process.is_alive():
            return
        if self._connection is not None:
            self._connection.close()
        # This serialization check is provably pre-spawn and therefore the only
        # startup rejection for which execute() may immediately return admission.
        # Process.start failures are not classifiable from pid alone: CPython can
        # spawn inside Popen.__init__ before BaseProcess assigns self._popen.
        if (
            self._context.get_start_method() in {"spawn", "forkserver"}
            and self._function is not None
        ):
            pickle.dumps(self._function)
        parent, child = self._context.Pipe()
        self.generation = uuid.uuid4().hex
        self._process = self._context.Process(
            target=_worker_loop, args=(child, self.generation, self._function), daemon=True,
        )
        try:
            self._process.start()
        except BaseException as exc:
            parent.close()
            child.close()
            if getattr(self._process, "_popen", None) is None:
                # Ambiguous boundary: Popen.__init__ may already have created an
                # OS child even though BaseProcess.pid still reports None.
                self.quarantined = True
                raise WorkerQuarantined(
                    "worker start failed after entering OS-spawn boundary; "
                    "cleanup cannot be proven and lease is retained",
                    pid=None,
                ) from exc
            raise
        child.close()
        self._connection = parent
        from factor_engine.runtime.resource_broker import register_heavy_worker, NoActiveHeavyRunGuard
        try:
            register_heavy_worker(self._process.pid)
        except NoActiveHeavyRunGuard:
            # Non-v2 callers may legitimately have no active heavy-run guard.
            pass
        except BaseException:
            self._retire()
            self._connection.close()
            raise
        self.quarantined = False

    def execute(self, function: Callable[..., Any] | Any, *args: Any,
                timeout_seconds: float, lease: Any = None, **kwargs: Any) -> WorkerResult:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("worker already has an active request")
        try:
            return self._execute(function, args, kwargs, timeout_seconds, lease)
        finally:
            self._call_lock.release()

    def _retire(self) -> None:
        """Observe both process and IPC thread exit before releasing ownership."""
        assert self._process is not None
        if self._process.is_alive():
            self._process.terminate()
        deadline = time.monotonic() + self.exit_observation_seconds
        self._process.join(max(0.0, deadline - time.monotonic()))
        if self._transport is not None:
            self._transport.join(max(0.0, deadline - time.monotonic()))
        if self._process.is_alive() or (self._transport is not None and self._transport.is_alive()):
            self.quarantined = True
            raise WorkerQuarantined(
                "worker or IPC thread did not exit; cleanup pending and lease retained",
                pid=self._process.pid)

    def _execute(self, function, args, kwargs, timeout_seconds, lease):
        try:
            self.start()
        except BaseException:
            # Admission was transferred by execute(), including startup. A
            # spawn/pickling failure must not leak it; a partially started or
            # untracked process must still be observed dead before release.
            if self.quarantined:
                raise
            if self._process is not None and self._process.pid is not None:
                self._retire()
            if lease is not None:
                lease.release()
            raise
        assert self._process is not None
        request_id = uuid.uuid4().hex
        completed = threading.Event()
        response = {}

        def exchange():
            try:
                if self._function is None:
                    payload = pickle.dumps((function, args, kwargs))
                else:
                    payload = pickle.dumps(((function,) + args, kwargs))
                self._connection.send((request_id, payload))
                del payload
                rid, generation, ok, payload = self._connection.recv()
                response["result"] = (rid, generation, ok, pickle.loads(payload))
            except BaseException as exc:
                response["error"] = exc
            finally:
                completed.set()

        # poll() only guarantees a header, not a complete message. Supervise
        # the full serialization/send/receive/decode operation instead.
        self._transport = threading.Thread(target=exchange, daemon=True,
                                           name="factor-v2-worker-ipc")
        self._transport.start()
        if completed.wait(timeout_seconds):
            self._transport.join()
            if "error" in response:
                self._retire()
                if lease is not None:
                    lease.release()
                error = response["error"]
                raise WorkerTransportFailed(
                    f"worker transport failed after proven retirement: {type(error).__name__}: {error}"
                ) from error
            rid, generation, ok, value = response["result"]
            if rid != request_id or generation != self.generation:
                self._retire()
                if lease is not None:
                    lease.release()
                raise WorkerStaleResponse("stale worker response rejected; worker retirement proven")
            if lease is not None:
                lease.release()
            if ok:
                return WorkerResult(value, generation)
            if not isinstance(value, BaseException):
                raise RuntimeError("invalid worker exception response")
            raise value
        # Cooperative phase: invalidate write authority before termination.
        old_generation = self.generation
        self.generation = uuid.uuid4().hex
        completed.wait(self.cancel_grace_seconds)
        # Even a late cooperative reply belongs to the expired generation.
        # Retire it rather than reusing a process with a stale generation.
        self._retire()
        response.clear()
        if lease is not None:
            lease.release()
        raise WorkerTimedOut(f"worker generation {old_generation} terminated after deadline")

    def close(self) -> None:
        if self._process is None:
            return
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("worker still has an active request; close cannot release ownership")
        try:
            if self._process.is_alive() and not self.quarantined:
                completed = threading.Event()
                response = {}

                def close_exchange():
                    try:
                        self._connection.send(("__close__", None))
                        rid, generation, ok, payload = self._connection.recv()
                        if rid != "__close__" or generation != self.generation:
                            raise WorkerStaleResponse("cleanup response generation mismatch")
                        value = pickle.loads(payload)
                        if not ok:
                            raise value
                    except BaseException as exc:
                        response["error"] = exc
                    finally:
                        completed.set()

                # Cleanup includes source/cache lease release, but a broken
                # cleanup hook or partial IPC message must not hang shutdown.
                self._transport = threading.Thread(target=close_exchange, daemon=True,
                                                   name="factor-v2-worker-close")
                self._transport.start()
                acknowledged = completed.wait(self.cancel_grace_seconds)
                if acknowledged:
                    self._transport.join()
                    self._process.join(self.exit_observation_seconds)
                self._retire()
                if acknowledged and "error" in response:
                    raise response["error"]
            else:
                self._retire()
        finally:
            if not self.quarantined and self._connection is not None:
                self._connection.close()
            self._call_lock.release()
