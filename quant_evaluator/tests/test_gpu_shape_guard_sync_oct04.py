"""Host-readback contract for the shape-linear CUDA risk guard."""
from types import SimpleNamespace

import pytest


class _FakeScalar:
    def __init__(self, value, reads, error=None):
        self.value = value
        self.reads = reads
        self.error = error

    def item(self):
        self.reads.append(self.value)
        if self.error is not None:
            raise self.error
        return self.value


class _FakeArray:
    _next_ptr = 1000

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)
        self.flags = SimpleNamespace(c_contiguous=True)
        self.device = SimpleNamespace(id=0)
        self.nbytes = 8 * (shape[0] if shape else 1)
        self.data = SimpleNamespace(ptr=_FakeArray._next_ptr)
        _FakeArray._next_ptr += self.nbytes + 100


class _FakeIds:
    def astype(self, dtype, copy=False):
        return self

    def __getitem__(self, key):
        return self


class _FakeCupy:
    ndarray = _FakeArray
    float64 = "float64"
    int32 = "int32"
    int64 = "int64"
    uint8 = "uint8"

    def __init__(self, *, risk_count, exact_error=0, sum_error=None):
        self.reads = []
        self.risk_count = risk_count
        self.exact_error = exact_error
        self.sum_error = sum_error
        self.cuda = SimpleNamespace(Device=lambda: SimpleNamespace(id=0))

    def RawKernel(self, source, name, options):
        return _FakeKernel(self, name)

    def empty(self, shape, dtype):
        return _FakeArray(shape, dtype)

    def zeros(self, shape, dtype):
        return _FakeScalar(0, self.reads)

    def sum(self, values, dtype):
        return _FakeScalar(self.risk_count, self.reads, error=self.sum_error)

    def nonzero(self, values):
        return (_FakeIds(),)


class _FakeKernel:
    attributes = {"local_size_bytes": 0}

    def __init__(self, cp, name):
        self.cp = cp
        self.name = name

    def compile(self):
        pass

    def __call__(self, grid, block, args):
        if self.name == "shape_linear_exact":
            args[3].value = self.cp.exact_error


def _run_repair(monkeypatch, *, risk_count, exact_error=0, sum_error=None):
    import sys

    from quant_evaluator.kernels.gpu import shape_linear_numeric

    cp = _FakeCupy(
        risk_count=risk_count, exact_error=exact_error, sum_error=sum_error,
    )
    monkeypatch.setitem(sys.modules, "cupy", cp)
    values = _FakeArray((3, 1), cp.float64)
    output = _FakeArray((1,), cp.float64)
    result = shape_linear_numeric.repair_shape_linear_gpu(values, output, "curvature")
    assert result is output
    return cp.reads


@pytest.mark.parametrize(
    ("risk_count", "expected_host_reads"),
    [(0, [0]), (1, [1, 0])],
)
def test_guard_has_no_host_readback_but_risky_repair_keeps_exact_error_readback(
    monkeypatch, risk_count, expected_host_reads,
):
    """A guard-only item read adds a sync; exact repair still checks its flag."""
    reads = _run_repair(monkeypatch, risk_count=risk_count)
    assert reads == expected_host_reads


def test_exact_repair_error_flag_still_fails_closed(monkeypatch):
    with pytest.raises(ValueError, match="exact repair exceeded its admitted domain"):
        _run_repair(monkeypatch, risk_count=1, exact_error=1)


def test_risk_count_readback_propagates_deferred_cuda_launch_error(monkeypatch):
    with pytest.raises(RuntimeError, match="deferred CUDA launch failure"):
        _run_repair(
            monkeypatch, risk_count=0,
            sum_error=RuntimeError("deferred CUDA launch failure"),
        )
