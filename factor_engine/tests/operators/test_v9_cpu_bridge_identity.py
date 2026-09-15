from __future__ import annotations

import subprocess
import sys
import hashlib
import json
from pathlib import Path


COMMON = r'''
import hashlib
import inspect
from factor_engine.backend.contracts import ExecutionKind
from factor_engine.backend.polars_backend_kind import (
    PolarsImplementationKind, get_physical_spec, polars_backend_kind,
)
spec = get_physical_spec(op)
assert spec is not None
assert spec.validation_errors() == ()
assert spec.execution_kind is ExecutionKind.DELEGATE_PYTHON
assert not spec.is_production_eligible()
assert hashlib.sha256(inspect.getsource(source_callable).encode()).hexdigest() == spec.implementation_source_hash
assert polars_backend_kind(op, production_mode=False) is PolarsImplementationKind.UNSUPPORTED
assert polars_backend_kind(op, production_mode=True) is PolarsImplementationKind.UNSUPPORTED
'''


def test_peer_bridge_has_content_bound_non_native_spec():
    setup = r'''
from factor_engine.cleaned_operators.cross_section import polars_peer
from factor_engine.cleaned_operators.registry import OperatorRegistry
op = OperatorRegistry.get("group_peer_beta_deviation", "polars", mode="any")
source_callable = polars_peer._group_peer_beta_deviation
'''
    subprocess.run([sys.executable, "-c", setup + COMMON], check=True)


def test_expectile_numpy_kernel_has_transitive_source_bound_spec():
    setup = r'''
import hashlib
from pathlib import Path
from factor_engine.cleaned_operators.polars_native.ts_advanced_batch1 import TSExpectileBetaPolarsNative
from factor_engine.cleaned_operators import advanced_expectile, rolling_pack
from factor_engine.cleaned_operators.common import expectile_native
from factor_engine.backend.contracts import ExecutionKind
from factor_engine.backend.polars_backend_kind import get_physical_spec
spec=get_physical_spec(TSExpectileBetaPolarsNative())
assert spec.execution_kind is ExecutionKind.POLARS_NUMPY_KERNEL
assert not spec.is_production_eligible()
expected=hashlib.sha256(Path(advanced_expectile.__file__).read_bytes()
    +Path(rolling_pack.__file__).read_bytes()+Path(expectile_native.__file__).read_bytes()).hexdigest()
assert spec.implementation_source_hash==expected
'''
    subprocess.run([sys.executable, "-c", setup], check=True)


def test_declaration_payload_hashes_are_reproducible():
    path = Path(__file__).resolve().parents[3] / "evidence/r2/V9-CPU-BRIDGE-IDENTITY-PAYLOADS.json"
    records = json.loads(path.read_text())
    for canonical in ("group_peer_beta_deviation", "ts_expectile_beta"):
        for kind in ("parameter_domain", "semantic_contract"):
            assert hashlib.sha256(records[canonical][kind + "_payload"].encode()).hexdigest() == records[canonical][kind + "_hash"]
