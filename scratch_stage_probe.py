import os
import sys
import importlib
from pathlib import Path

_FE_ROOT = Path('/home/sunhaiwei/quant_projects/factor_engine')
sys.path.insert(0, str(_FE_ROOT))

from factor_engine import cleaned_operators as co
from factor_engine.cleaned_operators.registry import OperatorRegistry as Prod

# Mirror _stage_test_registry with the conftest's EXACT attribute set
TestRegistry = type('_ExactStage', (Prod,), {})
for attr in (
    '_operators', '_aliases', '_catalog', '_first_registered', '_overwrite_log',
    '_override_chain', '_canonical_manifests', '_DECLARED_OVERRIDE_MANIFEST',
    '_DECLARED_OVERRIDE_CONTRACT_HASHES',
):
    setattr(TestRegistry, attr, {})
from factor_engine.cleaned_operators.registry import _BOOTSTRAP_TOKEN
TestRegistry._lifecycle = Prod.Lifecycle.BUILDING
TestRegistry._version = 0
TestRegistry._mutation_token = _BOOTSTRAP_TOKEN
TestRegistry._frozen = None
TestRegistry._DECLARED_OVERRIDE_SOURCES = frozenset(Prod._DECLARED_OVERRIDE_SOURCES)
TestRegistry._hard_fail_duplicates = True
TestRegistry._auto_pin_declared_bootstrap = True
TestRegistry._enforce_override_chain_pinning = True

# Swap public bindings like sessionstart does
co.OperatorRegistry = TestRegistry
import factor_engine.cleaned_operators.registry as rmod
rmod.OperatorRegistry = TestRegistry

print('staged registry:', TestRegistry.__name__, 'lifecycle:', TestRegistry.lifecycle())

# Import the bootstrap modules in order (close to conftest's loop)
from factor_engine.cleaned_operators import BOOTSTRAP_MODULE_SPECS
mod = None
try:
    for spec in BOOTSTRAP_MODULE_SPECS:
        mod = spec.module
        importlib.import_module(mod)
    print('ALL bootstrap modules imported on exact-staged subclass OK')
except Exception as e:
    print('MODULE FAILED:', mod, '->', type(e).__name__, str(e)[:160])
    print('ops before failure:', len(TestRegistry._operators))
    print('catalog before failure:', len(TestRegistry._catalog))
    print('ts_mean_abs_deviation in ops?', 'ts_mean_abs_deviation' in TestRegistry._operators)
    print('has ts_mad?', 'ts_mad' in TestRegistry._operators)