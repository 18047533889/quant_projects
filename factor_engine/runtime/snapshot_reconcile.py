"""兼容 shim：runtime.reconcile / quality"""
import sys
import factor_engine.runtime.reconcile.snapshot_reconcile as _mod
sys.modules[__name__] = _mod
