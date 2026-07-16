"""兼容 shim：runtime.reconcile / quality"""
import sys
import runtime.reconcile.snapshot_reconcile as _mod
sys.modules[__name__] = _mod
