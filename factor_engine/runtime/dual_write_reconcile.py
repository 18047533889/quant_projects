"""兼容 shim：runtime.reconcile / quality"""
import sys
import runtime.reconcile.dual_write_reconcile as _mod
sys.modules[__name__] = _mod
