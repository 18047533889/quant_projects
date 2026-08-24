"""兼容 shim：storage.sources.staging_loader"""
import sys
import factor_engine.storage.sources.staging_loader as _mod
sys.modules[__name__] = _mod
