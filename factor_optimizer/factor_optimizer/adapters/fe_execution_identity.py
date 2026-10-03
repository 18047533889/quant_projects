"""Shared construction of execution identities for resolved FE operators."""
from __future__ import annotations

import hashlib
import inspect
from collections.abc import Mapping


def build_fe_operator_identity(*, canonical: str, backend: str, mode: str,
                               operator, adapter, input_contract: str,
                               semantic_version, versions: Mapping[str, str]) -> dict:
    """Build a binding after the caller has selected the exact runtime operator.

    Backend selection intentionally stays with each adapter so this helper
    cannot change fallback order or substitute a registration.
    """
    from factor_engine.cleaned_operators.registry import (
        _contract_hash, _impl_source_hash,
    )

    try:
        adapter_source = inspect.getsource(adapter)
        implementation_hash = _impl_source_hash(operator)
        contract_hash = _contract_hash(operator)
    except Exception as exc:
        raise RuntimeError("cannot certify the selected FactorEngine implementation") from exc
    if not isinstance(implementation_hash, str) or not implementation_hash:
        raise RuntimeError("FactorEngine implementation identity is empty")
    if not isinstance(contract_hash, str) or not contract_hash:
        raise RuntimeError("FactorEngine contract identity is empty")
    if any(not isinstance(key, str) or not isinstance(value, str) or not value
           for key, value in versions.items()):
        raise RuntimeError("FactorEngine execution version identity is incomplete")
    if not isinstance(semantic_version, str):
        raise RuntimeError("FactorEngine semantic version identity is uncertifiable")

    return {
        "canonical": canonical,
        "backend": backend,
        "mode": mode,
        "semantic_version": semantic_version,
        "operator_type": {
            "module": type(operator).__module__,
            "qualname": type(operator).__qualname__,
        },
        "operator_implementation_hash": implementation_hash,
        "operator_contract_hash": contract_hash,
        "adapter_implementation_hash": hashlib.sha256(
            adapter_source.encode("utf-8")).hexdigest(),
        "input_contract": input_contract,
        **dict(versions),
    }


__all__ = ["build_fe_operator_identity"]
