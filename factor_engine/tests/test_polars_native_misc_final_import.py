from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parents[1]


def test_misc_final_imports_with_authoritative_metadata_abi(monkeypatch) -> None:
    import factor_engine.cleaned_operators.base_polars as base_polars

    monkeypatch.setattr(base_polars, "register_operator", lambda **_kwargs: lambda cls: cls)
    module_name = "_test_polars_native_misc_final"
    spec = importlib.util.spec_from_file_location(
        module_name,
        _ROOT / "cleaned_operators/polars_native/misc_final.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    metadata_fields = set(base_polars.OperatorMetadata.__dataclass_fields__)
    exported_classes = [getattr(module, name) for name in module.__all__]
    assert len(exported_classes) == 9
    assert all(set(vars(cls.metadata)) <= metadata_fields for cls in exported_classes)
