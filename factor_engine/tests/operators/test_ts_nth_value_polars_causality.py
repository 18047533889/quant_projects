from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import types

import pytest

pl = pytest.importorskip("polars")


def _load_target_class():
    package_name = "factor_engine.cleaned_operators.polars_native"
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(__file__).parents[2] / "cleaned_operators" / "polars_native")]
    sys.modules[package_name] = package

    source_path = Path(package.__path__[0]) / "ts_batch1.py"
    source = source_path.read_text()
    tree = ast.parse(source, source_path)
    selected = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        or (isinstance(node, ast.ClassDef) and node.name == "TSNthValuePolarsNative")
    ]
    module_name = f"{package_name}._ts_nth_value_test_target"
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(source_path)
    sys.modules[module_name] = module
    exec(compile(ast.Module(selected, type_ignores=[]), source_path, "exec"), module.__dict__)
    return module.TSNthValuePolarsNative


TSNthValuePolarsNative = _load_target_class()


def test_ts_nth_value_positive_n_is_historical_and_prefix_causal() -> None:
    op = TSNthValuePolarsNative()
    prefix = pl.Series("x", [10.0, 20.0, 30.0, 40.0])
    poisoned = prefix.append(pl.Series("x", [1_000_000.0, -1_000_000.0]))

    actual = op.calculate(prefix, 2).to_list()
    poisoned_prefix = op.calculate(poisoned, 2).head(len(prefix)).to_list()
    assert actual[:2] == [None, None]
    assert actual[2] == pytest.approx(10.0)
    assert actual[3] == pytest.approx(20.0)
    assert poisoned_prefix[:2] == [None, None]
    assert poisoned_prefix[2] == pytest.approx(10.0)
    assert poisoned_prefix[3] == pytest.approx(20.0)


@pytest.mark.parametrize("n", [0, -1, 1.5, True])
def test_ts_nth_value_rejects_invalid_lag(n: object) -> None:
    op = TSNthValuePolarsNative()

    with pytest.raises((TypeError, ValueError)):
        op.calculate(pl.Series("x", [10.0, 20.0, 30.0]), n)


def test_ts_nth_value_registration_metadata_is_pit_safe_lag() -> None:
    metadata = TSNthValuePolarsNative.metadata

    assert "pit_safe" in metadata.tags
    assert "lag" in metadata.tags
    assert metadata.param_specs["n"].min == 1
    assert metadata.param_specs["n"].history_semantics == "exact_rows"
