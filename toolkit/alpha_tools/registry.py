"""Registry and runtime facade for active alpha tools."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

from toolkit.alpha_tools import library

_ROOT = Path(__file__).resolve().parent
_GENERATED_REGISTRY_PATH = _ROOT / "generated_registry.json"
_INACTIVE_REGISTRY_PATH = _ROOT / "inactive_registry.json"


SEED_TOOL_SPECS: dict[str, dict[str, Any]] = {
    "decompose_overnight_intraday": {
        "version": 1,
        "economic_module": "return_decomposition",
        "input_contract": ["series", "series"],
        "return_contract": {
            "kind": "tuple",
            "items": [
                {"name": "overnight_ret", "kind": "series"},
                {"name": "intraday_ret", "kind": "series"},
            ],
        },
        "parameters": {},
        "source": "seed_report_disclosed",
    },
    "classify_volume_regime": {
        "version": 1,
        "economic_module": "volume_state",
        "input_contract": ["series"],
        "return_contract": {
            "kind": "tuple",
            "items": [
                {"name": "is_high_vol", "kind": "series"},
                {"name": "is_low_vol", "kind": "series"},
                {"name": "vol_ratio", "kind": "series"},
            ],
        },
        "parameters": {
            "window": {"type": "int", "default": 20, "min": 2, "max": 252},
            "high_threshold": {"type": "float", "default": 1.5, "min": 1.0, "max": 10.0},
            "low_threshold": {"type": "float", "default": 0.6, "min": 0.0, "max": 1.0},
        },
        "source": "seed_report_disclosed",
    },
}


class AlphaToolsFacade:
    """Runtime object exposing only active alpha tool callables."""

    __slots__ = ("_functions",)

    def __init__(self, functions: dict[str, Any]) -> None:
        object.__setattr__(self, "_functions", MappingProxyType(dict(functions)))

    def __getattr__(self, name: str) -> Any:
        functions = object.__getattribute__(self, "_functions")
        if name in functions:
            return functions[name]
        raise AttributeError(f"alpha_tools has no active tool {name!r}.")

    def __dir__(self) -> list[str]:
        return sorted(object.__getattribute__(self, "_functions"))

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ARG002
        raise AttributeError("alpha_tools is read-only.")


def build_alpha_tools_facade() -> AlphaToolsFacade:
    """Build the read-only runtime facade for generated alpha functions."""

    return AlphaToolsFacade(get_active_tool_functions())


def get_active_tool_names() -> tuple[str, ...]:
    return tuple(sorted(get_active_tool_specs()))


def get_inactive_tool_names() -> tuple[str, ...]:
    return tuple(sorted(load_inactive_tool_specs()))


def get_active_tool_specs() -> dict[str, dict[str, Any]]:
    specs = _with_hashes(SEED_TOOL_SPECS, get_seed_tool_functions())
    generated = load_generated_tool_specs()
    specs.update(generated)
    return specs


def get_active_tool_functions() -> dict[str, Any]:
    functions = get_seed_tool_functions()
    functions.update(load_generated_tool_functions())
    return functions


def get_seed_tool_functions() -> dict[str, Any]:
    return {name: getattr(library, name) for name in SEED_TOOL_SPECS}


def load_generated_tool_specs() -> dict[str, dict[str, Any]]:
    payload = _read_json(_GENERATED_REGISTRY_PATH, {"active_tools": {}})
    specs = payload.get("active_tools", {})
    functions = load_generated_tool_functions()
    callable_specs = {name: spec for name, spec in specs.items() if name in functions}
    return _with_hashes(callable_specs, functions)


def load_generated_tool_functions() -> dict[str, Any]:
    try:
        module = importlib.import_module("toolkit.alpha_tools.generated_library")
    except Exception:  # noqa: BLE001 - generated file must never break startup
        return {}
    specs = _read_json(_GENERATED_REGISTRY_PATH, {"active_tools": {}}).get(
        "active_tools", {}
    )
    functions: dict[str, Any] = {}
    for name in specs:
        value = getattr(module, name, None)
        if callable(value):
            functions[name] = value
    return functions


def load_inactive_tool_specs() -> dict[str, dict[str, Any]]:
    payload = _read_json(_INACTIVE_REGISTRY_PATH, {"inactive_tools": {}})
    return payload.get("inactive_tools", {})


def validate_tool_parameter_value(
    tool_name: str,
    parameter_name: str,
    value: object,
    bound_parameters: dict[str, object] | None = None,
) -> str | None:
    """Return an error message if a literal parameter value violates registry bounds."""

    spec = get_active_tool_specs().get(tool_name)
    if not spec:
        return f"Unknown active alpha tool {tool_name!r}."
    parameter = spec.get("parameters", {}).get(parameter_name)
    if parameter is None:
        return f"Unknown parameter {parameter_name!r} for alpha_tools.{tool_name}."
    if value is None and parameter.get("default") is None:
        return None
    expected = parameter.get("type")
    if expected == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{tool_name}.{parameter_name} must be an integer literal."
    elif expected == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{tool_name}.{parameter_name} must be a numeric literal."
    minimum = parameter.get("min")
    maximum = parameter.get("max")
    if "max_ref" in parameter and bound_parameters:
        maximum = bound_parameters.get(parameter["max_ref"], maximum)
    if value is not None and minimum is not None and value < minimum:
        return f"{tool_name}.{parameter_name} must be >= {minimum}."
    if value is not None and maximum is not None and value > maximum:
        return f"{tool_name}.{parameter_name} must be <= {maximum}."
    return None


def _with_hashes(
    specs: dict[str, dict[str, Any]],
    functions: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name, spec in specs.items():
        enriched = dict(spec)
        enriched["name"] = name
        enriched.setdefault("status", "active")
        enriched.setdefault("return_contract", {"kind": "series", "name": name})
        if "input_contract" in enriched:
            enriched["data_args"] = len(enriched["input_contract"])
        function = functions.get(name)
        if function is not None:
            try:
                source = inspect.getsource(function)
            except OSError:
                source = repr(function)
            enriched["hash"] = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
        output[name] = enriched
    return output


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
