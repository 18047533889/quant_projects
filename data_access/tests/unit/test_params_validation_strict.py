from __future__ import annotations

import pytest

from data_access.core.exceptions import ValidationError
from data_access.registry.params_validation import ParamSpec, parse_params_schema, validate_params


def test_int_parameter_rejects_fractional_values() -> None:
    specs = {"year": ParamSpec(name="year", type="int", min_int=2000, max_int=2100)}

    with pytest.raises(ValidationError, match="必须是整数"):
        validate_params("demo", specs, {"year": 2024.5})
    with pytest.raises(ValidationError, match="必须是整数"):
        validate_params("demo", specs, {"year": "2024.5"})

    assert validate_params("demo", specs, {"year": "2024"}) == {"year": 2024}


def test_path_like_parameter_rejects_traversal_segments() -> None:
    specs = {"factor_id": ParamSpec(name="factor_id", type="str")}

    for value in (".", "..", "a/b", r"a\b"):
        with pytest.raises(ValidationError):
            validate_params("factor_lake", specs, {"factor_id": value})


def test_schema_definition_is_validated_eagerly() -> None:
    with pytest.raises(ValidationError, match="仅支持"):
        parse_params_schema({"factor_id": {"type": "float"}})
    with pytest.raises(ValidationError, match="合法正则"):
        parse_params_schema({"factor_id": {"type": "str", "pattern": "["}})
    with pytest.raises(ValidationError, match="min 不能大于 max"):
        parse_params_schema({"year": {"type": "int", "min": 2030, "max": 2020}})
    with pytest.raises(ValidationError, match="合法 Python 标识符"):
        parse_params_schema({"factor-id": "str"})


def test_enum_and_default_path_segment_pattern() -> None:
    specs = parse_params_schema(
        {
            "kind": {
                "type": "str",
                "values": ["trades_v1", "quotes_v1"],
                "path_segment": True,
            }
        }
    )

    assert validate_params("ticks", specs, {"kind": "trades_v1"}) == {
        "kind": "trades_v1"
    }
    with pytest.raises(ValidationError, match="必须是"):
        validate_params("ticks", specs, {"kind": "other"})
