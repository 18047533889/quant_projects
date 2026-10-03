"""Parameter guards shared by treatment recipe compilation."""
from __future__ import annotations

import inspect
from collections.abc import Mapping
from inspect import Parameter
from typing import Any, Callable

from factor_preprocess.errors import InvalidContractError

_RUNTIME_INPUT_ALIASES = frozenset({"values", "x", "data"})
_FIXED_POSITIONAL_KINDS = (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)


def validate_recipe_runtime_input_binding(
    func: Callable,
    parameters: Mapping[str, Any],
    *,
    transform_name: str,
) -> None:
    """Reject recipe keyword configuration of the transform's runtime input.

    The first fixed positional parameter is the runtime input by convention.
    When the signature has no fixed positional parameter, recognized input
    aliases provide the boundary; variadic keyword functions reserve those
    aliases. Other configured parameters remain available to the transform.
    """
    if not callable(func):
        raise InvalidContractError(
            f"Transform {transform_name!r} has no callable signature to validate"
        )
    if not isinstance(parameters, Mapping):
        raise InvalidContractError("recipe transform parameters must be a mapping")
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError) as exc:
        raise InvalidContractError(
            f"Transform {transform_name!r} has no inspectable runtime input signature"
        ) from exc

    declared = tuple(signature.parameters.values())
    first_positional = next(
        (parameter for parameter in declared if parameter.kind in _FIXED_POSITIONAL_KINDS),
        None,
    )
    if first_positional is not None:
        input_name = first_positional.name
    else:
        named_alias = next(
            (parameter for parameter in declared if parameter.name in _RUNTIME_INPUT_ALIASES),
            None,
        )
        if named_alias is not None:
            input_name = named_alias.name
        elif any(parameter.kind is Parameter.VAR_KEYWORD for parameter in declared):
            shadowed_aliases = _RUNTIME_INPUT_ALIASES.intersection(parameters)
            if shadowed_aliases:
                names = ", ".join(sorted(shadowed_aliases))
                raise InvalidContractError(
                    f"recipe parameters for transform {transform_name!r} bind reserved "
                    f"runtime input alias(es): {names}"
                )
            return
        else:
            raise InvalidContractError(
                f"Transform {transform_name!r} has no identifiable runtime input parameter"
            )

    if input_name in parameters:
        raise InvalidContractError(
            f"recipe parameters for transform {transform_name!r} cannot configure its "
            f"runtime input parameter {input_name!r}"
        )


__all__ = ["validate_recipe_runtime_input_binding"]
