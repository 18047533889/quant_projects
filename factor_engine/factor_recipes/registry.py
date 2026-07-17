# -*- coding: utf-8 -*-
"""Declarative recipes expanded to primitive operators before Factor DSL execution."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FactorRecipe:
    name: str
    category: str
    description: str
    expression: str
    parameters: tuple[str, ...]
    status: str = "production"
    replacement_for: tuple[str, ...] = ()


class FactorRecipeRegistry:
    _recipes: dict[str, FactorRecipe] = {}

    @classmethod
    def register(cls, recipe: FactorRecipe) -> None:
        if recipe.name in cls._recipes:
            raise ValueError(f"duplicate factor recipe: {recipe.name}")
        cls._recipes[recipe.name] = recipe

    @classmethod
    def get(cls, name: str) -> FactorRecipe | None:
        return cls._recipes.get(name)

    @classmethod
    def list_names(cls, *, status: str | None = None) -> list[str]:
        return sorted(
            name
            for name, recipe in cls._recipes.items()
            if status is None or recipe.status == status
        )

    @classmethod
    def catalog(cls) -> dict[str, dict]:
        return {name: deepcopy(asdict(recipe)) for name, recipe in sorted(cls._recipes.items())}
