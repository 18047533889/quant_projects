"""Immutable, content-addressed static symbol adjacency matrices."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StaticAdjacency:
    """A directed N×N graph whose row is source and column is destination."""

    symbols: tuple[Any, ...]
    matrix: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        symbols_list = []
        for symbol in self.symbols:
            if isinstance(symbol, (bool, np.bool_)) or not isinstance(symbol, (str, Integral)):
                raise TypeError("StaticAdjacency symbols must be strings or integers (not booleans)")
            symbols_list.append(int(symbol) if isinstance(symbol, Integral) else symbol)
        symbols = tuple(symbols_list)
        unique = len(set(symbols)) == len(symbols)
        if not unique:
            raise ValueError("StaticAdjacency symbols must be unique")
        matrix_rows = []
        for row in self.matrix:
            converted = []
            for value in row:
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                    raise TypeError("StaticAdjacency weights must be real numbers (not booleans or strings)")
                converted.append(float(value))
            matrix_rows.append(tuple(converted))
        matrix = tuple(matrix_rows)
        n = len(symbols)
        if len(matrix) != n or any(len(row) != n for row in matrix):
            raise ValueError(f"StaticAdjacency matrix must be square {n}x{n}")
        if not all(np.isfinite(value) for row in matrix for value in row):
            raise ValueError("StaticAdjacency weights must be finite")
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "StaticAdjacency":
        """Freeze a square DataFrame with identical unique row/column axes."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("similarity must be StaticAdjacency or a pandas DataFrame")
        if not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError("adjacency source and destination symbol axes must be unique")
        if list(frame.index) != list(frame.columns):
            raise ValueError("adjacency index and columns must contain the same symbols in the same order")
        values = frame.to_numpy(dtype=object)
        return cls(tuple(frame.index), tuple(tuple(row) for row in values))

    def stable_hash(self) -> str:
        """Stable content hash used by tests, manifests, and plan identities."""
        typed_symbols = [
            ["integer", symbol] if isinstance(symbol, int) else ["string", symbol]
            for symbol in self.symbols
        ]
        payload = {"symbols": typed_symbols, "matrix": [list(row) for row in self.matrix]}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def values_for(self, symbols: Any) -> np.ndarray:
        requested = tuple(symbols)
        if requested != self.symbols:
            raise ValueError("StaticAdjacency symbols/order must exactly match x columns")
        return np.asarray(self.matrix, dtype=float)


def coerce_static_adjacency(value: Any) -> StaticAdjacency:
    if isinstance(value, StaticAdjacency):
        return value
    if isinstance(value, pd.DataFrame):
        return StaticAdjacency.from_frame(value)
    raise TypeError("similarity must be StaticAdjacency or a square pandas DataFrame")
