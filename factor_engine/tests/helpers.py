from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from storage.datasource import DataSource


@dataclass
class InMemorySeriesSource(DataSource):
    data: dict[str, pd.Series]

    def load_column(self, name: str) -> Any:
        return self.data[name]

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        return {n: self.data[n] for n in names if n in self.data}
