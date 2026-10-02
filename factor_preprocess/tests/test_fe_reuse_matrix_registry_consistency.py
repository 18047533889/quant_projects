"""Keep the FE reuse matrix routing summary aligned with the live registry."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

from factor_preprocess.registry.transforms import get_default_registry


def test_routing_summary_matches_registry_counts_and_names():
    registry = get_default_registry()
    origins = ("FE_OPERATOR", "FE_COMPOSITE", "FP_NATIVE")
    actual = {
        origin: {
            meta.name for meta in registry.all_transforms()
            if meta.implementation_origin == origin
        }
        for origin in origins
    }
    counts = Counter(
        meta.implementation_origin for meta in registry.all_transforms()
    )
    assert set(counts) == set(origins), "document every registered routing category"

    matrix = Path(__file__).resolve().parents[1] / "docs" / "FE_OPERATOR_REUSE_MATRIX.md"
    text = matrix.read_text(encoding="utf-8")
    summary = text.split("## Routing summary (R61-FI-041)", 1)[1].split(
        "\nParity note", 1
    )[0]

    for origin in origins:
        match = re.search(
            rf"\*\*{origin} \((\d+)\):\*\*(.*?)(?=\n- \*\*|\Z)",
            summary,
            flags=re.DOTALL,
        )
        assert match, f"missing {origin} routing summary"
        documented_count = int(match.group(1))
        documented_names = set(re.findall(r"`([^`]+)`", match.group(2)))
        assert documented_count == counts[origin] == len(actual[origin])
        if origin != "FP_NATIVE":
            assert documented_names == actual[origin]
