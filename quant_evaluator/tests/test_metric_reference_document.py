from pathlib import Path
from quant_evaluator.registry.metrics import list_metrics


def test_document_covers_every_registered_metric_once():
    path = Path(__file__).resolve().parents[1] / "docs/METRIC_REFERENCE.md"
    text = path.read_text()
    for metric_id in list_metrics():
        assert text.count(f'<a id="metric-{metric_id}"></a>') == 1


def test_every_metric_has_readable_math_without_source_dump():
    path = Path(__file__).resolve().parents[1] / "docs/METRIC_REFERENCE.md"
    text = path.read_text()
    assert "```python" not in text
    for section in text.split('<a id="metric-')[1:]:
        definition = section.split('\n## ')[1]
        assert "### 数学公式与计算口径" in definition
        assert "$$" in definition


def test_generated_reference_is_current():
    from quant_evaluator.scripts.build_metric_reference import OUTPUT, render
    assert OUTPUT.read_text() == render()


def test_formula_sources_have_exact_coverage_and_balanced_math():
    import json
    import re
    folder = Path(__file__).resolve().parents[1] / "docs"
    entries = {}
    for path in sorted(folder.glob("formulas_*.json")):
        part = json.loads(path.read_text())
        assert not (set(entries) & set(part))
        entries.update(part)
    assert set(entries) == set(list_metrics())
    for metric_id, definition in entries.items():
        assert not re.search(r"[\x00-\x08\x0b-\x1f]", definition), metric_id
        assert definition.count("$$") >= 2, metric_id
        assert definition.count("$$") % 2 == 0, metric_id
        for math in definition.split("$$")[1::2]:
            depth = 0
            for token in re.findall(r"\\.|[{}]", math):
                if token == "{":
                    depth += 1
                elif token == "}":
                    depth -= 1
                assert depth >= 0, metric_id
            assert depth == 0, metric_id
