from pathlib import Path
import re
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
        assert "```math" in definition


def test_generated_reference_is_current():
    from quant_evaluator.scripts.build_metric_reference import OUTPUT, render
    assert OUTPUT.read_text() == render()


def test_github_identifier_and_comparison_regressions():
    from quant_evaluator.scripts.build_metric_reference import portable_math
    for label in ("train_predictive_dimension", "validation_predictive_dimension"):
        result = portable_math("$$D_f=A." + r"\mathrm{" + label + "}_f$$")
        assert label not in result
        assert label.replace("_", r"\_") in result
    result = portable_math(r"$$e=\mathrm{mean}_{t<m}IC_t,\quad M=|e-l|$$")
    assert r"_{t\lt m}" in result
    assert "<" not in result
    assert r"\gt " in portable_math(r"$a>b$")


def test_portable_math_regressions_from_github_screenshots():
    from quant_evaluator.scripts.build_metric_reference import portable_math
    example = r"$$H=Q(\bar x^*)-Q(\bar x^*)+\operatorname{mean}(x)$$"
    fixed = portable_math(example)
    assert r"\operatorname" not in fixed
    assert "*" not in fixed
    assert fixed.count(r"x^{\ast}") == 2
    assert r"\mathrm{mean}" in fixed
    assert portable_math(fixed) == fixed
    inline = portable_math(r"$r_1-\operatorname{mean}(r_2,r_3,r_4)$")
    assert inline == r"$`r_1-\mathrm{mean}(r_2,r_3,r_4)`$"
    assert portable_math(inline) == inline


def test_published_math_has_no_known_markdown_or_macro_hazards():
    folder = Path(__file__).resolve().parents[1] / "docs"
    for name in ("METRIC_REFERENCE.md", "METRIC_CONVENTIONS.md"):
        text = (folder / name).read_text()
        for match in re.finditer(r"```math\n([\s\S]*?)\n```|(?<!\$)\$([^$\n]+)\$(?!\$)", text):
            body = match.group(1) if match.group(1) is not None else match.group(2)
            assert r"\operatorname" not in body, name
            assert "*" not in body, name


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
