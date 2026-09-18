from pathlib import Path
from quant_evaluator.registry.metrics import list_metrics


def test_document_covers_every_registered_metric_once():
    path = Path(__file__).resolve().parents[1] / "docs/METRIC_REFERENCE.md"
    text = path.read_text()
    for metric_id in list_metrics():
        assert text.count(f'<a id="metric-{metric_id}"></a>') == 1
