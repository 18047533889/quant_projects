import csv
import gzip
import pytest
from export_review import export_review
from test_export_review import packed


@pytest.mark.parametrize('filename', ['review.csv', 'review.csv.gz'])
def test_failed_reconciliation_does_not_publish_partial_csv(tmp_path, filename):
    row = {'source_row': 2, 'id': 'A', 'formula': 'close'}
    source = packed(tmp_path / 'source.gz', [row, dict(row, source_row=3, id='B')])
    audit = packed(tmp_path / 'audit.gz', [dict(row, status='PARSED_FIELDS_BOUND')])
    output = tmp_path / filename
    with pytest.raises(ValueError, match='row counts'):
        export_review(source, audit, output)
    assert not output.exists()
    assert not list(tmp_path.glob('.review-*'))


def test_retry_success_keeps_batch_failure_visible(tmp_path):
    row = {'source_row': 2, 'id': 'A', 'formula': 'close'}
    source = packed(tmp_path / 'source.gz', [row])
    audit = packed(tmp_path / 'audit.gz', [dict(row, status='PARSED_FIELDS_BOUND')])
    smoke = packed(tmp_path / 'smoke.gz', [dict(
        row, executed_formula='close', status='EXECUTED',
        retry_after_batch_abort=True, batch_abort_error_type='RuntimeError',
        batch_abort_error='missing shared result', value_count=4, finite_count=4)])
    output = tmp_path / 'review.csv.gz'
    export_review(source, audit, output, [smoke])
    with gzip.open(output, 'rt', encoding='utf-8-sig', newline='') as stream:
        checked = next(csv.DictReader(stream))
    assert checked['execution_status'] == 'EXECUTED'
    assert checked['retry_after_batch_abort'] == 'True'
    assert checked['batch_abort_error_type'] == 'RuntimeError'
    assert checked['batch_abort_error'] == 'missing shared result'


def test_export_has_current_physical_fields_without_losing_originals(tmp_path):
    row = {'source_row': 2, 'id': 'A', 'formula': 'Close',
           'fields': 'Close', 'tables': 'price_volume'}
    source = packed(tmp_path / 'source.gz', [row])
    audit = packed(tmp_path / 'audit.gz', [dict(
        row, status='PARSED_FIELDS_BOUND', current_formula='close',
        bindings=[{'table': 'StockDailyBarAdj', 'column': 'AdjClose'}])])
    output = tmp_path / 'review.csv.gz'
    export_review(source, audit, output)
    with gzip.open(output, 'rt', encoding='utf-8-sig', newline='') as stream:
        checked = next(csv.DictReader(stream))
    assert checked['original_fields'] == 'Close'
    assert checked['original_tables'] == 'price_volume'
    assert checked['current_fields'] == 'StockDailyBarAdj.AdjClose'
    assert checked['current_tables'] == 'StockDailyBarAdj'
    assert checked['execution_status'] == 'NOT_RUN'
