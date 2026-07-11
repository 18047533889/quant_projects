from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
yaml = pytest.importorskip("yaml")

from runtime.config import load_config
from runtime.engine import FactorEngine


def test_run_from_config_with_kline_parquet_source(tmp_path: Path):
    root = tmp_path / 'day_aggs_v1' / '2024' / '01'
    root.mkdir(parents=True)

    for day, rows in {
        '2024-01-01': [('AAA', 10.0), ('BBB', 20.0)],
        '2024-01-02': [('AAA', 11.0), ('BBB', 19.0)],
        '2024-01-03': [('AAA', 12.0), ('BBB', 18.0)],
    }.items():
        frame = pd.DataFrame(
            {
                'ticker': [ticker for ticker, _ in rows],
                'window_start': [pd.Timestamp(day, tz='UTC').value for _ in rows],
                'close': [close for _, close in rows],
                'open': [close for _, close in rows],
                'high': [close + 1 for _, close in rows],
                'low': [close - 1 for _, close in rows],
                'volume': [1000.0 for _ in rows],
                'transactions': [10.0 for _ in rows],
            }
        )
        frame.to_parquet(root / f'{day}.parquet', index=False)

    config_path = tmp_path / 'factor.yaml'
    config_path.write_text(
        yaml.safe_dump(
            {
                'factor': {
                    'name': 'mom_2_rank',
                    'expr': 'rank(ts_mean(col("close"), 2))',
                },
                'data_source': {
                    'type': 'parquet_kline',
                    'root': str(tmp_path / 'day_aggs_v1'),
                    'instrument_column': 'ticker',
                    'timestamp_column': 'window_start',
                    'fields': {'close': 'close'},
                },
                'backend': {'type': 'pandas'},
                'engine': {'enable_cache': True},
            }
        )
    )

    out = FactorEngine.run_from_config(config_path)
    result = out['result']

    assert out['factor'].name == 'mom_2_rank'
    assert out['analysis'].lookback == 2
    assert result.loc[(pd.Timestamp('2024-01-03'), 'AAA')] == 0.5
    assert result.loc[(pd.Timestamp('2024-01-03'), 'BBB')] == 1.0


def test_load_config_defaults_materialization_lake_root_to_workspace_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    workspace_root = tmp_path / "workspace_data"
    monkeypatch.setenv("QUANTSOCIETY_WORKSPACE_DATA_ROOT", str(workspace_root))

    config_path = tmp_path / 'factor_with_materialization.yaml'
    config_path.write_text(
        yaml.safe_dump(
            {
                'factor': {
                    'name': 'demo_factor',
                    'expr': 'col("close")',
                },
                'data_source': {
                    'type': 'parquet_kline',
                    'root': str(tmp_path / 'day_aggs_v1'),
                },
                'materialization': {},
            }
        )
    )

    config = load_config(config_path)

    assert config.materialization is not None
    assert config.materialization.lake_root == str(workspace_root / 'factors' / 'lake')
