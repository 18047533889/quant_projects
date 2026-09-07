import pyarrow as pa
import pyarrow.parquet as pq

from factor_engine.storage.parquet_batch_writer import BatchParquetWriter


def test_schema_inference_does_not_drop_first_iterator_batch(tmp_path):
    batches = (pa.record_batch({"value": [value]}) for value in (11., 22., 33.))
    path = tmp_path / "values.parquet"
    result = BatchParquetWriter.write_batches(batches, None, path, compression="ZSTD-1")
    assert result.num_rows == 3
    assert pq.read_table(path)["value"].to_pylist() == [11., 22., 33.]


def test_single_batch_schema_inference_keeps_only_batch(tmp_path):
    path = tmp_path / "one.parquet"
    BatchParquetWriter.write_batches(iter([pa.record_batch({"value": [7.]})]), None, path)
    assert pq.read_table(path)["value"].to_pylist() == [7.]
