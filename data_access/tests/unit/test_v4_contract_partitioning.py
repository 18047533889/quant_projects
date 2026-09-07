from types import SimpleNamespace

import pytest

from data_access.core.exceptions import ValidationError
from data_access.read.contract_ir import build_contract_ir
from data_access.read.partition_planner import PartitionSpec, TimePartitionSpec


def ir_for(partitioning):
    dataset = SimpleNamespace(partitioning=partitioning, schema={"value": "double"})
    registry = SimpleNamespace(names=lambda: ["demo"], get=lambda name: dataset)
    return build_contract_ir(registry, contracts={})


def test_typed_and_mapping_partition_contracts_have_equal_fingerprints():
    typed = ir_for(PartitionSpec(time=TimePartitionSpec(frequency="monthly"), hive=("year", "month")))
    mapping = ir_for({"time": {"frequency": "monthly"}, "hive": ["year", "month"]})
    assert typed.to_dict() == mapping.to_dict()
    assert typed.fingerprint() == mapping.fingerprint()
    assert typed.to_dict()["demo"]["partitioning"] == {
        "time": {"source": "filename", "field": "date", "frequency": "monthly", "pattern": None},
        "hive": ("year", "month"),
    }
    changed = ir_for(PartitionSpec(time=TimePartitionSpec(frequency="daily"), hive=("year", "month")))
    assert changed.fingerprint() != typed.fingerprint()


def test_absent_partitioning_stays_empty():
    assert ir_for(None).to_dict()["demo"]["partitioning"] == {}


def test_invalid_partitioning_is_not_silently_serialized():
    with pytest.raises(ValidationError, match="未知"):
        ir_for({"unknown_partition_key": "monthly"})
