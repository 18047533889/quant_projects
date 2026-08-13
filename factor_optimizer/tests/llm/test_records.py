"""Tests for LLM records module."""

from datetime import datetime
import json

from factor_optimizer.llm.records import LLMRecord, RecordStore, compute_hash


def test_compute_hash():
    """Test content hashing."""
    content1 = "test content"
    content2 = "test content"
    content3 = "different content"

    hash1 = compute_hash(content1)
    hash2 = compute_hash(content2)
    hash3 = compute_hash(content3)

    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 64  # SHA256 produces 64 hex chars


def test_llm_record_creation():
    """Test basic LLM record creation."""
    record = LLMRecord(
        record_id="rec_001",
        timestamp=datetime(2026, 8, 14, 10, 30, 0),
        model="claude-opus-4",
        model_version="20260801",
        prompt_template_id="mutation_proposal",
        prompt_template_version="1.0.0",
        prompt_hash="abc123",
        input_tokens=500,
        output_tokens=200,
        latency_ms=1234.5,
        response_hash="def456",
        metadata={"temperature": 0.7},
    )

    assert record.record_id == "rec_001"
    assert record.model == "claude-opus-4"
    assert record.input_tokens == 500
    assert record.metadata["temperature"] == 0.7


def test_llm_record_validation():
    """Test LLM record validation."""
    try:
        LLMRecord(
            record_id="",  # Invalid
            timestamp=datetime.now(),
            model="test",
            model_version=None,
            prompt_template_id="test",
            prompt_template_version="1.0.0",
            prompt_hash="hash",
            input_tokens=None,
            output_tokens=None,
            latency_ms=None,
            response_hash="hash",
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "record_id is required" in str(e)


def test_llm_record_serialization():
    """Test record serialization and deserialization."""
    original = LLMRecord(
        record_id="rec_002",
        timestamp=datetime(2026, 8, 14, 12, 0, 0),
        model="gpt-4",
        model_version=None,
        prompt_template_id="test_template",
        prompt_template_version="2.0.0",
        prompt_hash="prompt_hash_123",
        input_tokens=100,
        output_tokens=50,
        latency_ms=500.0,
        response_hash="response_hash_456",
        metadata={"max_tokens": 1000},
    )

    # Serialize
    data = original.to_dict()
    assert data["record_id"] == "rec_002"
    assert data["model"] == "gpt-4"
    assert isinstance(data["timestamp"], str)

    # Deserialize
    restored = LLMRecord.from_dict(data)
    assert restored.record_id == original.record_id
    assert restored.timestamp == original.timestamp
    assert restored.model == original.model
    assert restored.metadata == original.metadata


def test_record_store_basic_operations():
    """Test basic record store operations."""
    store = RecordStore()

    record1 = LLMRecord(
        record_id="rec_1",
        timestamp=datetime(2026, 8, 14, 10, 0, 0),
        model="model_a",
        model_version=None,
        prompt_template_id="template_1",
        prompt_template_version="1.0.0",
        prompt_hash="hash_1",
        input_tokens=100,
        output_tokens=50,
        latency_ms=100.0,
        response_hash="resp_1",
    )

    record2 = LLMRecord(
        record_id="rec_2",
        timestamp=datetime(2026, 8, 14, 11, 0, 0),
        model="model_b",
        model_version=None,
        prompt_template_id="template_2",
        prompt_template_version="1.0.0",
        prompt_hash="hash_2",
        input_tokens=200,
        output_tokens=100,
        latency_ms=200.0,
        response_hash="resp_2",
    )

    # Add records
    store.add(record1)
    store.add(record2)

    # Get by ID
    retrieved = store.get("rec_1")
    assert retrieved is not None
    assert retrieved.record_id == "rec_1"

    # List all
    all_records = store.list_all()
    assert len(all_records) == 2

    # Get non-existent
    assert store.get("rec_999") is None


def test_record_store_duplicate_prevention():
    """Test that duplicate record IDs are rejected."""
    store = RecordStore()

    record1 = LLMRecord(
        record_id="rec_dup",
        timestamp=datetime.now(),
        model="test",
        model_version=None,
        prompt_template_id="test",
        prompt_template_version="1.0.0",
        prompt_hash="hash",
        input_tokens=None,
        output_tokens=None,
        latency_ms=None,
        response_hash="hash",
    )

    record2 = LLMRecord(
        record_id="rec_dup",  # Same ID
        timestamp=datetime.now(),
        model="test",
        model_version=None,
        prompt_template_id="test",
        prompt_template_version="1.0.0",
        prompt_hash="hash",
        input_tokens=None,
        output_tokens=None,
        latency_ms=None,
        response_hash="hash",
    )

    store.add(record1)

    try:
        store.add(record2)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Duplicate record_id" in str(e)


def test_record_store_filtering():
    """Test filtering by model and template."""
    store = RecordStore()

    records = [
        LLMRecord(
            record_id=f"rec_{i}",
            timestamp=datetime.now(),
            model="model_a" if i % 2 == 0 else "model_b",
            model_version=None,
            prompt_template_id="template_1" if i < 3 else "template_2",
            prompt_template_version="1.0.0",
            prompt_hash=f"hash_{i}",
            input_tokens=100,
            output_tokens=50,
            latency_ms=100.0,
            response_hash=f"resp_{i}",
        )
        for i in range(6)
    ]

    for record in records:
        store.add(record)

    # Filter by model
    model_a_records = store.list_by_model("model_a")
    assert len(model_a_records) == 3
    assert all(r.model == "model_a" for r in model_a_records)

    # Filter by template
    template_1_records = store.list_by_template("template_1")
    assert len(template_1_records) == 3
    assert all(r.prompt_template_id == "template_1" for r in template_1_records)


def test_record_store_token_usage():
    """Test token usage calculation."""
    store = RecordStore()

    records = [
        LLMRecord(
            record_id=f"rec_{i}",
            timestamp=datetime.now(),
            model="test",
            model_version=None,
            prompt_template_id="test",
            prompt_template_version="1.0.0",
            prompt_hash="hash",
            input_tokens=100 * (i + 1),
            output_tokens=50 * (i + 1),
            latency_ms=100.0,
            response_hash="hash",
        )
        for i in range(3)
    ]

    for record in records:
        store.add(record)

    usage = store.total_tokens()
    assert usage["input_tokens"] == 600  # 100 + 200 + 300
    assert usage["output_tokens"] == 300  # 50 + 100 + 150
    assert usage["total_tokens"] == 900


def test_record_store_clear():
    """Test clearing the store."""
    store = RecordStore()

    record = LLMRecord(
        record_id="rec_1",
        timestamp=datetime.now(),
        model="test",
        model_version=None,
        prompt_template_id="test",
        prompt_template_version="1.0.0",
        prompt_hash="hash",
        input_tokens=None,
        output_tokens=None,
        latency_ms=None,
        response_hash="hash",
    )

    store.add(record)
    assert len(store.list_all()) == 1

    store.clear()
    assert len(store.list_all()) == 0
    assert store.get("rec_1") is None


def test_record_store_export_import():
    """Test JSON export and import."""
    store = RecordStore()

    record = LLMRecord(
        record_id="rec_export",
        timestamp=datetime(2026, 8, 14, 15, 0, 0),
        model="test_model",
        model_version="v1",
        prompt_template_id="test_template",
        prompt_template_version="1.0.0",
        prompt_hash="test_hash",
        input_tokens=150,
        output_tokens=75,
        latency_ms=250.0,
        response_hash="resp_hash",
        metadata={"test": "value"},
    )

    store.add(record)

    # Export
    json_str = store.export_json()
    assert "rec_export" in json_str
    assert "test_model" in json_str

    # Import to new store
    new_store = RecordStore()
    count = new_store.import_json(json_str)
    assert count == 1

    # Verify imported record
    imported = new_store.get("rec_export")
    assert imported is not None
    assert imported.model == "test_model"
    assert imported.input_tokens == 150
    assert imported.metadata["test"] == "value"


def test_record_store_import_skips_duplicates():
    """Test that import skips duplicate IDs."""
    store = RecordStore()

    record = LLMRecord(
        record_id="rec_dup",
        timestamp=datetime.now(),
        model="test",
        model_version=None,
        prompt_template_id="test",
        prompt_template_version="1.0.0",
        prompt_hash="hash",
        input_tokens=None,
        output_tokens=None,
        latency_ms=None,
        response_hash="hash",
    )

    store.add(record)
    json_str = store.export_json()

    # Import again should skip the duplicate
    count = store.import_json(json_str)
    assert count == 0
    assert len(store.list_all()) == 1
