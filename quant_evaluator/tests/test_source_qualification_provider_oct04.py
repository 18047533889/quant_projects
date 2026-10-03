from types import SimpleNamespace

import pytest

from quant_evaluator.runtime import source_qualification_provider as provider

FINGERPRINT = "a" * 64
OTHER_FINGERPRINT = "b" * 64


def _report(fingerprint=FINGERPRINT):
    context = SimpleNamespace(request_content_sha256=fingerprint)
    records = (SimpleNamespace(context=context), SimpleNamespace(context=context))
    return SimpleNamespace(records=records, manifest_sha256="c" * 64)


def test_configure_is_lazy_and_lookup_returns_unverified_candidate(tmp_path, monkeypatch):
    path = tmp_path / "does-not-exist.json"
    loaded = []
    monkeypatch.setattr(provider, "load_source_profile_report",
                        lambda item: loaded.append(item) or _report())

    configured = provider.FileSourceQualificationProvider({FINGERPRINT: path})
    assert loaded == []
    provider.configure_source_qualification_provider(configured)
    result = provider.lookup_source_qualification_candidate(FINGERPRINT)

    assert result.reason_code == "candidate_found"
    assert result.candidate.records == _report().records
    assert result.candidate.candidate_id.startswith("abba:")
    assert str(path) not in result.candidate.candidate_id
    assert loaded == [path]
    assert not hasattr(result.candidate, "validated")


def test_lookup_requires_exact_report_context_fingerprint(tmp_path, monkeypatch):
    path = tmp_path / "profile.json"
    monkeypatch.setattr(provider, "load_source_profile_report",
                        lambda _path: _report(OTHER_FINGERPRINT))
    provider.configure_source_qualification_provider(
        provider.FileSourceQualificationProvider({FINGERPRINT: path}))

    result = provider.lookup_source_qualification_candidate(FINGERPRINT)
    assert result.candidate is None
    assert result.reason_code == "fingerprint_mismatch"


def test_lookup_maps_reader_errors_to_safe_reason_code(tmp_path, monkeypatch):
    path = tmp_path / "profile.json"

    def fail(_path):
        raise ValueError(f"secret path: {path}")

    monkeypatch.setattr(provider, "load_source_profile_report", fail)
    provider.configure_source_qualification_provider(
        provider.FileSourceQualificationProvider({FINGERPRINT: path}))

    result = provider.lookup_source_qualification_candidate(FINGERPRINT)
    assert result.candidate is None
    assert result.reason_code == "report_invalid"
    assert str(path) not in result.reason_code


def test_configuration_is_bounded_and_rejects_invalid_fingerprints(tmp_path):
    paths = {f"{index:064x}": tmp_path / f"{index}.json" for index in range(33)}
    with pytest.raises(ValueError, match="entry_limit"):
        provider.FileSourceQualificationProvider(paths)
    with pytest.raises(ValueError, match="fingerprint_invalid"):
        provider.FileSourceQualificationProvider({"../report": tmp_path / "x"})


def test_unconfigured_and_unmapped_lookups_are_safe_misses():
    provider.clear_source_qualification_provider()
    assert provider.lookup_source_qualification_candidate(FINGERPRINT).reason_code == "provider_not_configured"
    provider.configure_source_qualification_provider(provider.FileSourceQualificationProvider({}))
    assert provider.lookup_source_qualification_candidate(FINGERPRINT).reason_code == "candidate_not_found"


def test_oversized_report_is_rejected_by_existing_strict_reader(tmp_path):
    from quant_evaluator.scripts.source_profile_report_reader import MAX_REPORT_BYTES

    path = tmp_path / "large.json"
    path.write_bytes(b" " * (MAX_REPORT_BYTES + 1))
    provider.configure_source_qualification_provider(
        provider.FileSourceQualificationProvider({FINGERPRINT: path}))
    result = provider.lookup_source_qualification_candidate(FINGERPRINT)
    assert result.candidate is None
    assert result.reason_code == "report_invalid"


def test_provider_configuration_is_cleared_in_fork(tmp_path):
    import multiprocessing

    provider.configure_source_qualification_provider(
        provider.FileSourceQualificationProvider({FINGERPRINT: tmp_path / "x.json"}))
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("fork is unavailable")
    context = multiprocessing.get_context("fork")
    output = context.Queue()
    process = context.Process(target=lambda: output.put(
        provider.get_source_qualification_provider() is None))
    process.start()
    process.join(timeout=5)
    assert process.exitcode == 0
    assert output.get(timeout=1) is True
    assert provider.get_source_qualification_provider() is not None


def test_lookup_uses_strict_reader_and_returns_typed_records(tmp_path):
    from test_source_profile_report_reader_oct04 import _json, _payload
    from quant_evaluator.runtime.source_route_profiles import CounterbalancedRouteProfileRecord

    path = tmp_path / "profile.json"
    path.write_text(_json(_payload()), encoding="utf-8")
    provider.configure_source_qualification_provider(
        provider.FileSourceQualificationProvider({FINGERPRINT: path}))

    result = provider.lookup_source_qualification_candidate(FINGERPRINT)
    assert result.reason_code == "candidate_found"
    assert all(type(record) is CounterbalancedRouteProfileRecord
               for record in result.candidate.records)


@pytest.fixture(autouse=True)
def _clear_provider_after_test():
    provider.clear_source_qualification_provider()
    yield
    provider.clear_source_qualification_provider()
