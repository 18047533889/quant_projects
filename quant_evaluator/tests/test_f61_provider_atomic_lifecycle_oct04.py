"""Provider claim must not overwrite another caller between check and install."""
import json

from quant_evaluator.runtime import source_qualification_provider as api
from quant_evaluator.scripts import verify_real_cos_f61_provider as cli
from test_f61_source_profile_report_reader import _payload


def test_other_caller_winning_install_gap_is_preserved(tmp_path, monkeypatch):
    payload = _payload()
    axis, profile, output = (tmp_path / name for name in ("axis.json", "profile.json", "out.json"))
    axis.write_text("{}")
    profile.write_text(json.dumps(payload))
    monkeypatch.setattr(cli.tiles, "MANIFEST_SHA256", payload["manifest_sha256"])
    monkeypatch.setattr(cli, "preflight", lambda: {"pass": True})
    monkeypatch.setattr(cli, "_runtime_ready", lambda: True)
    monkeypatch.setattr(cli, "has_validated_records", lambda: False)
    monkeypatch.setattr(cli, "warm_source_profile", lambda *a: None)
    monkeypatch.setattr(cli.tiles, "read_manifest", lambda *a: ())
    monkeypatch.setattr(cli.tiles, "select_source_records", lambda *a: tuple(range(61)))
    dates, assets = tuple(range(2586)), tuple(range(5461))
    monkeypatch.setattr(cli.tiles, "read_axis_index", lambda *a: (dates, assets, ()))
    monkeypatch.setattr(cli.tiles, "load_labels", lambda *a: (dates, assets, object()))
    class Source:
        def close(self):
            pass
    monkeypatch.setattr(cli.source_batch, "_make_cos_source", lambda *a, **kw: Source())
    monkeypatch.setattr(cli, "reference_source_all24", lambda *a, **kw: object())
    calls = []
    def harmless_backend(*a, **kw):
        calls.append(a)
        return object(), {}
    monkeypatch.setattr(cli, "_checked_run_backend", harmless_backend)
    monkeypatch.setattr(cli, "verify_provider_run", lambda *a, **kw: {})
    real_class = api.FileSourceQualificationProvider
    sentinel = real_class({"b" * 64: profile})
    def concurrent_constructor(paths):
        # This exact seam follows the final freshness check. Keep real
        # global provider state and setters; only external preparation is fake.
        api.configure_source_qualification_provider(sentinel)
        return real_class(paths)
    monkeypatch.setattr(cli, "FileSourceQualificationProvider", concurrent_constructor)
    try:
        result = cli.main(["--run", "--axis-index", str(axis),
                           "--profile-report", str(profile), "--output", str(output)])
        assert result == 1, "lost provider ownership must abort"
        assert api.get_source_qualification_provider() is sentinel
        assert calls == [] and not output.exists()
    finally:
        if api.get_source_qualification_provider() is sentinel:
            api.clear_source_qualification_provider()


def test_atomic_claim_allows_only_one_concurrent_owner(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    providers = tuple(api.FileSourceQualificationProvider({key * 64: tmp_path / key})
                      for key in ("a", "b"))
    barrier = Barrier(2)
    def claim(provider):
        barrier.wait(timeout=5)
        return api.configure_source_qualification_provider_if_absent(provider)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = tuple(pool.map(claim, providers))
        assert sorted(outcomes) == [False, True]
        winner = providers[outcomes.index(True)]
        loser = providers[outcomes.index(False)]
        assert api.get_source_qualification_provider() is winner
        assert api.clear_source_qualification_provider_if_current(loser) is False
        assert api.get_source_qualification_provider() is winner
        assert api.clear_source_qualification_provider_if_current(winner) is True
        assert api.get_source_qualification_provider() is None
    finally:
        for provider in providers:
            api.clear_source_qualification_provider_if_current(provider)


def test_stale_owner_release_preserves_replacement(tmp_path):
    original = api.FileSourceQualificationProvider({"a" * 64: tmp_path / "old"})
    replacement = api.FileSourceQualificationProvider({"b" * 64: tmp_path / "new"})
    try:
        assert api.configure_source_qualification_provider_if_absent(original) is True
        api.configure_source_qualification_provider(replacement)
        assert api.clear_source_qualification_provider_if_current(original) is False
        assert api.get_source_qualification_provider() is replacement
    finally:
        api.clear_source_qualification_provider_if_current(replacement)
