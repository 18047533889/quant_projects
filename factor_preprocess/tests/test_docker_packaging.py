"""Regression checks for the FactorPreprocess container build contract."""

from pathlib import Path


DOCKERFILE = Path(__file__).parents[2] / "docker" / "factor_preprocess" / "Dockerfile"


def test_dockerfile_builds_flat_package_and_stays_running_without_cli() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "COPY factor_preprocess/ factor_preprocess/" in dockerfile
    assert "COPY src/ src/" not in dockerfile
    assert '["tail", "-f", "/dev/null"]' in dockerfile


def test_preprocess_has_no_service_entrypoint() -> None:
    package_dir = DOCKERFILE.parents[2] / "factor_preprocess"
    assert not (package_dir / "__main__.py").exists()
