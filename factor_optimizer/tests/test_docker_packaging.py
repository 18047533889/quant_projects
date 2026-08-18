"""Regression checks for the FactorOptimizer container build contract."""

from pathlib import Path


DOCKERFILE = Path(__file__).parents[2] / "docker" / "factor_optimizer" / "Dockerfile"


def test_dockerfile_builds_flat_package_and_stays_running_without_cli() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "COPY factor_optimizer/ factor_optimizer/" in dockerfile
    assert "COPY src/ src/" not in dockerfile
    assert '["python", "-m", "factor_optimizer"]' not in dockerfile
    assert '["tail", "-f", "/dev/null"]' in dockerfile
    assert "package_info()" not in dockerfile


def test_optimizer_has_no_service_entrypoint() -> None:
    package_dir = DOCKERFILE.parents[2] / "factor_optimizer"
    assert not (package_dir / "__main__.py").exists()
