"""Regression checks for the QuantEvaluator container build contract."""

from pathlib import Path


DOCKERFILE = Path(__file__).parents[2] / "docker" / "quant_evaluator" / "Dockerfile"


def test_dockerfile_builds_flat_package_and_stays_running_without_cli() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "COPY quant_evaluator/ quant_evaluator/" in dockerfile
    assert "COPY src/ src/" not in dockerfile
    assert '["tail", "-f", "/dev/null"]' in dockerfile


def test_evaluator_has_no_service_entrypoint() -> None:
    package_dir = DOCKERFILE.parents[2] / "quant_evaluator"
    assert not (package_dir / "__main__.py").exists()
