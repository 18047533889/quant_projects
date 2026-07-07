from __future__ import annotations

from pathlib import Path

from factor_layer.factor_admission.admission import admit_evaluation_run
from factor_layer.factor_admission.config import FactorAdmissionConfig, load_config


def run_from_config(config_path: str | Path) -> dict[str, object]:
    config: FactorAdmissionConfig = load_config(config_path)
    return admit_evaluation_run(config)