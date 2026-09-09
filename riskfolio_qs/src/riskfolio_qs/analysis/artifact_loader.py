"""Load and validate optimizer artifacts without invoking an optimizer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ..core.contracts import OutputBundle
from .contracts import OptimizationArtifacts


CORE_FILES = {
    "target_positions.parquet",
    "trades.parquet",
    "summary.parquet",
    "metadata.json",
    "run_manifest.yaml",
}


def _normalize_wide(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    result = frame.copy()
    try:
        index = pd.DatetimeIndex(pd.to_datetime(result.index))
    except Exception as exc:
        raise ValueError(f"{name} index cannot be parsed as dates") from exc
    if index.tz is not None:
        index = index.tz_localize(None)
    result.index = index.normalize()
    result.index.name = "date"
    result.columns = pd.Index([str(value) for value in result.columns], name="asset")
    if result.index.has_duplicates:
        raise ValueError(f"{name} contains duplicate dates")
    if result.columns.has_duplicates:
        raise ValueError(f"{name} contains duplicate assets")
    if not result.index.is_monotonic_increasing:
        result = result.sort_index()
    return result


def _normalize_summary(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    index = pd.DatetimeIndex(pd.to_datetime(result.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    result.index = index.normalize()
    result.index.name = "date"
    if result.index.has_duplicates:
        raise ValueError("summary contains duplicate dates")
    return result.sort_index()


def _normalize_trades(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels != 2:
        raise ValueError("trades must use MultiIndex(date, asset)")
    if "delta_weight" not in frame.columns:
        raise ValueError("trades must contain delta_weight")
    dates = pd.DatetimeIndex(pd.to_datetime(frame.index.get_level_values(0)))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    assets = pd.Index(
        [str(value) for value in frame.index.get_level_values(1)], dtype=object
    )
    result = frame.copy()
    result.index = pd.MultiIndex.from_arrays(
        [dates.normalize(), assets], names=["date", "asset"]
    )
    if result.index.has_duplicates:
        raise ValueError("trades contains duplicate date-asset rows")
    return result.sort_index()


def _require_finite(frame: pd.DataFrame, name: str) -> None:
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains missing or infinite values")


def validate_artifacts(
    artifacts: OptimizationArtifacts,
    *,
    reconstruction_tolerance: float = 1e-10,
) -> OptimizationArtifacts:
    """Normalize artifacts and enforce the cross-file contract."""

    target = _normalize_wide(artifacts.target_positions, "target_positions")
    trades = _normalize_trades(artifacts.trades)
    summary = _normalize_summary(artifacts.summary)
    _require_finite(target, "target_positions")
    _require_finite(trades[["delta_weight"]], "trades.delta_weight")

    if target.empty or len(target.columns) == 0:
        raise ValueError("target_positions must not be empty")
    if not summary.index.equals(target.index):
        raise ValueError("summary dates must exactly match target_positions dates")

    expected_index = pd.MultiIndex.from_product(
        [target.index, target.columns], names=["date", "asset"]
    )
    missing = expected_index.difference(trades.index)
    extra = trades.index.difference(expected_index)
    if len(missing) or len(extra):
        raise ValueError(
            "trades date-asset universe must exactly match target_positions; "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    trade_matrix = (
        trades["delta_weight"]
        .reindex(expected_index)
        .unstack("asset")
        .reindex(index=target.index, columns=target.columns)
    )
    previous = target.shift(1)
    previous.loc[target.index[0]] = target.iloc[0] - trade_matrix.iloc[0]
    residual = target - previous - trade_matrix
    maximum_error = float(np.abs(residual.to_numpy(dtype=float)).max(initial=0.0))
    if maximum_error > float(reconstruction_tolerance):
        raise ValueError(
            "target_positions and trades are inconsistent; "
            f"maximum reconstruction error={maximum_error:.3e}, "
            f"tolerance={reconstruction_tolerance:.3e}"
        )

    manifest = dict(artifacts.run_manifest or {})
    if manifest:
        expected_dates = manifest.get("date_count")
        expected_assets = manifest.get("asset_count")
        if expected_dates is not None and int(expected_dates) != len(target):
            raise ValueError(
                "run_manifest date_count does not match target_positions"
            )
        if expected_assets is not None and int(expected_assets) != len(target.columns):
            raise ValueError(
                "run_manifest asset_count does not match target_positions"
            )

    artifacts.target_positions = target
    artifacts.trades = trades
    artifacts.summary = summary
    return artifacts


def _load_yaml(path: Path, *, required: bool) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"required artifact does not exist: {path}")
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a mapping")
    return dict(payload)


def load_artifact_directory(
    input_dir: str | Path,
    *,
    reconstruction_tolerance: float = 1e-10,
) -> OptimizationArtifacts:
    """Load one ``riskfolio-qs optimize`` output directory."""

    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"optimization output directory does not exist: {root}")
    missing = sorted(name for name in CORE_FILES if not (root / name).exists())
    if missing:
        raise FileNotFoundError(f"optimization output is missing artifacts: {missing}")

    metadata_raw = pd.read_json(root / "metadata.json")
    if "date" in metadata_raw.columns:
        metadata = metadata_raw.set_index("date")
        metadata.index = pd.DatetimeIndex(pd.to_datetime(metadata.index, utc=True))
        metadata.index.name = "date"
    else:
        metadata = metadata_raw

    params_payload = _load_yaml(root / "resolved_params.yaml", required=False)
    params = params_payload.get("resolved_params", params_payload)
    if not isinstance(params, dict):
        params = {}

    artifacts = OptimizationArtifacts(
        target_positions=pd.read_parquet(root / "target_positions.parquet"),
        trades=pd.read_parquet(root / "trades.parquet"),
        summary=pd.read_parquet(root / "summary.parquet"),
        metadata=metadata,
        run_manifest=_load_yaml(root / "run_manifest.yaml", required=True),
        resolved_config=_load_yaml(
            root / "resolved_cli_config.yaml", required=False
        ),
        resolved_params=dict(params),
        input_dir=root,
    )
    return validate_artifacts(
        artifacts, reconstruction_tolerance=reconstruction_tolerance
    )


def artifacts_from_output_bundle(
    bundle: OutputBundle,
    *,
    reconstruction_tolerance: float = 1e-10,
) -> OptimizationArtifacts:
    artifacts = OptimizationArtifacts(
        target_positions=bundle.target_positions.copy(),
        trades=bundle.trades.copy(),
        summary=bundle.summary.copy(),
        metadata=bundle.metadata.copy(),
    )
    return validate_artifacts(
        artifacts, reconstruction_tolerance=reconstruction_tolerance
    )


__all__ = [
    "artifacts_from_output_bundle",
    "load_artifact_directory",
    "validate_artifacts",
]
