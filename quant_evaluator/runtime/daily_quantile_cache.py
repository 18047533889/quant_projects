"""Bounded request-local reuse for immutable daily quantile panels."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any
from quant_evaluator.contracts.artifact_types import DailyQuantileReturnArtifact
from quant_evaluator.contracts.factor_batch import FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.metrics.quantile import _validate_min_assets, _validate_quantile_count

MAX_RETAINED_BYTES = 256 * 1024 * 1024

@dataclass(frozen=True)
class DailyQuantilePanelKey:
    input_binding: tuple[Any, ...]
    n_quantiles: int
    min_assets: int
    tie_method: str = "max"
    producer_version: str = "1.0.0"

class RequestDailyQuantilePanelCache:
    """Request-local LRU. Entries retain their exact source contracts strongly."""
    def __init__(self, root_factor_batch=None, root_label_bundle=None,
                 max_retained_bytes: int = MAX_RETAINED_BYTES, *, max_entries: int = 64):
        if isinstance(max_retained_bytes, bool) or not isinstance(max_retained_bytes, int) or max_retained_bytes < 0:
            raise ValueError("max_retained_bytes must be a non-negative integer")
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = max_entries
        self._artifacts = OrderedDict()
        self._owners = {}
        self._entry_bytes = {}
        self._retained_bytes = 0
        self._root_factor_batch = root_factor_batch
        self._root_label_bundle = root_label_bundle
        self._max_retained_bytes = max_retained_bytes

    @property
    def retained_bytes(self) -> int:
        """Conservatively counted retained array payload, not whole-process RSS."""
        return self._retained_bytes

    @property
    def entry_count(self) -> int:
        return len(self._artifacts)

    @staticmethod
    def _key(factor_batch, label_bundle, n_quantiles, min_assets):
        _validate_quantile_count(n_quantiles)
        _validate_min_assets(min_assets)
        if not isinstance(factor_batch, FactorBatch) or not isinstance(label_bundle, LabelBundle):
            raise TypeError("daily quantile cache requires FactorBatch and LabelBundle")
        from quant_evaluator.metrics.registry_adapters import _daily_quantile_input_binding
        return DailyQuantilePanelKey(_daily_quantile_input_binding(factor_batch, label_bundle),
                                     int(n_quantiles), int(min_assets))

    def get_or_build(self, factor_batch, label_bundle, n_quantiles, min_assets) -> DailyQuantileReturnArtifact:
        key = self._key(factor_batch, label_bundle, n_quantiles, min_assets)
        owners = self._owners.get(key)
        if owners is not None and owners[0] is factor_batch and owners[1] is label_bundle:
            self._artifacts.move_to_end(key)
            from quant_evaluator.metrics.registry_adapters import validate_daily_quantile_return_artifact
            artifact = self._artifacts[key]
            validate_daily_quantile_return_artifact(artifact, factor_batch, label_bundle, n_quantiles, min_assets)
            return artifact
        from quant_evaluator.metrics.registry_adapters import build_daily_quantile_return_artifact
        artifact = build_daily_quantile_return_artifact(
            factor_batch, label_bundle, n_quantiles=n_quantiles, min_assets=min_assets,
            min_periods=1, _bind_request_inputs=True)
        retained = artifact.values.nbytes + artifact.counts.nbytes + artifact.valid_mask.nbytes
        # The builder also freezes the per-bucket valid-day array in provenance.
        retained += artifact.provenance["valid_period_counts_qf"].nbytes
        if factor_batch is not self._root_factor_batch:
            retained += factor_batch.values.nbytes + (factor_batch.validity.nbytes if factor_batch.validity is not None else 0)
            for axis in (factor_batch.time_axis, factor_batch.asset_axis):
                if axis.values is not None:
                    retained += axis.values.nbytes
        if label_bundle is not self._root_label_bundle:
            retained += label_bundle.values.nbytes + (label_bundle.validity.nbytes if label_bundle.validity is not None else 0)
            if label_bundle.asset_axis is not None and label_bundle.asset_axis.values is not None:
                retained += label_bundle.asset_axis.values.nbytes
        # Count shared non-root arrays conservatively per entry. The separate
        # entry cap bounds Python owners/key/coordinate overhead for tiny panels.
        if retained > self._max_retained_bytes:
            return artifact
        while self._artifacts and (self._retained_bytes + retained > self._max_retained_bytes
                                   or len(self._artifacts) >= self._max_entries):
            old, _ = self._artifacts.popitem(last=False)
            self._retained_bytes -= self._entry_bytes.pop(old)
            self._owners.pop(old, None)
        self._artifacts[key] = artifact
        self._owners[key] = (factor_batch, label_bundle)
        self._entry_bytes[key] = retained
        self._retained_bytes += retained
        return artifact
