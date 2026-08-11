# -*- coding: utf-8 -*-
"""Universal UniverseMask contract for cross-sectional operators (P1-001).

The audit found that cross-sectional operators (``rank``/``zscore``/
``cs_regression``/``group_*``/``cs_multi_robust_resid``/KNN/local-Moran/
transport/group-spectrum) each decided their own "universe" by "whatever is
finite".  That lets delisted / ST / suspended / non-CS names leak into the
cross-section and biases every rank / zscore / regression / size-neutralization.

This module fixes the ONE universe contract every cross-sectional pass must
apply BEFORE ranking, and provides the shape-preserving helper.  It does not
rewrite every operator: the runtime / mining pipeline applies
``apply_universe_mask`` to the input panel once, then feeds the masked panel to
the operator (operators keep their NaN fail-closed policy on top).

Composition per market (2026-08-08 COS dicts):

A-share (anchor = DailyBar/List member):
    ``tradability_state`` (NOT IsSuspend)      AND
    ``public_status`` listed policy            AND
    finite daily bar (Close/Ret present)

US (anchor = StockList.type == CS):
    ``stock_list.type == "CS"``                AND
    ``universe_daily`` membership              AND
    finite daily bar (Close/Ret present)

A share's trading panel is NOT identical to the Status universe: Status contains
terminated/ST/not-yet-listed codes, so the trading panel anchors on the daily
bar / list.  US DailyBar additionally contains codes outside the ordinary CS
universe and has PreClose/Ret gaps, hence the type + universe_daily + valid bar
triple.  ``is_ticker_halt`` is far too sparse to be an IsSuspend equivalent; it
is an optional minute-level enhancement only, never the A-side daily mask.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from backend.universe_spec import UNIVERSE_SPEC, cross_section_shape_preserving


class UniverseContractError(Exception):
    """Production universe-contract violation.

    R10-P0-010: raised when a production universe mask is missing a required
    component (or the panel/mask axes diverge under strict alignment) — the
    reverse of silently passing every name through.
    """

# Source concepts each market's mask is composed from.
A_SHARE_MASK_FIELDS: tuple[str, ...] = (
    "tradability_state",  # NOT IsSuspend (StockDailyBar.IsSuspend negated)
    "public_status",      # listed-state policy (ashare_stock_status)
    "close",              # finite daily bar anchors the trading panel
)
US_MASK_FIELDS: tuple[str, ...] = (
    "stock_list.type",    # == "CS"
    "universe_daily",     # universe membership
    "close",              # finite daily bar
)

_MASK_FIELDS = {
    "ashare": A_SHARE_MASK_FIELDS,
    "us": US_MASK_FIELDS,
}


@dataclass(frozen=True)
class UniverseMaskContract:
    """The documented universe composition for one market.

    ``required_fields`` are the source concepts the runtime must load to build
    the mask.  ``shape_preserving`` mirrors ``backend.universe_spec``: after the
    mask is applied the panel keeps its (date × instrument) shape, out-of-universe
    names become NaN — never dropped rows.
    """

    market: str
    required_fields: tuple[str, ...]
    notes: str = ""
    shape_preserving: bool = True


@dataclass(frozen=True)
class MaskComponentSpec:
    """Declarative per-component mask predicate (R10-P0-011).

    Not every universe component is a "non-zero = in-pool" boolean.

    * ``is_suspend`` is ``1`` = suspended, ``0`` = tradable  -> ``==0``;
    * ``stock_list.type`` is a categorical  -> ``=="CS"``;
    * ``close`` / ``universe_daily`` / ``tradability_state`` -> ``!=0``.

    ``apply_mask_predicate`` evaluates ``predicate`` against a component panel;
    NaN / Inf always fail closed to out-of-universe.
    """

    field: str
    predicate: str = "!=0"


@dataclass(frozen=True)
class UniverseMembership:
    """R37-P0-012：PIT universe 成员身份。

    禁止把 ``Universe="CSI300"`` 直接等价成今天的一张静态成员列表去回算历史。
    本对象把成员关系绑到 ``valid_time``（成分有效期）与 ``knowledge_time``
    （成分名单获知时点）两个时点：

    - 成分加入前（decision_time < valid_time）不能出现；
    - 成分退出后不能被今天成员表重写过去；
    - ``membership_hash`` 进 FactorSemanticIdentity / DataKnowledgeIdentity，
      same expr + different universe => 不同 cache/checkpoint identity。
    """

    universe: str
    instrument: str = ""
    valid_time: str = ""        # 成分在该日/时段内有效
    knowledge_time: str = ""    # 名单在什么时点才被知道（PIT 可及）
    membership_hash: str = ""

    def effective_at(self, decision_time: str) -> bool:
        """decision_time 时该成分是否**已知且有效**。

        硬规则：decision_time 必须 >= knowledge_time 且 >= valid_time ——
        名单未公布前（future knowledge）或成分尚未生效前都不可用。
        """
        if not self.valid_time and not self.knowledge_time:
            return True  # 无时点信息 = 旧行为（静态），不判无效
        for t in (self.valid_time, self.knowledge_time):
            if t and decision_time < t:
                return False
        return True

    def identity_key(self) -> str:
        """成员级 identity（进 membership hash）。"""
        return f"{self.universe}|{self.instrument}|{self.valid_time}|{self.knowledge_time}"


def universe_membership_identity(universe: str, members: tuple[str, ...],
                                 *, as_of: str = "", version: str = "") -> str:
    """R37-P0-012：把命名 universe + 成员列表 + as_of 折叠成 membership hash。

    不同 as_of / 成员列表 => 不同 hash => 不同 cache/checkpoint 身份。
    ``version`` 可带名单 revision 号（成分修正触发受影响的日重算）。
    """
    import hashlib

    payload = {
        "universe": universe,
        "members": tuple(sorted(members)),
        "as_of": as_of,
        "version": version,
    }
    raw = ",".join(f"{k}={v}" for k, v in sorted(payload.items(), key=lambda x: x[0]))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def apply_mask_predicate(arr: Any, predicate: str) -> np.ndarray:
    """Evaluate a declared mask predicate on a component array.

    Supported predicates:

    * ``!=0``  — finite and non-zero (default for mask / close / membership);
    * ``==0``  — finite and zero (``IsSuspend``: 1 suspended / 0 tradable);
    * ``>0``   — finite and strictly positive;
    * ``=="CS"`` — exact string equality (``StockList.type``).

    NaN / ±Inf always evaluate to False — an unknown universe state is never an
    in-pool member (``bool(np.nan) is True`` must not leak through).
    """
    pred = str(predicate).strip()
    arr = np.asarray(arr)
    if pred.startswith('=="') and pred.endswith('"'):
        return arr == pred[3:-1]
    try:
        f = arr.astype(float)
    except (ValueError, TypeError):
        raise ValueError(
            f"mask component for predicate {predicate!r} must be numeric"
        ) from None
    finite = np.isfinite(f)
    if pred == "!=0":
        return finite & (f != 0)
    if pred == "==0":
        return finite & (f == 0)
    if pred == ">0":
        return finite & (f > 0)
    raise ValueError(f"unsupported mask predicate: {predicate!r}")


def universe_mask_contract(market: str) -> UniverseMaskContract:
    key = str(market).strip().lower()
    if key == "a_share" or key == "cn":
        key = "ashare"
    try:
        fields = _MASK_FIELDS[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"no universe mask contract for market {market!r}") from exc
    notes = (
        "A: DailyBar/List anchor AND NOT IsSuspend AND PublicStatus policy AND valid bar"
        if key == "ashare"
        else "US: StockList.type==CS AND universe_daily AND valid bar"
    )
    return UniverseMaskContract(
        market=key,
        required_fields=fields,
        notes=notes,
        shape_preserving=cross_section_shape_preserving(),
    )


def apply_universe_mask(
    panel: Any,
    mask: Any,
    *,
    shape_preserving: bool | None = None,
    strict_alignment: bool = False,
) -> Any:
    """Mask a (date × instrument) panel to its universe, keeping its shape.

    ``panel`` and ``mask`` are aligned boolean/numeric arrays: True = in-universe.
    Out-of-universe names become NaN (shape-preserving, matching
    ``backend.universe_spec``), never dropped rows.  ``None`` / all-NaN mask
    fail-closes the whole row/column via the NaN result.

    NaN/±Inf in the mask are treated as OUT-of-universe (fail-closed): an
    unknown universe state must never become an in-pool member
    (``bool(np.nan) is True`` — audit P0).  A DataFrame mask is aligned to the
    panel by label (index AND columns): same shape but a different column order
    can no longer pair A's data with B's mask.

    ``strict_alignment=True`` (R10-P0-013, production): a DataFrame mask must
    have EXACTLY the panel's index AND columns — no silent ``reindex``.  A mask
    that is missing a name or has extra names raises :class:`UniverseContractError`
    so the caller learns the mask source is incomplete instead of auto-filling
    the gaps with NaN.  The default (``False``) keeps the label-reindexing
    research behaviour.

    Accepts 2-D numpy arrays or pandas DataFrames (same axes).
    """
    keep_shape = UNIVERSE_SPEC.shape_preserving_cross_section if shape_preserving is None else shape_preserving
    del keep_shape  # shape is preserved by construction (never dropped rows)

    import pandas as pd

    if isinstance(panel, pd.DataFrame):
        if isinstance(mask, pd.DataFrame):
            if strict_alignment:
                if not mask.index.equals(panel.index) or not mask.columns.equals(panel.columns):
                    raise UniverseContractError(
                        "strict universe mask alignment failed: mask must carry "
                        "EXACTLY the panel's index and columns (no reindex). "
                        f"panel index {len(panel.index)}x{len(panel.columns)}, "
                        f"mask index {len(mask.index)}x{len(mask.columns)}"
                    )
                m = mask.to_numpy(dtype=float)
            else:
                # Label-based alignment: a mask with the same shape but a
                # different column order must NOT be applied positionally.
                # Reindex by label; cells the mask does not cover become NaN.
                m = mask.reindex(index=panel.index, columns=panel.columns).to_numpy(
                    dtype=float
                )
        else:
            m = np.asarray(mask, dtype=float)
        if m.shape != panel.shape:
            raise ValueError(
                f"panel/mask shape mismatch: {panel.shape} vs {m.shape}"
            )
        # Fail-closed truth: finite & nonzero = in-universe; NaN/Inf/0 = out.
        m_bool = np.isfinite(m) & (m != 0)
        out = panel.to_numpy(dtype=float).copy()
        out[~m_bool] = np.nan
        return pd.DataFrame(out, index=panel.index, columns=panel.columns, dtype=float)

    p = np.asarray(panel, dtype=float).copy()
    m = np.asarray(mask, dtype=float)
    if p.shape != m.shape:
        raise ValueError(f"panel/mask shape mismatch: {p.shape} vs {m.shape}")
    m_bool = np.isfinite(m) & (m != 0)
    p[~m_bool] = np.nan
    return p


def apply_universe_mask_to_panel(
    panel: Any,
    market: str,
    mask_columns: dict[str, Any] | None = None,
    *,
    mode: str = "research",
    predicates: Mapping[str, str] | None = None,
    strict_alignment: bool | None = None,
) -> tuple[Any, Any, float]:
    """Build and apply the universe mask for ``panel`` (P0-034).

    Returns ``(masked_panel, mask_df, coverage_ratio)`` where ``mask_df`` is the
    combined boolean mask (same axes as ``panel``) and ``coverage_ratio`` is the
    fraction of finite panel cells that stay in-universe after masking.

    ``mask_columns`` keys the per-market mask components by name — e.g.
    ``{"tradability_state": <panel>, "close": <panel>}`` for A-share or
    ``{"stock_list.type": <panel>, "universe_daily": <panel>, "close": <panel>}``
    for US (see ``universe_mask_contract``).  Each value is a 2-D array/DataFrame
    aligned to ``panel`` or a 1-D Series/array of per-date masks (broadcast
    across instruments), or a :class:`MaskComponentSpec` (its ``predicate``
    wins over ``predicates``).  A NaN/Inf cell in any supplied component fails
    closed to OUT-of-universe.

    ``mode="production"`` (R10-P0-010) is the reverse of the old allow-all
    default: a missing REQUIRED component — or no components at all — raises
    :class:`UniverseContractError` instead of passing every name through.  In
    ``"research"`` mode a partial mask is allowed (missing components contribute
    no constraint).

    ``predicates`` maps a component name to a :func:`apply_mask_predicate`
    expression (``"==0"`` for ``is_suspend``, ``'=="CS"'`` for
    ``stock_list.type``, default ``"!=0"``) — R10-P0-011: not every component is
    a "non-zero = in-pool" boolean.

    ``strict_alignment`` (defaults to ``mode == "production"``) passes through
    to :func:`apply_universe_mask` (R10-P0-013): production requires EXACT mask
    axis identity and never silently reindexes.

    This is the single integration point the CS materialization entry calls
    BEFORE ranking/regression so the universe mask is applied once and the
    coverage is recorded in the run lineage (P1-24).
    """
    import numpy as np
    import pandas as pd

    contract = universe_mask_contract(market)  # raises on unknown market
    shape = np.asarray(panel).shape
    is_production = str(mode).strip().lower() == "production"
    strict = bool(contract.shape_preserving) if strict_alignment is None else bool(strict_alignment)
    if is_production:
        strict = True
        supplied = set((mask_columns or {}).keys())
        missing = set(contract.required_fields) - supplied
        if missing:
            raise UniverseContractError(
                f"production universe mask for market {market!r} is missing "
                f"required component(s): {sorted(missing)}; supplied: "
                f"{sorted(supplied)}. A partial mask is not allowed in "
                "production — load every required component or fail."
            )
    if mask_columns is None or not mask_columns:
        if is_production:
            raise UniverseContractError(
                f"production universe mask for market {market!r} requires "
                f"component(s) {list(contract.required_fields)} but none were "
                "supplied (the old behaviour silently passed every name through)"
            )
        mask = np.ones(shape, dtype=float)
    else:
        mask = np.ones(shape, dtype=float)
        for name, comp in mask_columns.items():
            if comp is None:
                continue
            # R10-P0-011: per-component predicate.  ``predicates[name]`` wins;
            # otherwise the default ``!=0``.  ``MaskComponentSpec`` is the
            # declarative descriptor (``field`` + ``predicate``); build the
            # ``predicates`` dict from it, e.g.
            # ``predicates={"is_suspend": MaskComponentSpec("is_suspend", "==0").predicate}``.
            predicate = "!=0"
            if predicates is not None and name in predicates:
                predicate = predicates[name]
            is_string_pred = str(predicate).strip().startswith('=="')
            if isinstance(comp, pd.DataFrame):
                if strict and (
                    not comp.index.equals(getattr(panel, "index", None))
                    or not comp.columns.equals(getattr(panel, "columns", None))
                ):
                    raise UniverseContractError(
                        f"universe mask component {name!r} does not carry the "
                        "panel's exact axes (production strict alignment); "
                        "fix the mask source instead of reindexing"
                    )
                comp_arr = comp.to_numpy(dtype=float) if not is_string_pred else comp.to_numpy(dtype=object)
            elif isinstance(comp, pd.Series):
                comp_arr = comp.reindex(getattr(panel, "index", None)).to_numpy(dtype=float)
                if comp_arr.shape[0] == shape[0] and shape[1] > 1:
                    comp_arr = np.tile(comp_arr.reshape(-1, 1), (1, shape[1]))
            else:
                comp_arr = np.asarray(comp, dtype=float if not is_string_pred else object)
                if comp_arr.ndim == 1 and comp_arr.shape[0] == shape[0] and shape[1] > 1:
                    comp_arr = np.tile(comp_arr.reshape(-1, 1), (1, shape[1]))
            if np.asarray(comp_arr).shape != shape:
                raise ValueError(
                    f"universe mask component {name!r} has shape "
                    f"{np.asarray(comp_arr).shape}, panel is {shape}"
                )
            keep = apply_mask_predicate(np.asarray(comp_arr), predicate)
            mask = mask * keep.astype(float)

    masked = apply_universe_mask(panel, mask, strict_alignment=strict)
    panel_finite = np.isfinite(np.asarray(panel, dtype=float))
    in_universe = np.isfinite(np.asarray(masked, dtype=float))
    total = int(panel_finite.sum())
    coverage_ratio = float(in_universe.sum() / total) if total else 0.0
    return masked, mask, coverage_ratio


@dataclass(frozen=True)
class UniverseCoverageMetrics:
    """Three distinct coverage ratios (R10-P0-027).

    The old ``coverage_ratio`` conflated three different questions:

    * ``universe_breadth`` — fraction of ALL panel cells that are in-universe
      (the cross-section the factor is eligible on).  A field with values on
      only 100 of 5000 names must NOT report 100% just because those 100 are
      all in-universe.
    * ``field_coverage``   — finite factor cells among the eligible
      (in-universe) cells: does the FACTOR carry values where the universe
      expects them.
    * ``mask_retention``   — retained finite cells / pre-mask finite cells:
      how much of the original data the mask keeps (the historical
      ``coverage_ratio``).

    ``coverage_ratio`` is retained as an alias for ``mask_retention``.
    """

    universe_breadth: float = 0.0
    field_coverage: float = 0.0
    mask_retention: float = 0.0

    @property
    def coverage_ratio(self) -> float:
        """Backward-compatible alias: the mask retention."""
        return self.mask_retention

    def to_dict(self) -> dict[str, float]:
        return {
            "universe_breadth": self.universe_breadth,
            "field_coverage": self.field_coverage,
            "mask_retention": self.mask_retention,
        }


def coverage_metrics(
    panel: Any,
    masked: Any,
    mask: Any,
) -> UniverseCoverageMetrics:
    """Split the single coverage ratio into three metrics (R10-P0-027)."""
    panel_f = np.asarray(panel, dtype=float)
    masked_f = np.asarray(masked, dtype=float)
    mask_b = np.asarray(mask, dtype=float) != 0
    total = int(panel_f.size)
    eligible = int(mask_b.sum())
    pre_finite = int(np.isfinite(panel_f).sum())
    retained = int(np.isfinite(masked_f).sum())
    eligible_finite = int(np.isfinite(panel_f[mask_b]).sum())
    return UniverseCoverageMetrics(
        universe_breadth=float(eligible / total) if total else 0.0,
        field_coverage=float(eligible_finite / eligible) if eligible else 0.0,
        mask_retention=float(retained / pre_finite) if pre_finite else 0.0,
    )


def _matches_concept(required: str, name: str) -> bool:
    """Canonical FieldID comparison — R10-P0-012.

    A required concept is matched by EXACT canonical identity: the field name
    itself, or its last dotted component.  Substring matching is forbidden —
    ``required="close"`` must NOT be satisfied by ``"preclose"`` (it is only
    satisfied by ``"Close"`` / ``"StockDailyBar.Close"`` / ``"daily_close"``,
    whose last canonical component equals ``close``).
    """
    low = str(name).strip().lower()
    if low == required:
        return True
    last = low.rsplit(".", 1)[-1]
    return last == required


def is_market_eligible(market: str, field_names: tuple[str, ...]) -> bool:
    """True when every required universe field for ``market`` is available.

    R10-P0-012: capability matching uses canonical field identity only — never
    ``"close" in name`` substring matching (which let ``preclose`` impersonate
    ``close``).
    """
    required = universe_mask_contract(market).required_fields
    return all(
        any(_matches_concept(f, name) for name in field_names) for f in required
    )


__all__ = [
    "A_SHARE_MASK_FIELDS",
    "US_MASK_FIELDS",
    "MaskComponentSpec",
    "UniverseContractError",
    "UniverseCoverageMetrics",
    "UniverseMaskContract",
    "UniverseMembership",
    "apply_mask_predicate",
    "apply_universe_mask",
    "apply_universe_mask_to_panel",
    "coverage_metrics",
    "is_market_eligible",
    "universe_mask_contract",
    "universe_membership_identity",
]
