# -*- coding: utf-8 -*-
"""Systemic tail / group relation dynamics (2026-08 market language, P1/P2).

* ``group_tail_centrality`` — per-stock tail systemic centrality: given that a
  name is in its own trailing extreme state, how often do its group peers enter
  theirs on the same day (co-exceedance, not just correlation beta).
* ``group_tail_lead_score``  — conditional probability that peers enter their
  extreme state ``lag`` days later, minus the baseline (a stress leader).
* ``relation_diffusion_score`` — finite-step graph diffusion of a signal over a
  group adjacency (each stock spreads uniformly to its peers), returning the
  ``(1-alpha) sum alpha^{k-1} P^k x`` cascade.  Without a true PIT supply/customer
  graph, the group membership defines the adjacency (production-feasible form).

Extreme states use each stock's own trailing-window quantile (PIT-safe).  All
operators are deterministic and fail closed to NaN for stocks with no peers or
no observed extreme state.

Group membership is read **per day** from the ``group_id`` panel: a historical
day ``d`` is always evaluated with the membership that was in force *on day d*,
so an industry change / index migration never relabels past days (review
P1-46a).  Callers must supply a genuinely historical per-day ``group_id`` panel
(PIT membership), not a single current-day snapshot — a today-only snapshot
fed for every row would reintroduce the look-back relabel the per-day panel
avoids.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_udf

_EPS = 1e-12


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    cost: int,
    extra_tags: tuple[str, ...] = (),
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="tail_systemic",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "tail_systemic", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic", *extra_tags,
            f"signature:{','.join(params)}->series", "domain:price_volume",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _extreme_indicator_panel(xv: np.ndarray, w: int, q: float, side: str, min_periods: int, prior_threshold: bool = True) -> np.ndarray:
    """Per-stock trailing-quantile extreme indicator (0/1, NaN for missing).

    ``min_periods`` guards the cold start: with a one-row history the trailing
    quantile equals the row itself, so the first observation is *mechanically*
    its own extreme (``Q_0.1(x) == x``).  Until enough history is visible the
    indicator is NaN, never a spurious extreme=1 (review P0-14).

    ``prior_threshold`` (round-7 P0, review §25): by default the quantile
    threshold is estimated on the window *excluding* the current row, then the
    current observation is classified against it ("is today extreme relative to
    *history*").  ``prior_threshold=False`` restores the legacy self-normalising
    estimate where an extreme current value inflates its own threshold.
    """
    rows, cols = xv.shape
    E = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            lo = max(0, r - w + 1)
            if prior_threshold:
                if r - 1 < lo:
                    continue  # no strictly-prior rows yet
                est = xv[lo:r, c]
            else:
                est = xv[lo : r + 1, c]
            if int(np.isfinite(est).sum()) < min_periods:
                continue
            thr = float(np.nanquantile(est, q if side == "lower" else 1.0 - q))
            if not np.isfinite(thr) or not np.isfinite(xv[r, c]):
                continue
            if side == "lower":
                E[r, c] = 1.0 if xv[r, c] <= thr else 0.0
            else:
                E[r, c] = 1.0 if xv[r, c] >= thr else 0.0
    return E


def _group_index_by_day(gv: np.ndarray) -> list[dict[Any, np.ndarray]]:
    """Per-day mapping of group key -> member column indices."""
    days: list[dict[Any, np.ndarray]] = []
    rows, cols = gv.shape
    for r in range(rows):
        members: dict[Any, list[int]] = {}
        for c in range(cols):
            key = gv[r, c]
            if key is None or (isinstance(key, float) and np.isnan(key)):
                continue
            members.setdefault(key, []).append(c)
        days.append({k: np.asarray(v, dtype=int) for k, v in members.items()})
    return days


def _tail_centrality_series(xv: np.ndarray, gv: np.ndarray, w: int, q: float, side: str, min_periods: int, prior_threshold: bool = True) -> np.ndarray:
    rows, cols = xv.shape
    E = _extreme_indicator_panel(xv, w, q, side, min_periods, prior_threshold)
    by_day = _group_index_by_day(gv)
    out = np.full((rows, cols), np.nan, dtype=float)
    # Rolling-window centrality: per-day contributions live in a ring buffer and
    # leave the running sums once they age out of ``[r-w+1, r]``.  The earlier
    # implementation accumulated from the sample start (an *expanding* mean)
    # even though ``window`` was advertised as a rolling window.
    day_acc = np.zeros((rows, cols), dtype=float)
    day_cnt = np.zeros((rows, cols), dtype=float)
    run_acc = np.zeros(cols, dtype=float)
    run_cnt = np.zeros(cols, dtype=float)
    for r in range(rows):
        if r - w >= 0:
            drop = r - w
            run_acc -= day_acc[drop]
            run_cnt -= day_cnt[drop]
        members = by_day[r]
        for key, idx in members.items():
            if idx.size < 2:
                continue
            E_cur = E[r, idx]
            peer_count = np.isfinite(E_cur).sum()
            if peer_count < 2:
                continue
            for pos, i in enumerate(idx):
                Ei = E_cur[pos]
                if not np.isfinite(Ei) or Ei <= 0.0:
                    continue
                others = np.concatenate([E_cur[:pos], E_cur[pos + 1:]])
                others = others[np.isfinite(others)]
                if others.size == 0:
                    continue
                frac = float(others.mean())
                if np.isfinite(frac):
                    day_acc[r, i] += frac
                    day_cnt[r, i] += 1.0
        run_acc += day_acc[r]
        run_cnt += day_cnt[r]
        for i in range(cols):
            if run_cnt[i] > 0:
                out[r, i] = run_acc[i] / run_cnt[i]
    return out


@register_operator(
    name="group_tail_centrality",
    category="tail_systemic",
    business_category="tail_systemic",
    canonical="group_tail_centrality",
    source="tail_systemic",
)
class GroupTailCentrality(SeriesOperator):
    """尾部系统性中心度（个股进入极值状态时同组股票的共振程度）。

    ``TC_i = mean over window of (P(E_i,d=1) 时同组 peer 同日极值比例)``。
    比 correlation beta 更能识别"极端行情核心股"。极值用个股自身 trailing
    分位定义（PIT 安全）。P1。
    """

    metadata = _metadata(
        "group_tail_centrality",
        "尾部系统性中心度（个股极值与同组极值共振程度）。",
        ["x", "group_id", "window", "quantile", "side", "min_periods", "prior_threshold"],
        unit="probability",
        cost=6,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        group_id: pd.DataFrame,
        window: int = 120,
        quantile: float = 0.1,
        side: str = "lower",
        min_periods: int = 10,
        prior_threshold: bool = True,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        q = float(quantile)
        mp = int(min_periods)
        # ``quantile`` defines an *extreme/tail state* — q > 0.5 is not a tail
        # (the upper side mirrors at 1-q, so (0, 0.5] spans every meaningful
        # tail fraction).
        if not (0.0 < q <= 0.5):
            raise ValueError("group_tail_centrality requires 0 < quantile <= 0.5 (tail state)")
        if side not in ("lower", "upper"):
            raise ValueError("side must be 'lower' or 'upper'")
        if w < 5:
            raise ValueError("group_tail_centrality requires window >= 5")
        if not 1 <= mp <= w:
            raise ValueError("group_tail_centrality requires 1 <= min_periods <= window")
        return frame_like(
            x,
            _tail_centrality_series(x.to_numpy(dtype=float), group_id.to_numpy(dtype=object), w, q, side, mp, bool(prior_threshold)),
        )


def _tail_lead_series(xv: np.ndarray, gv: np.ndarray, w: int, q: float, side: str, lag: int, min_periods: int = 10, prior_threshold: bool = True) -> np.ndarray:
    rows, cols = xv.shape
    E = _extreme_indicator_panel(xv, w, q, side, min_periods, prior_threshold)
    by_day = _group_index_by_day(gv)
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        acc = np.zeros(cols, dtype=float)
        cnt = np.zeros(cols, dtype=float)
        base = np.zeros(cols, dtype=float)
        base_cnt = np.zeros(cols, dtype=float)
        for d in range(max(0, r - w + 1), r + 1):
            f = d + int(lag)
            # PIT: only *completed* forward observations.  ``f >= rows`` guards
            # the data edge; ``f > r`` guards the current row — without it the
            # output at row ``r`` reads ``E[f]`` for f in (r, rows), a genuine
            # future function.
            if f >= rows or f > r:
                continue
            members = by_day[d]
            if not members:
                continue
            Ef = E[f]
            for key, idx in members.items():
                if idx.size < 2:
                    continue
                E_cur = E[d, idx]
                for pos, i in enumerate(idx):
                    Ei = E_cur[pos]
                    if not np.isfinite(Ei):
                        continue
                    others = np.array([Ef[j] for j in idx if j != i], dtype=float)
                    others = others[np.isfinite(others)]
                    if others.size == 0:
                        continue
                    # baseline = unconditional peer-extreme probability P(E_peer, d+lag)
                    # (the natural baseline for "peer tail lead excess probability");
                    # the earlier code subtracted P(self extreme), which conflated
                    # self extreme-ness with peer lead.
                    base[i] += float(others.mean())
                    base_cnt[i] += 1.0
                    if Ei <= 0.0:
                        continue
                    acc[i] += float(others.mean())
                    cnt[i] += 1.0
        for i in range(cols):
            if cnt[i] > 0 and base_cnt[i] > 0:
                out[r, i] = (acc[i] / cnt[i]) - (base[i] / base_cnt[i])
    return out


@register_operator(
    name="group_tail_lead_score",
    category="tail_systemic",
    business_category="tail_systemic",
    canonical="group_tail_lead_score",
    source="tail_systemic",
    status="experimental",
)
class GroupTailLeadScore(SeriesOperator):
    """尾部领先分（个股极值领先同组极值 lag 天的超额概率）。

    ``lead_i = P(E_peer,d+lag=1 | E_i,d=1) - P(E_peer,d+lag=1)``（基线为 peer
    无条件尾部概率，不是 P(E_i)——文档与实现已对齐，review P1-25），只用
    ``d+lag <= 当前行`` 的 completed 观测（PIT 安全）。高 = 该股是 stress
    leader：它一进入极值，板块随后常跟着进入。P2 / Research。
    """

    metadata = _metadata(
        "group_tail_lead_score",
        "尾部领先分 P(E_peer,d+lag | E_i,d) - P(E_peer,d+lag)（completed-only）。",
        ["x", "group_id", "window", "quantile", "side", "lag", "min_periods", "prior_threshold"],
        unit="probability",
        cost=7,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        group_id: pd.DataFrame,
        window: int = 120,
        quantile: float = 0.1,
        side: str = "lower",
        lag: int = 1,
        min_periods: int = 10,
        prior_threshold: bool = True,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        q = float(quantile)
        lg = int(lag)
        mp = int(min_periods)
        if not (0.0 < q <= 0.5):
            raise ValueError("group_tail_lead_score requires 0 < quantile <= 0.5 (tail state)")
        if side not in ("lower", "upper"):
            raise ValueError("side must be 'lower' or 'upper'")
        if lg < 1:
            raise ValueError("group_tail_lead_score requires lag >= 1")
        if w < lg + 2:
            raise ValueError("group_tail_lead_score requires window >= lag + 2")
        if not 1 <= mp <= w:
            raise ValueError("group_tail_lead_score requires 1 <= min_periods <= window")
        return frame_like(
            x,
            _tail_lead_series(x.to_numpy(dtype=float), group_id.to_numpy(dtype=object), w, q, side, lg, mp, bool(prior_threshold)),
        )


def _diffusion_series(xv: np.ndarray, gv: np.ndarray, alpha: float, steps: int) -> np.ndarray:
    rows, cols = xv.shape
    by_day = _group_index_by_day(gv)
    out = np.full((rows, cols), np.nan, dtype=float)
    for r in range(rows):
        for key, idx in by_day[r].items():
            if idx.size < 2:
                continue
            vals = np.asarray([xv[r, i] for i in idx], dtype=float)
            # Local fail-close (review P1-46b): a member whose ``x`` is missing
            # drops out of that day's adjacency instead of NaN-ing the whole
            # group's cascade.  The finite subgraph is re-normalised — each
            # remaining stock still spreads uniformly to its finite peers — and
            # missing members keep their NaN output.
            finite_mask = np.isfinite(vals)
            idx = idx[finite_mask]
            if idx.size < 2:
                continue
            m = idx.size
            P = np.full((m, m), 1.0 / (m - 1), dtype=float)
            np.fill_diagonal(P, 0.0)
            cascade = np.zeros(m, dtype=float)
            v = vals[finite_mask].copy()
            for k in range(1, int(steps) + 1):
                v = P @ v
                cascade += (1.0 - alpha) * (alpha ** (k - 1)) * v
            # Truncation remainder assigned to the terminal power: the scalar
            # geometric weights sum to 1-α^K, so without the remainder a constant
            # graph signal is shrunk by (1-α^K) and every stock gets the same
            # spurious "diffusion residual".  Adding α^K·(P^steps x) makes the
            # weights sum to exactly 1 and preserves constants — P0-05.
            cascade += (alpha ** int(steps)) * v
            for pos, i in enumerate(idx):
                out[r, i] = float(cascade[pos])
    return out


@register_operator(
    name="relation_diffusion_score",
    category="tail_systemic",
    business_category="tail_systemic",
    canonical="relation_diffusion_score",
    source="tail_systemic",
    status="experimental",
)
class RelationDiffusionScore(SeriesOperator):
    """有限步图扩散分（(1-α)·Σ_{k=1..K} α^{k-1} P^k x，P 为组内均匀邻接）。

    group 成员定义邻接（PIT graph 可用前的最务实形态）：每只股票向其同组 peer
    均匀扩散信号，输出 K 步衰减级联。``x - D`` 即 diffusion residual recipe。
    fail-close 为**局部**：组内某只股票 x 缺失只剔除该节点、在剩余有限成员
    上重归一化传播，不再让整组级联变 NaN（review P1-46b）。P2 / Research。
    """

    metadata = _metadata(
        "relation_diffusion_score",
        "组内有限步图扩散分 (1-α)Σ α^{k-1} P^k x。",
        ["x", "group", "alpha", "steps"],
        unit="same_as:target",
        cost=6,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        group: pd.DataFrame,
        alpha: float = 0.5,
        steps: int = 2,
        **_: Any,
    ) -> pd.DataFrame:
        al = float(alpha)
        st = int(steps)
        if not (0.0 < al < 1.0):
            raise ValueError("relation_diffusion_score requires 0 < alpha < 1")
        if st < 1 or st > 5:
            raise ValueError("relation_diffusion_score requires 1 <= steps <= 5")
        return frame_like(
            x,
            _diffusion_series(x.to_numpy(dtype=float), group.to_numpy(dtype=object), al, st),
        )


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({"group_tail_centrality"})
    _surface.extend_research_only({"group_tail_lead_score", "relation_diffusion_score"})
    for _canon in ("group_tail_centrality", "group_tail_lead_score", "relation_diffusion_score"):
        register_polars_udf(_canon)


_register_surface()
