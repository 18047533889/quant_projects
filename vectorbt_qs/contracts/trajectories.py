"""Immutable QE-facing portfolio trajectories built from executed ledgers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import json
import math
from typing import Mapping, Optional, Sequence
from types import MappingProxyType

from .costs import CostScope


@dataclass(frozen=True)
class TrajectoryRefs:
    factor_ids: tuple[str, ...]
    source_fingerprint: str
    portfolio_ref: str
    cost_ref: str
    benchmark_ref: Optional[str]
    execution_ref: Optional[str] = None
    borrow_ref: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self,"factor_ids",tuple(self.factor_ids))
        if not self.factor_ids or len(set(self.factor_ids)) != len(self.factor_ids):
            raise ValueError("trajectory factor refs must be nonempty and unique")
        if not all(isinstance(value,str) and value for value in (
                self.source_fingerprint,self.portfolio_ref,self.cost_ref)):
            raise ValueError("trajectory source/portfolio/cost refs are required")


@dataclass(frozen=True)
class PortfolioTrajectory:
    scenario_id: str
    scope: CostScope
    profile: str
    dates: tuple[str, ...]
    gross_return: tuple[float, ...]
    net_return: tuple[float, ...]
    nav: tuple[float, ...]
    benchmark_return: Optional[tuple[float, ...]]
    active_return: Optional[tuple[float, ...]]
    relative_wealth: Optional[tuple[float, ...]]
    contributions: Mapping[str, tuple[float, ...]]
    refs: TrajectoryRefs
    statuses: tuple[str, ...]
    executable_certified: bool = False
    cost_component_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scope,CostScope) or not isinstance(self.refs,TrajectoryRefs):
            raise TypeError("trajectory requires typed scope and refs")
        if self.profile not in {"LONG_ONLY_RESEARCH","LONG_SHORT_RESEARCH"}:
            raise ValueError("unknown portfolio trajectory profile")
        object.__setattr__(self,"dates",tuple(self.dates))
        object.__setattr__(self,"statuses",tuple(self.statuses))
        if any(self.dates[i] >= self.dates[i+1] for i in range(len(self.dates)-1)):
            raise ValueError("trajectory dates must be strictly increasing")
        for name in ("gross_return","net_return","nav","benchmark_return","active_return","relative_wealth"):
            value=getattr(self,name)
            if value is not None:
                value=tuple(float(x) for x in value)
                if not all(math.isfinite(x) for x in value):
                    raise ValueError("trajectory must retain explicit invalid-valuation status before finite metric evaluation")
                object.__setattr__(self,name,value)
        frozen={str(k):tuple(float(x) for x in v) for k,v in self.contributions.items()}
        if any(len(v)!=len(self.dates) or not all(math.isfinite(x) for x in v) for v in frozen.values()):
            raise ValueError("trajectory contributions must be finite and axis-aligned")
        object.__setattr__(self,"contributions",MappingProxyType(frozen))
        names=tuple(self.cost_component_names)
        if len(set(names))!=len(names) or any(k not in frozen for k in names):
            raise ValueError("declared cost components must be unique and present")
        if any(x<0 for k in names for x in frozen[k]):
            raise ValueError("declared costs must be nonnegative")
        object.__setattr__(self,"cost_component_names",names)
        n = len(self.dates)
        for name in ("gross_return", "net_return", "nav"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must align with dates")
        if self.scope == CostScope.NET_EXECUTABLE and not self.executable_certified:
            raise ValueError("NET_EXECUTABLE requires actual fills/rules/borrow evidence certification")
        if self.scope == CostScope.NET_EXECUTABLE and not self.refs.execution_ref:
            raise ValueError("NET_EXECUTABLE requires an actual execution ledger ref")
        if self.scope == CostScope.NET_EXECUTABLE and self.profile == "LONG_SHORT_RESEARCH" and not self.refs.borrow_ref:
            raise ValueError("executable long-short requires borrow availability evidence")
        if self.benchmark_return is not None:
            if len(self.benchmark_return)!=n or len(self.active_return or ())!=n or len(self.relative_wealth or ())!=n:
                raise ValueError("benchmark and relative trajectories must align")
            if self.active_return is None or self.relative_wealth is None:
                raise ValueError("benchmark trajectory requires active return and relative wealth")
            if any(abs(self.active_return[i] - (self.net_return[i] - self.benchmark_return[i])) > 1e-12 for i in range(n)):
                raise ValueError("active return identity is inconsistent")
        wealth = 1.0
        for i, value in enumerate(self.net_return):
            if value < -1:
                raise ValueError("negative portfolio capital requires a separate contract")
            wealth *= 1.0 + value
            if abs(self.nav[i] - wealth) > 1e-12:
                raise ValueError("NAV identity is inconsistent")
        if self.benchmark_return is not None:
            benchmark_nav=1.0
            for i,value in enumerate(self.benchmark_return):
                benchmark_nav*=1.0+value
                if benchmark_nav<=0 or abs(self.relative_wealth[i]-self.nav[i]/benchmark_nav)>1e-12:
                    raise ValueError("relative wealth identity is inconsistent")

    @property
    def artifact_id(self) -> str:
        payload = {"scenario": self.scenario_id, "scope": self.scope.value, "profile": self.profile,
                   "dates": self.dates, "gross": self.gross_return, "net": self.net_return,
                   "nav":self.nav,"benchmark":self.benchmark_return,"active":self.active_return,
                   "relative":self.relative_wealth,"contributions":dict(self.contributions),
                   "refs": self.refs.__dict__, "statuses": self.statuses,
                   "executable_certified":self.executable_certified,
                   "cost_component_names":self.cost_component_names}
        return sha256(json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def build_research_trajectory(
    *, scenario_id: str, profile: str, dates: Sequence[str], gross_return: Sequence[float],
    benchmark_return: Optional[Sequence[float]], cost_contributions: Mapping[str, Sequence[float]],
    refs: TrajectoryRefs, scope: CostScope = CostScope.NET_ASSUMED,
    long_contribution: Optional[Sequence[float]] = None,
    short_contribution: Optional[Sequence[float]] = None,
) -> PortfolioTrajectory:
    """Bind holding PnL and actual-ledger cost rows without recreating fills.

    For LONG_ONLY_RESEARCH the gross series represents 1.0 capital.  For
    LONG_SHORT_RESEARCH callers supply signed +0.5/-0.5 contribution series;
    their sum must equal gross return.
    """
    gross = tuple(float(x) for x in gross_return)
    n = len(gross)
    costs = {k: tuple(float(x) for x in v) for k, v in cost_contributions.items()}
    if any(not math.isfinite(x) or x<0 for v in costs.values() for x in v):
        raise ValueError("research cost contributions must be finite and nonnegative")
    if any(len(v) != n for v in costs.values()):
        raise ValueError("cost contributions must align with gross returns")
    contrib = dict(costs)
    if profile == "LONG_SHORT_RESEARCH":
        if long_contribution is None or short_contribution is None:
            raise ValueError("LS trajectory requires signed long and short contributions")
        long_c, short_c = tuple(map(float, long_contribution)), tuple(map(float, short_contribution))
        if any(abs(long_c[i] + short_c[i] - gross[i]) > 1e-12 for i in range(n)):
            raise ValueError("signed +0.5/-0.5 leg contributions must sum to LS gross return")
        contrib.update(long=long_c, short=short_c)
    total_cost = tuple(sum(costs[k][i] for k in costs) for i in range(n))
    net = gross if scope == CostScope.GROSS_DIAGNOSTIC else tuple(gross[i] - total_cost[i] for i in range(n))
    nav, wealth = [], 1.0
    for value in net:
        wealth *= 1.0 + value
        nav.append(wealth)
    benchmark = None if benchmark_return is None else tuple(map(float, benchmark_return))
    active = relative = None
    if benchmark is not None:
        if len(benchmark) != n:
            raise ValueError("benchmark returns must align")
        active = tuple(net[i] - benchmark[i] for i in range(n))
        pwealth = bwealth = 1.0
        rel = []
        for rp, rb in zip(net, benchmark):
            pwealth *= 1.0 + rp
            bwealth *= 1.0 + rb
            if bwealth == 0:
                rel.append(math.nan)
            else:
                rel.append(pwealth / bwealth)
        relative = tuple(rel)
    statuses = ("ASSUMED_COST", "NOT_EXECUTABLE_CERTIFICATION") if scope == CostScope.NET_ASSUMED else ()
    return PortfolioTrajectory(scenario_id, scope, profile, tuple(map(str, dates)), gross, net,
                               tuple(nav), benchmark, active, relative, contrib, refs, statuses,
                               cost_component_names=tuple(costs))


def actual_calendar_carry_cost(
    balances: Sequence[float], dates: Sequence[str], annual_rate: float, *, negative_only: bool
) -> tuple[float, ...]:
    """ACTUAL/calendar-day carry; financing applies only to negative cash."""
    if len(balances) != len(dates):
        raise ValueError("balances and dates must align")
    if not math.isfinite(annual_rate) or annual_rate<0 or not all(math.isfinite(x) for x in balances):
        raise ValueError("carry rates and balances must be finite; rates nonnegative")
    if not dates:
        return ()
    out = [0.0]
    parsed = [date.fromisoformat(str(d)[:10]) for d in dates]
    if any(parsed[i]>=parsed[i+1] for i in range(len(parsed)-1)):
        raise ValueError("carry dates must be strictly increasing")
    for i in range(1, len(parsed)):
        base = max(-float(balances[i - 1]), 0.0) if negative_only else abs(float(balances[i - 1]))
        out.append(base * float(annual_rate) * (parsed[i] - parsed[i - 1]).days / 365.0)
    return tuple(out)


def build_trajectory_from_execution_plan(
    *, plan, order_price, gross_return: Sequence[float], benchmark_return: Sequence[float],
    initial_nav: float, slippage: float, scenario_id: str, refs: TrajectoryRefs,
    valuation_price=None,
) -> PortfolioTrajectory:
    """Compatibility entry; delegate to the single actual filled-ledger authority.

    Caller-supplied gross returns are an assertion, never the source of PnL.
    Explicit held valuation prices are mandatory; execution quotes are not
    silently assumed to be daily marks.
    """
    import numpy as np
    if valuation_price is None:
        raise ValueError("valuation_price is required for actual ledger reconciliation")
    result=build_trajectory_from_filled_ledger(
        plan=plan,order_price=order_price,valuation_price=valuation_price,
        initial_nav=initial_nav,slippage=slippage,scenario_id=scenario_id,refs=refs,
        benchmark_return=benchmark_return)
    expected=np.asarray(gross_return,dtype=float)
    if expected.shape != (len(result.dates),) or not np.allclose(expected,result.gross_return,atol=1e-12,rtol=1e-10):
        raise ValueError("supplied gross returns disagree with actual filled ledger")
    return result


def build_trajectory_from_filled_ledger(
    *, plan, order_price, valuation_price, initial_nav: float, slippage: float,
    scenario_id: str, refs: TrajectoryRefs, benchmark_return: Sequence[float],
    scope: CostScope = CostScope.NET_ASSUMED,
    annual_cash_borrow_rate: float = .04,
) -> PortfolioTrajectory:
    """Revalue fixed actual fills, shares, dividends and cash into a daily ledger.

    The no-cost shadow keeps the identical fills, so scenario costs cannot
    change its gross NAV. Net cash pays price slippage once, statutory fees on
    effective fill notionals, and negative-cash financing on actual calendar days.
    This remains research evidence, not proof of real-account fills/borrow.
    """
    import numpy as np
    if not math.isfinite(initial_nav) or initial_nav<=0 or not math.isfinite(slippage) or slippage<0:
        raise ValueError("initial_nav must be positive and slippage finite nonnegative")
    if not math.isfinite(annual_cash_borrow_rate) or annual_cash_borrow_rate<0:
        raise ValueError("cash borrowing rate must be finite nonnegative")
    if scope not in {CostScope.NET_ASSUMED,CostScope.GROSS_DIAGNOSTIC}:
        raise ValueError("filled-ledger research repricing cannot certify executable scope")
    if len(refs.factor_ids)!=1:
        raise ValueError("one filled ledger binds exactly one factor")
    index,columns=plan.order_size.index,plan.order_size.columns
    if len(benchmark_return)!=len(index):
        raise ValueError("benchmark must align with the execution calendar")
    for name,frame in (("order prices",order_price),("valuation prices",valuation_price),
                       ("fees",plan.fees),("fixed fees",plan.fixed_fees),("holdings",plan.real_holdings),
                       ("cash deposits",plan.cash_deposits),("asset deposits",plan.asset_deposits)):
        if not frame.index.equals(index) or not frame.columns.equals(columns):
            raise ValueError(f"{name} must match the actual execution axes")
    dates=tuple(str(stamp.date()) for stamp in index)
    if any(dates[i]>=dates[i+1] for i in range(len(dates)-1)):
        raise ValueError("filled ledger requires one ordered observation per calendar date")
    gross_cash=net_cash=float(initial_nav)
    previous_gross=previous_net=float(initial_nav)
    gross_returns=[]; net_returns=[]; nav=[]; investment=[]
    fees=[]; price_cost=[]; financing=[]; cash_path=[]
    carried_shares=np.zeros(len(columns),dtype=float)
    parsed=[date.fromisoformat(d) for d in dates]
    for i in range(len(index)):
        sizes=np.asarray(plan.order_size.iloc[i],dtype=float)
        sizes=np.where(np.isnan(sizes),0.,sizes)
        prices=np.asarray(order_price.iloc[i],dtype=float)
        observed_holdings=np.asarray(plan.real_holdings.iloc[i],dtype=float)
        share_deposits=np.asarray(plan.asset_deposits.iloc[i],dtype=float)
        if np.any(~np.isfinite(share_deposits)):
            raise ValueError("asset distributions must be finite")
        carried_shares+=share_deposits+sizes
        observed=np.isfinite(observed_holdings)
        if not np.allclose(observed_holdings[observed],carried_shares[observed],atol=1e-8,rtol=1e-12):
            raise ValueError("actual holdings disagree with filled-order/share-distribution ledger")
        holdings=carried_shares.copy()
        valuations=np.asarray(valuation_price.iloc[i],dtype=float)
        traded=sizes!=0
        held=holdings!=0
        if (not np.all(np.isfinite(sizes)) or not np.all(np.isfinite(holdings)) or np.any(holdings<0)
                or np.any(~np.isfinite(prices[traded])) or np.any(prices[traded]<=0)
                or np.any(~np.isfinite(valuations[held])) or np.any(valuations[held]<=0)):
            raise ValueError("filled LO ledger has invalid orders/held valuations")
        safe_prices=np.where(traded,prices,0.)
        effective=safe_prices*(1+np.sign(sizes)*slippage)
        fee_rates=np.asarray(plan.fees.iloc[i],dtype=float)
        fixed=np.asarray(plan.fixed_fees.iloc[i],dtype=float)
        if np.any(~np.isfinite(fee_rates[traded])) or np.any(fee_rates[traded]<0) or np.any(~np.isfinite(fixed)) or np.any(fixed<0):
            raise ValueError("actual fees must be finite and nonnegative")
        explicit=float(np.sum(np.abs(sizes[traded])*effective[traded]*fee_rates[traded])+np.sum(fixed[traded]))
        slip=float(np.sum(np.abs(sizes)*safe_prices)*slippage)
        carry=0. if i==0 else max(-net_cash,0.)*annual_cash_borrow_rate*(parsed[i]-parsed[i-1]).days/365.
        deposit=float(np.asarray(plan.cash_deposits.iloc[i],dtype=float).sum())
        if not math.isfinite(deposit):
            raise ValueError("cash distributions must be finite")
        gross_cash+=deposit-float(np.sum(sizes*safe_prices))
        net_cash+=deposit-float(np.sum(sizes*effective))-explicit-carry
        market_value=float(np.sum(holdings[held]*valuations[held]))
        gross_equity=gross_cash+market_value
        net_equity=net_cash+market_value
        if min(gross_equity,net_equity,previous_gross,previous_net)<=0:
            raise ValueError("nonpositive capital requires an explicit default/financing contract")
        gross_returns.append(gross_equity/previous_gross-1.)
        net_returns.append(net_equity/previous_net-1.)
        nav.append(net_equity/initial_nav)
        investment.append(market_value/(gross_equity if scope==CostScope.GROSS_DIAGNOSTIC else net_equity))
        fees.append(explicit/previous_net); price_cost.append(slip/previous_net); financing.append(carry/previous_net)
        cash_path.append(net_cash)
        previous_gross,previous_net=gross_equity,net_equity
    selected=tuple(gross_returns if scope==CostScope.GROSS_DIAGNOSTIC else net_returns)
    selected_nav=[]; wealth=1.
    for value in selected:
        wealth*=1+value
        selected_nav.append(wealth)
    benchmark=tuple(float(x) for x in benchmark_return)
    active=tuple(r-b for r,b in zip(selected,benchmark))
    relative=[]; bwealth=1.
    for p,b in zip(selected_nav,benchmark):
        bwealth*=1+b
        if bwealth<=0:
            raise ValueError("relative NAV is undefined for nonpositive benchmark capital")
        relative.append(p/bwealth)
    return PortfolioTrajectory(scenario_id,scope,"LONG_ONLY_RESEARCH",dates,tuple(gross_returns),selected,
        tuple(selected_nav),benchmark,active,tuple(relative),
        {"explicit_fees":tuple(fees),"implementation_slippage":tuple(price_cost),"financing":tuple(financing),
         "investment_fraction":tuple(investment),"cash_cny":tuple(cash_path)},refs,
        ("FIXED_ACTUAL_ORDER_REPRICING","NOT_EXECUTABLE_CERTIFICATION"),
        cost_component_names=("explicit_fees","implementation_slippage","financing"))


__all__ = ["TrajectoryRefs", "PortfolioTrajectory", "build_research_trajectory", "build_trajectory_from_execution_plan", "build_trajectory_from_filled_ledger", "actual_calendar_carry_cost"]
