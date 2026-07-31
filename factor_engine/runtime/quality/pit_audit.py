# -*- coding: utf-8 -*-
"""Point-in-Time safety audit for operator IR and logical SourceRef dependencies."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from ir.nodes import IRNode

class PitSafetyError(RuntimeError):
    def __init__(self,violations:list[str]):self.violations=violations;super().__init__("PIT 安全审计失败: "+", ".join(sorted(set(violations))))
@dataclass
class PitAuditReport:
    violations:list[str]=field(default_factory=list);checked_ops:list[str]=field(default_factory=list)
    @property
    def passed(self)->bool:return not self.violations
    def to_dict(self)->dict[str,Any]:return {"passed":self.passed,"violations":list(self.violations),"checked_ops":list(self.checked_ops)}

_FORWARD_FILL_OPS=frozenset({"ffill","fillna_ffill","forward_fill","fill_forward","bfill","fillna_backfill","backfill"})
_POSITIVE_LAG_PARAMS={"ts_delay":("n",1),"ts_delta":("n",1),"ts_pct":("d",1)}
_FINANCIAL_TABLES=frozenset({"StockIncome","StockCashFlow","StockBalance"})
_EXACT_DAILY_TABLES=frozenset({"DailyBar","StockDailyBar","BenchmarkIndexDailyBar","SizeDaily","EtfDailyBar"})
_ASOF_DAILY_TABLES=frozenset({"IndustryDaily"});_MINUTE_TABLES=frozenset({"StockMinuteBar","MinuteBar"})


def _literal_number(node:IRNode,*,attr:str,input_index:int)->float|None:
    raw=(node.attrs or {}).get(attr)
    if raw is None and input_index<len(node.inputs):
        child=node.inputs[input_index]
        if child.op=="literal":raw=(child.attrs or {}).get("value")
    if isinstance(raw,bool) or not isinstance(raw,(int,float)):return None
    return float(raw)

def _audit_derived_field(field,*,forbid_forward_fill,fail_on_missing,violations,checked,source_guard):
    key=f"DerivedField:{field}"
    if key in source_guard:violations.append(f"{key}(cycle)");return
    source_guard.add(key)
    try:
        from runtime.derived_field_registry import load_derived_field_definition
        from api.dsl_parser import parse_factor
        from ir.analyzer import Analyzer
        definition=load_derived_field_definition(field);factor=parse_factor(definition.expression,name=f"pit::{field}@{definition.version}",surface="compat",dialect="lqtp",dialect_version="2026-07-19")
        report=audit_ir(Analyzer().lower(factor.expr).ir,forbid_forward_fill=forbid_forward_fill,fail_on_missing=fail_on_missing,_source_guard=source_guard)
        checked.extend(f"{key}->{op}" for op in report.checked_ops);violations.extend(f"{key}->{item}" for item in report.violations)
    except Exception as exc:violations.append(f"{key}(unverifiable:{type(exc).__name__})")
    finally:source_guard.discard(key)

def _valid_hhmm(text:str)->bool:
    try:
        h,m=(int(x) for x in text.split(":"));return 0<=h<=23 and 0<=m<=59
    except Exception:return False

def _audit_intraday_feature(label:str,tparams:dict[str,Any],violations:list[str])->None:
    feature=str(tparams.get("feature") or "").strip()
    if not feature:violations.append(f"{label}(missing_feature)")
    try:bar_minutes=int(tparams.get("bar_minutes",5))
    except (TypeError,ValueError):bar_minutes=0
    if bar_minutes<=0:violations.append(f"{label}(bar_minutes<=0)")
    try:coverage=float(tparams.get("min_coverage",0.8))
    except (TypeError,ValueError):coverage=0.0
    if not 0<coverage<=1:violations.append(f"{label}(invalid_min_coverage)")
    cutoff=str(tparams.get("cutoff_time") or "session_close")
    if cutoff!="session_close" and not _valid_hhmm(cutoff):violations.append(f"{label}(invalid_cutoff_time)")
    for key in ("session_open","session_close","split_time"):
        if key in tparams and tparams[key] is not None and not _valid_hhmm(str(tparams[key])):violations.append(f"{label}(invalid_{key})")
    for key in ("minutes","history_days","session_minutes","lag"):
        if key in tparams:
            try:value=int(tparams[key])
            except (TypeError,ValueError):value=0
            if value<=0:violations.append(f"{label}({key}<=0)")
    if feature in {"profile_zscore","profile_deviation","abnormal_volume_profile","abnormal_return_profile","abnormal_vol_profile"}:
        try:history=int(tparams.get("history_days",0))
        except (TypeError,ValueError):history=0
        if history<2:violations.append(f"{label}(profile_history_days<2)")
    if "q" in tparams:
        try:q=float(tparams["q"])
        except (TypeError,ValueError):q=0.0
        if not 0<q<1:violations.append(f"{label}(q_not_in_0_1)")

def _audit_source_ref(name,*,forbid_forward_fill,fail_on_missing,violations,checked,source_guard)->bool:
    from api.source_ref import decode_source_ref
    spec=decode_source_ref(name)
    if spec is None:return False
    label=f"SourceRef[{spec.table}.{spec.field}]";checked.append(label);params,tparams=spec.params_dict(),spec.transform_params_dict()
    if spec.table in _EXACT_DAILY_TABLES:
        if spec.table=="BenchmarkIndexDailyBar" and not str(params.get("index","")).strip():violations.append(f"{label}(missing_index)")
        if spec.transform is not None:violations.append(f"{label}(unexpected_transform={spec.transform})")
        return True
    if spec.table in _ASOF_DAILY_TABLES:
        if spec.transform not in {None,"asof_backward"}:violations.append(f"{label}(unsupported_transform={spec.transform})")
        return True
    if spec.table in _FINANCIAL_TABLES:
        if spec.transform not in {None,"financial_asof","financial_lag"}:violations.append(f"{label}(unsupported_transform={spec.transform})")
        if spec.transform=="financial_lag":
            try:q=int(tparams.get("quarters",1))
            except (TypeError,ValueError):q=0
            if q<=0:violations.append(f"{label}(financial_lag_quarters<=0)")
        return True
    if spec.table in _MINUTE_TABLES:
        if spec.transform=="intraday_feature":_audit_intraday_feature(label,tparams,violations);return True
        if spec.transform not in {"minute_at","minute_range","minute_bar","minute_resample"}:violations.append(f"{label}(minute_transform_required)");return True
        if spec.transform=="minute_at" and not str(tparams.get("hhmm","")).strip():violations.append(f"{label}(missing_hhmm)")
        if spec.transform=="minute_range":
            start,end=str(tparams.get("start","")),str(tparams.get("end",""))
            if not start or not end or start>=end:violations.append(f"{label}(invalid_minute_range)")
        if spec.transform in {"minute_bar","minute_resample"}:
            try:period,index=int(tparams.get("period",1)),int(tparams.get("index",0))
            except (TypeError,ValueError):period,index=0,-1
            if period<=0 or index<0:violations.append(f"{label}(invalid_minute_period_or_index)")
        return True
    if spec.table=="Intermediate":
        try:
            from runtime.intermediate_registry import intermediate_dependency_lineage
            intermediate_dependency_lineage(str(params.get("name","")),int(params.get("version",0)))
        except Exception as exc:violations.append(f"{label}(unversioned:{type(exc).__name__})")
        return True
    if spec.table=="DerivedField":_audit_derived_field(spec.field,forbid_forward_fill=forbid_forward_fill,fail_on_missing=fail_on_missing,violations=violations,checked=checked,source_guard=source_guard);return True
    if spec.table=="TurnoverBaseDaily":
        if spec.transform is not None:violations.append(f"{label}(unexpected_transform={spec.transform})")
        return True
    violations.append(f"{label}(unknown_availability_contract)");return True

def audit_ir(ir:IRNode,*,forbid_forward_fill:bool=False,fail_on_missing:bool=True,_source_guard:set[str]|None=None)->PitAuditReport:
    from backend.cleaned_bridge import ensure_cleaned_loaded
    from cleaned_operators.operator_policy import infer_operator_policy
    from cleaned_operators.registry import OperatorRegistry
    ensure_cleaned_loaded();violations=[];checked=[];source_guard=_source_guard if _source_guard is not None else set()
    def walk(node:IRNode):
        if node.op=="column":_audit_source_ref(str((node.attrs or {}).get("name") or ""),forbid_forward_fill=forbid_forward_fill,fail_on_missing=fail_on_missing,violations=violations,checked=checked,source_guard=source_guard);return
        if node.op in {"literal","plan_ref"}:return
        checked.append(node.op);canon=OperatorRegistry.resolve_canonical_optional(node.op);op_impl=OperatorRegistry.get(canon)
        if op_impl is None:
            if fail_on_missing:violations.append(f"{canon}(missing_runtime)")
            return
        policy=infer_operator_policy(op_impl,canonical=canon)
        if not policy.pit_safe or policy.lag<0:violations.append(canon)
        rule=_POSITIVE_LAG_PARAMS.get(canon)
        if rule is not None:
            name,index=rule;value=_literal_number(node,attr=name,input_index=index)
            if value is not None and value<1:violations.append(f"{canon}({name}={value:g})")
        if canon in {"bfill","backfill","fillna_backfill","fillna_bfill","lead","Lead","next"}:violations.append(f"{canon}(future_data)")
        if forbid_forward_fill and canon in _FORWARD_FILL_OPS:violations.append(f"{canon}(forward_fill)")
        for child in node.inputs:walk(child)
    walk(ir);return PitAuditReport(violations=violations,checked_ops=checked)
def assert_pit_safe(ir:IRNode,*,enforce:bool=True,forbid_forward_fill:bool=False,fail_on_missing:bool=True)->PitAuditReport:
    report=audit_ir(ir,forbid_forward_fill=forbid_forward_fill,fail_on_missing=fail_on_missing)
    if enforce and not report.passed:raise PitSafetyError(report.violations)
    return report
