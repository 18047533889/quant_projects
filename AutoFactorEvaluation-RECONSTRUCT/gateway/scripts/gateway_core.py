"""Gateway 核心 — 7 步审查流水线。

设计原则:
  - 输入: manifest dict + 缓存状态 + 初始化结果
  - 输出: 可直接插入 afv.json 的 Gateway 段 dict
  - 落盘: 唯一的 IO 操作是写入去重缓存报告到 ``{cache_dir}/dedup_cache.json``
    （仅非 DUPLICATED 时写入，供后续因子去重检测使用）

流程:
  Step 1: 校验 manifest 字段完整性
  Step 2: 构建 factor_engine YAML 配置 dict
  Step 3: 去重检测（通过传入的缓存状态）
  Step 4: 因子引擎小体量试运行
  Step 5: 未来函数拦截（DeepSeek）
  Step 6: 复杂度评分（算子权重表 + 深度惩罚）
  Step 7: 数据质量检测（覆盖率检查）

截断规则: 任一 Step 标记非 PASS，立即截断，后续 Steps 标记为 N/A。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from utils.deepseek_client import deepseek_chat


# ============================================================
# 数据模型
# ============================================================


@dataclass
class StepResult:
    """单步审查结果。"""
    step: str
    status: str  # PASS | REJECTED | DUPLICATED | TEMP | N/A
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


ALL_STEPS = [
    "Step1_ParseManifest", "Step2_YamlConfig", "Step3_Dedup",
    "Step4_TinyRun", "Step5_FutureScan", "Step6_Complexity", "Step7_DataQuality",
]


# ============================================================
# 主入口
# ============================================================


def run_gateway(
    manifest: dict,
    *,
    campaign_config: dict | None = None,
    init_result: dict[str, Any] | None = None,
    dedup_cache: dict[str, dict[str, Any]] | None = None,
    market_data_root: str | None = None,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """运行 Gateway 7 步审查流水线。

    Args:
        manifest: 已解析的 manifest dict。
        campaign_config: campaign config dict（补充 manifest 缺失字段）。
        init_result: 初始化结果（含 complexity_table）。
        dedup_cache: 去重缓存 {hash: info}。None=不查重。
        market_data_root: 行情数据根路径。None=跳过数据质量检测。
        cache_dir: 去重缓存报告写入目录。Gateway 在非 DUPLICATED 时
            将去重缓存条目写入 ``{cache_dir}/dedup_cache.json``。

    Returns:
        dict —— 可直接插入 afv.json 的 "Gateway" 段:
        {
            "Status": "PASS",
            "RunId": "gw_...",
            "CheckedAt": "...",
            "NonPassStep": null,
            "Steps": { ... },
            "DedupCacheWritten": bool,  # 是否写入了去重缓存文件
        }
    """
    run_id = f"gw_{datetime.now():%Y%m%d_%H%M%S_%f}"
    checked_at = datetime.now(timezone.utc).isoformat()

    out = {
        "Status": "PASS",
        "RunId": run_id,
        "CheckedAt": checked_at,
        "NonPassStep": None,
        "Steps": {},
    }
    dedup_update: dict[str, dict[str, Any]] = {}

    # Step 1
    sr = _step1_validate(manifest, campaign_config)
    out["Steps"]["Step1_ParseManifest"] = asdict(sr)
    if sr.status != "PASS":
        return _finalize(out, sr, "Step1_ParseManifest")

    # Step 2
    sr = _step2_build_yaml(manifest)
    out["Steps"]["Step2_YamlConfig"] = asdict(sr)
    if sr.status != "PASS":
        return _finalize(out, sr, "Step2_YamlConfig")

    # Step 3
    sr, dedup_update = _step3_dedup(manifest, dedup_cache)
    out["Steps"]["Step3_Dedup"] = asdict(sr)
    if sr.status != "PASS":
        # DUPLICATED 时不写入缓存报告
        r = _finalize(out, sr, "Step3_Dedup")
        r["DedupCacheWritten"] = False
        return r

    # Step 4
    sr = _step4_tiny_run(manifest)
    out["Steps"]["Step4_TinyRun"] = asdict(sr)
    if sr.status != "PASS":
        _write_dedup_cache(cache_dir, dedup_update)
        r = _finalize(out, sr, "Step4_TinyRun")
        r["DedupCacheWritten"] = True
        return r

    # Step 5
    sr = _step5_future_scan(manifest)
    out["Steps"]["Step5_FutureScan"] = asdict(sr)
    if sr.status != "PASS":
        _write_dedup_cache(cache_dir, dedup_update)
        r = _finalize(out, sr, "Step5_FutureScan")
        r["DedupCacheWritten"] = True
        return r

    # Step 6
    sr = _step6_complexity(manifest, init_result)
    out["Steps"]["Step6_Complexity"] = asdict(sr)
    if sr.status != "PASS":
        _write_dedup_cache(cache_dir, dedup_update)
        r = _finalize(out, sr, "Step6_Complexity")
        r["DedupCacheWritten"] = True
        return r

    # Step 7
    sr = _step7_data_quality(manifest, market_data_root)
    out["Steps"]["Step7_DataQuality"] = asdict(sr)
    if sr.status != "PASS":
        _write_dedup_cache(cache_dir, dedup_update)
        r = _finalize(out, sr, "Step7_DataQuality")
        r["DedupCacheWritten"] = True
        return r

    # 全部 PASS — 写入去重缓存
    _write_dedup_cache(cache_dir, dedup_update)
    out["DedupCacheWritten"] = True
    return out


def _finalize(out: dict, sr: StepResult, step_name: str) -> dict:
    out["Status"] = sr.status
    out["NonPassStep"] = step_name
    found = False
    for s in ALL_STEPS:
        if s == step_name:
            found = True
        elif found and s not in out["Steps"]:
            out["Steps"][s] = asdict(StepResult(step=s, status="N/A", reason="前置步骤截断"))
    return out


# ============================================================
# 去重缓存写入（唯一的 IO 操作）
# ============================================================


def _write_dedup_cache(
    cache_dir: str | None,
    dedup_update: dict[str, dict[str, Any]],
) -> None:
    """将去重缓存条目写入 ``{cache_dir}/dedup_cache.json``。

    仅在非 DUPLICATED 时调用。文件为增量追加式 JSON 对象。
    """
    if not cache_dir or not dedup_update:
        return
    from pathlib import Path
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    cache_path = p / "dedup_cache.json"

    import json
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}

    cache.update(dedup_update)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================
# Step 1: 校验 manifest
# ============================================================


def _step1_validate(manifest: dict, campaign_config: dict | None) -> StepResult:
    required = ["schema_version", "candidate_id", "formula"]
    missing = [f for f in required if f not in manifest]
    if missing:
        return StepResult("Step1_ParseManifest", "REJECTED", f"manifest 缺少必填字段: {missing}")
    if manifest.get("schema_version") != "disk.v1":
        return StepResult("Step1_ParseManifest", "REJECTED",
                          f"不支持的 schema_version: {manifest.get('schema_version')}")
    if campaign_config:
        for key in ["market", "universe_id", "domain_root", "domain",
                     "frequency_bucket", "signal_structure", "asset_class",
                     "generator_name", "generator_version", "mined_by"]:
            if key not in manifest and key in campaign_config:
                manifest[key] = campaign_config[key]
    return StepResult("Step1_ParseManifest", "PASS", detail={"manifest": manifest})


# ============================================================
# Step 2: 构建 YAML 配置 dict（不落盘）
# ============================================================


def _step2_build_yaml(manifest: dict) -> StepResult:
    try:
        expr = manifest.get("formula", "")
        expr_type = manifest.get("expression_type", "dsl")
        calc_mode = "code" if expr_type in ("python", "code") else "expr"
        market = manifest.get("market", "")
        fields = {"close": "Close", "open": "Open", "high": "High", "low": "Low",
                  "volume": "Volume", "vwap": "Vwap"}
        ts_col, inst_col = ("TradeDate", "Symbol") if market == "ashare" else ("window_start", "ticker")
        yaml_config = {
            "factor": {"name": manifest.get("candidate_id", "unknown"), "expr": expr,
                       "calc_mode": calc_mode, "freq": "1d",
                       "description": manifest.get("description", "")},
            "data_source": {"type": "parquet", "root": "",
                            "timestamp_col": ts_col, "instrument_col": inst_col,
                            "fields": fields, "max_files": 2},
            "backend": {"type": "pandas"},
            "engine": {"enable_cache": False, "tiny_run": True},
        }
        return StepResult("Step2_YamlConfig", "PASS", detail={"yaml_config": yaml_config})
    except Exception as e:
        return StepResult("Step2_YamlConfig", "REJECTED", f"YAML 构建失败: {e}")


# ============================================================
# Step 3: 去重检测（纯内存）
# ============================================================


def _step3_dedup(
    manifest: dict, dedup_cache: dict[str, dict[str, Any]] | None,
) -> tuple[StepResult, dict[str, dict[str, Any]]]:
    formula = manifest.get("formula", "")
    universe = manifest.get("universe_id", "")
    freq = manifest.get("frequency_bucket", "")
    payload = f"{' '.join(formula.split())}|{universe}|{freq}"
    h = hashlib.sha256(payload.encode()).hexdigest()
    cache = dedup_cache or {}
    if h in cache:
        return (StepResult("Step3_Dedup", "DUPLICATED",
                           f"重复因子: 首次提交于 {cache[h].get('first_seen','unknown')}",
                           detail={"candidate_hash": h}), {})
    upd = {h: {"candidate_id": manifest.get("candidate_id"),
               "campaign_id": manifest.get("campaign_id"),
               "first_seen": datetime.now(timezone.utc).isoformat()}}
    return (StepResult("Step3_Dedup", "PASS", detail={"candidate_hash": h, "is_new": True}), upd)


# ============================================================
# Step 4: 因子引擎小体量试运行
# ============================================================


def _step4_tiny_run(manifest: dict) -> StepResult:
    try:
        import sys, tempfile, yaml as _yaml
        from pathlib import Path
        p = Path(__file__).resolve().parents[2]
        fe = str(p / "factor_engine")
        if fe not in sys.path:
            sys.path.insert(0, fe)

        # 构建临时 YAML（此时需要填充 data_source.root）
        market = manifest.get("market", "")
        data_root = ""
        if market == "ashare":
            import os; data_root = os.environ.get("ASHARE_DATA_ROOT", "")
        else:
            import os; data_root = os.environ.get("US_STOCK_DATA_ROOT", "")

        expr_type = manifest.get("expression_type", "dsl")
        calc_mode = "code" if expr_type in ("python", "code") else "expr"
        fields = {"close": "Close", "open": "Open", "high": "High", "low": "Low",
                  "volume": "Volume", "vwap": "Vwap"}
        ts_col, inst_col = ("TradeDate", "Symbol") if market == "ashare" else ("window_start", "ticker")

        yaml_config = {
            "factor": {"name": manifest.get("candidate_id", "unknown"),
                       "expr": manifest.get("formula", ""),
                       "calc_mode": calc_mode, "freq": "1d"},
            "data_source": {"type": "parquet", "root": data_root,
                            "timestamp_col": ts_col, "instrument_col": inst_col,
                            "fields": fields, "max_files": 2},
            "backend": {"type": "pandas"},
            "engine": {"enable_cache": False, "tiny_run": True},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            _yaml.dump(yaml_config, f, allow_unicode=True, default_flow_style=False)
            tmp = f.name

        from factor_engine.runtime.engine import FactorEngine
        engine, factor = FactorEngine.from_config(tmp)
        result = engine.run(factor)
        Path(tmp).unlink(missing_ok=True)
        series = result.get("result")
        if series is None:
            return StepResult("Step4_TinyRun", "REJECTED", "引擎返回为空")
        nn = int(series.notna().sum()) if hasattr(series, "notna") else 0
        return StepResult("Step4_TinyRun", "PASS", detail={"rows": len(series), "non_null": nn})
    except ImportError as e:
        return StepResult("Step4_TinyRun", "REJECTED", f"因子引擎导入失败: {e}")
    except Exception as e:
        import traceback
        return StepResult("Step4_TinyRun", "REJECTED", f"引擎试运行失败: {e}",
                          detail={"traceback": traceback.format_exc()})


# ============================================================
# Step 5: 未来函数拦截 (DeepSeek)
# ============================================================


FUTURE_SCAN_PROMPT = """你是一个量化因子未来函数检测专家。

请分析下面的因子表达式/代码，判断是否包含"未来函数"——即在当前时间点使用了未来不可知信息的操作。

未来函数包括：
1. 显式未来引用函数: REFX, LEAD, PEEK, FUTURE, FWD_LOOK, TS_LEAD, NEXT, FORWARD, LOOKAHEAD, AHEAD
2. 正向时间偏移: t+1, ts+5, shift(x,-1), lag(x,-3), delay(x,负参数)
3. 未来数据后缀: 字段名含 _next_, _forward_, _future_
4. 前向引用: close(+1), price(+5)

注意: delay(x, N) 中 N>0 是正常回溯，不是未来函数。
shift(x, N) 中 N>0 是向前取未来数据，属于未来函数。
ts_mean, rank, zscore 等标准算子不是未来函数。

请严格按 JSON 格式返回:
{"has_future": true/false, "reason": "说明"}

因子表达式/代码:
"""


def _step5_future_scan(manifest: dict) -> StepResult:
    try:
        formula = manifest.get("formula", "")
        result = deepseek_chat([
            {"role": "system", "content": "你是一个精确的 JSON 生成器，只输出 JSON。"},
            {"role": "user", "content": FUTURE_SCAN_PROMPT + formula},
        ])
        result = result.strip()
        if result.startswith("```"):
            result = result.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        parsed = json.loads(result)
        if parsed.get("has_future", False):
            return StepResult("Step5_FutureScan", "REJECTED",
                              f"检测到未来函数: {parsed.get('reason', '')}",
                              detail={"has_future": True, "llm_reason": parsed.get("reason")})
        return StepResult("Step5_FutureScan", "PASS", detail={"has_future": False})
    except Exception as e:
        return StepResult("Step5_FutureScan", "REJECTED", f"未来函数检测异常: {e}")


# ============================================================
# Step 6: 复杂度评分
# ============================================================


def _step6_complexity(manifest: dict, init_result: dict[str, Any] | None) -> StepResult:
    try:
        formula = manifest.get("formula", "")
        ct: dict[str, float] = (init_result or {}).get("complexity_table", {})
        if not ct:
            return StepResult("Step6_Complexity", "TEMP", "复杂度评分表为空", detail={"score": None})
        ops_found = re.findall(r'\b([A-Za-z][A-Za-z0-9_]*)\s*\(', formula)
        total = 0.0
        matched: dict[str, float] = {}
        for op in ops_found:
            w = ct.get(op, 1.0); total += w; matched[op] = w
        md = _calc_nesting_depth(formula)
        dp = max(0, md - 1) * 0.5
        total += dp
        import os
        thr = float(os.environ.get("GATEWAY_MAX_COMPLEXITY", "20.0"))
        if total > thr:
            return StepResult("Step6_Complexity", "REJECTED",
                              f"复杂度 {total:.1f} 超过阈值 {thr}",
                              detail={"score": round(total, 2), "threshold": thr,
                                      "matched_ops": matched, "depth_penalty": dp, "max_depth": md})
        return StepResult("Step6_Complexity", "PASS",
                          detail={"score": round(total, 2), "threshold": thr,
                                  "matched_ops": matched, "depth_penalty": dp, "max_depth": md})
    except Exception as e:
        return StepResult("Step6_Complexity", "TEMP", f"复杂度评分异常: {e}")


def _calc_nesting_depth(expr: str) -> int:
    md = c = 0
    for ch in expr:
        if ch == '(':
            c += 1; md = max(md, c)
        elif ch == ')':
            c = max(0, c - 1)
    return md


# ============================================================
# Step 7: 数据质量检测
# ============================================================


def _step7_data_quality(manifest: dict, market_data_root: str | None) -> StepResult:
    try:
        if not market_data_root:
            return StepResult("Step7_DataQuality", "TEMP", "未提供行情数据路径")
        formula = manifest.get("formula", "")
        refs: set[str] = set()
        for m in re.finditer(r"""col\s*\(\s*['"](\w+)['"]\s*\)""", formula):
            refs.add(m.group(1))
        for m in re.finditer(r'\b([a-z][a-z0-9_]*)\b', formula):
            name = m.group(1)
            rest = formula[m.end():].lstrip()
            if not rest.startswith('(') and name not in ('and', 'or', 'not', 't', 'ts'):
                refs.add(name)
        if not refs:
            return StepResult("Step7_DataQuality", "PASS", "未检测到需要检查的字段")
        from pathlib import Path
        dr = Path(market_data_root)
        if not dr.exists():
            return StepResult("Step7_DataQuality", "TEMP", f"数据路径不存在: {dr}")
        files = sorted(dr.rglob("*.parquet"))
        if not files:
            return StepResult("Step7_DataQuality", "TEMP", f"未找到 parquet 文件: {dr}")
        import pandas as pd
        df = pd.read_parquet(files[0])
        results: dict[str, dict] = {}
        ok_all = True
        for field in sorted(refs):
            if field in df.columns:
                total = len(df); nn = df[field].notna().sum()
                cov = nn / total * 100 if total > 0 else 0
                ok = cov >= 98.0
                results[field] = {"coverage_pct": round(cov, 2), "total": int(total),
                                  "non_null": int(nn), "ok": ok}
                if not ok:
                    ok_all = False
            else:
                results[field] = {"coverage_pct": 0, "error": f"字段 '{field}' 不存在", "ok": False}
                ok_all = False
        if ok_all:
            return StepResult("Step7_DataQuality", "PASS", detail={"fields": results})
        return StepResult("Step7_DataQuality", "TEMP", "部分字段数据覆盖率不足 98%", detail={"fields": results})
    except Exception as e:
        return StepResult("Step7_DataQuality", "TEMP", f"数据质量检测异常: {e}")
