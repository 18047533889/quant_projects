#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本周新挖因子：发布详情页 / 重写首页新挖栏目 / 相关性分析 / COS 上传。

只读主链路已产出的 report_manifest.json 与 npz artifact，不重算任何指标。
"""
from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path("/home/sunhaiwei/quant_projects")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "jobs"))

REPORTS = Path(os.environ.get(
    "NEW_MINING_REPORTS",
    "/home/sunhaiwei/quant_project_archives/factor_engine-docs/reports/2026-08-23"))
INDEX = REPORTS / "index.html"
MATRICES = ROOT / "weekly_backtest_output/factor_matrices_all"
WORK = ROOT / "work/newmining_20260919"
CANDIDATES = WORK / "candidates.json"
STATE = WORK / "state.json"
CORR_JSON = WORK / "correlation_new_vs_pool.json"
COS_ROOT = "cos://qs-cold/candidate_pool/sunhaiwei/lqtp_show"
SOURCE_META = "cos://qs-cold/candidate_pool/xlubs/mining_outputs/metadata"

THRESHOLD_TEXT = "RankIC &gt; 0.015 或 RankIC IR &gt; 0.15（方向冻结后的带符号阈值，未取绝对值）"

# ---------------------------------------------------------------------------
# batch policy
# ---------------------------------------------------------------------------
# The report already states its own convention on the pool page: "本周指登记
# 来源『本周新增』的累计提交批次，不按本周日历日期重新划分".  This section obeys
# the same rule.  The mining campaign timestamp inside a factor id
# (alphasage_YYYYMMDD…) is surfaced as a per-batch breakdown for traceability,
# but it never demotes a candidate to "pool stock": every candidate of the round
# being published is new by construction (nothing was published before), so a
# calendar-week split would just hide genuinely new factors.
def _campaign_map():
    """factor name -> campaign (mining) id, from the frozen candidate roster."""
    out = {}
    for path in (CANDIDATES, WORK / "admitted.json"):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        records = payload.get("candidates") if isinstance(payload, dict) else payload
        for r in records or []:
            name = r.get("page_name") or r.get("factor_name")
            if name and r.get("campaign"):
                out.setdefault(str(name), str(r["campaign"]))
    return out


def batch_label(campaign):
    """20260909T062020Z -> 2026-09-09 (campaign timestamps are the mining time)."""
    s = str(campaign or "")
    if len(s) >= 8 and s[:8].isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return "未标注"


def batch_breakdown(rows):
    """date -> [candidates, landed, evaluated], oldest first."""
    out = {}
    for r in rows:
        key = batch_label(r.get("campaign"))
        slot = out.setdefault(key, [0, 0, 0])
        slot[0] += 1
        if r.get("entry"):
            slot[1] += 1
            if html_eligible(r["entry"]):
                slot[2] += 1
    return dict(sorted(out.items()))


def now_cn():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def esc(text):
    from html import escape
    return escape(str(text))


def html_eligible(entry):
    import incremental_factor_intake as intake
    return bool(intake.html_eligible(entry))


def _manifest_paths():
    """Map factor name -> its per-factor manifest, across root and every shard.

    Sharded runs write their results under ``work/<campaign>/shards/<id>/factors``
    while the earlier serial run wrote to ``work/<campaign>/factors``; publishing
    must union both or it silently reports a fraction of the batch.
    Manifests younger than a few seconds are skipped so a publish triggered while
    a wave is still flushing cannot read a half-written JSON.
    """
    bases = [WORK / "factors"]
    shards = WORK / "shards"
    if shards.is_dir():
        bases += sorted(p for p in shards.glob("*/factors") if p.is_dir())
    found = {}
    cutoff = time.time() - 5.0
    for base in bases:
        for manifest in sorted(base.glob("*/report_manifest.json")):
            try:
                if manifest.stat().st_size <= 2 or manifest.stat().st_mtime > cutoff:
                    continue
            except OSError:
                continue
            found.setdefault(manifest.parent.name, manifest)
    return found


def _merged_state():
    """Union of the campaign state and every shard state (root wins on clash)."""
    parts = [STATE]
    shards = WORK / "shards"
    if shards.is_dir():
        parts += sorted(shards.glob("*/state.json"))
    merged = {}
    for path in parts:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        for name, info in (payload.get("factors") or {}).items():
            merged.setdefault(name, dict(info))
    return merged


def _entry_status(entry):
    import incremental_factor_intake as intake
    if entry.get("status") == "unavailable":
        return "unavailable"
    return "evaluated" if intake.html_eligible(entry) else "below_gate"


def _frozen_names():
    """The frozen candidate roster — the authoritative denominator for the batch.

    Shard state files only mention the waves a shard has already started, so
    deriving the total from them would make the reported denominator drift as
    the batch progresses.
    """
    for path in (CANDIDATES, WORK / "admitted.json"):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        records = payload.get("candidates") if isinstance(payload, dict) else payload
        if not records:
            continue
        names = [r.get("page_name") or r.get("factor_name") for r in records]
        names = [n for n in names if n]
        if names:
            return names
    return []


def load_results():
    """Collect every evaluated factor from its isolated per-factor manifest.

    Disk is authoritative: after sharded runs the per-factor manifests are the
    single source of truth, while ``state.json`` (possibly shard-local) may lag.
    State entries without a manifest (e.g. failed or still-running waves) and
    frozen candidates that no wave has touched yet are merged in so the roster
    is complete and the denominator does not drift mid-batch.
    """
    state = _merged_state()
    campaigns = _campaign_map()
    rows = {}
    for name, manifest in _manifest_paths().items():
        try:
            payload = json.loads(manifest.read_text())
        except Exception:
            continue
        entry = (payload.get("factors") or {}).get(name)
        if not entry:
            continue
        info = state.get(name, {})
        rows[name] = dict(
            name=name, entry=entry, manifest=str(manifest),
            provenance=payload.get("evaluation_provenance"),
            metrics=entry.get("metrics") or {},
            direction=entry.get("direction"), is_flipped=entry.get("is_flipped"),
            evaluated_at=(payload.get("evaluation_provenance") or {}).get("completed_at"),
            # The on-disk manifest is the truth for a factor that has one; shard
            # state files are only used for candidates with no manifest yet.
            status=_entry_status(entry),
            reason=info.get("reason"), elapsed_s=info.get("elapsed_s"),
            exit_code=info.get("exit_code"), wave=info.get("wave"),
            campaign=campaigns.get(name),
        )
    for name, info in state.items():
        if name not in rows:
            rows[name] = dict(name=name, status=info.get("status"), reason=info.get("reason"),
                              elapsed_s=info.get("elapsed_s"), exit_code=info.get("exit_code"),
                              wave=info.get("wave"), campaign=campaigns.get(name))
    for name in _frozen_names():
        if name not in rows:
            rows[name] = dict(name=name, status="queued", campaign=campaigns.get(name))
    return rows


# ---------------------------------------------------------------------------
# display naming: never expose the miner tag in user-facing names
# ---------------------------------------------------------------------------
_MINER_NAME_RE = re.compile(r"^[a-z][a-z0-9]*_\d{12,14}_([0-9a-f]{6,16})$")
_MINER_TOKEN_RE = re.compile(r"[a-z][a-z0-9]*_\d{12,14}_[0-9a-f]{6,16}|alphasage_[A-Za-z0-9_]+")


def display_name(name):
    """Strip the mining-source/timestamp prefix from a factor id for display.

    Internal ids like ``alphasage_20260909062020_241df17a`` key the matrices,
    manifests and state files and must stay stable; every user-facing surface
    (page filenames, section tables, COS keys) shows ``nm_241df17a`` instead —
    no miner name, no mining timestamp, only the content hash of the formula.
    Legacy ids without a timestamp (``alphasage_rank_amount``) become
    ``nm_rank_amount``.  Pool-native ids (e.g. ``weekly_f8d14e51d8525a54``)
    pass through unchanged.  The hash is a content digest, so cross-miner
    collisions are not a practical concern.
    """
    text = str(name)
    m = _MINER_NAME_RE.match(text)
    if m:
        return f"nm_{m.group(1)}"
    if text.startswith("alphasage_"):
        return "nm_" + text[len("alphasage_"):]
    return text


def scrub_names(text):
    """Replace every miner-prefixed id appearing inside free text (error
    messages, reasons) with its display form."""
    return _MINER_TOKEN_RE.sub(lambda m: display_name(m.group(0)), str(text))


# ---------------------------------------------------------------------------
# 1. detail pages
# ---------------------------------------------------------------------------
def render_pages():
    import incremental_factor_intake as intake
    rows = load_results()
    published, skipped, errors = [], [], {}
    (REPORTS / "factors").mkdir(parents=True, exist_ok=True)
    for name, row in sorted(rows.items()):
        entry = row.get("entry")
        if not entry or not html_eligible(entry):
            reason = (row.get("status") or "no_manifest") if not entry else "below_gate_or_ineligible"
            skipped.append((name, reason))
            # 撤下陈旧详情页（可恢复归档），避免站点残留旧口径页面
            try:
                intake.withdraw_factor_html(REPORTS, name)
                intake.withdraw_factor_html(REPORTS, display_name(name))
            except Exception:
                pass
            continue
        dname = display_name(name)
        try:
            with tempfile.TemporaryDirectory(prefix="publish_") as staging:
                staging = Path(staging)
                source = staging / "manifest.json"
                source.write_text(json.dumps(dict(
                    schema_version=1, evaluation_version=entry.get("evaluation_version"),
                    evaluation_provenance=row.get("provenance"),
                    factors={name: entry}), ensure_ascii=False))
                intake.publish_report_from_manifest(source, report_dir=staging, update_homepage=False)
                page = staging / "factors" / f"factor_{name}.html"
                html_bytes = page.read_bytes().replace(name.encode(), dname.encode())
                target = REPORTS / "factors" / f"factor_{dname}.html"
                temporary = target.with_suffix(".html.tmp")
                temporary.write_bytes(html_bytes)
                os.replace(temporary, target)
            # remove the page published earlier under the internal id, if any
            stale = REPORTS / "factors" / f"factor_{name}.html"
            if stale.exists():
                stale.unlink()
            published.append(dname)
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(dict(published=len(published), skipped=len(skipped), errors=len(errors),
                          error_sample=dict(list(errors.items())[:5])), ensure_ascii=False))
    return published


# ---------------------------------------------------------------------------
# 2. homepage
# ---------------------------------------------------------------------------
def _badge_new():
    return '<span class="tag" style="background:#dcfce7;color:#166534">新</span>'


def _flip_badge(flipped):
    return '<span class="tag tag-flip">翻正</span>' if flipped else ""


def _fmt(value, spec="+.4f"):
    import math
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    return format(value, spec)


def _pct(value):
    import math
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    return f"{value:.1%}"


def _redundancy_map():
    """factor -> correlation row (vs pool / among new), from the correlation step."""
    if not CORR_JSON.exists():
        return {}
    try:
        payload = json.loads(CORR_JSON.read_text())
    except Exception:
        return {}
    return {r.get("factor"): r for r in payload.get("new_factors", [])
            if r.get("status") == "ok"}


POOL_RHO_REDUNDANT = 0.70     # |rho| vs any pool factor at or above this -> demote
TWIN_RHO = 0.95               # |rho| vs another new factor at or above this -> keep best only


def _repair_summary():
    """Counts of the repairs this run performed, for the evidence block."""
    kinds, dsl, fallbacks = {}, [], 0
    state = _merged_state()
    for name, info in state.items():
        kind = ((info or {}).get("repair") or {}).get("kind")
        if kind:
            kinds[kind] = kinds.get(kind, 0) + 1
    for name, manifest in _manifest_paths().items():
        try:
            entry = (json.loads(manifest.read_text()).get("factors") or {}).get(name)
        except Exception:
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("dsl_repair"):
            dsl.append(name)
        if entry.get("landing_backend_fallbacks"):
            fallbacks += 1
    return kinds, dsl, fallbacks


def build_new_mining_section(rows):
    """本周新挖栏目：完整台账，含未过门槛/失败的因子，链接仅对已发布详情页给出。

    Batch policy: "本周新增" is the intake round being published — same rule the
    pool page states ("不按本周日历日期重新划分").  The campaign timestamps are
    reported as a per-batch breakdown for traceability, and they never demote a
    candidate to pool stock.

    Redundancy policy: a candidate that merely re-packages an existing pool
    factor (|rho| >= 0.70 vs the pool) or duplicates a stronger sibling inside
    this batch (|rho| >= 0.95, e.g. turnover_ratio vs 1/turnover_ratio) is
    demoted out of the headline list and shown in a collapsed evidence block.
    """
    scope = list(rows)
    batches = batch_breakdown(scope)

    evaluated = [r for r in scope if r.get("entry") and html_eligible(r["entry"])]
    evaluated.sort(key=lambda r: -(r["metrics"].get("rank_ic") or -9))
    landed = [r for r in scope if r.get("entry")]
    pending = [r for r in scope if not (r.get("entry") and html_eligible(r["entry"]))]

    corr = _redundancy_map()
    ic_of = {r["name"]: (r["metrics"].get("rank_ic") or 0.0) for r in evaluated}
    demoted, headline = [], []
    for r in evaluated:
        info = corr.get(r["name"])
        rho = info.get("max_abs_rho_vs_pool") if info else None
        twin = None
        if info:
            for pair in info.get("top_new_pairs") or []:
                a = abs(pair.get("rho") or 0.0)
                if a < TWIN_RHO:
                    continue
                other = pair.get("factor")
                oi = ic_of.get(other, 0.0)
                mi = ic_of.get(r["name"], 0.0)
                if oi > mi or (oi == mi and other < r["name"]):
                    twin = (other, pair.get("rho"))
                    break
        if rho is not None and rho >= POOL_RHO_REDUNDANT:
            demoted.append((r, "pool", rho, info.get("most_correlated")))
        elif twin is not None:
            demoted.append((r, "twin", twin[1], twin[0]))
        else:
            headline.append(r)

    lines = []
    for i, r in enumerate(headline, 1):
        m = r["metrics"]
        dname = display_name(r["name"])
        lines.append(
            f'<tr><td class="rank">{i}</td>'
            f'<td><a href="factors/factor_{esc(dname)}.html"><code>{esc(dname)}</code></a>{_badge_new()}{_flip_badge(r.get("is_flipped"))}</td>'
            f'<td class="{"pos" if (m.get("rank_ic") or 0) >= 0 else "neg"}">{_fmt(m.get("rank_ic"))}</td>'
            f'<td>{_fmt(m.get("ic_ir"), "+.3f")}</td>'
            f'<td>{_fmt(m.get("ls_sharpe"), "+.2f")}</td>'
            f'<td>{_pct(m.get("ls_annual"))}</td>'
            f'<td>{_pct(m.get("ls_mdd"))}</td>'
            f'<td>{esc(r.get("direction"))}</td>'
            f'<td style="font-size:0.72rem;color:#64748b">{esc(r.get("evaluated_at") or "—")}</td></tr>')
    body = "\n".join(lines) if lines else '<tr><td colspan="9">暂无通过主链路重评并达到筛选门槛的因子</td></tr>'
    fail_lines = []
    for r in sorted(pending, key=lambda x: x["name"]):
        st = r.get("status") or "pending"
        why = r.get("reason") or (r["entry"].get("reason") if r.get("entry") else "") or ""
        fail_lines.append(f'<li><code>{esc(display_name(r["name"]))}</code> — {esc(st)}；{esc(scrub_names(str(why))[:220])}</li>')
    import collections as _collections
    breakdown = _collections.Counter((r.get("status") or "pending") for r in pending)
    summary = "、".join(f"{k} {v}" for k, v in breakdown.most_common())
    fail_block = (f'<details><summary>未达门槛 / 未完成的 {len(fail_lines)} 个候选（{esc(summary)}）'
                  f'—— 保留失败证据，不进入推荐列表</summary>'
                  f'<ul style="font-size:0.78rem">{"".join(fail_lines)}</ul></details>') if fail_lines else ""
    dem_lines = []
    for r, kind, rho, other in demoted:
        label = ('与存量池已有因子高度相关（|ρ|≥0.70，疑似换手率/流动性等经典因子换个公式重报）'
                 if kind == "pool" else '与本批另一个新挖因子高度重复（|ρ|≥0.95，仅保留 RankIC 更高者）')
        dem_lines.append(
            f'<li><code>{esc(display_name(r["name"]))}</code> — {esc(label)}：max|ρ|={rho:.3f}'
            f'（最相关：<code>{esc(display_name(other))}</code>）；本轮重评 RankIC={_fmt(r["metrics"].get("rank_ic"))}，'
            f'已并入存量口径，不在上方列表重复展示。</li>')
    dem_block = (f'<details><summary>去冗余：{len(dem_lines)} 个因子因与已有因子高度相关被降级'
                 f'（证据保留，总表仍收录）</summary>'
                 f'<ul style="font-size:0.78rem">{"".join(dem_lines)}</ul></details>') if dem_lines else ""

    # ---- traceability: which mining batch each candidate came from ----
    batch_rows = "".join(
        f'<tr><td>{esc(d)}</td><td>{n}</td><td>{landed_n}</td><td>{eval_n}</td></tr>'
        for d, (n, landed_n, eval_n) in batches.items())
    batch_block = (
        f'<details><summary>本轮候选的挖掘批次分布（{len(batches)} 个批次，共 {len(scope)} 个候选）'
        f'——仅作溯源，不作为"新/存量"的划分依据</summary>'
        '<table style="font-size:0.78rem"><thead><tr><th>挖掘批次（上游 campaign）</th>'
        '<th>候选数</th><th>已落值</th><th>达门槛</th></tr></thead>'
        f'<tbody>{batch_rows}</tbody></table>'
        '<p style="font-size:0.78rem;color:#64748b">上游（xlubs mining_outputs）本周期共产出以上批次，'
        '最新一批的挖掘时间是 2026-09-16；COS 上 09-17 之后没有新批次上传，'
        '因此若强行按"最近 7 个自然日"切分，本周只剩 09-16 那一批，那不是报告丢数据，'
        '而是上游本周还没有新产出。本栏目与存量页口径一致：按提交批次（而非日历周）认定"本周新增"。</p>'
        '</details>')

    kinds, dsl_fixed, fallbacks = _repair_summary()
    cov = kinds.get("coverage", 0)
    res = kinds.get("resource", 0)
    repair_block = (
        '<details><summary>本轮"不可用"逐条归因与修复（含已修好的证据）</summary>'
        '<ul style="font-size:0.78rem">'
        f'<li><b>预热边界（已彻底取消该要求）</b>：旧判定规则要求因子在窗口首日 2016-01-04 就要有有效截面，'
        '而行情数据本身最早就是 2016-01-04——任何带 lookback 的算子（连 <code>ts_delta(x, 1)</code> 也要前一天）'
        '在首日必然是空值，于是被判"不可用"。这不是参数比可用历史长，而是规则要求"零预热"。'
        '现在的口径是<b>完全不设预热要求、也不再设 260 天之类的固定宽限</b>：因子从它自己的第一个有效值开始算、开始落值，'
        '左边界只记录不限制（<code>factor_report_sources.MIN_USABLE_DAYS</code> 只管"有效交易日太少会让 IC 变成噪声"这一条统计底线，'
        '默认 250 个交易日，设为 0 即完全不卡）。改口径后仅复用既有落值矩阵重评，该类因子即有 {cov} 个转为可用。</li>'
        f'<li><b>准入拒绝（{res} 个，串行重落值中）</b>：并发落值时引擎的内存准入控制器按主机 MemAvailable '
        '与外部负载判定，返回 ResourceBudgetExceeded。这类不是公式问题，改为串行重落值。</li>'
        f'<li><b>公式缺陷（3 类，全部已处理）</b>：① <code>winsorize(x, 1)</code> 的第二参是 (0,1) 分位，'
        '作者写 1（本意 1%），引擎按 lower=1 &gt; upper=0.95 拒绝 → 已按作者本意修为 '
        f'<code>winsorize(x, 0.01)</code> 并在因子页标注（{len(dsl_fixed)} 个）；'
        '② <code>ts_regression_slope(y, &lt;非列表达式&gt;, w)</code> 被 polars/SQL 计划的'
        '"第一个非列子节点即 window"启发式误判 → 已把计划参数错误纳入后端回退条件，'
        f'自动改用 pandas 桥接路径落值（本轮已回退 {fallbacks} 个）；'
        '③ <code>log(turnover_ratio)</code> 缺零值保护：只要有 1 只股票换手率为 0，'
        '<code>log(0)=-inf</code> 就把整个截面的 zscore 变成 NaN（实测 2588 天里 2580 天含零值股票 → '
        '全部不可用，仅 8 天无零值）→ 改用引擎自带的 <code>safe_log</code>。</li>'
        '<li><b>真退化（2 个，如实标注、不做假修复）</b>：'
        '<code>cs_resid(turnover_ratio, turnover_ratio)</code> 是把 x 对自己回归，残差恒等于 0；'
        '<code>ts_pct(is_nan(net_profit), 60)</code> 的输入恒为常数，滚动分位退化成 0/0。'
        '两者实测横截面标准差 = 0（常数），定义上就没有信息量，与落值/DSL 无关；'
        '任何"修复"都会变成另一个因子，故保留原始公式并标注为挖掘侧缺陷。</li>'
        '<li><b>分类缺陷（已修）</b>：因子矩阵搜索目录与既有周度回测因子库共用，于是"我方从未成功落值、'
        '但该目录里躺着同名遗留文件"的候选，会被判成"有矩阵但覆盖不足"而<b>永久移出重试集</b>'
        '（本轮实测 13 个受此影响）。已改为按 manifest 的 <code>matrix_path</code> 判定矩阵归属：'
        '没有自产矩阵一律视为未落值并重新落值。该判别在现有台账上零误差——'
        '76 个可用因子 100% 持有自产矩阵，101 个不可用因子 100% 不持有。'
        '重落值后仍不达标者（滚动窗口叠加事件字段的定义域限制），按实情标注，不再反复重试。</li>'
        '</ul></details>')

    return f'''
<section id="new-mining">
<h2>🆕 本周新增因子（{len(headline)} 个达筛选门槛且非冗余 / 共 {len(scope)} 个本轮候选）</h2>
<p style="font-size:0.82rem;color:#64748b">
本轮进度：<b>{len(landed)}/{len(scope)}</b> 个候选已完成落值 + 评估
（达门槛 {len(evaluated)} 个，其中 {len(demoted)} 个因与存量/本批高度相关被降级去冗余，实际展示 {len(headline)} 个；
未达标 {len(landed) - len(evaluated)} 个；尚未完成 {len(scope) - len(landed)} 个，详见下方折叠清单）。<br/>
<b>批次口径</b>：与存量页一致——"本周"指登记来源"本周新增"的<b>累计提交批次</b>，<b>不按日历周重新划分</b>。
本轮候选来自上游 <b>{len(batches)}</b> 个挖掘批次（{esc(" → ".join(batches))}），
全部为本报告周期首次纳入：上一版首页（2026-09-14 版面）没有任何新挖栏目，故这 {len(scope)} 条中不存在"上周已发布"的重复项，
也不把任何一条降级为存量。你点名的 <code>rank(StockValuationDaily.TurnoverRatio * 0.01)</code> 一类，
属于存量池登记因子（<code>weekly_*</code>），不在本轮候选内，仍按存量展示。<br/>
候选来源：<code>{esc(SOURCE_META)}</code>（按公式哈希去重后与本报告存量池无重叠）。<br/>
准入阈值：{THRESHOLD_TEXT}。<br/>
本栏目展示的是<b>本轮用本报告同一主链路重新落值 + quant_evaluator 重评</b>的结果：因子值经 FactorEngine 全窗（2016-01-04 → 2026-08-27）落值，
评估口径 <code>equal-amount-gross100-actual-volume-v3-20260907</code>，方向仅由 2016-01-04→2018-06-30 训练窗确定。
挖掘方自报指标不参与本栏目的 RankIC/IR 展示，仅作对照保留在详情页。<br/>
栏目生成时间（北京时间）：{now_cn()}。
</p>
{repair_block}
<table>
  <thead><tr><th>#</th><th>因子</th><th>RankIC</th><th>IR</th><th>LS Sharpe</th><th>LS 年化</th><th>LS 回撤</th><th>方向</th><th>本轮评估时间</th></tr></thead>
  <tbody>{body}</tbody>
</table>
{dem_block}
{fail_block}
{batch_block}
</section>
'''


def _master_table_rows(rows, start_rank):
    out = []
    rank = start_rank
    for r in sorted(rows, key=lambda x: x["name"]):
        if not (r.get("entry") and html_eligible(r["entry"])):
            continue
        m = r["metrics"]
        dname = display_name(r["name"])
        out.append(
            f'<tr><td class="rank">{rank}</td>'
            f'<td><a href="factors/factor_{esc(dname)}.html"><code>{esc(dname)}</code></a>{_badge_new()}{_flip_badge(r.get("is_flipped"))}</td>'
            f'<td class="{"pos" if (m.get("rank_ic") or 0) >= 0 else "neg"}">{_fmt(m.get("rank_ic"))}</td>'
            f'<td>{_fmt(m.get("ic_ir"), "+.3f")}</td>'
            f'<td>{_fmt(m.get("ls_sharpe"), "+.2f")}</td>'
            f'<td>{_pct(m.get("ls_annual"))}</td>'
            f'<td>{_pct(m.get("ls_mdd"))}</td>'
            f'<td>{_pct(m.get("ls_winrate"))}</td>'
            f'<td>{_pct(m.get("g10_annual"))}</td>'
            f'<td>本轮重评 · 未纳入 2026 稳健性重算</td></tr>')
        rank += 1
    return out, rank


def update_index():
    rows = list(load_results().values())
    html = INDEX.read_text(encoding="utf-8")
    section = build_new_mining_section(rows)

    # 1) replace the weekly new-mining section entirely
    html, n = re.subn(r'<section\b[^>]*id="new-mining"[^>]*>.*?</section>', '', html, flags=re.DOTALL)
    anchor = '<section id="robustness-2026">'
    if anchor in html:
        html = html.replace(anchor, section + '\n' + anchor, 1)
    else:
        html = html.replace('</main>', section + '</main>', 1)

    # 2) append rows to master table and bump its count
    match = re.search(r'<h2 id="all-factors">全部 (\d+) 个因子</h2>', html)
    if not match:
        raise ValueError("homepage master table heading missing")
    declared = int(match.group(1))
    # idempotency matches on both naming schemes: rows injected by earlier runs
    # carry the internal id in their href, new runs write display names.
    # Counts must use the *number of factors*, not the size of this union.
    eligible = [r for r in rows if r.get("entry") and html_eligible(r["entry"])]
    n_eligible = len(eligible)
    added_names = {display_name(r["name"]) for r in eligible} | {r["name"] for r in eligible}

    ref = html.find('<h2 id="all-factors">')
    table_end = html.find('</table>', ref)
    if ref < 0 or table_end < 0:
        raise ValueError("homepage master table missing")
    # The master table lists a non-contiguous subset of ranked factors, so the
    # heading is a nominal count, not the row count.  Derive the delta from the
    # rows a previous run already injected: re-running must not inflate it.
    pre_tbody = html[html.find('<tbody>', ref):table_end]
    already = sum(1 for name in re.findall(r'href="factors/factor_([^"<>]+)\.html"', pre_tbody)
                  if name in added_names)
    total = declared + n_eligible - already
    # Deterministic start rank, derived from the (idempotent) nominal total so
    # re-runs renumber exactly the same rows instead of drifting upward.
    added, _next_rank = _master_table_rows(rows, total - n_eligible + 1)

    # remove any earlier copy of these rows (idempotent re-runs)
    def keep_row(m):
        row = m.group(0)
        links = re.findall(r'href="factors/factor_([^"<>]+)\.html"', row)
        return '' if any(n in added_names for n in links) else row
    # Scope every master-table edit to the master table itself.  The weekly
    # new-mining section lives *after* this table, so a document-wide row
    # rewrite would silently delete the rows that were just inserted above.
    head, table, rest = html[:ref], html[ref:table_end], html[table_end:]
    table = re.sub(r'<tr\b[^>]*>.*?</tr>', keep_row, table, flags=re.DOTALL)
    tbody_end = table.find('</tbody>')
    if tbody_end < 0:
        raise ValueError("homepage master table tbody missing")
    table = table[:tbody_end] + ('\n' + '\n'.join(added) + '\n') + table[tbody_end:]
    table = re.sub(r'<h2 id="all-factors">全部 \d+ 个因子</h2>',
                   f'<h2 id="all-factors">全部 {total} 个因子</h2>', table, count=1)
    html = head + table + rest

    # 3) header / card counts (best effort, only where the pattern is unambiguous)
    html = re.sub(r'<b>(\d+)</b><span>报告因子数</span>', f'<b>{total}</b><span>报告因子数</span>', html)
    html = re.sub(r'<b>(\d+)</b><span>报告因子总数</span>', f'<b>{total}</b><span>报告因子总数</span>', html)

    # 4) miner-free naming everywhere: rename last week's legacy pages and
    #    scrub any residual internal id from free text (error reasons etc.)
    html = _legacy_page_rename(html)
    html = scrub_names(html)

    tmp = INDEX.with_suffix(".html.tmp")
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, INDEX)
    print(json.dumps(dict(previous_section_removed=n, master_rows_added=len(added),
                          master_total=total, already_present=already,
                          new_mining_evaluated=n_eligible),
                     ensure_ascii=False))


def _legacy_page_rename(html):
    """Rename last-week's miner-prefixed pages and rewrite every reference.

    The two factors promoted from last week's new-mining section (plus their
    siblings) were published under ``factor_alphasage_*`` filenames; the user
    asked for miner-free display names everywhere, so the page files are
    renamed with their content rewritten, and all homepage references updated.
    """
    ids = sorted(set(re.findall(r'href="factors/factor_(alphasage_[A-Za-z0-9_]+)\.html"', html)))
    renamed = []
    for name in ids:
        dname = display_name(name)
        if dname == name:
            continue
        old = REPORTS / "factors" / f"factor_{name}.html"
        new = REPORTS / "factors" / f"factor_{dname}.html"
        if old.exists() and not new.exists():
            new.write_bytes(old.read_bytes().replace(name.encode(), dname.encode()))
            old.unlink()
            renamed.append(dname)
        html = html.replace(name, dname)
    if renamed:
        print(json.dumps(dict(legacy_pages_renamed=renamed), ensure_ascii=False))
    return html


# ---------------------------------------------------------------------------
# 3. correlation analysis
# ---------------------------------------------------------------------------
def _sample_grid(n_dates=180, min_assets=200):
    """A common sampling grid over the widest available post-window span.

    The reference must not be whichever matrix happens to sort first: some pool
    blocks only start in 2019, and any warm-up-repaired factor carries extra
    pre-window rows.  Read one column per candidate (the date index is the same
    for every column, so this stays cheap) across a spread of the pool and keep
    the widest coverage.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    files = sorted(MATRICES.glob("*.parquet"))
    stride = max(1, len(files) // 60)
    best = None
    for path in files[::stride]:
        try:
            column = pq.ParquetFile(path).schema_arrow.names[0]
            index = pd.read_parquet(path, columns=[column]).index
        except Exception:
            continue
        span = index[index >= "2016-01-04"]
        if best is None or len(span) > len(best):
            best = span
    if best is None or len(best) == 0:
        best = pd.read_parquet(files[0]).index
        best = best[best >= "2016-01-04"]
    step = max(1, len(best) // n_dates)
    return best[::step]


def correlate(n_dates=180, min_overlap=120):
    """Spearman (rank) correlation on a common date/asset sample.

    NaN is filled with 0 after per-date demeaning of ranks, so a factor that is
    missing on a cell contributes no signal rather than a fake value.
    """
    import numpy as np
    import pandas as pd

    rows = load_results()
    pool_files = {p.stem: p for p in MATRICES.glob("*.parquet")}
    new_names = [n for n in rows if n in pool_files]
    new_names.sort()
    pool_names = [n for n in pool_files if n not in set(new_names)]
    names = new_names + pool_names
    print(f"[corr] new={len(new_names)} pool={len(pool_names)}", flush=True)

    dates = _sample_grid(n_dates)
    frames = {}
    coverage = {}
    for i, name in enumerate(names):
        path = pool_files.get(name)
        if path is None:
            continue
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        frame = frame.reindex(index=dates)
        if frame.shape[1] < 50:
            continue
        ranks = frame.rank(axis=1, na_option="keep")
        ranks = ranks.sub(ranks.mean(axis=1), axis=0)
        ranks = ranks.div(ranks.abs().sum(axis=1).replace(0, np.nan), axis=0)
        frames[name] = ranks.astype("float32")
        coverage[name] = int(np.isfinite(frame.to_numpy(dtype=np.float32)).sum())
        if (i + 1) % 100 == 0:
            print(f"[corr] loaded {i + 1}/{len(names)}", flush=True)

    # Every factor has its own asset universe, so the flattened vectors only
    # line up on a shared column index.  Use the union of all universes; cells
    # outside a factor's own universe become NaN and then 0, i.e. they carry no
    # signal instead of a fake cross-sectional value.
    universe = sorted(set().union(*(set(f.columns) for f in frames.values()))) if frames else []
    print(f"[corr] shared universe={len(universe)} assets", flush=True)
    vectors = {}
    for name, ranks in frames.items():
        aligned = ranks.reindex(columns=universe)
        vectors[name] = np.nan_to_num(aligned.to_numpy(dtype=np.float32), nan=0.0).reshape(-1)
    del frames

    usable = [n for n in names if n in vectors]
    matrix = np.vstack([vectors[n] for n in usable])
    print(f"[corr] matrix={matrix.shape} bytes={matrix.nbytes / 1024**3:.2f} GiB", flush=True)
    gram = matrix @ matrix.T
    norms = np.sqrt(np.diag(gram))
    norms[norms == 0] = np.nan
    corr = gram / np.outer(norms, norms)
    np.nan_to_num(corr, copy=False)
    index = {n: i for i, n in enumerate(usable)}

    new_rows = []
    new_set = set(new_names)
    for n in new_names:
        if n not in index:
            new_rows.append(dict(factor=n, status="no_matrix"))
            continue
        row = corr[index[n]]
        # Two separate maxima: against the pre-existing pool, and inside this
        # mining batch.  Redundancy is judged on whichever is larger, so a new
        # factor duplicating another new factor is flagged too.
        pool_hit, pool_rho = None, 0.0
        for other in pool_names:
            j = index.get(other)
            if j is None:
                continue
            rho = float(row[j])
            if abs(rho) > abs(pool_rho):
                pool_hit, pool_rho = other, rho
        batch_hit, batch_rho = None, 0.0
        for other in new_names:
            if other == n:
                continue
            j = index.get(other)
            if j is None:
                continue
            rho = float(row[j])
            if abs(rho) > abs(batch_rho):
                batch_hit, batch_rho = other, rho
        overall_hit, overall_rho = pool_hit, pool_rho
        kind = "与存量池"
        if abs(batch_rho) > abs(pool_rho):
            overall_hit, overall_rho, kind = batch_hit, batch_rho, "新挖内部"
        new_new = [(o, float(row[index[o]])) for o in new_names
                   if o != n and o in index]
        new_new.sort(key=lambda kv: -abs(kv[1]))
        new_rows.append(dict(
            factor=n, status="ok",
            max_abs_rho_vs_pool=abs(pool_rho), max_rho_vs_pool=pool_rho,
            most_correlated=overall_hit, most_correlated_kind=kind,
            max_abs_rho_overall=abs(overall_rho), max_rho_overall=overall_rho,
            most_correlated_pool=pool_hit, max_abs_rho_vs_new=abs(batch_rho),
            most_correlated_new=batch_hit,
            coverage_cells=coverage.get(n),
            top_new_pairs=[dict(factor=k, rho=v) for k, v in new_new[:5]],
            redundant=bool(overall_hit is not None and abs(overall_rho) >= 0.7),
        ))

    payload = dict(generated_at=now_cn(), sample_dates=int(len(dates)),
                   n_factors=len(usable), n_new=len(new_names), n_pool=len(pool_names),
                   method="per-date rank → demean → L1-normalise → cosine (= Spearman on the sampled panel)",
                   new_factors=new_rows)
    CORR_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    red = sum(1 for r in new_rows if r.get("redundant"))
    print(json.dumps(dict(written=str(CORR_JSON), factors=len(usable), redundant_vs_pool=red),
                     ensure_ascii=False))

    # small self-contained heatmap over the new block
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for font_path in ("/home/sunhaiwei/.fonts/NotoSansSC-Regular.otf",
                          "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
            if os.path.exists(font_path):
                try:
                    font_manager.fontManager.addfont(font_path)
                    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
                    plt.rcParams["axes.unicode_minus"] = False
                    break
                except Exception:
                    pass
        block_names = [n for n in new_names if n in index]
        idx = [index[n] for n in block_names]
        sub = corr[np.ix_(idx, idx)]
        fig, ax = plt.subplots(figsize=(7, 6), dpi=130)
        im = ax.imshow(sub, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title("本周新挖因子两两相关系数（Spearman，采样面板）", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
        buf = io.BytesIO()
        fig.tight_layout(); fig.savefig(buf, format="png"); plt.close(fig)
        (WORK / "fig_new_mining_correlation.png").write_bytes(buf.getvalue())
        print("[corr] heatmap ->", WORK / "fig_new_mining_correlation.png")
    except Exception as exc:
        print("[corr] heatmap skipped:", exc)
    return payload


def correlation_section():
    if not CORR_JSON.exists():
        return ""
    payload = json.loads(CORR_JSON.read_text())
    rows = payload["new_factors"]
    ok = [r for r in rows if r.get("status") == "ok"]
    ok.sort(key=lambda r: -(r.get("max_abs_rho_overall") or r["max_abs_rho_vs_pool"]))
    red = sum(1 for r in ok if r.get("redundant"))
    lines = []
    for r in ok[:400]:
        kind = {"新挖内部": "新挖内部", "与存量池": "与存量池"}.get(r.get("most_correlated_kind"), "—")
        tag = ('<span class="tag" style="background:#fee2e2;color:#991b1b">高度相关</span>'
               if r["redundant"] else
               '<span class="tag" style="background:#dcfce7;color:#166534">相对独立</span>')
        overall = r.get("max_abs_rho_overall", r["max_abs_rho_vs_pool"])
        lines.append(f'<tr><td><code>{esc(display_name(r["factor"]))}</code></td>'
                     f'<td>{(overall if overall is not None else float("nan")):.3f}</td>'
                     f'<td>{r["max_abs_rho_vs_pool"]:.3f}</td>'
                     f'<td>{(r.get("max_abs_rho_vs_new") or 0.0):.3f}</td>'
                     f'<td><code>{esc(display_name(r["most_correlated"]) if r["most_correlated"] else "—")}</code> <span style="color:#94a3b8;font-size:0.72rem">{esc(kind)}</span></td>'
                     f'<td>{tag}</td></tr>')
    heat = WORK / "fig_new_mining_correlation.png"
    img = ""
    if heat.exists() and heat.stat().st_size < 900_000:
        img = ('<img style="max-width:100%" alt="新挖因子相关性热力图" src="data:image/png;base64,'
               + base64.b64encode(heat.read_bytes()).decode() + '"/>')
    return f'''
<section id="new-mining-correlation">
<h2>🔗 本周新挖因子相关性分析</h2>
<p style="font-size:0.82rem;color:#64748b">
方法：{esc(payload["method"])}；采样 {payload["sample_dates"]} 个交易日 × 全市场，共 {payload["n_factors"]} 个有矩阵的因子
（新挖 {payload["n_new"]} + 存量 {payload["n_pool"]}）。生成时间：{esc(payload["generated_at"])}。<br/>
覆盖范围：本轮全部新挖因子 × 新挖因子两两、以及新挖 × 全部存量因子。<br/>
判读：|ρ| ≥ 0.70 视为与已有因子高度相关（近似冗余），应谨慎重复入池；该阈值沿用报告既有聚类口径。
本批 <b>{len(ok)}</b> 个因子中 <b>{red}</b> 个被判为高度相关。<br/>
缺失值不填充为 0 以外的任何假设值；采样面板口径与逐日全样本口径存在差异，属已知可比性限制。
</p>
{img}
<table>
  <thead><tr><th>新挖因子</th><th>全库最大 |ρ|</th><th>与存量池 |ρ|</th><th>与新挖 |ρ|</th><th>最相关因子</th><th>判读</th></tr></thead>
  <tbody>{''.join(lines)}</tbody>
</table>
</section>
'''


def inject_correlation():
    section = correlation_section()
    if not section:
        print("[corr] no correlation payload; skipped")
        return
    html = INDEX.read_text(encoding="utf-8")
    html = re.sub(r'<section\b[^>]*id="new-mining-correlation"[^>]*>.*?</section>', '', html, flags=re.DOTALL)
    anchor = '<section id="clusters">'
    if anchor in html:
        html = html.replace(anchor, section + '\n' + anchor, 1)
    else:
        html = html.replace('</main>', section + '</main>', 1)
    tmp = INDEX.with_suffix(".html.tmp")
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, INDEX)
    print("[corr] section injected")


# ---------------------------------------------------------------------------
# 4. COS upload
# ---------------------------------------------------------------------------
def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _name_mapping(rows):
    """Internal id -> display id, persisted next to the published artifacts."""
    mapping = {}
    for name in sorted(rows):
        dname = display_name(name)
        if dname != name:
            mapping[name] = dname
    return mapping


def cos_upload(limit=None, what="all"):
    """Upload factor values / evaluations / report pages to the authorized COS prefix."""
    receipts_path = WORK / "cos_receipts.json"
    receipts = json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
    rows = load_results()
    mapping_path = WORK / "name_mapping.json"
    mapping_path.write_text(json.dumps(_name_mapping(rows), ensure_ascii=False, indent=1),
                            encoding="utf-8")
    targets = [(mapping_path, "metadata/name_mapping.json")]
    if what in {"all", "values"}:
        for name in sorted(rows):
            matrix = MATRICES / f"{name}.parquet"
            if matrix.exists() and not matrix.is_symlink():
                targets.append((matrix, f"factor_values/{display_name(name)}.parquet"))
    if what in {"all", "evaluations"}:
        manifests = _manifest_paths()
        for name in sorted(manifests):
            manifest = manifests[name]
            if manifest.exists():
                dname = display_name(name)
                targets.append((manifest, f"evaluations/{dname}/report_manifest.json"))
                payload = json.loads(manifest.read_text())
                artifact = payload.get("factors", {}).get(name, {}).get("artifact")
                if artifact and Path(artifact).exists():
                    targets.append((Path(artifact), f"evaluations/{dname}/{Path(artifact).name}"))
    if what in {"all", "report"}:
        # Only this round's pages plus the shared entry points.  Re-uploading
        # every historical detail page would be thousands of redundant objects.
        for name in sorted(rows):
            path = REPORTS / "factors" / f"factor_{display_name(name)}.html"
            if path.exists():
                targets.append((path, f"report/factor_engine/docs/reports/2026-08-23/factors/{path.name}"))
        if limit is None:
            for path in (INDEX, REPORTS / "robustness_2026/robustness_2026.html",
                         REPORTS / "lqtp_submissions.html"):
                if path.exists():
                    rel = path.relative_to(REPORTS)
                    targets.append((path, f"report/factor_engine/docs/reports/2026-08-23/{rel}"))
    if limit:
        targets = targets[:limit]
    print(f"[cos] objects to upload: {len(targets)}", flush=True)

    done = 0
    errors = {}
    for path, rel in targets:
        key = f"{path}::{os.path.getsize(path)}"
        try:
            sha = _sha256(path)
            if receipts.get(str(path), {}).get("sha256") == sha and receipts[str(path)].get("verified"):
                done += 1
                continue
            uri = f"{COS_ROOT}/{rel}"
            result = subprocess.run(["admin-cos", "cp", str(path), uri],
                                    capture_output=True, timeout=1800)
            if result.returncode != 0:
                errors[str(path)] = result.stderr.decode()[-300:]
                continue
            with tempfile.TemporaryDirectory(prefix=".cos_verify_", dir=WORK) as tmp:
                back = Path(tmp) / "object"
                subprocess.run(["admin-cos", "cp", uri, str(back)], check=True,
                               capture_output=True, timeout=1800)
                if _sha256(back) != sha:
                    errors[str(path)] = "readback sha256 mismatch"
                    continue
            receipts[str(path)] = dict(uri=uri, sha256=sha, bytes=os.path.getsize(path),
                                       verified=True, at=now_cn())
            done += 1
            if done % 25 == 0:
                receipts_path.write_text(json.dumps(receipts, ensure_ascii=False, indent=1))
                print(f"[cos] uploaded {done}/{len(targets)}", flush=True)
        except Exception as exc:
            errors[str(path)] = f"{type(exc).__name__}: {exc}"
    receipts_path.write_text(json.dumps(receipts, ensure_ascii=False, indent=1))
    print(json.dumps(dict(uploaded_or_cached=done, total=len(targets), errors=len(errors),
                          error_sample=dict(list(errors.items())[:5])), ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", action="store_true")
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--correlate", action="store_true")
    ap.add_argument("--corr-inject", action="store_true")
    ap.add_argument("--cos", action="store_true")
    ap.add_argument("--cos-what", default="all", choices=("all", "values", "evaluations", "report"))
    ap.add_argument("--cos-limit", type=int, default=None)
    ap.add_argument("--ndates", type=int, default=180)
    args = ap.parse_args()
    if args.pages:
        render_pages()
    if args.correlate:
        correlate(n_dates=args.ndates)
    if args.corr_inject:
        inject_correlation()
    if args.index:
        update_index()
    if args.cos:
        cos_upload(limit=args.cos_limit, what=args.cos_what)
    if not any([args.pages, args.index, args.correlate, args.corr_inject, args.cos]):
        ap.print_help()


if __name__ == "__main__":
    main()
