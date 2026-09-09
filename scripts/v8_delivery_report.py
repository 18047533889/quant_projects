"""Summarize frozen execution and explicitly retain unclosed acceptance clauses."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/v8'


def main():
    ledger = json.loads((OUT/'acceptance_ledger.json').read_text())
    execution = json.loads((OUT/'final_regression/execution.json').read_text())
    manifest = json.loads((OUT/'final_source/source_manifest.json').read_text())
    rows = []
    total = dict(passed=0, failures=0, errors=0, skipped=0)
    for name, result in execution['suites'].items():
        cases = list(ET.parse(OUT/f'final_regression/{name}.xml').getroot().iter('testcase'))
        counts = {key:sum(c.find(tag) is not None for c in cases)
                  for key,tag in [('failures','failure'),('errors','error'),('skipped','skipped')]}
        counts['passed'] = len(cases)-sum(counts.values())
        for key in total: total[key] += counts[key]
        rows.append(f"| {name} | {counts['passed']} | {counts['failures']+counts['errors']} | {counts['skipped']} | {result['returncode']} |")
    incomplete = [(key,row) for key,row in ledger['acceptances'].items() if row['status'] != 'VERIFIED']
    lines = [
        '# V8 整改与验收记录', '',
        '## 当前结论', '',
        '代码已在独立整改目录修改并完成下列冻结版本回归；这不等于 174 项验收全部闭合，也不等于生产认证。', '',
        f"验收状态：`{json.dumps(ledger['counts'],ensure_ascii=False)}`。未闭合项逐项列在下文，不能按通过处理。", '',
        '主要改动：QE 数值、身份和适用域修正；FA 唯一连续评分与配对选择合同；FO 强制调用统一决策与预算结算；FP 正则化、状态输入与因果复用；生命周期授权、持久通知与重试保护。', '',
        '## 冻结版本回归', '',
        '| 测试组 | 通过 | 失败/错误 | 跳过或预期失败 | 退出码 |',
        '|---|---:|---:|---:|---:|', *rows, '',
        f"合计：{total['passed']} 通过，{total['failures']+total['errors']} 失败/错误，{total['skipped']} 跳过或预期失败。JUnit 的 skipped 包括 pytest 的预期失败；不是通过。", '',
        '另有隔离临时 PostgreSQL 真库回归 23 项通过，见 `postgres_full/tests.xml`；未连接生产数据库。', '',
        '首轮冻结回归在平台租约测试出现 1 项失败：20 毫秒真实时钟租约在测试线程准备阶段已过期。仅将测试改成受控时钟和事件同步，未放宽生产校验；随后重新冻结并完整重跑。首轮源码与日志完整保留于 `initial_frozen_source/` 和 `initial_frozen_regression/`。', '',
        f"冻结源码清单：{execution['source_manifest_sha256']}；测试前后源码一致：{not execution['source_changed']}。", '',
        f"分支 `{manifest['branch']}`，基线 HEAD `{manifest['head']}`。HEAD 未提交本轮修改，不能用 HEAD 单独代表交付源码。", '',
        '## 仍未闭合的验收', '',
    ]
    for key,row in incomplete:
        note = row.get('gap') or row.get('evidence') or row.get('domain')
        lines.extend([f"### {key} — {row['status']}", '', str(row['required']), '', str(note), ''])
    lines.extend([
        '## 使用、历史数据与回退边界', '',
        '- 所有修改位于 `/home/sunhaiwei/quant_projects_v3_remediation`；未提交、推送、部署或切换生产策略。原项目代码和历史证据保持不变。',
        '- `final_source/source_snapshot.tar.gz` 是源码快照；`current_vs_HEAD.patch` 包含继承的 V3–V7 修改，不是 V8 独立补丁。不要把整份补丁当作仅撤销 V8 的工具。',
        '- 数值定义改变的指标需要沿受影响血缘重新计算，并重新取得资格；纯政策变更仅可复用身份和适用域仍有效的原始数值。不得混用旧资格、新方法和旧决策。',
        '- 回退应在独立目录恢复完整且一致的源码、依赖、政策及证据组合后测试；不要覆盖当前整改工作区，也不要只替换单个评分文件。',
        '- 六份数值回执仅涵盖已执行的精确合成测试夹具；未安装进生产授权解析器，不覆盖所有参数或真实市场。',
        '- 单张 NVIDIA L20 的实测范围写在 `gpu_resource_probe.json`；没有十万因子、多 GPU 或异步能力认证。',
        '- 真实数据校准、空因子负控和新旧影子比较仍需用户指定未封存的开发数据路径、日期和股票池；未访问封存测试集。', '',
        '## 证据入口', '',
        '- `acceptance_ledger.json`：71 工单及 174 验收逐项映射。',
        '- `final_regression/`：分库日志、JUnit 和执行状态。',
        '- `final_source/`：源码清单、状态、完整差异和快照。',
        '- `numerical_receipts.json`：精确夹具范围的数值回执。',
        '- `fa/`、`fo/`、`fp_regime/`：GPT-5.6-sol 子任务的细项证据。', '',
    ])
    (OUT/'V8_整改与验收报告.md').write_text('\n'.join(lines))
    payload = dict(tests=total,acceptance_counts=ledger['counts'],
                   source_manifest_sha256=execution['source_manifest_sha256'],
                   report_sha256=hashlib.sha256((OUT/'V8_整改与验收报告.md').read_bytes()).hexdigest())
    (OUT/'delivery_summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload))


if __name__ == '__main__':
    main()
