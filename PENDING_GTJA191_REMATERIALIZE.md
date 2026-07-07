# 待办：GTJA191 全量 force 落盘（算子改完后）

**状态：** 已暂停落盘，等待 factor_engine 算子改完。

**排期：** 约 1 小时后（`scripts/schedule_gtja191_rematerialize.sh 1h`）会生成就绪标记：
- 标记文件：`.scheduled_gtja191_rematerialize.ready`
- 日志：`logs/deferred_gtja191_rematerialize.log`

**下次 AI 会话应执行：**

1. 检查 `factor_engine/cleaned_operators/` 等近期改动（尤其 `elementwise_math.py`）
2. 跑测试：`unset FACTOR_LAKE_ROOT && python3 -m pytest factor_engine/tests/ -q`
3. 确认无误后全量落盘：
   ```bash
   source ~/quant_projects/env.sh
   unset FACTOR_LAKE_ROOT
   python3 scripts/materialize_gtja191_factors.py --force --batch-size 1
   ```
4. 校验：`python3 scripts/verify_gtja191_factor_lake.py --sample-size 25`

**取消排期：** `bash scripts/schedule_gtja191_rematerialize.sh cancel`

**说明：** Cursor AI 无法自动在 1 小时后“醒来”；排期脚本只会写入就绪标记。请在新对话中 @ 此文件或说「继续 GTJA191 落盘」。
