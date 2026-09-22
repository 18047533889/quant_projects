# 可指定 COS 清单的真实批量审查入口

## 使用
在 server-c /home/sunhaiwei/quant_projects 使用已有 DataAccess COS 配置：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python factor_optimizer/examples/cos_batch_audit.py \
  --manifest cos://qs-cold/candidate_pool/sunhaiwei/lqtp_show/metadata/35e3558548de591e9dda003ad236ff4846051b73628c1d980c7cf3f9cb73067e/landing_manifest.json \
  --factors 3 --assets 64 --max-factor-mib 16 --optimize
```

此示例本轮实测使用 DATA_ACCESS_COS_CLI=admin-cos，
DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research，
ASHARE_PARQUET_ROOT=/home/sunhaiwei/cos_data，DATA_ACCESS_SKIP_COS_MIRROR=1。
这些是现有授权配置，不新增凭据或授权目标。

不指定参数仍使用旧固定清单、2 因子、256 资产和 8 MiB 单对象筛选上限。
load_cos_sample 同样提供 manifest_uri、max_factor_bytes 关键字参数。
--optimize 在 TRAIN 选择一个冻结方案，再做 VALIDATION 确认；不评分 TEST。
去掉 --optimize 仅进行训练诊断。

## 边界
- 清单必须是指定池 metadata/64位小写SHA256/landing_manifest.json 精确 URI；
  禁止跨桶、目录穿越、查询参数或本地路径。网络读取前验证。
- 因子数 1..16；单因子筛选上限最大 64 MiB，与 DataAccess 原有硬上限一致。
- 选中因子对象声明大小之和最多 128 MiB，超过时整体拒绝；不偷偷少选或换候选。
  这是压缩对象字节限制，不是整个进程 RSS 上限；解码膨胀仍受 DataAccess 自身
  结果/扫描预算约束，发生拒绝时不自动扩大预算。
- 不改 DataAccess 预算，不绕过对象内容 hash、来源状态和路径校验。
- 筛选只接受 verified 且未被质量阻断的记录，按因子 ID 排序确定样本，
  不根据收益挑样本。当前是指定清单的入口，不是全池自动发现服务。
- 每个因子使用同一组达到 TRAIN 90% 覆盖的资产，不足请求数时明确拒绝。
  资产筛选和优化复用同一 TRAIN 划分，不用验证/测试收益挑资产。

## 本轮真实数据与结果
新清单的 volume_state_persistence、weekly_b1258507388baa03、
weekly_da68888feb0a6a1a，分别约 10.15、11.14、11.14 MB，通过 DataAccess 读取。
最初请求 256 资产失败：三个因子满足 TRAIN 90% 覆盖的资产数分别为 5081、66、66。
随后仅依据训练覆盖改为 64 资产，未降低 90% 门槛，未依据收益调整样本。

500 日 × 64 资产，2024-08-02..2026-08-25；TRAIN 267 日；
默认 bootstrap 499 次。三因子最终均保留 RAW：
- volume_state_persistence：验证最差分段表现触发退化门槛。
- 两只 weekly：联合改善置信下界分别约 -0.62449、-0.10769，不能确认提升。
  它们的复合 DSL 谱系仍不完整，已识别排名，因此禁止重复 CS-rank，
  未把基础处理缺口伪装成完成。后续仍需补充有依据的复合表达式解释。
- 64 资产不足以作为充分的二十层分组验收，不能宣传为已覆盖。
- 原始因子 PIT 认证、真实中性化暴露、DSL 重编译和最终 TEST 仍未完成。

逐方法 3 × 65 = 195 案例：171 executed、24 requires_additional_inputs_or_control、
0 failed。已执行案例检查前缀不变性和资产排列不变性，比较含成本训练指标。
24 项包括每只因子的三类中性化、三类 DSL 重编译、ABANDON 和 drop 控制。
不是所有方法都完成真实验收，也不是所有已执行方法都有收益改善。

自动优化及来源证据：[JSON](cos_manifest_sampling_20260922.json)；
逐方法参数和检查结果：[JSON](cos_manifest_methods_20260922.json)。
后者压缩存储，省去重复 RAW 指标，不等于省略未执行案例。

## 回归与性能声明
新增测试先观察失败：CLI 缺少参数的断言失败，新调用参数不受支持；
修复后 22 项针对性测试通过。完整优化库+预处理库：
1837 passed、1 xfailed、16 warnings，106.20 秒。
HP 非因果滤波为预期失败，限制 OFFLINE_ONLY。
全仓库既有 14 项收集错误未在本轮解决，不宣称全平台通过。
本轮改进为可复用批量入口，无受控性能 A/B，不声明加速倍数。
