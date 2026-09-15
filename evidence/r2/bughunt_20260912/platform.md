# 平台第二轮 bughunt

正式目录 /home/sunhaiwei/quant_projects，main。无分支/复制/commit/push/生产动作；基线 HEAD e94ac507d670fd1c16b1d6a63fc5d6286daa5970。

## 修复

1. reconcile_candidates 消耗一次性迭代器后再求 fingerprint，错误得到空批指纹：入口将有界传入批固化为 tuple。
2. 同批同内容不同语义、或同语义不同内容未检查，可能执行冲突记录并覆盖 durable discovery：逐记录分类，并对本批歧义双方均拒绝。完全相同的记录保留第一条 NEW、其余 DUPLICATE；不再用 content_hash 字典把一条的分类覆盖到另一条。
3. batch_fingerprint 只包含 spec hash，使换过语义声明的记录被当作成功重放：merkle-v2 绑定两个生产者携带的引用。无公式/语义计算权威转移。
4. durable consumed seed 以当前输入伪装历史 known entry，会掩盖已恢复的真实语义冲突：只有确无历史完整身份时才使用旧 hash-only 兼容分支。
5. raw 中合法 datetime 被 json.dumps 拒绝：持久化已规范化 carrier，时间为 ISO 字符串，避免无关原始 Python 对象打断批次。
6. malformed candidate/ref/list 字段未规范拒绝：加逐记录 string 和 string-sequence 验证；坏记录与正常邻居隔离。
7. AdmissionRequest.context 与 AdmissionVerdict.detail 内部仍可变：使用已有递归冻结器，断开外部嵌套 alias。
8. OPS08 shadow fixture 原来把 implementation name 当 semantic ID：按 FP 注册表修正为 FILL:forward，stage=missingness，implementation=forward_fill；未放宽 FP 新门。

新反例集中于 quant_platform/tests/test_bughunt_intake_identity.py。旧 observability fixture 原来在批内制造歧义却期待一条 duplicate，已改为独立 known conflict 以保持原收集器计数测试目的，并另有明确歧义拒绝反例。

## 测试

- platform_before：9 failed（修复前真实反例）。
- platform_after：115 passed / 1 failed（旧测试固定 merkle-v1 前缀；已按新版本更新）。
- platform_full：496 passed / 2 failed / 1 skipped；分别是旧歧义 fixture 和 FP 新语义门抓到 OPS08 错声明；均已修。
- platform_contract_final：138 passed，源码稳定 PASS。
- platform_regression：499 passed / 1 skipped；pytest exit0，但期间新增其他 FE 测试文件，严格保留 SOURCE_CHANGED_DURING_RUN。
- live PostgreSQL 测试文件明确排除；另一个 PostgreSQL fencing 例因无 disposable DSN 跳过。

JSON/log 保留源哈希、运行期变更及实际结果，未重写失败历史。

## 兼容与未闭合项

merkle-v1 旧记录未删除；新批 token 使用 v2，历史 consumption 仍参加去重。缺少原始 discovery 的老 hash-only 记录不能凭空补回旧语义，须单独审计/重评。

跨包 FittedState 新 identity 和 QE artifact kind/hash 的迁移见 fp.md/qe.md；没有改写历史资产。

另确认更广泛公共 _contenthash 编码碰撞。这个文件本轮尚未修改，不能标修复完成；具体复现及必须保留的 v1/v2 迁移边界见 OPEN_FINDINGS.md。这不是 merkle-v2 新引入的问题：新批指纹只输入固定类型的独立 SHA 字符串，结构碰撞存在于公共 codec 的嵌套值/不同类型场景。
