# 版本标量身份边界（2026-09-13 02:47 heartbeat）

server-c main原树，剩776 GiB；FE/DA另一任务active，已发送平台合同文件范围协调，不触其文件。
保留已有改动，未commit/push/建分支/复制/改写生产记录。

FeatureSetVersion原接受list/dict/int/bool作为身份、版本、consumer/label/data引用，冻结DTO仍可保留可变对象。
现按已有str/optional str合同校验，合法空值保持兼容，不更改哈希公式或历史记录。
新增21项：version_scalar_before为20 failed / 1 passed。
version_scalar_final为58 passed，但FE测试文件在运行中改变，标记SOURCE_CHANGED_DURING_RUN，不当作稳定PASS。
version_scalar_recheck：58 passed，源码before=after，无并发变化。
复测源码digest：a963394a486b768c5478d12386b0334ff5d89ad95bad4a8ff5ea3f9b4c9f8583。
复测日志SHA256：c20479aee832c4961a2f797af8788dd35ac940c3f624fbbe3e65235a9eb89624。

最终平台version_scalar_platform_full：679 passed / 1 skipped / 3 warnings，源码before=after与复测一致。
平台日志SHA256：6913053f572e555bf1757673df019c906f4c508a1cc5c6dfffb4170837d41db8。
排除live PostgreSQL文件并移除QP_PG_DSN；1项disposable fencing测试跳过，不作为通过或生产认证。
git diff --check（quant_platform与本轮证据目录）通过。

旧错误类型会被拒绝，不自动迁移。哈希v2迁移与其他待查项不因此关单。
