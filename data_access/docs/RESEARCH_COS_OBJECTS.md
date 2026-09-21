# COS 平铺因子对象的有界研究读取

入口：`data_access.cos.research.read_declared_cos_object`。
用途：跨账号 COS 只能通过服务器能力网关访问时，读取已登记数据集中的一个
明确 parquet 对象，返回 Arrow 表和远端来源记录。不是新的生产发布接口。

## 使用约束

- 必须传入已有 DataAccessStore 和其已登记的数据集名，参数经登记表验证；
  storage 必须声明 COS 根，解析结果必须是单个 parquet 对象，不接受通配扫描。
- 显式 `allow_research=True`；production/strict 上下文拒绝。
- 在任何网络访问前检查数据集读取授权、路径参数及读取预算；
  不读取或解析 COS 凭据，不修改网关权限。
- 优先使用部署配置的 `DATA_ACCESS_COS_CLI`。server-c 的 admin-cos
  网关允许 /home/{user} 与 /srv/quant 下的本地下载路径，不允许 /tmp；
  因此研究命令应配置
  `DATA_ACCESS_COS_CACHE_ROOT=/home/sunhaiwei/.cache/quant-dataaccess/research`。
- 先取单对象元数据，再下载，再复查元数据。对象 key、ETag 和大小不一致拒绝；
  单部分 MD5 ETag 还会校验实际文件摘要。所有结果都带实际内容 SHA-256。
- 网关的人类可读大小可能四舍五入，预估上界采用
  `floor(displayed_bytes * 1.01) + 1024`。这是保守准入预算，不是网络传输硬限速；
  下载后再次检查实际文件大小。调用者预算可以收紧默认 64 MiB 文件上限。
- 返回数据默认最多 2,000,000 行、128 MiB；请求总时限至多 60 秒。
  远端请求预算至少为 3（读取元数据、GET、复查元数据）。
- DataAccess 使用原数据集契约执行列、时间、股票过滤，返回前完全物化 Arrow；
  成功、网关失败、文件损坏和查询失败都会清理本次创建的私有临时目录。
  不复制整个池，不留下持久行情或因子副本。

返回 ResearchObjectRead 包含 table、source_uri、source_etag、content_sha256、
downloaded_bytes、dataset。元数据重查与内容摘要不等于上游因子 PIT 认证；
该研究接口不能被包装成生产 sealed-test 或发布凭据。

可执行真实例子及环境设置见 factor_optimizer 的
`examples/cos_batch_audit.py` 和 `docs/COS_DIAGNOSTICS_20260922.md`。
