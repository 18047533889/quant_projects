# 有界 COS 研究元数据读取

`read_declared_cos_object` 除 Parquet 外，支持注册声明的 JSON / JSONL /
NDJSON。必须显式 `allow_research=True`；生产/严格语义不允许使用此研究入口。
原注册数据集的授权、字段投影和读取预算仍然生效。

JSON 扫描上限 2 MiB，结果上限 8 MiB，最多 10,000 行；调用方只能收紧预算。
流程是精确对象 HEAD → GET → HEAD，对象 ETag/大小改变即拒绝；单段 MD5
ETag 与实际内容校验，同时返回 SHA256。临时文件在成功和失败时均清理。
这不等于证明上游数据正确或满足历史时点可得性。

元数据按内容哈希分目录时，可显式使用
`cos_cli_ls(prefix, recursive=True)` 发现文件。默认仍不递归。
目录本身不是文件；不递归返回空文件列表，不能据此断言远端不存在元数据。
该列表是 best-effort；网关错误也可能返回空列表。
仅枚举元数据，不下载整批文件。

## 因子配方绑定

存储说明 `report_storage.json` 不是因子谱系。
`landing_manifest.json` 中的 `factors` 可以含
`uri`、`sha256`、`verified`、`fe_dsl`、`status`。
使用配方前必须与实际因子对象 URI 和内容 SHA256 一一匹配；
不能仅凭因子名字或“最新”时间绑定。禁止执行元数据里的 Python 源代码。

2026-09-22 的真实读取已核验以下 manifest 内容摘要与父目录哈希一致：
`0135a2f864f7b00d3d95a8055e760e4190f45462c1e1d068cdb9c8607e576046`。
其中 `weekly_1f8db9c3aace0c93` 声明的因子值摘要为
`e4938ee90b2ec0ccbc8b996c610ded191e638c72c96b43facf81de43cd3b014f`。
本记录只证明 manifest 已读取；尚未核验该因子值本体，因此不将它自动标成已知谱系。
其 DSL 含条件分支和内部排名，不能把语法上出现 rank 解释为最终输出已统一排名。
对象 HEAD 大小约 106,493,379 字节，超过 64 MiB 研究读取上限；
本次没有放宽限制或下载该对象。
