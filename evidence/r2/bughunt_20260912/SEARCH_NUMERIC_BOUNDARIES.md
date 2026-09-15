# 搜索配置数值边界（2026-09-13 11:53 heartbeat）

server-c正式main原树，776 GiB可用，FE/DA任务idle，保留已有修改。
未commit/push/建分支/复制/改写生产记录。

SearchConfig的plateau_threshold原仅比较负值，接受NaN/Inf；plateau_window与max_concurrency接受bool及1.0等非int。
现显式要求窗口/并发为int且排除bool，阈值为有限int/float且非负。合法默认和零阈值、窗口1不变。
新增16项，search_numeric_before为14 failed / 2 passed，保留前置证据。
search_numeric_final：搜索目录765 passed / 12 warnings，无skip，源码before=after。
源码digest：8784a1971afc1ea5f4d668204199328268aa0645efeb6d292bf575ac7337bc8f。
日志SHA256：69e9aca1d15f471eb94808843c72abd6ce0517031174990e6150009ee237033a。
git diff --check（factor_optimizer与本轮证据）通过。

旧错误类型配置会明确拒绝，不自动转换。运行期并发隔离、公共hash迁移等仍开放，不作为生产认证。
