# Label 工程实现

## 核心流程

```
load_factor_meta(source_factor_dir)
+ load_evaluation_summary(evaluation_summary_path)
    → generate_tags()
        → collect_rule_tags()    确定性规则标签
        → DeepSeek.request_semantic_tags()  语义标签
        → normalize_semantic_label()  标准化
    → route_factor()
        → decide_admission()     准入决策
        → build_route_record()   路由记录
        → build_lifecycle_event()  生命周期事件
    → [可选] materialize_routed_factor_package()  物化入库
    → persist_label_state()  持久化标签状态
```

## 关键设计

- **DeepSeek 真实客户端**：`deepseek_client.py` 使用真实 API，支持重试和退避，无 mock 路径
- **规则标签确定性**：`tagging.py` 中的规则标签基于配置的策略文件
- **materialize=False 模式**：pipeline 编排时只算路由信息，不执行实际入库
- **日志分层**：按 label/indicator/timeseries/pipeline 分文件
