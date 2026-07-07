# INTERFACE

当前模块已经具备模块级 pipeline，但还没有完成服务层。

说明：
- 已实现的入口与产物写在对应章节。
- 没有做的功能在对应章节直接留白。

## 1. Module Responsibility

负责消费 `factor_evaluation` 已生成的评估结果，产出准入决策，并将 run / summary / decision 写入 catalog。

不负责：
- 因子计算
- 评估指标生成
- 异步任务调度

## 2. Public Service Entries

- `pipeline.run`
- `pipeline.run_pipeline`
- `pipeline.run_from_config`
- `pipeline.run_config_directory`
- `run_pipeline.py config`
- `run_pipeline.py config-dir`
- `run_from_config.py`

## 3. Request / Response Contract

### `pipeline.run_from_config(config_path)`

Request:
- `config_path`
- `output_root` optional

Response:
- `summary`
- `results`
- `output_root`
- `run_summary_path`
- `config_snapshot`

### `pipeline.run_config_directory(config_dir)`

Request:
- `config_dir`
- `pattern` optional
- `output_root` optional
- `stop_on_error` optional

Response:
- `summary`
- `results`
- `output_root`
- `run_summary_path`
- `config_snapshots`

## 4. Backing Python APIs

- `config.load_config`
- `admission.admit_evaluation_run`
- `catalog.AdmissionCatalog`

## 5. Artifacts

- `<output_root>/run_summary.json`
- `<output_root>/config_snapshot.yaml`
- `<output_root>/config_snapshots/*.yaml`
- `<output_root>/results/*.json`
- `<evaluation_root>/<factor_id>/<run_id>/admission_decision.json`
- `<factor_lake_root>/_catalog.sqlite`

## 6. Manifest Contract

## 7. Events

## 8. Error Contract

- `CONFIG_INVALID`
- `EVALUATION_RUN_NOT_FOUND`
- `SUMMARY_NOT_FOUND`
- `FACTOR_ID_MISMATCH`
- `RUN_ID_MISMATCH`
- `PRIMARY_HORIZON_NOT_FOUND`
- `FACTOR_NOT_REGISTERED`

## 9. Upstream / Downstream

Upstream:
- `factor_evaluation` 运行目录
- `summary.json`
- factor lake catalog

Downstream:
- `admission_decision.json`
- `factor_admission_status`
- `factor_admission_decisions`
- 策略层与审计查询

## 10. Forbidden Direct Calls

## 11. Unresolved Issues
