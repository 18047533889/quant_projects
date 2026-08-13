# 01 — Master Architecture

## 1.1 端到端目标

```text
Manual Research / Auto Miners / LLM / External Corpus
                         |
                         v
                  +---------------+
                  | FactorEngine  |
                  | validate/calc |
                  +-------+-------+
                          | FactorBatch
                          v
                  +---------------+
                  |QuantEvaluator |
                  |Evidence/Diag  |
                  +------+--------+
                         |
          +--------------+----------------+
          |                               |
          v                               v
 +----------------+              +----------------+
 |FactorOptimizer |--- child --->| FactorEngine   |
 |mutate/search   |              +-------+--------+
 +----------------+                      |
          ^                              v
          +---------------------- QuantEvaluator
                                         |
                                         v
                                +----------------+
                                | FactorAssets   |
                                |govern/assemble |
                                +-------+--------+
                                        | FactorSet
                                        v
                                +----------------+
                                |FactorPreprocess|
                                |representation  |
                                +-------+--------+
                                        |
                                  ModelInputBundle
                                        |
                                      Modeling
```

`FactorOptimizer` 是 feedback loop，不是所有因子的必经步骤。

## 1.2 Runtime Dependencies

目标不是形成硬依赖链，而是形成协议链：

- QE core 不强依赖 DA/FE；接受 Arrow/NumPy/Polars/Pandas 数据。
- QE 提供 optional adapters 对接 DA/FE。
- FO core 依赖 `EvaluatorProtocol`、`FactorExecutorProtocol` 抽象；官方 adapter 对接 QE/FE。
- FA core 可以只消费 `EvaluationBundle`/`EvidenceRef`；需要 raw value 的 exact similarity/novelty 通过 provider/adapter 请求 QE。
- FP core 消费 `FactorSet + values + exposure context`，不反向依赖 FA internal。

禁止循环依赖。

## 1.3 Schema Strategy

根 `/schemas` 保存语言中立、版本化的 schema；每个 package 构建时复制/生成自己需要的 contract types。

不要建立新的 `quant_contracts` runtime package，避免多一个所有人都必须安装的依赖。

所有跨库 contract 必须包含：

- `schema_version`
- `producer`
- `created_at`
- `run_id`
- `factor_ids`
- 时间/市场/频率/股票池元数据（适用时）
- parent/reference IDs
- config/code hash（适用时）

但禁止把大规模 factor values 嵌进 JSON metadata；values 用 Arrow/NumPy/Polars 或外部 ref。

## 1.4 Time Semantics

不要新建 PIT Engine，但任何时间相关对象必须明确：

- `decision_time`
- `signal_available_time`
- `execution_time` / `label_start_time`
- `label_end_time`
- fitted transform 的 `fit_start_time`, `fit_end_time`

规则：

- DA 负责“数据在当时能不能读到”。
- FE 负责“公式是否 causal / production legal”。
- QE 只验证 caller 提供的 evaluation timing 是否一致，不自行进行数据库 PIT join。
- FP 负责所有 learned/fitted preprocessing 不跨越训练边界。

## 1.5 No Universal Common Package

禁止建立 `common/`, `shared/`, `core/`, `helpers/` 作为跨包杂物堆。

如果跨包确实共享：

1. 先判断是否属于 schema contract。
2. 如果只是一个库需要，放库内 `_utils`。
3. 如果是 Data/Factor semantics，应下沉 DA/FE。
4. 如果只是两个库都“碰巧需要”，优先重复极少量纯 protocol/type，而不是建立巨大公共依赖。

## 1.6 Human/AI/Public Interfaces

每个 package 最终至少提供：

- Python public API
- CLI/doctor（仅诊断，不造平台）
- README + API examples
- package-local tests
- benchmark entrypoint
- `package_info()`/version capability 查询（可选）

AI 调用应优先通过稳定的结构化 API，而不是读取内部数据库或 private module。
