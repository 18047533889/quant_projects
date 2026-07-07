# 量化因子引擎 Agent 框架

研报 PDF → AI 生成 YAML 配置 → 评价脚本评分 → 不达标则 AI 修订 YAML 迭代，达标则以当前 YAML 为最终版本。

## 目标流程

```
研报 PDF → AI → YAML v1 → 配置文件评价脚本.py → 评分
                ↑                    │
                └── 不达标 ←──────────┘
                     AI → YAML v2 … → YAML vN
                达标 → 结束，yaml vN 为最终版本
```

## 六层架构与本项目对应

| 层 | 职责 | 本项目对应 |
|----|------|-------------|
| **1** | CLAUDE.md / rules / memory — 长期上下文，「是什么」 | `config/`、`docs/CLAUDE.md`、`memory/` |
| **2** | Tools / MCP — 动作能力，「能做什么」 | `tools/` |
| **3** | Skills — 按需加载的方法论，「怎么做」 | `skills/` |
| **4** | Hooks — 强制执行，不依赖模型判断 | `hooks/` |
| **5** | Subagents — 隔离上下文的工作者 | `planning/sub_agents/` |
| **6** | Verifiers — 验证闭环，可验 / 可回滚 / 可审计 | `verifiers/`、`scripts/配置文件评价脚本.py` |

## 目录结构

```
factor_agent/
├── README.md                 # 本说明
├── docs/
│   └── CLAUDE.md             # 第1层：系统约定、YAML schema 说明、达标标准
├── config/                   # 第1层：配置与规则
│   ├── settings.py
│   ├── yaml_schema.py        # YAML 结构约定
│   ├── message_schema.py
│   └── logger.py
├── memory/                   # 第1层：长期上下文与压缩
│   ├── context_compression.py
│   ├── store.py
│   └── policies.py
├── tools/                    # 第2层：Tools / MCP
│   ├── registry.py
│   ├── pdf_tools.py
│   ├── yaml_tools.py
│   └── eval_tools.py         # 调用评价脚本
├── skills/                   # 第3层：Skills
│   ├── base.py
│   ├── registry.py
│   ├── report_to_yaml_skill.py
│   └── score_feedback_revise_skill.py
├── hooks/                    # 第4层：Hooks
│   ├── base.py
│   ├── pre_yaml_hooks.py
│   └── post_pass_hooks.py
├── planning/                 # 第5层：任务与子 Agent
│   ├── todo.py
│   ├── task_system.py
│   ├── task_schema.py
│   └── sub_agents/
│       ├── base.py
│       ├── extract_agent.py
│       └── revise_agent.py
├── verifiers/                # 第6层：Verifiers
│   ├── evaluator.py
│   └── audit.py
├── core/                     # Agent 循环与状态
│   ├── agent_loop.py
│   ├── state.py
│   └── autonomous.py
├── scripts/
│   └── 配置文件评价脚本.py   # 评分入口
└── main.py                   # 主入口
```

## 使用方式

### 1. 直接运行主入口

```bash
python factor_layer/factor_agent/main.py "请根据当前上下文生成首版配置文件 YAML，并写入 output/config_v1.yaml，然后调用 run_eval 评分。"
```

如果不传 query，`main.py` 会使用内置默认 query。

### 2. 选择模型提供方

当前 `main.py` 支持：

- `--provider anthropic`
- `--provider minimax`

也可以通过环境变量控制：

- `LLM_PROVIDER`
- `LLM_MODEL`
- `MINIMAX_API_KEY`
- `MINIMAX_MODEL`

### 3. 输出位置

从当前 `config/settings.py` 看，运行期默认使用：

- `output/`：存放 YAML 版本和评分结果
- `scripts/配置文件评价脚本.py`：评分入口

## 依赖与运行

- Python 3.x
- 需要安装 `anthropic`
- 使用 Anthropic 时需要 `ANTHROPIC_API_KEY`
- 使用 Minimax 时需要 `MINIMAX_API_KEY`

当前默认阈值与循环参数见：

- `config/settings.py`

其中包括：

- `MAX_ROUNDS = 10`
- `SCORE_THRESHOLD = 0.7`
- `OUTPUT_DIR = output/`

## 适合什么时候看这个目录

- 你想把研报内容自动翻译成 factor_engine 可消费的 YAML
- 你想了解“生成 YAML → 调评价脚本 → 根据分数修订”的闭环
- 你在排查评分脚本、tool、skill 或 verifier 的协作关系
