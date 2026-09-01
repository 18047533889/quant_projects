# lightgbm_qs — LightGBM 因子训练与回测管线（量化研究）

A 股因子研究主链路：**因子矩阵 → LightGBM 训练（GPU/CPU）→ vwap-to-vwap 标签 →
组合优化 → vectorbt 回测 → 报告**。本仓库是研究脚本集（非库），运行于
`quant_projects` monorepo 内，直接复用各平台库。

**Repo:** https://github.com/HKUST-QUANT-SOCIETY/lightgbm_qs (private)

> ⚠️ 本仓库**只同步代码/文档/测试**（`scripts/`、`tests/`、`MISTAKES_AND_LEAKAGE_LESSONS.md`）。
> `data/`（222G parquet 因子/面板）与 `outputs/`、`snapshots/`（回测图/快照）为
> 本地再生产物，**不进入 Git**。

---

## 与平台库的关系（谁提供什么）

| 本仓库脚本使用 | 来自哪个库 | 用途 |
|---|---|---|
| `factor_preprocess.transforms.cross_sectional` | [factor_preprocess](../factor_preprocess/README.md) | `cs_winsor` / `cs_rank` / `cs_zscore` 截面预处理 |
| `vectorbt_qs.mvp.data.adapter.set_data_root` | [vectorbt_qs](../vectorbt_qs/README.md) | 回测数据根配置 |
| `vectorbt_qs.mvp.engine.runner.run_backtest / portfolio_report` | [vectorbt_qs](../vectorbt_qs/README.md) | accurate 回测与绩效报告 |
| `vectorbt_qs.mvp.visualization.build_nav_curves` | [vectorbt_qs](../vectorbt_qs/README.md) | 净值曲线 |
| `import lightgbm as lgb` | 第三方 | GPU/CPU 训练 |
| 因子矩阵 / 行情 | [data_access](../data_access/README.md) 与本地 `data/panel` | 特征与标签 |

## 口径（硬性，改前必读 `MISTAKES_AND_LEAKAGE_LESSONS.md`）

- **标签一律 vwap-to-vwap 后复权**：`label[t] = vwap_adj[t+10]/vwap_adj[t] - 1`
  （`build_adj_label.py` 重建，修正未复权 vwap 跳空问题）。
- 训练窗口 2016 起，**滚动 3 个月重训**，预测下一个 3 个月（walk-forward）。
- 特征：全部入选因子，NaN 保留（LightGBM 原生处理）。
- GPU：`device='gpu'`、`max_bin=63`。机器 32 核 92G：并行 ≤31 workers。

## 主要脚本

- `build_adj_label.py` — 重建后复权 10 日前向标签（vwap-to-vwap）
- `build_ohlcv_adj.py` / `build_ohlcv_adj_from_table.py` — 后复权 OHLCV
- `build_features*.py` / `preprocess_features*.py` — 82 特征矩阵构建与预处理
- `build_dataset.py` / `build_factor_matrix.py` — 因子矩阵 / 数据集
- `train_lightgbm_gpu.py` / `train_lgb_cpu.py` / `train_parallel.py` — 训练入口
- `walkforward` 相关 — 滚动切分清单
- `merge_*.py` — 分块因子/面板合并
- `backtest_final_vectorbt.py` — 组合 → vectorbt 回测
- `portfolio_and_backtest.py` — 仓位生成 + 回测
- `audit_adj_consistency.py` — 复权口径审计
- `recompute_lqtp_adj*.py` — LQTP 后复权重算
- `run_*.sh` — 全链路（528full / 818full / final）一键脚本

## 测试

```bash
cd /home/sunhaiwei/quant_projects
PYTHONPATH=. pytest lightgbm_qs/tests/ -q     # test_purge / test_walkforward_leak
```

## 布局

```
scripts/          # 全部研究脚本（build/train/backtest/audit）
tests/            # 泄漏/切分回归测试
data/             # 本地 parquet（222G，不入 git）
outputs/          # 回测图/报告（不入 git）
snapshots/        # 训练快照（不入 git）
MISTAKES_AND_LEAKAGE_LESSONS.md   # ⭐防再犯手册（动回测/训练前必读）
```

## 相关仓库（完整平台）

见 `quant_projects` 根目录 `HANDOVER.md` 交接文档 —— 12+ 个库的依赖与接口全景。
