---
name: quant-walk-forward
description: 时序交叉验证（walk-forward）的唯一正确方式 — 必须用 modeling.split（date_bounded_split / split_by_date_cutoff / assert_date_authoritative）+ leakage_guard（TrainOnlyFitGuard / assert_frozen_preprocessing）+ PURGE_TRADING_DAYS 防泄漏。禁止 sklearn 默认 shuffle split 做时间序列。当任务涉及"训练/测试切分、walk-forward、滚动预测、OOS、样本外、purge、embargo、交叉验证"时先读此 skill。
version: 1.0.0
---

# Walk-Forward 交叉验证（modeling.split + leakage_guard）

**前提**：已读 quant-ml-modeling（modeling/ 库总览）+ debug-pit-leakage（泄漏排查）。

## 为什么必须用 modeling.split（不是 sklearn）

- **日期权威**：`assert_date_authoritative` 强制切分以日期为准（不是行索引、不是 asset 顺序）。行顺序在 reindex/去重后不可靠。
- **10 日前瞻 label** → 训练/验证边界必须 purge：label 区间跨越 cut 的样本会把未来信息带进 train 段（`PURGE_TRADING_DAYS=10`，见 `factor_selection.py`）。
- **选择泄漏**：因子选择也必须逐折在 train 段内做（`factor_selection.py --fold`），**禁止全样本选特征再 walk-forward 训练**（P0-B 已修，别回退）。
- 预处理/标准化 **fit 只能 train 段**：`TrainOnlyFitGuard` / `assert_frozen_preprocessing`。

## 标准切分代码（真实 API）

```python
import sys; sys.path.insert(0, '/home/sunhaiwei/quant_projects')
from modeling.split import date_bounded_split, split_by_date_cutoff, assert_date_authoritative
from modeling.leakage_guard import TrainOnlyFitGuard, assert_frozen_preprocessing

# 1. 日期权威校验（硬门，先跑）
assert_date_authoritative(X, date_col="datetime")   # X 必须含日期轴

# 2. 单 cut 切分（参考 lightgbm_qs/scripts/factor_selection.py）
train, test = split_by_date_cutoff(
    X, y, cutoff="2019-04-01",          # 切点
    purge_days=10,                       # PURGE_TRADING_DAYS：前瞻收益边界
    embargo_days=0,                      # 需要时可设（防相邻窗口信息重叠）
)

# 3. 多折 walk-forward（date_bounded_split）
folds = date_bounded_split(
    X, n_splits=5,
    min_train_days=504,                  # 至少 2 年训练
    gap_days=0,
    purge_days=10,
)
for fold in folds:
    fit_preprocessing(fold.train_dates, spec)   # 只 fit train 段 → FrozenPreprocessing
    train_model(fold)
```

## 训练-预测循环（不重造）

```python
# 每折：
#   fit_preprocessing(train_ds, spec) → FrozenPreprocessing（train-only fit，guard 校验）
#   model.fit(X_train_frozen, y_train)
#   y_pred_oos = model.predict(X_test_frozen)   # OOS 用冻结的预处理，禁止重 fit
#   y_pred_oos 累积 → 拼 OOS 预测矩阵 → 喂 riskfolio_qs/portfolio_and_backtest.py
```

## 现成管线（别重造，直接跑）

```bash
cd /home/sunhaiwei/quant_projects
# 逐折因子选择（防选择泄漏的正确模式）
.venv/bin/python lightgbm_qs/scripts/factor_selection.py --fold 2019-04-01
# 全特征构建 + 去重（corr>0.98 簇内留 rank_ic 最高）
.venv/bin/python lightgbm_qs/scripts/build_full_features.py
# LightGBM GPU 训练（自带 walk-forward）
.venv/bin/python lightgbm_qs/scripts/train_lightgbm_gpu.py
# 最终训练
.venv/bin/python lightgbm_qs/scripts/train_final.py
```

## 折叠评估规范

- 每折 OOS 段算 `rank_ic`/`ic_ir`（用 quant_evaluator，不走手写 corr）。
- 报告必须含：每折 IS/OOS 对比、折间波动（稳定性）、全部 OOS 合并的因子衰减。
- 若某折 OOS 显著差于 IS → 查泄漏（debug-pit-leakage），不是直接调参掩盖。
- 证据 YAML：git_sha、折叠定义、PURGE 参数、每个 NOT_RUN 项如实写。

## 禁止

- ❌ `train_test_split` / sklearn `TimeSeriesSplit` 代替 modeling.split（日期权威唯一真源）
- ❌ 全样本选特征喂 walk-forward（选择泄漏）
- ❌ 预处理/标准化在全样本 fit 后用于 test（必须 Frozen + guard）
- ❌ 不 purge 就切分（10 日前瞻 label 的样本跨边界）
- ❌ OOS 段重算预处理（标准化用 train 段参数）
