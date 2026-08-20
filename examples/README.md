# `examples` — 示例与配置样例（详尽说明)

本目录提供 **可运行的 Python 示例** 与 **YAML 配置片段**，演示 **`FactorEngine`**、**多因子**、**Joblib**、**回测** 等用法。  
运行前请在 **`factor_engine` 仓库根** 设置 **`PYTHONPATH=.`**，或 **`pip install -e .`**；**数据路径** 需改为你本机路径。

### 协作者速览（约 5 分钟）

1. **本目录在干什么**：**可复制运行的脚本** + 根目录 **示例 YAML**，覆盖 **最小因子**、**配置驱动**、**多因子/Joblib**、**真实数据冒烟**、**单多资产回测**。
2. **从哪下手**：想最快跑通 → **`simple_factor.py`**；想 **YAML** → **`config_driven_factor.yaml`** + **`FactorEngine.run_from_config`**；本地 COS 镜像 → **`config_local_stock_daily_bar_smoke.yaml`**；回测 → **`backtest_*.py`**。
3. **数据集模板**：子目录 **`configs/`**（[`configs/README.md`](configs/README.md)）按 **11+ 数据源** 分文件，改 **`root`** 即可试。
4. **常见坑**：忘记 **`PYTHONPATH=.`**；路径仍指向作者机器。

---

## 1. Python 脚本逐项说明

| 文件 | 演示内容 | 关键依赖 |
|------|----------|----------|
| [`simple_factor.py`](simple_factor.py) | 最小 **`Factor` + `FactorEngine.run`** | pandas、本地/内存数据 |
| [`pandas_factor.py`](pandas_factor.py) | Pandas 后端 + **`KlineParquetSource` 类数据源** | 需有效 `root` |
| [`multi_factor_dag.py`](multi_factor_dag.py) | 多因子、**共享子式**（DAG） | 同 pandas 后端 |
| [`run_factors_joblib.py`](run_factors_joblib.py) | **`run_many_parallel`**、Chunk 配置 | **`factor-engine[parallel]`** |
| [`run_real_data_factor_smoke.py`](run_real_data_factor_smoke.py) | 真实 parquet 冒烟 | 路径、`RUN_*` 环境变量 |
| [`backtest_single_asset.py`](backtest_single_asset.py) | **`run_single_asset_backtest`**、`target_position` | **`factor-engine[backtest]`** |
| [`backtest_multi_asset.py`](backtest_multi_asset.py) | **`run_multi_asset_backtest`**、`target_weights` | pandas |

---

## 2. 根目录 YAML

| 文件 | 用途 |
|------|------|
| [`config_driven_factor.yaml`](config_driven_factor.yaml) | **`FactorEngine.run_from_config`** 完整示例（K 线类） |
| [`notebook_config_smoke.yaml`](notebook_config_smoke.yaml) | Notebook/CI 烟测，字段最小化 |

---

## 3. 子目录 [`configs/`](configs/README.md)

- **11+ 个数据集** 的 **YAML 模板**（日 K、分钟、报价、成交、多张基本面表）。  
- 与根 [`README.md`](../README.md)「数据集列表」一一对应。

---

## 4. 典型命令

```bash
cd /path/to/factor_engine
PYTHONPATH=. python examples/simple_factor.py
PYTHONPATH=. python -c "from runtime.engine import FactorEngine; from runtime.config import load_config; ..."
# 单资产回测示例在 monorepo：backtest_layer/examples/backtest_single_asset.py（见 backtest_layer/single_asset_backtest/README.md 配置 PYTHONPATH）
```

配置驱动：

```bash
PYTHONPATH=. python -c "
from runtime.engine import FactorEngine
FactorEngine.run_from_config('examples/config_driven_factor.yaml')
"
```

（需 YAML 内 **`data_source.root`** 等指向真实路径。）

---

## 5. 与回测子系统的关系

- **回测** 实现在 monorepo **`backtest_layer/single_asset_backtest/`**（包名 **`single_asset_backtest`**）；示例在 **`../../../backtest_layer/examples/`**。  
- 深读见 [`../../../backtest_layer/single_asset_backtest/README.md`](../../../backtest_layer/single_asset_backtest/README.md)。

---

## 6. 延伸阅读

- [`runtime/README.md`](../runtime/README.md)  
- [`storage/README.md`](../storage/README.md)  
- [`docs/operators_semantics.md`](../docs/operators_semantics.md)  
