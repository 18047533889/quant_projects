# rolling_ic：只计算请求的滚动均值

公共指标 `rolling_ic` 原先调用完整 `compute_rolling_ic_stats`，随后只取其中的均值。现在由独立模块 `metrics/rolling_ic_mean.py` 计算，避免未请求的方差、IR及其他统计。完整统计函数保持不变。

## 公式与契约

对因子 f、交易序列位置 t，取从 max(0,t-window+1) 到 t 的尾随窗口，记其有限 IC 的位置集合为 S。

\[
\mu_{t,f}=\frac{1}{|S|}\sum_{i\in S} IC_{i,f},\qquad |S|\geq\mathrm{min\_periods}.
\]

不足最少观测数时为 NaN；NaN、正负无穷不计入 S。常量 IC 的均值仍有效，不因方差为零而丢失。默认 window=60、min_periods=20，输出 float64 (T,F)，一维输入作为单因子输出 (T,1)。复用原累计求和均值内核，不更改公式、窗口、缺失值口径或完整统计 API。

## 可复查验证

- 新测试文件：`tests/test_rolling_ic_mean_only_oct04.py`。
- 同命令先失败：1 failed / 46 passed，具体因仅请求均值仍触发5次标准差计算。
- 改后同命令：47 passed / 0.09s；包含独立手算结果、常量、NaN/Inf、短历史、空时间轴、非连续视图、不同窗口和最少观测数，以及一维输入。
- 加上 `tests/metrics/test_catalog_bindings.py`、`tests/test_ic_summary_calendar_index.py`：123 passed / 0.58s。

## 轻量 A/B：不是端到端认证

CPU 合成 IC 序列 T2586×F48，seed20261004，12常量列、36随机列，混入 NaN/Inf；window60/min_periods20，线程限制1。三组相反顺序配对，每次均值输出与完整统计逐位一致。

完整统计秒数：0.343183874967508、0.34319778496865183、0.34920770302414894。
均值独立路径秒数：0.003466535999905318、0.003195354016497731、0.004437686991877854。
中位数约343.20ms与3.47ms。这里没有 COS 读取、排名、完整评估入口或 GPU，不能将该差异当作真实整链加速倍数。

该改动改变可执行源码指纹；先前固定版本 F48 性能证据只能作为历史记录，不能为修改后的源码自动授权 CUDA。新源码仍需重新测量、独立对照和当前资格接入。
