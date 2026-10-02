# ANN 余弦计算的数值边界

`FaissANNIndex` 使用 `IndexFlatIP`，对存储的 float32 单位向量做精确平面搜索。
其 `capability` 是 `EXACT_FLAT`。`AnnoyANNIndex` 使用 angular 索引，
`capability` 是 `APPROXIMATE`；你不能把候选召回结果当作全库精确相似度图。
两者都接收因子证据嵌入，不直接接收股票历史收益。

## 归一化顺序与公式

你传入的向量必须为实数、有限、非零。复数、字符串、NaN 和无穷值会触发
`ValueError`。建库时还需保证因子 ID 唯一，嵌入行数与 ID 数一致，维数符合配置。
查询向量需满足同样的数值规则及维数检查，`k` 必须为正整数。

实现先按行最大幅度缩放，再求单位向量，最后转换成后端的 float32：

$$s_i=\max_j|x_{ij}|,\qquad
v_{ij}=x_{ij}/s_i,\qquad
u_{ij}=v_{ij}/\sqrt{\sum_jv_{ij}^{2}}.$$

这与 $x_i/\lVert x_i\rVert_2$ 的数学方向一致。先缩放避免平方和溢出，
也避免极小非零向量的平方和下溢为零。直接将 $10^{300}$ 转成 float32
会得到无穷值，因此维护者不能把转换移到归一化之前。实现不修改调用者数组。

FAISS 的单位向量内积和距离定义为：

$$c=u_q^\top u_i,\qquad d=1-c.$$

Annoy 返回 angular 距离，统一余弦分数定义为：

$$c=1-d_{\mathrm{angular}}^2/2.$$

float32 舍入可能让 $c$ 越过 $[-1,1]$ 的端点。实现仅允许
$16\epsilon_{32}$ 的数值越界，再将分数截到 $[-1,1]$；非有限分数或更大的
越界会触发 `ValueError`。这项容差处理不改变相似度阈值的含义。

## 验证范围

`tests/similarity/test_ann_extreme_vectors.py` 比较 $10^{300}$、$10^{-300}$
及 float64 最小正次正规数的归一化方向，并用实际安装的 FAISS/Annoy 检查
同向、正交倾向和反向向量的分数。测试同时覆盖非法输入及端点舍入。
这些测试验证数值合同，没有证明 Annoy 全量召回率或真实全库吞吐率。

2026-10-02，维护者在 server-c 的 `.venv/bin/python` 下运行 ANN 两个测试模块，
77 项通过；FactorAssets 全量回归 1,638 项通过、97 条警告，116.42 秒。
这次全量回归使用正式主路径的当前工作树，包含其他 AI 尚未提交的 FE 改动。
早期回归曾因 FE 正在修改的事件算子初始化失败；核对修复后的工作树后重跑通过。
