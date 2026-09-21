# 研究专用分层衰减内核

接口：factor_preprocess.transforms.layered_decay.layered_decay。
输入是同形状的日期×资产 values 与整数 layers，以及 1–20 个 [1,60] 内
有限半衰期。必须明确 allow_research=True；该方法不加入生产变换注册表。
预处理库只做数值变换，不读取收益或自行拟合尺度。优化库负责 TRAIN
诊断、QE 分箱、冻结二十维参数、验证确认和成本后选优。

对每一资产维护每层独立的分子 N 与质量 D，初始为零：

$$\alpha_q=1-2^{-1/h_q},\quad \rho_q=1-\alpha_q,$$
$$N_{t,q}=\rho_qN_{t-1,q}+\alpha_qx_{t-1}\mathbf1[b_{t-1}=q],$$
$$D_{t,q}=\rho_qD_{t-1,q}+\alpha_q\mathbf1[b_{t-1}=q],$$
$$z_t=\frac{\sum_qN_{t,q}}{\sum_qD_{t,q}}.$$

每条历史观测保留进入时的层归属；今天换层不会重新归类历史。
权重含进入层的 alpha，输出是调整并归一化后的滤波，不等于 adjust=False
普通 EWMA。只读取前一行信号和层号；首行缺失。前一行非有限或者层号 -1
会清空该资产全部状态，输出缺失，避免旧状态在缺口后重新出现。

半衰期单位是面板行。调用者须提供完整交易日轴，不得删掉全市场缺日后
仍将其称为自然日或交易日衰减。分层只能用当时已知信号，不能用未来标签。
数值实现按资产幅度缩放分子，防止有限极值的累加溢出，额外递归状态为
O(资产数×层数)。没有引入持仓限制、借券、冲击成本或生产准入保证。

训练提案、跨层迁移和验证规则的完整说明位于优化库
[LAYER_DECAY.md](https://github.com/HKUST-QUANT-SOCIETY/factor_optimizer/blob/main/docs/LAYER_DECAY.md)。
