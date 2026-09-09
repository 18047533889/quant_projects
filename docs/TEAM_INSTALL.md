# 组员安装与版本核对

推荐从 `18047533889/quant_projects` 总仓获取代码。先核对本次发布回执 `evidence/team_sync_20260909/push_both_result.json`；它记录总仓源码提交及 13 个 HKUST 镜像的提交和内容树。包自身的 `0.1.0` 等版本号不足以证明源码相同。

## 核心 8 库

范围：DataAccess、FactorEngine、QuantEvaluator、FactorPreprocess、FactorOptimizer、FactorAssets、QuantPlatform、Modeling。目标环境为 **CPython 3.12.3、Linux x86_64**。以下命令都在总仓根目录执行，使用新的组员环境；不要直接覆盖 server-c 正在使用的环境。

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install pip==26.2.1 setuptools==84.0.0 wheel==0.48.0
python -m pip install --no-build-isolation \
  --config-settings editable_mode=compat -r requirements.txt
python scripts/check_team_environment.py
```

`requirements.txt` 是统一入口；`requirements/core-editable.txt` 把全部内部库一起从当前源码安装，避免误取公共索引中的同名包。`requirements/core-py312-linux.txt` 精确约束核心外部传递依赖。相对源码路径以调用目录为准，因此不要在其他目录直接运行上述命令。

锁使用已测核心环境版本作为基线，并通过真实联合依赖解析验证。**依赖解析成功不等于新的完整安装环境已经跑过此前 5,191 项回归。** 本轮没有替换 server-c 的现有 Python 环境，也没有为所有操作系统、GPU 或生产配置背书。锁固定版本但不包含全部发行文件哈希，不是供应链签名证明。

可先执行不安装的解析检查：

```sh
python -m pip install --dry-run --ignore-installed --no-build-isolation \
  --config-settings editable_mode=compat -r requirements.txt
```

环境检查器会核对完整核心外部版本、内部 8 库版本及 editable 来源目录、`pip check`、工作区是否干净，以及当前已提交的 13 个子目录内容树是否符合发布回执。回执本身入库后会有元数据提交，因此比较子目录树，而不是错误地要求当前 HEAD 必须等于回执的源码提交。该检查离线读取上次发布证据，不声称重新查询了 GitHub 实时状态。

仅检查源码同步、不检查依赖时：

```sh
python scripts/check_team_environment.py --source-only
```

## 其他环境分开安装

- **Riskfolio 与 VectorBT：** 阅读各库 README。Riskfolio/QS 兼容档要求 NumPy < 2，而 vendored VectorBT 的新版 metadata 有另一组要求；不能把互斥档装入核心环境，也不能用 `--no-deps` 绕过后就声称兼容。各库 requirements 是声明/入口，不等于新的统一精确锁。
- **AlphaProbe：** 使用独立 CPython 3.11/Linux 环境，并遵循嵌套目录 `alphaprobe/ashare/alphaprobe-dev` 的 PDM 配置与原锁。其 Torch/TorchVision/TorchAudio 指定 CUDA 12.1 的 cp311 发行文件，不适用于核心 3.12 环境或任意 CUDA/Mac 平台。PDM 安装应使用冻结锁模式并核对本机驱动条件。
- **LightGBM：** 是研究脚本集合；`lightgbm_qs/requirements.txt` 列出外部依赖，内部集成源码及 CPU/GPU 系统前提见其 README。不要假设一个通用 LightGBM wheel 自动具备 GPU 能力。
- **前端：** `platform_web` 使用 Node，不用 Python requirements。已在 Node 22.23.2 下验证本轮锁；从该目录运行 `npm ci --ignore-scripts`，随后 `npm test -- --maxWorkers=2 --minWorkers=1` 和 `npm run build`。以提交的 `package-lock.json` 固定依赖，不使用任意更新的 `npm install` 替代团队复现。
- 核心的 GPU、报告、数据库服务及测试 extras 是可选功能，不包含在最小核心安装完成声明中；增加 extras 前须对照核心约束重新解析、验证。

## 旧文件与上传

根目录和 FactorEngine 的历史 `requirements-production.lock` 不作为本次组员安装入口：其中存在解释器伪依赖、旧版本/缺失值或哈希模式说明问题。保留它们供历史证据核查，不建议直接执行其中的安装示例。

`push_both.sh` 是上传工具，不负责自动提交未保存代码或安装依赖。发布前先 `bash push_both.sh --dry-run`，审查 13 库差异，确认后再正式执行。详见 `docs/TEAM_SYNC_20260909.md`。
