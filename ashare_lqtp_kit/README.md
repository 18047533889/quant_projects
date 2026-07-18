# A 股 LQTP 平台客户端工具包

可独立外发的最小可用包：登录、RunFactor、TopK OPEN 回测、连通性探活。

## 官方入口

| 项 | 内容 |
|----|------|
| 因子服务（gRPC） | `110.42.223.26:50051`（文档里的 `http://...` 实际是 **insecure gRPC**，不是浏览器网页） |
| 用户手册 | https://my.feishu.cn/docx/T2DLdqcT4oTZSexCtSucWdYmn9f |
| Python gRPC 调用文档 | https://my.feishu.cn/docx/X9Uxd9hpfoaVzHxpLX6cuGi0nD0 |

```bash
export LQTP_SERVER=110.42.223.26:50051
export LQTP_USERNAME=your@email.com
export LQTP_PASSWORD=your_password
```

## 目录

```text
ashare_lqtp_kit/
├── README.md / requirements.txt / .env.example
├── protos/                 # gRPC protobuf（必带）
├── ashare_lqtp/            # Python 包（client + dsl_compat）
├── tools/
│   ├── probe.py            # Login + RunFactor + Backtest
│   ├── wait_and_probe.py   # 服务挂了可轮询等待
│   ├── demo_run_factor.py
│   └── demo_topk_backtest.py
└── examples/               # 官方示例（均已加 256MB message limit）
```

## 安装

```bash
cd ashare_lqtp_kit
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 快速验证

```bash
python3 tools/probe.py
# 或服务不稳定时：
python3 tools/wait_and_probe.py --interval 30
```

## 重要约定

| 项 | 说明 |
|----|------|
| 默认地址 | `110.42.223.26:50051` |
| 通道选项 | send/recv **256MB**（`GRPC_CHANNEL_OPTIONS` / `make_channel`） |
| TopK | 本地构权重 → 平台 OPEN 回测 |
| 凭证 | 只用环境变量，代码里不写死账号密码 |

## 版本

- kit `0.1.1`

## Week3 reports

Restored summary pack under `reports/week3/` (open `index.html`). Metrics recovered from prior run; chart images not included.

## Reports

- `reports/week3/` — Week3 20-factor presentation pack (restored metrics)
- `reports/all/` — full catalog summary (`production_summary.html`); 117/176 factors have metrics recovered from chat logs; charts not included
