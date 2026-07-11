# LQTP Python gRPC Examples

This package contains runnable Python clients for the LQTP backtest service.

## Layout

Keep `examples/` and `protos/` as sibling directories. The example clients import generated protobuf modules from `../protos`.

```text
lqtp-python-grpc-examples/
  examples/
  protos/
  requirements.txt
  README.md
```

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

Start the Rust service first:

```powershell
.\target\release\backtest.exe examples\config.yaml
```

Login:

```powershell
python examples\auth_client.py login --username admin --password change-me-before-first-start
$env:LQTP_TOKEN = "<access_token>"
```

Run a factor:

```powershell
python examples\factor_client.py run --token $env:LQTP_TOKEN --formula "close / delay(close, 1) - 1" --begin 20240102 --end 20240329 --warmup 1 --analyze
```

Run a backtest:

```powershell
python examples\backtest_client.py --token $env:LQTP_TOKEN --begin 20240102 --end 20240329 --return-orders --return-positions
```

Create a paper portfolio:

```powershell
python examples\portfolio_client.py create --token $env:LQTP_TOKEN --name demo
```

## Regenerate protobuf files

The generated `*_pb2.py` and `*_pb2_grpc.py` files are included. To regenerate them:

```powershell
python -m pip install grpcio-tools
python protos\generate_pb2.py
```
