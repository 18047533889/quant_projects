#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 Protobuf / gRPC Python 代码。

用法（在项目根目录执行）：
    python protos/generate_pb2.py

输出文件（覆盖写入）：
    protos/Struct_pb2.py
    protos/Struct_pb2_grpc.py
    protos/Factor_pb2.py
    protos/Factor_pb2_grpc.py
    protos/Auth_pb2.py
    protos/Auth_pb2_grpc.py

依赖：
    pip install grpcio-tools
"""

import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTO_DIR = os.path.join(ROOT, "protos")
PROTOS = ["Struct.proto", "Factor.proto", "Auth.proto", "Portfolio.proto"]

def main():
    for proto in PROTOS:
        proto_path = os.path.join(PROTO_DIR, proto)
        if not os.path.exists(proto_path):
            print(f"[SKIP] 找不到 {proto_path}", file=sys.stderr)
            continue

        cmd = [
            sys.executable, "-m", "grpc_tools.protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={PROTO_DIR}",
            f"--grpc_python_out={PROTO_DIR}",
            proto_path,
        ]
        print(f"[GEN]  {proto}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            sys.exit(result.returncode)

    print("\n生成完成：")
    for f in sorted(os.listdir(PROTO_DIR)):
        if f.endswith("_pb2.py") or f.endswith("_pb2_grpc.py"):
            print(f"  protos/{f}")

if __name__ == "__main__":
    main()
