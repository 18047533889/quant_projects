"""
Auth and factor-permission gRPC client.

Examples:
    python examples/auth_client.py login --username admin --password change-me

    python examples/auth_client.py me --token "$TOKEN"

    python examples/auth_client.py grant --token "$TOKEN" --user-id <uuid> \
        --definition-id "*" --actions execute_ad_hoc,run_analysis
"""

import argparse
import os
import sys

import grpc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protos"))

import Auth_pb2
import Auth_pb2_grpc


def add_common_args(parser):
    parser.add_argument("--server", default="localhost:50051", help="gRPC server address")
    parser.add_argument("--token", default=os.getenv("LQTP_TOKEN", ""), help="access token")


def parse_actions(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="LQTP 认证客户端")
    sub = parser.add_subparsers(dest="command", required=True)

    login = sub.add_parser("login", help="login and print tokens")
    add_common_args(login)
    login.add_argument("--username", required=True)
    login.add_argument("--password", required=True)

    refresh = sub.add_parser("refresh", help="refresh access token")
    add_common_args(refresh)
    refresh.add_argument("--refresh-token", required=True)

    logout = sub.add_parser("logout", help="revoke refresh token")
    add_common_args(logout)
    logout.add_argument("--refresh-token", required=True)

    me = sub.add_parser("me", help="show current user")
    add_common_args(me)

    create_user = sub.add_parser("create-user", help="创建用户，仅管理员可用")
    add_common_args(create_user)
    create_user.add_argument("--username", required=True)
    create_user.add_argument("--password", required=True)
    create_user.add_argument("--admin", action="store_true")

    grant = sub.add_parser("grant", help="授予因子权限，仅管理员可用")
    add_common_args(grant)
    grant.add_argument("--user-id", required=True)
    grant.add_argument("--definition-id", required=True, help='因子 UUID 或 "*"')
    grant.add_argument("--actions", required=True, type=parse_actions)

    revoke = sub.add_parser("revoke", help="撤销因子权限，仅管理员可用")
    add_common_args(revoke)
    revoke.add_argument("--user-id", required=True)
    revoke.add_argument("--definition-id", required=True, help='因子 UUID 或 "*"')
    revoke.add_argument("--actions", required=True, type=parse_actions)
    return parser.parse_args()


def metadata(token):
    return (("authorization", f"Bearer {token}"),) if token else None


def stub(args):
    channel = grpc.insecure_channel(
        args.server,
        options=[
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ],
    )
    return Auth_pb2_grpc.AuthServiceStub(channel)


def print_user(user):
    print(f"user_id : {user.user_id}")
    print(f"username: {user.username}")
    print(f"is_admin: {user.is_admin}")
    print(f"status  : {user.status}")


def command_login(args):
    response = stub(args).Login(
        Auth_pb2.LoginRequest(username=args.username, password=args.password),
        timeout=60,
    )
    print_user(response.user)
    print(f"access_token      : {response.access_token}")
    print(f"refresh_token     : {response.refresh_token}")
    print(f"expires_in_seconds: {response.expires_in_seconds}")


def command_refresh(args):
    response = stub(args).RefreshToken(
        Auth_pb2.RefreshTokenRequest(refresh_token=args.refresh_token),
        timeout=60,
    )
    print_user(response.user)
    print(f"access_token      : {response.access_token}")
    print(f"refresh_token     : {response.refresh_token}")
    print(f"expires_in_seconds: {response.expires_in_seconds}")


def command_logout(args):
    stub(args).Logout(
        Auth_pb2.LogoutRequest(refresh_token=args.refresh_token),
        timeout=60,
    )
    print("已退出登录")


def command_me(args):
    response = stub(args).GetCurrentUser(
        Auth_pb2.EmptyResponse(),
        metadata=metadata(args.token),
        timeout=60,
    )
    print_user(response)


def command_create_user(args):
    response = stub(args).CreateUser(
        Auth_pb2.CreateUserRequest(
            username=args.username,
            password=args.password,
            is_admin=args.admin,
        ),
        metadata=metadata(args.token),
        timeout=60,
    )
    print_user(response)


def command_permission(args, method_name):
    method = getattr(stub(args), method_name)
    response = method(
        Auth_pb2.GrantFactorPermissionRequest(
            user_id=args.user_id,
            definition_id=args.definition_id,
            actions=args.actions,
        ),
        metadata=metadata(args.token),
        timeout=60,
    )
    print(f"user_id      : {response.user_id}")
    print(f"definition_id: {response.definition_id}")
    print(f"actions      : {', '.join(response.actions)}")


def main():
    args = parse_args()
    try:
        if args.command == "login":
            command_login(args)
        elif args.command == "refresh":
            command_refresh(args)
        elif args.command == "logout":
            command_logout(args)
        elif args.command == "me":
            command_me(args)
        elif args.command == "create-user":
            command_create_user(args)
        elif args.command == "grant":
            command_permission(args, "GrantFactorPermission")
        elif args.command == "revoke":
            command_permission(args, "RevokeFactorPermission")
    except grpc.RpcError as error:
        raise SystemExit(f"gRPC 调用失败: {error.code()} - {error.details()}")


if __name__ == "__main__":
    main()
