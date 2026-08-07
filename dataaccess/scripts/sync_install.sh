#!/usr/bin/env bash
# 把 dataaccess 源码树同步到当前 Python 的 site-packages/data_access/。
#
# 为什么需要：pyproject.toml 把 data_access 包映射到仓库根（package_dir "."），
# 而仓库根目录名是 dataaccess（不是 data_access），所以 `import data_access`
# 在源码目录里无法直接命中，只能靠已安装副本。editable install 需要
# setuptools>=64（本机是 59，不支持 PEP 660），pip install . 又会被 setuptools 59
# 的 UNKNOWN/0.0.0 构建坑到。rsync 是最稳的同步方式，--delete 保证删除陈旧文件。
#
# 用法：bash scripts/sync_install.sh
set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$(python3 -c 'import site; print(site.getusersitepackages())')/data_access"

mkdir -p "$DEST"
rsync -a --delete \
  --exclude='tests' \
  --exclude='build' \
  --exclude='dist' \
  --exclude='*.egg-info' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='workspace_data' \
  --exclude='.git' \
  --exclude='deploy' \
  --exclude='Dockerfile' \
  --exclude='docker-compose.yml' \
  --exclude='.dockerignore' \
  --exclude='pyproject.toml' \
  --exclude='README.md' \
  --exclude='CHANGELOG.md' \
  --exclude='SYNC_REPORT.json' \
  --exclude='UPSTREAM.json' \
  --exclude='scripts' \
  --exclude='docs' \
  "$SRC/" "$DEST/"

# 清理可能残留的错误 dist-info（data_access-0.3.0.dist-info 保留）
rm -f "$(python3 -c 'import site; print(site.getusersitepackages())')/UNKNOWN-0.0.0.dist-info" \
  -rf 2>/dev/null || true

python3 - <<'PY'
import data_access, importlib.metadata as md
try:
    print("data-access", md.version("data-access"))
except Exception as e:
    print("version lookup:", e)
print("pkg at:", data_access.__file__)
PY
