"""R32-P0-107/108/109/110/111 —— DataAccess 终审报告（2026-08-12 02:xx UTC）

═══════════════════════════════════════════════════════════════════════════════
§1  状态汇总
═══════════════════════════════════════════════════════════════════════════════

R32-P0-107  CanonicalIdentityEncoder 全仓唯一        ✅ PROVEN (测试 + 代码审计)
R32-P0-108  关键 identity ≥128-bit                  ✅ PROVEN (stable_digest_full 64 字符)
R32-P0-109  版本改为 SCM/tag 驱动                    ⚠️  PARTIAL (配置就位，需 clean commit + tag)
R32-P0-110  R30 package 必须进 wheel                ✅ PROVEN (wheel inventory + clean-install smoke)
R32-P0-111  build SHA 在 build-time 固化           ⚠️  PARTIAL (dirty tree → commit_id=None)

本次任务在 **dirty 工作树**（67 uncommitted changes）上完成，setuptools-scm
在 dirty 状态下不写入 commit SHA（``__commit_id__ = None``），需 clean commit
后重新 build 方可证明 R32-P0-111。已配置就位，clean build 即可完整验证。


═══════════════════════════════════════════════════════════════════════════════
§2  实现细节
═══════════════════════════════════════════════════════════════════════════════

2.1  R32-P0-107：CanonicalIdentityEncoder（data_access/r30/_shared.py）
────────────────────────────────────────────────────────────────────────────────
  【修改】stable_digest / stable_digest_full
    - list/tuple **保序**（元素顺序有意义，不排序）
    - set/frozenset 显式 sorted（无序集合规范化）
    - dict 按 key sorted
    - 移除 repr fallback：未知类型 → ValueError（production fail-closed）
    - 支持 str/int/float/bool/None/bytes 基本类型

  【证明】tests/unit/test_r32_identity_version_2026_08.py
    - test_r32_p0_107_list_preserves_order：list 顺序不同 → digest 不同
    - test_r32_p0_107_set_sorts：set 元素相同 → digest 相同
    - test_r32_p0_107_no_repr_fallback：未知类型 → ValueError
    - test_r32_p0_107_basic_types_ok：基本类型 OK

  【wheel 验证】clean install 后 stable_digest(object()) → ValueError


2.2  R32-P0-108：关键 identity ≥128-bit
────────────────────────────────────────────────────────────────────────────────
  【实现】stable_digest_full 返回完整 64 字符 sha256（256-bit）
  【证明】test_r32_p0_108_full_digest_256bit：len(stable_digest_full("test")) == 64
  【wheel 验证】clean install → full_digest_len=64


2.3  R32-P0-109：版本改为 SCM/tag 驱动
────────────────────────────────────────────────────────────────────────────────
  【修改】pyproject.toml
    - version = "0.10.2" → dynamic = ["version"]
    - [build-system] requires += "setuptools-scm>=8.0"
    - [tool.setuptools_scm] 配置：
        version_file = "data_access/_build_info.py"
        tag_regex = "^dataaccess-v(?P<version>[0-9.]+)$"
        fallback_version = "0.11.0.dev0+untagged"

  【修改】data_access/__init__.py
    - 优先从 _build_info 读版本，fallback package metadata，最后 dev 兜底
    - __version__ 不再硬编码

  【修改】data_access/_build_meta.py
    - build_sha() 优先从 _build_info.__commit_id__ 读
    - 保留 env fallback + git fallback（dev 兼容）

  【状态】⚠️ PARTIAL
    - 配置已就位
    - dirty tree → setuptools-scm 不写入 commit SHA（__commit_id__ = None）
    - 需 clean commit + tag dataaccess-v0.11.0 后重新 build
    - 当前 wheel：version=0.11.0.dev0+untagged, build_sha=None

  【证明】test_r32_p0_109_version_scm_driven：__version__ 存在且非空


2.4  R32-P0-110：R30 package 必须进 wheel
────────────────────────────────────────────────────────────────────────────────
  【修改】pyproject.toml [tool.setuptools.packages.find]
    - 已显式包含 "data_access.r30"（用户/并发修改，本次保留）

  【证明】scripts/check_wheel_inventory.py
    - ✅ R26 wheel inventory OK: 含全部 12 个物理子包（包括 r30）

  【wheel 验证】clean install
    - import data_access.r30 → OK
    - unzip -l wheel | rg r30 → 19 个 r30/ 文件


2.5  R32-P0-111：build SHA 在 build-time 固化
────────────────────────────────────────────────────────────────────────────────
  【实现】data_access/_build_meta.py
    - build_sha() 优先从 _build_info.__commit_id__ 读（wheel 场景）
    - fallback env（CI/CD）→ git（dev）→ None（typed，不伪造）

  【状态】⚠️ PARTIAL
    - 代码就位，但 dirty tree → setuptools-scm 写入 __commit_id__ = None
    - clean commit 后重新 build wheel 即可证明

  【证明】test_r32_p0_111_build_sha_from_build_info：
    - __build_sha__ is None or isinstance(__build_sha__, str)（typed）


═══════════════════════════════════════════════════════════════════════════════
§3  测试执行（源码树 editable install）
═══════════════════════════════════════════════════════════════════════════════

pytest tests/unit/test_r32_identity_version_2026_08.py -v

  ✅ test_r32_p0_107_list_preserves_order PASSED
  ✅ test_r32_p0_107_set_sorts PASSED
  ✅ test_r32_p0_107_dict_sorts_keys PASSED
  ✅ test_r32_p0_107_tuple_preserves_order PASSED
  ✅ test_r32_p0_107_no_repr_fallback PASSED
  ✅ test_r32_p0_107_basic_types_ok PASSED
  ✅ test_r32_p0_108_full_digest_256bit PASSED
  ✅ test_r32_p0_109_version_scm_driven PASSED
  ✅ test_r32_p0_111_build_sha_from_build_info PASSED
  ✅ test_r32_p0_110_r30_in_packages PASSED

============================== 10 passed in 0.10s ==============================


═══════════════════════════════════════════════════════════════════════════════
§4  Wheel 构建与清单验证
═══════════════════════════════════════════════════════════════════════════════

构建命令：
  python3 -m build --wheel --outdir dist/

产物：
  dist/data_access-0.11.0.dev0+untagged-py3-none-any.whl (1.6 MB)

清单检查（scripts/check_wheel_inventory.py --wheel-dir dist/）：
  ✅ R26 wheel inventory OK: 含全部 12 个物理子包

Wheel 内容核查：
  unzip -l wheel | rg '(r30|_build_info)'
    ✅ data_access/_build_info.py (556 bytes)
    ✅ data_access/r30/__init__.py + 18 个 r30/ 模块

  _build_info.py 内容（setuptools-scm 生成）：
    __version__ = '0.11.0.dev0+untagged'
    __commit_id__ = None  ← dirty tree，需 clean commit 后重新 build


═══════════════════════════════════════════════════════════════════════════════
§5  Clean-Install 验证（/tmp/da_r32_clean_install + venv）
═══════════════════════════════════════════════════════════════════════════════

安装：
  python3 -m venv venv
  pip install /home/shw/quant_projects/dataaccess/dist/data_access-*.whl

Smoke 测试：
  import data_access
  ✅ data_access.__version__ = '0.11.0.dev0+untagged'
  ⚠️  data_access.__build_sha__ = None (dirty tree，需 clean commit)

  import data_access.r30
  ✅ r30 import OK

  from data_access.r30._shared import stable_digest, stable_digest_full
  ✅ stable_digest("test") = '039ff483cac47618'
  ✅ len(stable_digest_full("test")) = 64

  stable_digest(object())
  ✅ ValueError: stable_digest: unsupported type object. Production 必须显式序列化


═══════════════════════════════════════════════════════════════════════════════
§6  未证明项与后续动作
═══════════════════════════════════════════════════════════════════════════════

6.1  当前工作树状态
────────────────────────────────────────────────────────────────────────────────
  git status --porcelain | wc -l → 67
  【原因】并发多轮 R30/R31 改动 + 本次 R32 修改

  【影响】setuptools-scm 在 dirty 状态下不写入 commit SHA：
    - _build_info.__commit_id__ = None
    - wheel 安装后 data_access.__build_sha__ = None

6.2  完整验证 R32-P0-109/111 的后续步骤
────────────────────────────────────────────────────────────────────────────────
  1. 审查本次 R32 修改（pyproject.toml, r30/_shared.py, __init__.py, 
     _build_meta.py, 新增测试文件）
  2. git add + commit（clean tree）
  3. git tag dataaccess-v0.11.0
  4. python3 -m build --wheel --outdir dist/
  5. 验证 _build_info.py：
       __version__ = '0.11.0'
       __commit_id__ = '<40 字符 SHA>'
  6. clean-install 验证：
       data_access.__build_sha__ = '<SHA>'  （非 None）

6.3  NOT PROVEN 项（需 clean commit + tag）
────────────────────────────────────────────────────────────────────────────────
  ⚠️  R32-P0-109：版本号从 SCM tag 生成（配置就位，需 tag 触发）
  ⚠️  R32-P0-111：build SHA 固化进 wheel（dirty tree → commit_id=None）

  【当前可证明】
    - setuptools-scm 配置正确（构建成功，生成 _build_info.py）
    - fallback_version 生效（0.11.0.dev0+untagged）
    - _build_meta.build_sha() 优先读 _build_info（代码就位）
    - 运行时不依赖 .git（wheel 安装后无 .git 可访问）

  【需 clean build 证明】
    - tag commit → __version__ = '0.11.0'（无 +untagged）
    - clean tree → __commit_id__ = '<SHA>'（非 None）
    - wheel 安装后 __build_sha__ 返回构建期 SHA（非运行时 git）


═══════════════════════════════════════════════════════════════════════════════
§7  文件修改清单（本次 R32 任务）
═══════════════════════════════════════════════════════════════════════════════

Modified:
  pyproject.toml
    - [project] version = "0.10.2" → dynamic = ["version"]
    - [build-system] requires += "setuptools-scm>=8.0"
    - [tool.setuptools_scm] 配置（version_file, tag_regex, fallback_version）

  data_access/r30/_shared.py
    - stable_digest：list/tuple 保序，移除 repr fallback
    - stable_digest_full：同上逻辑，返回 64 字符

  data_access/__init__.py
    - __version__ 改为从 _build_info / package metadata 读
    - 移除 full_version() 调用

  data_access/_build_meta.py
    - build_sha() 优先从 _build_info.__commit_id__ 读
    - 保留 env/git fallback（dev 兼容）

Created:
  tests/unit/test_r32_identity_version_2026_08.py
    - 10 个测试覆盖 R32-P0-107/108/109/110/111


═══════════════════════════════════════════════════════════════════════════════
§8  签署
═══════════════════════════════════════════════════════════════════════════════

本报告诚实记录以下事实：

✅ R32-P0-107 已完整实现并证明（测试通过 + wheel smoke）
✅ R32-P0-108 已完整实现并证明（64 字符 digest）
✅ R32-P0-110 已完整实现并证明（wheel inventory + clean-install）

⚠️  R32-P0-109/111 配置已就位，但因 dirty tree 无法在当前 build 中证明 commit
   SHA 固化。需 clean commit + tag 后重新 build 方可完整验证。

未证明项已明确标注，不伪造任何状态。

────────────────────────────────────────────────────────────────────────────────
终审时间：2026-08-12 02:30 UTC
工作树状态：dirty (67 uncommitted changes)
Wheel 产物：dist/data_access-0.11.0.dev0+untagged-py3-none-any.whl
测试结果：10/10 passed
Clean-install：smoke OK (version + r30 + identity)
════════════════════════════════════════════════════════════════════════════════
"""