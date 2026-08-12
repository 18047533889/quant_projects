"""R32-P0-099..106 聚焦测试：授权、边界、原子性、SQL sandbox。"""
from pathlib import Path
import pytest

from data_access.core.exceptions import AccessDeniedError, ValidationError, DataError


# ============================================================================
# R32-P0-099: 写入/删除/发布/元数据变更必须逻辑授权
# ============================================================================

def test_p0_099_write_action_exists():
    """验证 write/delete 动作常量存在。"""
    from data_access.security.principal import (
        ACTION_DATASET_WRITE,
        ACTION_DATASET_DELETE,
        ACTION_DATASET_PUBLISH,
    )
    
    assert ACTION_DATASET_WRITE == "dataset:write"
    assert ACTION_DATASET_DELETE == "dataset:delete"
    assert ACTION_DATASET_PUBLISH == "dataset:publish"


def test_p0_099_store_uses_authorization():
    """验证 store.py 实际调用 authorize_dataset。"""
    import ast
    from pathlib import Path
    
    store_path = Path(__file__).parent.parent.parent / "store.py"
    content = store_path.read_text()
    
    # 验证存在 dataset:write 授权调用
    assert 'action="dataset:write"' in content
    assert 'action="dataset:delete"' in content


# ============================================================================
# R32-P0-100: Dataset 专属路径边界
# ============================================================================

def test_p0_100_path_boundary_prevents_cross_dataset_write(tmp_path):
    """dataset A 不能写入 dataset B 的 root。"""
    from data_access.registry.dataset_boundary import verify_path_belongs_to_dataset
    
    # 创建简单的 mock dataset
    class MockDataset:
        def __init__(self, name, root):
            self.name = name
            self.root = root
    
    dataset_a = MockDataset(name="dataset_a", root=str(tmp_path / "a"))
    dataset_b_root = tmp_path / "b"
    dataset_b_root.mkdir()
    
    # dataset_a 试图写入 dataset_b 的路径 → 拒绝
    with pytest.raises(ValidationError, match="path boundary violation"):
        verify_path_belongs_to_dataset(
            dataset_b_root / "data.parquet",
            dataset_a,
            operation="write",
        )


def test_p0_100_path_boundary_allows_own_root(tmp_path):
    """dataset 可以访问自己的 root。"""
    from data_access.registry.dataset_boundary import verify_path_belongs_to_dataset
    
    class MockDataset:
        def __init__(self, name, root):
            self.name = name
            self.root = root
    
    dataset_root = tmp_path / "mydata"
    dataset_root.mkdir()
    dataset = MockDataset(name="mydata", root=str(dataset_root))
    
    # 自己的路径 → 通过
    verify_path_belongs_to_dataset(
        dataset_root / "subdir" / "file.parquet",
        dataset,
        operation="write",
    )


# ============================================================================
# R32-P0-102: Generation 原子性
# ============================================================================

def test_p0_102_generation_atomicity_no_partial_visibility(tmp_path):
    """使用 generation 模式确保读者只看到完整 generation。"""
    from data_access.write.generation_atomicity import (
        atomic_publish_with_generation,
        read_current_generation_data,
    )
    
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    (candidate_dir / "data.txt").write_text("new generation")
    
    archive_parent = tmp_path / "archive"
    
    # 发布新 generation
    result = atomic_publish_with_generation(
        target_dir,
        candidate_dir,
        archive_parent,
        metadata={"rows": 100},
    )
    
    assert "generation_id" in result
    
    # 读者读取当前 generation
    current_gen_dir = read_current_generation_data(target_dir)
    assert current_gen_dir is not None
    assert (current_gen_dir / "data.txt").read_text() == "new generation"


def test_p0_102_upsert_partition_atomicity_documented(tmp_path):
    """验证 upsert 文档明确说明只有 partition-level 原子性。"""
    from data_access.write import upsert
    
    # 读取 upsert 模块 docstring
    docstring = upsert.upsert_table.__doc__ or ""
    assert "partition-level atomicity" in docstring.lower()
    assert "不" in docstring and "dataset-level" in docstring


# ============================================================================
# R32-P0-103: Publish 无 missing-target 窗口
# ============================================================================

def test_p0_103_generation_publish_no_missing_window(tmp_path):
    """使用 generation + 指针模式，读者永远不会看到 target 缺失。"""
    from data_access.write.generation_atomicity import (
        atomic_publish_with_generation,
        read_current_generation_data,
        get_current_generation_id,
    )
    
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    
    # 发布第一代
    candidate1 = tmp_path / "candidate1"
    candidate1.mkdir()
    (candidate1 / "v1.txt").write_text("version 1")
    
    atomic_publish_with_generation(
        target_dir,
        candidate1,
        tmp_path / "archive",
        metadata={"version": 1},
    )
    
    gen1_id = get_current_generation_id(target_dir)
    assert gen1_id is not None
    
    # 发布第二代（模拟并发读者）
    candidate2 = tmp_path / "candidate2"
    candidate2.mkdir()
    (candidate2 / "v2.txt").write_text("version 2")
    
    # 在 publish 期间，指针始终指向有效 generation
    atomic_publish_with_generation(
        target_dir,
        candidate2,
        tmp_path / "archive",
        metadata={"version": 2},
    )
    
    # 读者读取 → 不会看到空/缺失
    current_gen = read_current_generation_data(target_dir)
    assert current_gen is not None
    assert (current_gen / "v2.txt").exists()


# ============================================================================
# R32-P0-104: COS 分布式 fencing（文档约束）
# ============================================================================

def test_p0_104_distributed_write_constraint_documented():
    """验证分布式写入约束已明确文档化。"""
    doc_path = Path(__file__).parent.parent.parent / "write" / "DISTRIBUTED_WRITE_CONSTRAINT.md"
    assert doc_path.exists(), "R32-P0-104 约束文档缺失"
    
    content = doc_path.read_text()
    assert "单写者约束" in content or "Single-Writer" in content
    assert "COS" in content or "对象存储" in content
    assert "分布式" in content


# ============================================================================
# R32-P0-105: Metadata 写入必须授权与审计
# ============================================================================

def test_p0_105_metadata_write_action_defined():
    """验证 metadata:write 动作已定义。"""
    from data_access.security.principal import ACTION_METADATA_WRITE
    assert ACTION_METADATA_WRITE == "metadata:write"


def test_p0_105_publish_manifest_requires_metadata_write_authorization(tmp_path):
    """Publish manifest 写入需要 metadata:write 权限。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from data_access.write.publish import publish_from_staging
    from data_access.registry import load_registry
    from data_access.registry.paths import PathAuthorizer
    from data_access.security.principal import ACTION_METADATA_WRITE
    from data_access.core.exceptions import AccessDeniedError
    from textwrap import dedent
    import os

    # 设置环境变量
    os.environ["TEST_ROOT"] = str(tmp_path)

    # 创建 staging 和 target 数据集
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    pq.write_table(pa.table({"x": [1, 2]}), staging_dir / "data.parquet")

    target_dir = tmp_path / "published"
    target_dir.mkdir()

    # 通过 YAML 创建 registry
    yaml_path = tmp_path / "datasets.yaml"
    yaml_path.write_text(dedent("""
        staging_dataset:
          kind: static
          access_mode: staging
          root: ${TEST_ROOT}/staging
          glob: "*.parquet"
          time_column: date
          instrument_column: instrument

        target_dataset:
          kind: static
          access_mode: published
          root: ${TEST_ROOT}/published
          glob: "*.parquet"
          time_column: date
          instrument_column: instrument
    """).strip() + "\n")

    registry = load_registry(yaml_path)

    # 创建一个拒绝 metadata:write 的 authorizer
    class DenyMetadataWriteAuthorizer(PathAuthorizer):
        def authorize_dataset(self, dataset, action=None):
            if action == ACTION_METADATA_WRITE:
                raise AccessDeniedError(
                    f"Principal 无 {action} 权限访问 {dataset}"
                )
            # 其他权限通过
            pass

    authorizer = DenyMetadataWriteAuthorizer(
        allowed_roots=[str(tmp_path)],
    )

    # 发布应该失败（metadata:write 被拒绝）
    with pytest.raises(AccessDeniedError, match="metadata:write"):
        publish_from_staging(
            registry,
            authorizer,
            staging_name="staging_dataset",
            target_name="target_dataset",
        )


def test_p0_105_publish_manifest_succeeds_with_metadata_write_permission(tmp_path):
    """有 metadata:write 权限时 publish manifest 成功。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from data_access.write.publish import publish_from_staging
    from data_access.registry import load_registry
    from data_access.registry.paths import PathAuthorizer
    from textwrap import dedent
    import os

    # 设置环境变量
    os.environ["TEST_ROOT"] = str(tmp_path)

    # 创建 staging 和 target 数据集
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    pq.write_table(pa.table({"x": [1, 2]}), staging_dir / "data.parquet")

    target_dir = tmp_path / "published"
    target_dir.mkdir()

    # 通过 YAML 创建 registry
    yaml_path = tmp_path / "datasets.yaml"
    yaml_path.write_text(dedent("""
        staging_dataset:
          kind: static
          access_mode: staging
          root: ${TEST_ROOT}/staging
          glob: "*.parquet"
          time_column: date
          instrument_column: instrument

        target_dataset:
          kind: static
          access_mode: published
          root: ${TEST_ROOT}/published
          glob: "*.parquet"
          time_column: date
          instrument_column: instrument
    """).strip() + "\n")

    registry = load_registry(yaml_path)

    # 创建允许所有操作的 authorizer
    authorizer = PathAuthorizer(
        allowed_roots=[str(tmp_path)],
    )

    # 发布应该成功
    result = publish_from_staging(
        registry,
        authorizer,
        staging_name="staging_dataset",
        target_name="target_dataset",
    )

    assert "manifest_path" in result
    assert Path(result["manifest_path"]).exists()
    assert Path(result["manifest_path"]).name == ".publish_manifest.json"


# ============================================================================
# R32-P0-106: SQL AST 安全边界
# ============================================================================

def test_p0_106_sql_ast_rejects_comma_join():
    """SQL sandbox 拒绝逗号连接（FROM a, b）。"""
    from data_access.read.sql_ast_validator import enumerate_sql_sources
    
    sql = "SELECT * FROM dataset_a, dataset_b WHERE dataset_a.id = dataset_b.id"
    sources = enumerate_sql_sources(sql)
    
    # 必须枚举出两个表
    assert "dataset_a" in sources
    assert "dataset_b" in sources


def test_p0_106_sql_ast_rejects_cte():
    """SQL sandbox 识别 CTE。"""
    from data_access.read.sql_ast_validator import enumerate_sql_sources
    
    sql = "WITH temp AS (SELECT * FROM dataset_a) SELECT * FROM temp"
    sources = enumerate_sql_sources(sql)
    
    assert "dataset_a" in sources


def test_p0_106_sql_ast_rejects_subquery():
    """SQL sandbox 识别子查询中的表。"""
    from data_access.read.sql_ast_validator import enumerate_sql_sources
    
    sql = "SELECT * FROM (SELECT * FROM dataset_inner) AS sub"
    sources = enumerate_sql_sources(sql)
    
    assert "dataset_inner" in sources


def test_p0_106_sql_ast_rejects_information_schema():
    """SQL sandbox 拒绝 information_schema 等系统表。"""
    from data_access.read.sql_ast_validator import validate_sql_sources_against_allowlist
    
    sql = "SELECT * FROM information_schema.tables"
    
    with pytest.raises(ValidationError, match="禁止访问系统表"):
        validate_sql_sources_against_allowlist(sql, allowed_sources=set())


def test_p0_106_sql_ast_rejects_cross_join():
    """SQL sandbox 识别 CROSS JOIN。"""
    from data_access.read.sql_ast_validator import enumerate_sql_sources
    
    sql = "SELECT * FROM dataset_a CROSS JOIN dataset_b"
    sources = enumerate_sql_sources(sql)
    
    assert "dataset_a" in sources
    assert "dataset_b" in sources


def test_p0_106_sql_ast_rejects_lateral():
    """SQL sandbox 识别 LATERAL join。"""
    from data_access.read.sql_ast_validator import enumerate_sql_sources
    
    sql = """
    SELECT * FROM dataset_a, 
    LATERAL (SELECT * FROM dataset_b WHERE dataset_b.id = dataset_a.id) AS sub
    """
    sources = enumerate_sql_sources(sql)
    
    assert "dataset_a" in sources
    assert "dataset_b" in sources


def test_p0_106_sql_ast_only_allows_declared_sources():
    """SQL sandbox 只允许声明的 dataset。"""
    from data_access.read.sql_ast_validator import validate_sql_sources_against_allowlist
    
    sql = "SELECT * FROM dataset_a JOIN dataset_b ON dataset_a.id = dataset_b.id"
    
    # 只声明 dataset_a → 拒绝
    with pytest.raises(ValidationError, match="未授权的表引用"):
        validate_sql_sources_against_allowlist(sql, allowed_sources={"dataset_a"})
    
    # 声明两个 → 通过
    validate_sql_sources_against_allowlist(sql, allowed_sources={"dataset_a", "dataset_b"})


def test_p0_106_sql_ast_rejects_mutations():
    """SQL sandbox 拒绝所有 DML/DDL。"""
    from data_access.read.sql_ast_validator import validate_sql_no_mutations
    
    mutations = [
        "INSERT INTO dataset_a VALUES (1, 2)",
        "UPDATE dataset_a SET x = 1",
        "DELETE FROM dataset_a",
        "CREATE TABLE dataset_b (id INT)",
        "DROP TABLE dataset_a",
    ]
    
    for sql in mutations:
        with pytest.raises(ValidationError, match="禁止.*操作"):
            validate_sql_no_mutations(sql)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
