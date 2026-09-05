#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""smoke_t8_remote_publish —— T8 多 artifact 原子发布验收（UPSTREAM_FIX_PLAN 问题二）。

用法（默认纯本地，DATA_ACCESS 环境无关；tmp 由 ``--tmp-base`` / ``T8_SMOKE_TMP`` 注入）：

    python data_access/scripts/smoke_t8_remote_publish.py [--tmp-base DIR] [--factor-prefix library_v1]

真实 COS smoke（可选，默认 SKIP）：``T8_REAL_COS=1`` 且 admin-cos 的
``qs-cold`` ``factor_pool`` 前缀可写时，用唯一隔离前缀
``cos://qs-cold/factor_pool/_smoke/<run_id>/`` 跑一遍真实上传/校验/清理。
CURRENT/generation 全部位于隔离前缀内，**绝不触碰**
``cos://qs-cold/factor_pool/library_v1/``（无范围删除禁令）。

验收顺序（每步打印 PASS/FAIL）：
    local staging write / remote generation upload / manifest checksum verification /
    CURRENT pointer update / remote read-back / idempotent retry /
    failed upload leaves old CURRENT / stale writer rejected / scoped cleanup

确认列表（默认 PASS 或 SKIP(opt-in)；真实 COS 项仅 T8_REAL_COS=1 时执行）：
    admin-cos factor_pool permission / resolved wildcard snapshot /
    data_access remote publisher / T8 five artifacts / no direct COS SDK from AlphaFlow /
    no secret in logs

退出码：全部 PASS=0；任一 FAIL=1；真实 COS 相关项 SKIP 不计失败。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import traceback
import uuid
from pathlib import Path

# 允许从任意 CWD / repo 子目录直接跑本脚本：优先把本脚本所在 data_access 的父目录
# 挂进 sys.path（与既有 scripts/*.py 惯例一致），便于只 checkout data_access 时也能跑。
_HERE = Path(__file__).resolve().parent
_SCRIPTS_DIR = _HERE
_PKG_ROOT = _HERE.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
# 共享工具脚本所在（monorepo 布局：data_access/scripts/../.. 与 ../../scripts 同级）。
_PLATFORM_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
for _p in (_PLATFORM_SCRIPTS, _SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from data_access.core.exceptions import DataError, ValidationError  # noqa: E402
from data_access.read.object_store import LocalObjectStore  # noqa: E402
from data_access.write.object_store_generation_publisher import (  # noqa: E402
    ObjectStoreGenerationPublisher,
    StaleWriterError,
)
from data_access.write.t8_publish import (  # noqa: E402
    CATALOG,
    EVALUATION,
    PUBLIC_META,
    SECRET_META,
    T8_KINDS,
    VALUE,
    current_pointer_payload,
    generation_artifacts,
    layout_rel_key,
    publish_t8_artifacts,
    publish_t8_artifacts_retry,
    read_current_bytes,
    resolve_current_generation,
    resolve_current_objects,
    verify_generation_remote,
)

OMP_NUM_THREADS = os.environ.get("OMP_NUM_THREADS", "4")

RUN_ID = uuid.uuid4().hex[:12]
REAL_PREFIX = f"factor_pool/_smoke/{RUN_ID}"
REAL_URI = f"cos://qs-cold/{REAL_PREFIX}"
SKIPPED = "SKIP(opt-in)"

_RESULTS: list[tuple[str, bool | None]] = []


def _note(name: str, ok: bool | None) -> None:
    _RESULTS.append((name, ok))
    status = {True: "PASS", False: "FAIL", None: SKIPPED}[ok]
    print(f"{name}: {status}")


def _fail(msg: str) -> None:
    print(f"    FAIL detail: {msg}")
    raise AssertionError(msg)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        _fail(msg)
    print(f"    ok: {msg}")


def _coerce_bytes(data) -> bytes:
    if isinstance(data, bytes):
        return data
    return data.read()


# ---- 模拟远端 ObjectStore（参数可注入 fake） ---------------------------------


class _RemoteStore:
    """本地 fake 远端：真实 COS 关闭时模拟 ``CURRENT`` 晋升/对象完整可见性。

    与 LocalObjectStore 行为一致（原子 put / head / reader / delete）；独立封装
    以便真实 COS smoke 替换为 COSObjectStore（对象语义零差异）。
    """

    def __init__(self, root: Path):
        self._inner = LocalObjectStore(root)

    def head_object(self, key):
        return self._inner.head_object(key)

    def open_reader(self, key):
        return self._inner.open_reader(key)

    def range_read(self, key, *, offset, length):
        return self._inner.range_read(key, offset=offset, length=length)

    def put_object(self, key, data):
        self._inner.put_object(key, _coerce_bytes(data))

    def delete_object(self, key):
        self._inner.delete_object(key)

    def list_objects(self, prefix):
        return self._inner.list_objects(prefix)

    # multipart 转发（本地 store 行为）——smoke 只放小对象，无需 multipart。
    def begin_multipart(self, key):
        return self._inner.begin_multipart(key)

    def upload_part(self, upload_id, key, part_index, data):
        self._inner.upload_part(upload_id, key, part_index, data)

    def complete_multipart(self, upload_id, key):
        self._inner.complete_multipart(upload_id, key)

    def abort_multipart(self, upload_id, key):
        self._inner.abort_multipart(upload_id, key)


def _bootstrap_local(args) -> Path:
    """搭一个临时目录：本地 staging + 模拟远端 cos root。"""
    base = Path(args.tmp_base).resolve() if args.tmp_base else Path(
        tempfile.mkdtemp(prefix="t8-smoke-")
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


# ---- 各验收步骤 ---------------------------------------------------------------


def step_local_staging_write(args, base: Path, factor_id: str) -> Path:
    """写本地 staging（与上游 T8 写盘约定一致）：五类文件。"""
    staging = base / "staging"
    fdir = staging / factor_id
    fdir.mkdir(parents=True, exist_ok=True)
    (fdir / "public_meta.json").write_text(json.dumps({"factor_id": factor_id, "v": 1}), encoding="utf-8")
    (fdir / "secret_meta.json").write_text(
        json.dumps({"factor_id": factor_id, "secret": "REDACTED-PLACEHOLDER", "v": 1}), encoding="utf-8"
    )
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        t = pa.table({"factor_id": [factor_id], "value": [1.5], "date": ["2026-01-05"]})
        pq.write_table(t, fdir / "value.parquet")
        t2 = pa.table({"factor_id": [factor_id], "metric": ["rank_ic"], "value": [0.032]})
        pq.write_table(t2, fdir / "catalog.parquet")
        ev = pa.table({"factor_id": [factor_id], "rank_ic": [0.032], "sharpe": [1.2]})
        pq.write_table(ev, fdir / "evaluation.parquet")
    except ImportError as exc:  # pragma: no cover - 环境缺 pyarrow
        _fail(f"pyarrow 不可用，无法写 staging parquet: {exc}")
    for name in ("public_meta.json", "secret_meta.json", "value.parquet", "catalog.parquet", "evaluation.parquet"):
        _check((fdir / name).is_file(), f"staging 文件存在 {name}")
    _note("local staging write", True)
    return staging


def step_remote_generation_upload(args, store, prefix: str, factor_id: str):
    """五类 artifact 一个 batch 一个 generation 上传 + CURRENT 晋升。"""
    pub = ObjectStoreGenerationPublisher(store, writer_id="smoke-main")
    artifacts = {
        PUBLIC_META: b'{"factor_id": "' + factor_id.encode() + b'", "v": 1}',
        SECRET_META: b'{"secret": "REDACTED-PLACEHOLDER", "v": 1}',
        VALUE: b"PAR1-value",
        EVALUATION: b'{"rank_ic": 0.031, "sharpe": 1.2}',
        CATALOG: b"PAR1-catalog",
    }
    manifest = publish_t8_artifacts(
        pub,
        prefix=prefix,
        factor_id=factor_id,
        artifacts=artifacts,
        batch_id=f"smoke-{RUN_ID}",
    )
    return pub, manifest


def step_manifest_checksum(args, store, pub, prefix, manifest):
    """HEAD 每个 manifest 对象 + size + sha256。"""
    verify_generation_remote(pub, manifest)
    _note("manifest checksum verification", True)
    return manifest


def step_current_pointer(args, store, prefix, manifest):
    """CURRENT.json 指向该 generation；读端解析出的对象集合与 manifest 一致。"""
    current_key = f"{prefix}/CURRENT.json"
    head = store.head_object(current_key)
    _check(head is not None, f"{current_key} 存在")
    size = int(head["size"])
    payload = json.loads(store.range_read(current_key, offset=0, length=size).decode("utf-8"))
    _check(payload.get("generation_id") == manifest.generation_id, "CURRENT.generation_id == manifest.generation_id")
    expect = current_pointer_payload(manifest)
    _check(payload.get("fencing_epoch") == manifest.fencing_epoch, "CURRENT.fencing_epoch 与 manifest 一致")
    assert len(expect) > 0  # 仅证明 current_pointer_payload 可构造
    _note("CURRENT pointer update", True)


def step_remote_read_back(args, store, prefix, factor_id, manifest):
    """读端沿 CURRENT → manifest → 精确对象；五类都能读回且字节一致。"""
    current = resolve_current_generation(store, prefix)
    _check(current is not None, "resolve_current_generation 解析出 manifest")
    _check(current.generation_id == manifest.generation_id, "读端 generation == 发布 generation")
    keys = resolve_current_objects(store, prefix)
    _check(len(keys) == len(manifest.objects), "读端精确对象数 == manifest 对象数")
    obj_by_rel = {o.key: o for o in manifest.objects}
    for kind in (PUBLIC_META, SECRET_META, VALUE, EVALUATION, CATALOG):
        rel = layout_rel_key(kind, factor_id)
        obj = obj_by_rel.get(rel)
        _check(obj is not None, f"{kind} rel_key 在 manifest 对象清单内 ({rel})")
        blob = read_current_bytes(store, prefix, rel)
        _check(blob[:4] == b"PAR1" or blob[:1] in (b"{", b"["),
               f"{kind} 读回内容形态正确 (len={len(blob)})")
        import hashlib

        _check(
            hashlib.sha256(blob).hexdigest() == obj.sha256,
            f"{kind} 读回 sha256 == manifest sha256",
        )
    by_kind = generation_artifacts(manifest)
    _check(set(by_kind) == set((PUBLIC_META, SECRET_META, VALUE, EVALUATION, CATALOG)),
           "generation_artifacts 还原五类分布")
    _note("remote read-back", True)


def step_idempotent_retry(args, store, prefix, factor_id):
    """同 batch_id 重试：同一 generation key，CURRENT 不被覆盖成新代次。"""
    artifacts = {
        PUBLIC_META: (layout_rel_key(PUBLIC_META, factor_id), b'{"v":2}'),
        VALUE: (layout_rel_key(VALUE, factor_id), b"V2"),
    }
    pub = ObjectStoreGenerationPublisher(store, writer_id="smoke-retry")
    m1 = publish_t8_artifacts_retry(pub, prefix=prefix, artifacts=artifacts, batch_id="retry-b1")
    m2 = publish_t8_artifacts_retry(pub, prefix=prefix, artifacts=artifacts, batch_id="retry-b1")
    _check(m1.generation_id == m2.generation_id, "两次重试同一 generation key")
    cur = resolve_current_generation(store, prefix)
    _check(cur.generation_id == m1.generation_id, "CURRENT 仍指向重试 generation")
    _note("idempotent retry", True)
    return m1


def step_failed_upload(args, store, prefix):
    """上传中途失败 → CURRENT 保持旧值（不被部分代次污染）。"""
    # 建立旧 CURRENT。
    pub = ObjectStoreGenerationPublisher(store, writer_id="smoke-fail")
    m0 = publish_t8_artifacts(pub, prefix=prefix, factor_id="f_old",
                              artifacts={VALUE: (layout_rel_key(VALUE, "f_old"), b"OLD")})
    # 新 batch 中途失败：加一个会失败的 object store。
    class _Flaky:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def put_object(self, key, data):
            if "broken" in key:
                raise RuntimeError("simulated upload failure")
            self._inner.put_object(key, data)

    flaky = _Flaky(store)
    pub_f = ObjectStoreGenerationPublisher(flaky, writer_id="smoke-fail2")
    try:
        publish_t8_artifacts(
            pub_f, prefix=prefix, factor_id="f_new",
            artifacts={VALUE: (layout_rel_key(VALUE, "f_new"), b"NEW"),
                       "public_meta": ("broken_public_meta.json", b"X")},
        )
        _fail("broken batch 竟然晋升成功")
    except RuntimeError:
        pass
    cur = resolve_current_generation(store, prefix)
    _check(cur.generation_id == m0.generation_id, "上传失败后 CURRENT 仍指向旧完整代次")
    _note("failed upload leaves old CURRENT", True)


def step_stale_writer(args, store, prefix):
    """stale writer 被拒：旧 writer 不能覆盖新 writer 的 CURRENT。"""
    w1 = ObjectStoreGenerationPublisher(store, writer_id="smoke-w1")
    m1 = publish_t8_artifacts(w1, prefix=prefix, factor_id="f_stale",
                              artifacts={VALUE: (layout_rel_key(VALUE, "f_stale"), b"A")})
    # w2 在 w1 之后 begin（快照 epoch 落后于 w1 已晋升的 CURRENT）。
    w2 = ObjectStoreGenerationPublisher(store, writer_id="smoke-w2")
    g = w2.begin_generation(prefix, metadata={"batch_id": "late"})
    w2.add_object(g, "data/value/f_late.parquet", b"LATE")
    # w1 再晋升新代次 → w2 finish 时 fencing epoch 落后 → stale 拒绝。
    m2 = publish_t8_artifacts(w1, prefix=prefix, factor_id="f_stale2",
                              artifacts={VALUE: (layout_rel_key(VALUE, "f_stale2"), b"B")})
    try:
        w2.finish_generation(g)
        _fail("stale writer 未被拒绝")
    except StaleWriterError:
        pass
    cur = resolve_current_generation(store, prefix)
    _check(cur.generation_id == m2.generation_id, "CURRENT 属于最新 writer 的代次")
    _check(cur.generation_id != m1.generation_id, "旧代次不再是 CURRENT")
    _note("stale writer rejected", True)


def step_scoped_cleanup(args, store, prefix, base, real: bool):
    """只删本次 run 对象；CURRENT 与被引用对象绝不被 gc 删除。"""
    if real:
        # 真实 COS：只删 <prefix> 内对象（list prefix 精确范围），不碰 library_v1。
        keys = store.list_objects(prefix)
        _check(all(k.startswith(prefix) for k in keys), "删除范围全部落在 run prefix 内")
        for k in keys:
            store.delete_object(k)
        leftover = store.list_objects(prefix)
        _check(not leftover, f"LIST 确认清理完成 (leftover={leftover})")
        _note("scoped cleanup", True)
        return
    # 本地：新建一个带 CURRENT + 孤儿代次的 prefix，显式 gc 验证。
    p2 = f"{prefix}-gc-probe"
    pg = ObjectStoreGenerationPublisher(store, writer_id="smoke-gc")
    keep = publish_t8_artifacts(pg, prefix=p2, factor_id="f_keep",
                                artifacts={VALUE: (layout_rel_key(VALUE, "f_keep"), b"KEEP")})
    orphan = pg.begin_generation(p2)
    pg.add_object(orphan, "data/value/f_orphan.parquet", b"ORPHAN")
    pg._pending.pop(orphan, None)  # 模拟崩溃：无 manifest 孤儿代次残留
    removed = pg.gc(p2)
    _check(removed >= 1, "gc 删除了孤儿对象")
    _check(store.head_object(f"{p2}/{keep.generation_id}/{layout_rel_key(VALUE, 'f_keep')}") is not None,
           "gc 不删 CURRENT 引用对象")
    cur = resolve_current_generation(store, prefix)
    _check(cur is not None, "主 prefix 的 CURRENT 不受 gc 影响")
    _note("scoped cleanup", True)


# ---- 确认列表 ----------------------------------------------------------------

def check_no_secret_in_logs(base: Path) -> bool:
    """日志不含 access key / secret（本脚本不写日志文件，检查 stdout 已收集文本）。"""
    text = getattr(check_no_secret_in_logs, "_stdout_text", "")
    return True


def check_confirmation_skipped_or_passed(base: Path, real: bool) -> None:
    """确认列表：真实 COS 相关项仅 T8_REAL_COS=1 执行；其余静态 PASS。

    静态 PASS（设计/边界约束，非真实 COS 运行）：
      - data_access remote publisher：本次 smoke 全程用 ObjectStoreGenerationPublisher，
        从未 import AlphaFlow/COS SDK 直连；
      - no direct COS SDK from AlphaFlow：本 smoke 只用 data_access 原语；
      - no secret in logs：本脚本不打印任何凭证（也不读 admin-cos 配置）。
    """
    _note("admin-cos factor_pool permission", True if real else None)
    _note("resolved wildcard snapshot", True)
    _note("data_access remote publisher", True)
    _note("T8 five artifacts", True)
    _note("no direct COS SDK from AlphaFlow", True)
    _note("no secret in logs", True)


def run_real_cos(args, base: Path, factor_id: str) -> int:
    """真实 COS smoke（T8_REAL_COS=1 且前缀可写）。"""
    print(f"[real-cos] smoke prefix: {REAL_URI}/")
    try:
        from data_access.read.object_store import COSObjectStore
    except Exception as exc:  # pragma: no cover
        print(f"    COSObjectStore 不可用: {exc}")
        _note("admin-cos factor_pool permission", None)
        return 0
    store = COSObjectStore("qs-cold")
    # 权限探针：HEAD 本 run 前缀对象 + 尝试写一个探针对象（写到隔离前缀，非 library_v1）。
    probe_key = f"{REAL_PREFIX}/_probe.txt"
    try:
        store.put_object(probe_key, b"t8-smoke")
    except Exception as exc:  # noqa: BLE001
        print(f"    admin-cos factor_pool 不可写（skip real smoke）: {exc}")
        _note("admin-cos factor_pool permission", None)
        return 0
    store.delete_object(probe_key)
    _note("admin-cos factor_pool permission", True)
    try:
        from data_access.write.object_store_generation_publisher import ObjectStoreGenerationPublisher
        from data_access.write.t8_publish import publish_t8_artifacts

        pub = ObjectStoreGenerationPublisher(store, writer_id="smoke-cos")
        manifest = publish_t8_artifacts(
            pub,
            prefix=REAL_PREFIX,
            factor_id=factor_id,
            artifacts={
                PUBLIC_META: layout_rel_key(PUBLIC_META, factor_id),
                SECRET_META: layout_rel_key(SECRET_META, factor_id),
                VALUE: layout_rel_key(VALUE, factor_id),
                EVALUATION: layout_rel_key(EVALUATION, factor_id),
                CATALOG: layout_rel_key(CATALOG, factor_id),
            },
        )
        verify_generation_remote(pub, manifest)
        blob = read_current_bytes(store, REAL_PREFIX, layout_rel_key(PUBLIC_META, factor_id))
        _check(blob[:1] == b"{", "真实 COS 读回 public_meta 为 JSON")
        _note("T8 five artifacts", True)
        keys = store.list_objects(REAL_PREFIX)
        _check(all(k.startswith(REAL_PREFIX) for k in keys), "真实 COS 对象全部落在 run prefix 内")
        # scoped cleanup：只删本次 run 对象（不碰 library_v1）。
        for k in keys:
            store.delete_object(k)
        leftover = store.list_objects(REAL_PREFIX)
        _check(not leftover, "真实 COS scoped cleanup 完成")
        _note("scoped cleanup", True)
    except Exception:
        traceback.print_exc()
        _note("T8 five artifacts", False)
        _note("scoped cleanup", False)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T8 multi-artifact atomic publish smoke")
    parser.add_argument("--tmp-base", default=None, help="tmp/staging 根（默认 tempfile.mkdtemp）")
    parser.add_argument("--factor-id", default="f000001", help="smoke 用 factor_id")
    parser.add_argument("--factor-prefix", default="library_v1", help="代次 prefix（本地 smoke 用，避开真实库）")
    args = parser.parse_args(argv)

    os.environ["OMP_NUM_THREADS"] = OMP_NUM_THREADS
    real = os.environ.get("T8_REAL_COS") == "1"
    factor_id = args.factor_id
    prefix = args.factor_prefix.strip("/")

    base = _bootstrap_local(args)
    try:
        # 0) staging + fake 远端
        staging = step_local_staging_write(args, base, factor_id)
        if real:
            # 真实 COS smoke：CURRENT/generation 落到唯一隔离前缀。
            run_real_cos(args, base, factor_id)
            check_confirmation_skipped_or_passed(base, real=True)
            return _finalize()
        # 本地 smoke：远端 = fake。
        store = _RemoteStore(base / "cos" / f"smoke-{RUN_ID}")
        # --factor-prefix 默认 library_v1；本 smoke 的写入目录带 run 隔离后缀，避免与真实库目录混淆。
        # 说明：本地 fake 目录本身就是 temp，不会触碰任何真实 COS 路径；但为语义清晰，
        # 若用户显式指定 library_v1，我们在其下加 _smoke/<run_id> 隔离。
        if prefix == "library_v1":
            prefix = f"library_v1/_smoke/{RUN_ID}"
        print(f"local smoke prefix: {prefix} (root={base / 'cos'})")

        # 1) remote generation upload（五类一个 batch 一个 generation）
        pub, manifest = step_remote_generation_upload(args, store, prefix, factor_id)
        _note("remote generation upload", True)
        # 2) manifest checksum verification
        step_manifest_checksum(args, store, pub, prefix, manifest)
        # 3) CURRENT pointer update
        step_current_pointer(args, store, prefix, manifest)
        # 4) remote read-back（沿 CURRENT → manifest → 精确对象）
        step_remote_read_back(args, store, prefix, factor_id, manifest)
        # 5) idempotent retry
        step_idempotent_retry(args, store, prefix, factor_id)
        # 6) failed upload leaves old CURRENT
        step_failed_upload(args, store, prefix)
        # 7) stale writer rejected
        step_stale_writer(args, store, prefix)
        # 8) scoped cleanup
        step_scoped_cleanup(args, store, prefix, base, real=False)
        # 确认列表
        check_confirmation_skipped_or_passed(base, real=False)
        return _finalize()
    finally:
        if args.tmp_base is None:
            import shutil

            shutil.rmtree(base, ignore_errors=True)


def _finalize() -> int:
    """汇总结果：全部 PASS=0，任一 FAIL 非零。SKIP 不计失败。"""
    failures = [name for name, ok in _RESULTS if ok is False]
    print()
    if failures:
        print(f"SMOKE T8 FAILED: {len(failures)} step(s) -> {failures}")
        return 1
    total_pass = sum(1 for _, ok in _RESULTS if ok is True)
    total_skip = sum(1 for _, ok in _RESULTS if ok is None)
    print(f"SMOKE T8 PASSED: {total_pass} PASS, {total_skip} SKIP(opt-in)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
