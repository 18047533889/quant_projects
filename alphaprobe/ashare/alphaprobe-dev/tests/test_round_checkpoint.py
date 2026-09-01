"""Phase 8/11/13 测试：checkpoint v2 + RoundManager + trainer Test 封存（§3.3/§57/§84/§85）。"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from alphaprobe.trainer.checkpoint import (
    CheckpointMeta,
    save_checkpoint,
    load_checkpoint_if_exists,
    checkpoint_path,
    CHECKPOINT_VERSION,
)
from alphaprobe.checkpoint import load_checkpoint_v2

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "alphaprobe"


# ---------------------------------------------------------------------------
# §57 checkpoint v2：保存→加载→active_pool_factor_ids 一致 + v1 仍可读
# ---------------------------------------------------------------------------


def _write_v1_checkpoint(tmp: Path, *, completed_iteration: int = 3) -> Path:
    """手写最小 v1 checkpoint JSON（trainer/checkpoint.py 兼容旧文件）。"""
    payload = {
        "version": CHECKPOINT_VERSION,
        "completed_iteration": completed_iteration,
        "search_time": 10,
        "pool_capacity": 20,
        "pool": {
            "size": 2,
            "exprs": ["rank(close)", "ts_mean(close, 5)"],
            "topics": ["t1", "t2"],
            "descriptions": ["d1", "d2"],
            "single_ics": [0.01, 0.02],
            "icir": [0.1, 0.2],
            "weights": [0.5, 0.5],
            "best_ic_ret": 0.02,
            "eval_cnt": 0,
        },
        "graph": {
            "nodes": [
                {
                    "expression": "rank(close)",
                    "topic": "t1",
                    "description": "d1",
                    "depth": 1,
                    "ic": 0.01,
                    "icir": 0.1,
                    "times": 0,
                    "test_ic": 0.0,
                    "test_icir": 0.0,
                    "parent": None,
                    "children": [],
                }
            ]
        },
    }
    path = checkpoint_path(tmp)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_checkpoint_v2_roundtrip():
    from alphaprobe.checkpoint import save_checkpoint_v2

    with tempfile.TemporaryDirectory() as td:
        p = save_checkpoint_v2(
            td,
            run_id="run1",
            round_id="r2",
            campaign_id="c3",
            generation=7,
            active_pool_factor_ids=["f1", "f2"],
            pending_actions=[],
            scheduler_state={},
            budget_state={},
            config_hash="abc",
        )
        assert p.name == "checkpoint_v2.json"
        loaded = load_checkpoint_v2(td)
        assert loaded is not None
        assert loaded["checkpoint_version"] == 2
        assert loaded["active_pool_factor_ids"] == ["f1", "f2"]
        assert loaded["round_id"] == "r2"
        assert loaded["campaign_id"] == "c3"
        assert loaded["generation"] == 7
        assert loaded["config_hash"] == "abc"
        # scheduler/budget 空 dict（§57 本阶段先占位）
        assert loaded["scheduler_state"] == {}
        assert loaded["budget_state"] == {}


def test_v1_checkpoint_still_readable_by_trainer_loader():
    """v1 旧文件照常可读（兼容层不动）。"""
    with tempfile.TemporaryDirectory() as td:
        _write_v1_checkpoint(Path(td), completed_iteration=5)
        raw = json.loads(checkpoint_path(Path(td)).read_text(encoding="utf-8"))
        assert raw["version"] == CHECKPOINT_VERSION
        assert raw["completed_iteration"] == 5
        assert raw["pool"]["exprs"] == ["rank(close)", "ts_mean(close, 5)"]


# ---------------------------------------------------------------------------
# §84 RoundManager：2 轮 dry-run（campaign 回调 stub，不跑 LLM/数据）
# ---------------------------------------------------------------------------


class _FakeCalibrator:
    def __init__(self) -> None:
        self.frozen_round: str | None = None

    def freeze(self, round_id: str) -> None:
        self.frozen_round = round_id


class _FakeStore:
    """GlobalMemoryStore 最小 stub（不落 SQLite）。"""

    def __init__(self) -> None:
        self.packets = 0

    def build_memory_packet(self, **kwargs):
        self.packets += 1
        return {"round": kwargs.get("round_id")}


class _FakePatience:
    def observe(self, lineage_key: str, gains: dict) -> bool:
        return False


def test_round_manager_two_rounds_dry_run():
    from alphaprobe.continuous.round_manager import RoundManager

    seen = []
    calibrators = [_FakeCalibrator(), _FakeCalibrator()]

    def campaign_cb(round_no, calibrator, *a, **kw):
        seen.append(round_no)
        calibrators.append(calibrator)
        return {"pool_size": 3, "round_no": round_no}

    rm = RoundManager(
        store=_FakeStore(),
        calibrator_factory=lambda: calibrators[len(seen)] if len(seen) < 2 else _FakeCalibrator(),
        patience_tracker=_FakePatience(),
        campaign_callback=campaign_cb,
        max_rounds=2,
        max_hours=None,
        sleep_seconds=0,
    )
    rm.run()
    assert seen == [1, 2]
    assert rm.round_no == 2
    # calibrator frozen
    assert all(c.frozen_round is not None for c in calibrators)


# ---------------------------------------------------------------------------
# §3.3 trainer Test 封存：log() 在 test_data=None 下不抛、不写 test 指标
# ---------------------------------------------------------------------------


def test_alphaknowledge_logger_test_data_none_safe():
    """trainer 封存后：AlphaKnowledgeLogger(test_data=None).log() 不抛。

    trainer.py 依赖 torch/qlib（本测试环境无 GPU wheel）——这里用
    importlib 从源码文本加载（不执行模块顶层 import）。
    """
    trainer_mod = _load_trainer_module_text()

    class _Args:
        cuda = 0

    logger = trainer_mod.AlphaKnowledgeLogger(test_data=None, target=None, log_dir="/tmp", args=_Args())

    class _Node:
        pass

    class _Pool:
        state = {"exprs": []}
        size = 0
        single_ics = []
        expr2node = []
        exprs = []

        def to_dict(self):
            return {"exprs": []}

    with tempfile.TemporaryDirectory() as td:
        logger.log_dir = td
        # 不抛即可（空 pool，无 test 指标节点）
        logger.log(_Pool(), None, iteration=1)


def test_load_test_res_sealed():
    """load_test_res 封存：写一行说明并 return，不跑 qlib/不算 test 指标。"""
    trainer_mod = _load_trainer_module_text()

    class _Args:
        cuda = 0

    logger = trainer_mod.AlphaKnowledgeLogger(test_data=None, target=None, log_dir="/tmp", args=_Args())

    class _Pool:
        state = {"exprs": []}

    import io

    buf = io.StringIO()
    logger.load_test_res(_Pool(), buf)
    assert "sealed" in buf.getvalue().lower()


def test_grep_no_test_ic_writeback_in_trainer():
    """grep gate：trainer.py 无 node.test_ic / node.test_icir 赋值写回。"""
    src = (SRC_ROOT / "trainer" / "trainer.py").read_text(encoding="utf-8")
    # 注释里允许出现说明；只禁止赋值写回
    for line in src.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("node.test_ic ="), line
        assert not stripped.startswith("node.test_icir ="), line
    # 写回点（赋值）必须为零；注释说明词允许
    assert "node.test_ic =" not in src
    assert "node.test_icir =" not in src


# ---------------------------------------------------------------------------
# 辅助：无 torch 环境加载 trainer.py 模块源码（文本级，不执行顶层 import）
# ---------------------------------------------------------------------------

def _load_trainer_module_text():
    """从 trainer.py 源码文本加载，注入最小 stub 依赖，避免 import torch。"""
    import importlib.util
    import types

    # 构造最小 stub 模块（trainer.py 顶层 import 的符号）
    exp = types.ModuleType("shared.alphagen.data.expression")
    exp.Feature = lambda *a, **k: None
    exp.Expression = type("Expression", (), {})
    exp.Ref = lambda *a, **k: None
    exp.expression_edit_distance = lambda *a, **k: 0

    graph = types.ModuleType("shared.alphagen.data.expression_knowledge_graph")
    graph.ExpressionKnowledgeGraph = type("ExpressionKnowledgeGraph", (), {})
    graph.ExpressionNode = type("ExpressionNode", (), {})

    tree = types.ModuleType("shared.alphagen.data.tree")
    tree.ExpressionParser = type("ExpressionParser", (), {})
    tree.InvalidExpressionException = type("InvalidExpressionException", (Exception,), {})

    stock = types.ModuleType("shared.alphagen_qlib.stock_data")
    stock.StockData = type("StockData", (), {})

    utils = types.ModuleType("shared.alphagen.utils.correlation")
    utils.batch_pearsonr = lambda *a, **k: None
    utils.batch_spearmanr = lambda *a, **k: None
    utils.batch_ret = lambda *a, **k: None
    utils.batch_sharpe_ratio = lambda *a, **k: None
    utils.batch_max_drawdown = lambda *a, **k: None

    builder = types.ModuleType("baselines.gan.utils.builder")
    builder.exprs2tensor = lambda *a, **k: None

    torch_stub = types.ModuleType("torch")
    torch_stub.cuda = types.SimpleNamespace()
    torch_stub.cuda.empty_cache = lambda *a, **k: None
    torch_stub.linalg = types.SimpleNamespace()
    torch_stub.linalg.svd = lambda *a, **k: None
    torch_stub.linalg.qr = lambda *a, **k: None
    torch_stub.linalg.lstsq = lambda *a, **k: None
    torch_stub.isnan = lambda x: False
    torch_stub.Tensor = type("Tensor", (), {})
    torch_stub.device = type("device", (), {})

    # torch.backends / torch.nn 子模块 stub（shared 侧 import torch.backends.cudnn）
    backends = types.ModuleType("torch.backends")
    backends.cudnn = types.SimpleNamespace(benchmark=False, deterministic=False)
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.Module = type("Module", (), {})
    torch_stub.backends = backends
    torch_stub.nn = nn_mod

    import sys

    openai_stub = types.ModuleType("openai")
    openai_stub.OpenAI = lambda *a, **k: None
    llm_stub = types.ModuleType("shared.utils.llm")
    llm_stub.OpenAIModel = type("OpenAIModel", (), {})
    llm_stub.LLMQuotaExhaustedError = type("LLMQuotaExhaustedError", (Exception,), {})
    prompt_stub = types.ModuleType("shared.utils.prompt")
    prompt_stub.PROMPT_HEAD = ""
    prompt_stub.build_generation_user_prompt = lambda **k: ""
    parse_stub = types.ModuleType("alphaprobe.fe_bridge.expr_parse")
    parse_stub.parse_mining_expression = lambda *a, **k: None
    ckpt_stub = types.ModuleType("alphaprobe.trainer.checkpoint")
    ckpt_stub.save_checkpoint = lambda *a, **k: None
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.tqdm = lambda *a, **k: __import__("itertools").count()

    for name, mod in [
        ("shared.alphagen.data.expression", exp),
        ("shared.alphagen.data.expression_knowledge_graph", graph),
        ("shared.alphagen.data.tree", tree),
        ("shared.alphagen_qlib.stock_data", stock),
        ("shared.alphagen.utils.correlation", utils),
        ("baselines.gan.utils.builder", builder),
        ("torch", torch_stub),
        ("torch.backends", backends),
        ("torch.nn", nn_mod),
        ("openai", openai_stub),
        ("shared.utils.llm", llm_stub),
        ("shared.utils.prompt", prompt_stub),
        ("alphaprobe.fe_bridge.expr_parse", parse_stub),
        ("alphaprobe.trainer.checkpoint", ckpt_stub),
        ("tqdm", tqdm_stub),
    ]:
        sys.modules.setdefault(name, mod)

    src = (SRC_ROOT / "trainer" / "trainer.py").read_text(encoding="utf-8")
    spec = importlib.util.spec_from_loader("_alphaprobe_trainer_text", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(SRC_ROOT / "trainer" / "trainer.py")
    # 用 stub 占位注册 trainer 内部导入的 shared 子模块后 exec
    try:
        exec(compile(src, str(SRC_ROOT / "trainer" / "trainer.py"), "exec"), mod.__dict__)
    except ModuleNotFoundError as exc:
        # shared/alphagen/models 及其依赖在 exec 顶层 import 时也走 torch；
        # 把 trainer.pool 一并 stub 掉（AlphaKnowledgeLogger 不需要 pool 类型本体）
        pool_stub = types.ModuleType("alphaprobe.trainer.pool")
        pool_stub.AlphaKnowledgePool = type("AlphaKnowledgePool", (), {})
        sys.modules.setdefault("alphaprobe.trainer.pool", pool_stub)
        sys.modules.setdefault("alphaprobe.trainer", types.ModuleType("alphaprobe.trainer"))
        exec(compile(src, str(SRC_ROOT / "trainer" / "trainer.py"), "exec"), mod.__dict__)
    return mod
