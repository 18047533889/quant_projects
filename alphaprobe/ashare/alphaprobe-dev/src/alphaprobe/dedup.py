"""Identity 去重（任务书 §11）：四层去重 + atomic reservation。

Level 1 Canonical AST：安全化简（括号/±0/×1/÷1/pow(x,1)/常数折叠/交换律排序），
  禁止危险化简 x/x→1。
Level 2 Sign-invariant SignalEquivalenceID。
Level 3 ParameterFamily：ts_mean(close,19/20/21) 同族。
Level 4 Rank-equivalent fingerprint：256-bit SimHash（横截面 rank 签名）。
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Level 1: canonical 化简（对 FE DSL 文本做安全、确定性的代数化简）
# ---------------------------------------------------------------------------

_NUM = r"(?:\d+(?:\.\d*)?|\.\d+)"
# 单 token：标识符 + 可选整体调用（如 rank(...)、ts_mean(close, 20)）
IDENT_TAIL = r"[A-Za-z_$][\w$.]*(?:\([^()]*\))?"


def _fold_constant(expr: str) -> str:
    """常数折叠：只处理纯数字四则，不调用 eval。"""
    m = re.fullmatch(rf"\(\s*({_NUM})\s*([+\-*/])\s*({_NUM})\s*\)", expr)
    if not m:
        return expr
    try:
        a, op, b = float(m.group(1)), m.group(2), float(m.group(3))
    except ValueError:
        return expr
    if op == "+" :
        v = a + b
    elif op == "-":
        v = a - b
    elif op == "*":
        v = a * b
    else:
        if b == 0:
            return expr
        v = a / b
    if v == int(v):
        return str(int(v))
    return repr(v)


def _balanced_pairs(s: str) -> list[tuple[int, int]]:
    """括号配对（外层在前）。"""
    stack: list[int] = []
    pairs: list[tuple[int, int]] = []
    for i, ch in enumerate(s):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if stack:
                pairs.append((stack.pop(), i))
    pairs.reverse()  # stack.pop 产出内层在前；反转成外层优先
    return pairs


def _simplify_pairs(s: str, pattern: str, repl) -> str:
    """对每个平衡括号段尝试 fullmatch；命中即替换，从头重扫。

    仅对「真括号壳」（body 非空且整体被括号包裹的段）应用，
    函数调用自身的实参括号 rank(...) 不算壳（跳过其前一位是标识符字符的 '('）。
    """
    changed = True
    while changed:
        changed = False
        for a, b in _balanced_pairs(s):
            if a > 0 and (s[a - 1].isalnum() or s[a - 1] in "_."):
                continue  # 函数调用括号，不是壳
            body = s[a + 1 : b]
            m = re.fullmatch(pattern, body, re.S)
            if m:
                s = s[:a] + repl(m) + s[b + 1 :]
                changed = True
                break
    return s


def canonicalize_dsl(formula: str) -> str:
    """§11.1 安全化简。幂等：canonicalize(canonicalize(x)) == canonicalize(x)。

    只做文本层安全规则；深度 AST 化简由 factor_engine canonical 负责时，
    本函数保证不与之冲突（幂等且只做子集）。
    """
    s = " ".join(str(formula).split())
    # 括号折叠：内部括号平衡的整段去外壳
    prev = None
    while prev != s:
        prev = s
        s = _simplify_pairs(
            s, r"[A-Za-z_$][\w$.]*(?:\s*\([^()]*\))*",
            lambda m: m.group(0).strip(),
        )
        s = _simplify_pairs(
            s, r"[A-Za-z_$][\w$.]*\s*\((?:[^()]|\([^()]*\))*\)",
            lambda m: m.group(0).strip(),
        )
    # ×1 / ÷1 / +0 / -0 / ×(-1)
    s = _simplify_pairs(s, r"(.+?)\s*\*\s*1(?:\.0)?", lambda m: m.group(1).strip())
    s = _simplify_pairs(s, r"1(?:\.0)?\s*\*\s*(.+?)", lambda m: m.group(1).strip())
    s = _simplify_pairs(s, r"(.+?)\s*/\s*1(?:\.0)?", lambda m: m.group(1).strip())
    s = _simplify_pairs(s, r"(.+?)\s*\+\s*0(?:\.0)?", lambda m: m.group(1).strip())
    s = _simplify_pairs(s, r"0(?:\.0)?\s*\+\s*(.+?)", lambda m: m.group(1).strip())
    s = _simplify_pairs(s, r"(.+?)\s*-\s*0(?:\.0)?", lambda m: m.group(1).strip())
    s = _simplify_pairs(
        s, r"0(?:\.0)?\s*-\s*(.+?)", lambda m: "(-(" + m.group(1).strip() + "))"
    )
    s = _simplify_pairs(
        s, r"(.+?)\s*\*\s*\(-1(?:\.0)?\)", lambda m: "(-(" + m.group(1).strip() + "))"
    )
    s = _simplify_pairs(
        s, r"\(-1(?:\.0)?\)\s*\*\s*(.+?)", lambda m: "(-(" + m.group(1).strip() + "))"
    )
    # 双重负号：-(-(x)) -> x
    s = _simplify_pairs(s, r"-\s*\(\s*-\s*(.+?)\s*\)", lambda m: m.group(1).strip())
    # 一元负号规范：-(x) 保留；0-x 已转 -(x)
    s = _fold_constant(s)
    # 空白归一
    s = " ".join(s.split())
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"\(\s*", "(", s)
    s = re.sub(r"\s*\)", ")", s)
    # 一元负号紧凑形归一：(-rank(...)) → (-(rank(...)))，保证幂等
    s = _simplify_pairs(
        s, r"-\s*([A-Za-z_$][\w$.]*\s*\((?:[^()]|\([^()]*\))*\))",
        lambda m: "(-(" + m.group(1).strip() + "))",
    )
    return s


def canonical_ast_hash(formula: str) -> str:
    c = canonicalize_dsl(formula)
    return hashlib.sha256(c.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Level 2: sign-invariant
# ---------------------------------------------------------------------------


def sign_normalized(formula: str) -> str:
    """-(x)/(-x) 外壳剥离 → sign-invariant 规范形。"""
    s = canonicalize_dsl(formula)
    prev = None
    while prev != s:
        prev = s
        # (-(X)) → X（外壳整体为负号包一层）
        if s.startswith("(-(") and s.endswith("))"):
            inner = s[3:-2]
            if inner.count("(") == inner.count(")"):
                s = inner
                continue
        # -(X) → X
        if s.startswith("-(") and s.endswith(")"):
            inner = s[2:-1]
            if inner.count("(") == inner.count(")"):
                s = inner
    return s


def signal_equivalence_id(formula: str) -> str:
    return hashlib.sha256(sign_normalized(formula).encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Level 3: Parameter Family
# ---------------------------------------------------------------------------

_WINDOW_PARAM = re.compile(r",\s*(\d+)\s*\)")


def parameter_family_key(formula: str) -> str:
    """把最外层算子的窗口参数打码 → 同族 key。

    ts_mean(close,19)/ts_mean(close,20)/ts_mean(close,21) → 同一 family。
    """
    s = sign_normalized(formula)
    # 逐层把最后一个数字参数替换为 {W}
    s2 = _WINDOW_PARAM.sub(", {W})", s)
    return hashlib.sha256(s2.encode("utf-8")).hexdigest()[:32]


@dataclass
class ParameterFamily:
    family_id: str
    representative: str = ""
    best_parameter: str = ""
    best_score: float = float("-inf")
    explored: set[str] = field(default_factory=set)
    saturation: float = 0.0
    members: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Level 4: rank fingerprint（256-bit SimHash，量化 rank 签名）
# ---------------------------------------------------------------------------


def rank_fingerprint(cross_section_ranks_by_date: "list[dict[str, float]] | None") -> bytes | None:
    """§11.4：对固定跨年 sample dates 的横截面 rank 签名做 SimHash。

    输入：每个采样日的 {stock: rank}（0~1）。缺数据时返回 None。
    识别 f / 2f / 100f+5 / monotonic(f) / rank(f)。
    """
    if not cross_section_ranks_by_date:
        return None
    bits = [0] * 256
    for ranks in cross_section_ranks_by_date:
        for stock, r in ranks.items():
            h = hashlib.sha256(f"{stock}:{r:.3f}".encode()).digest()
            for byte_i, b in enumerate(h):
                for k in range(8):
                    if (b >> k) & 1:
                        bits[(byte_i * 8 + k) % 256] += 1
                    else:
                        bits[(byte_i * 8 + k) % 256] -= 1
    out = bytearray(32)
    for i, v in enumerate(bits):
        if v > 0:
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def fingerprint_hamming(a: bytes, b: bytes) -> int:
    return bin(int.from_bytes(a, "big") ^ int.from_bytes(b, "big")).count("1")


def fingerprint_hamming_array(q: "np.ndarray", mat: "np.ndarray") -> "np.ndarray":
    """向量化 hamming：查询 (32,) uint8 vs 位矩阵 (32, N) uint8 → (N,) 距离。

    按位 XOR → ``np.unpackbits`` 展开 → 列求和（256-bit SimHash 口径，
    与标量 :func:`fingerprint_hamming` 逐字节等价，保持 §11.4 语义）。
    """
    if mat.size == 0:
        return np.empty(0, dtype=np.int64)
    xor = np.bitwise_xor(q[:, None], mat)  # (32, N)
    bits = np.unpackbits(xor, axis=0)  # (256, N)
    return bits.sum(axis=0, dtype=np.int64)


# ---------------------------------------------------------------------------
# GlobalSeenIndex + Atomic Reservation（§11.6 / §34）
# ---------------------------------------------------------------------------


class GlobalSeenIndex:
    """跨轮 Global Seen/Dedup。线程安全；disk-backed 由 memory 层替换。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_signal: dict[str, str] = {}          # signal_id -> factor_id
        self._by_family: dict[str, set[str]] = defaultdict(set)  # family_id -> factor_ids
        self._fingerprints: dict[str, bytes] = {}     # factor_id -> fp
        self._rediscovery: dict[str, int] = defaultdict(int)  # signal_id -> count
        # 指纹位矩阵缓存（topk 向量化用）：版本号在每次写入时递增
        self._fp_cache: "np.ndarray | None" = None
        self._fp_cache_ids: list[str] = []
        self._fp_cache_version: int = -1
        self._fingerprints_version: int = 0

    def reserve(
        self,
        *,
        signal_id: str,
        factor_id: str,
        family_id: str | None = None,
        fingerprint: bytes | None = None,
    ) -> tuple[bool, str | None]:
        """§11.6 Atomic Reservation：NEW→RESERVED。

        Returns (reserved, existing_factor_id)。
        冲突时调用方必须立即停止昂贵回测。
        """
        with self._lock:
            existing = self._by_signal.get(signal_id)
            if existing is not None:
                self._rediscovery[signal_id] += 1
                return False, existing
            self._by_signal[signal_id] = factor_id
            if family_id:
                self._by_family[family_id].add(factor_id)
            if fingerprint is not None:
                self._fingerprints[factor_id] = fingerprint
                self._fingerprints_version += 1
            return True, None

    def lookup_signal(self, signal_id: str) -> str | None:
        with self._lock:
            return self._by_signal.get(signal_id)

    def family_members(self, family_id: str) -> set[str]:
        with self._lock:
            return set(self._by_family.get(family_id, ()))

    def mark_evaluated(self, factor_id: str, family_id: str | None = None) -> None:
        """RESERVED→EVALUATED：仅更新族统计。"""
        with self._lock:
            if family_id and factor_id in self._by_signal.values():
                self._by_family[family_id].add(factor_id)

    def rediscovery_count(self, signal_id: str) -> int:
        with self._lock:
            return self._rediscovery.get(signal_id, 0)

    def size(self) -> int:
        with self._lock:
            return len(self._by_signal)

    def topk_fingerprint_neighbors(
        self, fp: bytes, k: int = 5, max_hamming: int = 48
    ) -> list[tuple[str, int]]:
        """§11.5：只对 Top-K 邻居做精确确认，不做全库相关矩阵（§78）。

        numpy 向量化 hamming（位矩阵广播 popcount，复用库内既有 numpy 依赖，
        不引入新索引轮子）：全库距一次性算完，再流式取 top-k，
        峰值内存 O(N·fp_bytes) 而非 O(N²)。指纹库变化时惰性重打包。
        """
        if fp is None:
            return []
        with self._lock:
            if not self._fingerprints:
                return []
            fids = list(self._fingerprints.keys())
            fp_arr = self._packed_fingerprint_matrix(fids)
            d_arr = fingerprint_hamming_array(np.frombuffer(fp, dtype=np.uint8), fp_arr)
            cand = np.nonzero(d_arr <= max_hamming)[0]
            if cand.size == 0:
                return []
            if k < cand.size:
                # argpartition 取前 k 近，再精确排序（O(N) 而非 O(N log N)）
                part = np.argpartition(d_arr[cand], k - 1)[:k]
                cand = cand[part]
            pairs = sorted((int(d_arr[i]), fids[i]) for i in cand)
            return [(fid, d) for d, fid in pairs]

    def _packed_fingerprint_matrix(self, fids: list[str]) -> "np.ndarray":
        """缓存指纹位矩阵（256×N bits → 32×N uint8）；库变化时惰性重建。

        `_fingerprints_version` 追踪库版本：reserve/清空都递增，避免每次查询
        全量重打包。缓存仅在本实例生命周期内有效（无跨进程持久化）。
        """
        if (
            self._fp_cache is not None
            and self._fp_cache_version == self._fingerprints_version
            and self._fp_cache_ids == fids
        ):
            return self._fp_cache
        n = len(fids)
        mat = np.empty((n, 32), dtype=np.uint8)
        for i, fid in enumerate(fids):
            mat[i] = np.frombuffer(self._fingerprints[fid], dtype=np.uint8)
        self._fp_cache = mat.T.copy()  # (32, N)：查询向量广播减法按列对齐
        self._fp_cache_ids = fids
        self._fp_cache_version = self._fingerprints_version
        return self._fp_cache