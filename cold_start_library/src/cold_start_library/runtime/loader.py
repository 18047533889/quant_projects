"""冷启动库运行时加载（读 yaml、抽样），不依赖 domain 模板代码。"""

from __future__ import annotations

import random
import json
import hashlib
from importlib.resources import files
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from cold_start_library.runtime.dsl import validate_factor_engine_dsl
from cold_start_library.runtime.sampling import (
    norm_expr_structure,
    sample_structurally_diverse_entries,
)


@dataclass(frozen=True)
class ColdStartEntry:
    """冷启动条目：`expr` 为 factor_engine DSL（可直接 parse_expr）。"""

    expr: str
    topic: str
    description: str
    operators: tuple[str, ...] = ()
    factor_id: str = ""
    market: str = ""
    surface: str = ""
    availability_tier: str = ""
    catalog_ref: str = ""

    def __post_init__(self) -> None:
        if not self.expr:
            raise ValueError("cold-start expr is required")
        if self.market and self.market not in {"A", "US"}:
            raise ValueError("cold-start market must be A or US")
        if self.surface and self.surface not in {"daily", "extended"}:
            raise ValueError("cold-start surface must be daily or extended")
        if self.availability_tier and self.availability_tier not in {"core", "extended"}:
            raise ValueError("unsupported cold-start availability tier")
        object.__setattr__(self, "operators", tuple(self.operators))


def package_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_yaml_path(domain: str = "ashare", name: str = "backend_v9_core.yaml") -> Path:
    return package_root() / "data" / domain / name


def default_catalog_path() -> Path:
    """Resolve the shipped, audited V9 core pool without changing its tier."""
    source_path = package_root() / "library" / "production_default_core_v9.json"
    if source_path.is_file():
        return source_path
    installed = Path(str(files("cold_start_library").joinpath(
        "library", "production_default_core_v9.json"
    )))
    if installed.is_file():
        return installed
    raise FileNotFoundError(
        "authoritative production_default_core_v9.json is not installed; "
        "refusing to substitute another cold-start universe or tier"
    )


def sample_cold_start_entries(
    entries: Sequence[ColdStartEntry],
    n: int,
    *,
    seed: int | None = None,
    ensure_operator_coverage: bool = True,
    ensure_structural_diversity: bool = False,
) -> list[ColdStartEntry]:
    if n <= 0:
        return []
    pool = list(entries)
    if n >= len(pool):
        return pool

    if ensure_structural_diversity:
        return sample_structurally_diverse_entries(pool, n, seed=seed)

    rng = random.Random(seed)
    chosen: list[ColdStartEntry] = []
    chosen_keys: set[str] = set()

    if ensure_operator_coverage:
        by_op: dict[str, list[ColdStartEntry]] = {}
        for e in pool:
            for op in e.operators or ():
                by_op.setdefault(op, []).append(e)
        ops = list(by_op.keys())
        rng.shuffle(ops)
        for op in ops:
            if len(chosen) >= n:
                break
            candidates = [c for c in by_op[op] if c.expr not in chosen_keys]
            if not candidates:
                continue
            pick = rng.choice(candidates)
            chosen.append(pick)
            chosen_keys.add(pick.expr)

    remaining = [e for e in pool if e.expr not in chosen_keys]
    rng.shuffle(remaining)
    for e in remaining:
        if len(chosen) >= n:
            break
        chosen.append(e)
        chosen_keys.add(e.expr)
    return chosen[:n]


def entries_to_tuples(entries: Sequence[ColdStartEntry]) -> list[tuple[str, str, str]]:
    return [(e.expr, e.topic, e.description) for e in entries]


def load_cold_start_yaml(
    path: str | Path,
    *,
    validate: bool | None = None,
) -> list[ColdStartEntry]:
    """加载冷启动 yaml。

    ``validate``:
      - None: 若 yaml 标记 ``dsl_validated: true`` 则跳过 parse（训练热路径）
      - True/False: 强制开/关
    """
    raw = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
    do_validate = bool(validate) if validate is not None else not bool(raw.get("dsl_validated"))
    entries: list[ColdStartEntry] = []
    for item in raw.get("entries") or []:
        expr = str(item.get("expr") or "").strip()
        if not expr:
            continue
        if do_validate:
            ok, err = validate_factor_engine_dsl(expr)
            if not ok:
                raise ValueError(f"冷启动 DSL 无法 parse_expr: {expr[:120]} ({err})")
        ops = item.get("operators") or ()
        if isinstance(ops, str):
            ops = [x.strip() for x in ops.split(",") if x.strip()]
        entries.append(
            ColdStartEntry(
                expr=expr,
                topic=str(item.get("topic") or ""),
                description=str(item.get("description") or ""),
                operators=tuple(ops),
            )
        )
    return entries


def load_cold_start_catalog(
    path: str | Path,
    *,
    market: str | None = None,
    surface: str | None = None,
    availability_tier: str | None = None,
) -> list[ColdStartEntry]:
    """Load the authoritative V9 JSON catalog with identity/tier checks."""
    catalog_path = Path(path).expanduser()
    payload = catalog_path.read_bytes()
    raw = json.loads(payload)
    if not isinstance(raw, list):
        raise ValueError("cold-start JSON catalog must contain a list")
    catalog_ref = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    entries: list[ColdStartEntry] = []
    factor_ids: set[str] = set()
    expressions: set[str] = set()
    for row in raw:
        if not isinstance(row, dict):
            raise TypeError("cold-start catalog rows must be mappings")
        factor_id = str(row.get("factor_id") or "").strip()
        expr = str(row.get("formula_v9") or "").strip()
        if not factor_id or not expr:
            raise ValueError("catalog row requires factor_id and formula_v9")
        if row.get("v9_default_pool") != "core":
            raise ValueError("default catalog contains a row outside the V9 core pool")
        if row.get("v9_status") != "validated":
            raise ValueError("default catalog contains a non-validated V9 row")
        if market is not None and row.get("market") != market:
            continue
        if surface is not None and row.get("layer") != surface:
            continue
        if availability_tier is not None and row.get("availability_tier") != availability_tier:
            continue
        if factor_id in factor_ids or expr in expressions:
            raise ValueError("selected catalog factor IDs and formulas must be unique")
        factor_ids.add(factor_id)
        expressions.add(expr)
        operators = tuple(sorted(
            part.strip() for part in str(row.get("operators_latest") or "").split(",")
            if part.strip()
        ))
        entries.append(ColdStartEntry(
            expr=expr,
            topic=str(row.get("family") or ""),
            description=str(row.get("explanation") or ""),
            operators=operators,
            factor_id=factor_id,
            market=str(row.get("market") or ""),
            surface=str(row.get("layer") or ""),
            availability_tier=str(row.get("availability_tier") or ""),
            catalog_ref=catalog_ref,
        ))
    if not entries:
        raise ValueError("selected cold-start catalog domain is empty")
    return entries


def load_cold_start_entries(path: str | Path) -> list[ColdStartEntry]:
    resolved = Path(path).expanduser()
    if resolved.suffix.lower() == ".json":
        return load_cold_start_catalog(resolved)
    if resolved.suffix.lower() in {".yaml", ".yml"}:
        return load_cold_start_yaml(resolved)
    raise ValueError("cold-start catalog must be JSON or YAML")


def load_cold_start_for_training(
    yaml_path: str | Path | None = None,
    sample_size: int | None = None,
    seed: int | None = None,
) -> list[tuple[str, str, str]]:
    path = Path(yaml_path).expanduser() if yaml_path else default_catalog_path()
    entries = (load_cold_start_catalog(
        path, market="A", surface="daily", availability_tier="core",
    ) if yaml_path is None else load_cold_start_entries(path))
    if sample_size is None or sample_size >= len(entries):
        return entries_to_tuples(entries)
    picked = sample_cold_start_entries(entries, int(sample_size), seed=seed)
    return entries_to_tuples(picked)


def load_cold_start_mixed_for_training(
    total_size: int,
    pv_yaml: str | Path | None = None,
    alpha191_yaml: str | Path | None = None,
    pv_ratio: float = 0.7,
    seed: int | None = None,
) -> list[tuple[str, str, str]]:
    """兼容旧双库混合：主库 + 可选副库。"""
    n_pv = max(0, min(total_size, int(round(total_size * float(pv_ratio)))))
    n_a191 = max(0, total_size - n_pv)
    pv_path = Path(pv_yaml).expanduser() if pv_yaml else default_yaml_path()
    a191_path = (
        Path(alpha191_yaml).expanduser()
        if alpha191_yaml
        else default_yaml_path("alpha191", "gtja191.yaml")
    )
    rng = random.Random(seed)
    chosen: list[ColdStartEntry] = []
    seen: set[str] = set()

    def _take(path: Path, need: int, sub_seed: int | None) -> None:
        if need <= 0 or not path.is_file():
            return
        pool = [e for e in load_cold_start_yaml(path) if e.expr not in seen]
        pick = sample_cold_start_entries(pool, need, seed=sub_seed)
        for e in pick:
            seen.add(e.expr)
            chosen.append(e)

    pv_seed = rng.randint(0, 2147483647) if seed is not None else None
    a191_seed = rng.randint(0, 2147483647) if seed is not None else None
    _take(pv_path, n_pv, pv_seed)
    _take(a191_path, n_a191, a191_seed)
    rng.shuffle(chosen)
    return entries_to_tuples(chosen[:total_size])


def _allocate_mix_counts(total_size: int, ratios: dict[str, float]) -> dict[str, int]:
    keys = list(ratios.keys())
    raw = {k: max(0.0, float(ratios.get(k, 0.0))) for k in keys}
    s = sum(raw.values()) or 1.0
    norm = {k: v / s for k, v in raw.items()}
    counts = {k: int(total_size * norm[k]) for k in keys}
    remainder = total_size - sum(counts.values())
    order = sorted(keys, key=lambda k: (norm[k] - counts[k] / max(total_size, 1)), reverse=True)
    for i in range(max(0, remainder)):
        counts[order[i % len(order)]] += 1
    return counts


def load_cold_start_multi_library_for_training(
    total_size: int,
    ratios: dict[str, float] | None = None,
    library_paths: dict[str, str] | None = None,
    seed: int | None = None,
    pv_structural_diversity: bool = True,
) -> list[tuple[str, str, str]]:
    """
    多库按比例混合冷启动（默认 100% V9 A 股主库；也可配置经典四库）。

    PV 库默认启用结构多样性抽样：同一公式形状（仅数字参数不同）每批只取一条。
    """
    if total_size <= 0:
        return []

    default_ratios = {"pv": 1.0, "alpha101": 0.0, "alpha191": 0.0, "alpha158": 0.0}
    mix_ratios = dict(ratios or default_ratios)
    counts = _allocate_mix_counts(total_size, mix_ratios)

    default_paths = {
        "pv": default_yaml_path("ashare", "backend_v9_core.yaml"),
        "alpha101": default_yaml_path("alpha101", "wq101.yaml"),
        "alpha191": default_yaml_path("alpha191", "gtja191.yaml"),
        "alpha158": default_yaml_path("alpha158", "qlib158.yaml"),
    }
    paths: dict[str, Path] = dict(default_paths)
    if library_paths:
        for key, val in library_paths.items():
            if not val:
                continue
            paths[key] = Path(str(val)).expanduser()

    rng = random.Random(seed)
    chosen: list[ColdStartEntry] = []
    seen: set[str] = set()

    def _pick_from_yaml(
        key: str,
        count: int,
        *,
        structural: bool = False,
        sub_seed: int | None = None,
    ) -> None:
        if count <= 0:
            return
        path = Path(paths.get(key) or "")
        if not path.is_file():
            print(f"[cold_start] skip {key}: yaml not found {path}")
            return
        pool = load_cold_start_yaml(path)
        pick = sample_cold_start_entries(
            pool,
            count,
            seed=sub_seed,
            ensure_operator_coverage=not structural,
            ensure_structural_diversity=structural,
        )
        for e in pick:
            if e.expr in seen:
                continue
            seen.add(e.expr)
            chosen.append(e)

    for key, count in counts.items():
        sub_seed = rng.randint(0, 2147483647) if seed is not None else None
        structural = bool(pv_structural_diversity and key == "pv")
        _pick_from_yaml(key, count, structural=structural, sub_seed=sub_seed)

    if len(chosen) < total_size:
        pv_path = Path(paths["pv"])
        if pv_path.is_file():
            rest_pool = [e for e in load_cold_start_yaml(pv_path) if e.expr not in seen]
            need = total_size - len(chosen)
            extra = sample_structurally_diverse_entries(
                rest_pool,
                need,
                seed=(rng.randint(0, 2147483647) if seed is not None else None),
            )
            for e in extra:
                if e.expr in seen:
                    continue
                seen.add(e.expr)
                chosen.append(e)
                if len(chosen) >= total_size:
                    break

    rng.shuffle(chosen)
    result = entries_to_tuples(chosen[:total_size])
    structures = [norm_expr_structure(e[0]) for e in result]
    dup_struct = len(structures) - len(set(structures))
    if dup_struct > 0:
        print(f"[cold_start] warning: {dup_struct} duplicate structures in batch")
    else:
        print(f"[cold_start] structural diversity OK ({len(set(structures))} unique shapes)")
    return result


def validate_entries_factor_engine(
    entries: Sequence[ColdStartEntry],
) -> list[tuple[str, str]]:
    bad: list[tuple[str, str]] = []
    for e in entries:
        ok, msg = validate_factor_engine_dsl(e.expr)
        if not ok:
            bad.append((e.expr, msg))
    return bad
