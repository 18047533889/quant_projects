"""
语义去重器模块
"""

import hashlib
import json
import re
from typing import Dict, Any, Optional, Tuple, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import GatewayConfig


@dataclass
class DuplicateRecord:
    """重复因子记录"""
    candidate_id: str
    expr_hash: str
    candidate_hash: str
    checked_at: str
    historical_report_id: str
    similarity_score: float = 1.0


@dataclass
class DeduplicationResult:
    """去重结果"""
    is_duplicate: bool
    historical_report_id: Optional[str]
    similarity_score: float
    match_type: str
    matched_candidate_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_duplicate": self.is_duplicate,
            "historical_report_id": self.historical_report_id,
            "similarity_score": self.similarity_score,
            "match_type": self.match_type,
            "matched_candidate_id": self.matched_candidate_id,
        }


class ExpressionNormalizer:
    """表达式标准化器"""

    def __init__(self):
        self.space_pattern = re.compile(r'\s+')

    def normalize(self, expr: str) -> str:
        if not expr:
            return ""
        normalized = self.space_pattern.sub('', expr)
        return normalized.lower()

    def normalize_with_config(self, expr: str, config: Dict[str, Any]) -> str:
        normalized_expr = self.normalize(expr)
        config_str = json.dumps(config, sort_keys=True)
        return f"{normalized_expr}|{config_str}"


class HashDeduplicator:
    """哈希去重器"""

    def __init__(self, cache_size: int = 10000):
        self.cache_size = cache_size
        self._hash_cache: Dict[str, DuplicateRecord] = {}
        self._candidate_cache: Dict[str, DuplicateRecord] = {}
        self._access_order: List[str] = []
        self.persist_path: Optional[Path] = None

    def compute_expr_hash(self, expr: str) -> str:
        normalized = self._normalize_expression(expr)
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def compute_candidate_hash(self, expr: str, config: Dict[str, Any]) -> str:
        normalized_expr = self._normalize_expression(expr)
        normalized_config = json.dumps(config, sort_keys=True)
        combined = f"{normalized_expr}|{normalized_config}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()

    def _normalize_expression(self, expr: str) -> str:
        if not expr:
            return ""
        normalized = re.sub(r'\s+', '', expr)
        return normalized.lower()

    def add_record(self, record: DuplicateRecord) -> None:
        if len(self._hash_cache) >= self.cache_size:
            oldest = self._access_order.pop(0)
            if oldest in self._hash_cache:
                del self._hash_cache[oldest]
            to_remove = []
            for cid, r in self._candidate_cache.items():
                if r.expr_hash == oldest:
                    to_remove.append(cid)
            for cid in to_remove:
                del self._candidate_cache[cid]

        self._hash_cache[record.expr_hash] = record
        self._candidate_cache[record.candidate_hash] = record
        self._access_order.append(record.expr_hash)

    def check_expr_hash(self, expr_hash: str) -> Optional[DuplicateRecord]:
        return self._hash_cache.get(expr_hash)

    def check_candidate_hash(self, candidate_hash: str) -> Optional[DuplicateRecord]:
        return self._candidate_cache.get(candidate_hash)

    def load_from_persist(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                record = DuplicateRecord(
                    candidate_id=item['candidate_id'],
                    expr_hash=item['expr_hash'],
                    candidate_hash=item['candidate_hash'],
                    checked_at=item['checked_at'],
                    historical_report_id=item['historical_report_id'],
                    similarity_score=item.get('similarity_score', 1.0)
                )
                self.add_record(record)
        except Exception:
            pass

    def save_to_persist(self, path: Path) -> None:
        data = [
            {
                'candidate_id': r.candidate_id,
                'expr_hash': r.expr_hash,
                'candidate_hash': r.candidate_hash,
                'checked_at': r.checked_at,
                'historical_report_id': r.historical_report_id,
                'similarity_score': r.similarity_score,
            }
            for r in self._hash_cache.values()
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def clear(self) -> None:
        self._hash_cache.clear()
        self._candidate_cache.clear()
        self._access_order.clear()

    def size(self) -> int:
        return len(self._hash_cache)


class SemanticDeduplicator:
    """语义去重器（主类）"""

    def __init__(self, config: GatewayConfig, use_vector_db: bool = False):
        self.config = config
        self.use_vector_db = use_vector_db
        self.threshold = config.semantic_similarity_threshold
        self.hash_dedup = HashDeduplicator()
        self.normalizer = ExpressionNormalizer()
        self.vector_db = None
        if use_vector_db:
            self._init_vector_db()

    def _init_vector_db(self) -> None:
        pass

    def is_duplicate(
        self,
        expr: str,
        config_dict: Dict[str, Any],
        candidate_id: Optional[str] = None
    ) -> DeduplicationResult:
        expr_hash = self.hash_dedup.compute_expr_hash(expr)
        candidate_hash = self.hash_dedup.compute_candidate_hash(expr, config_dict)

        existing = self.hash_dedup.check_candidate_hash(candidate_hash)
        if existing:
            return DeduplicationResult(
                is_duplicate=True,
                historical_report_id=existing.historical_report_id,
                similarity_score=existing.similarity_score,
                match_type="hash",
                matched_candidate_id=existing.candidate_id
            )

        existing_expr = self.hash_dedup.check_expr_hash(expr_hash)
        if existing_expr:
            if self._is_config_equivalent(config_dict, existing_expr):
                return DeduplicationResult(
                    is_duplicate=True,
                    historical_report_id=existing_expr.historical_report_id,
                    similarity_score=existing_expr.similarity_score,
                    match_type="hash_expr",
                    matched_candidate_id=existing_expr.candidate_id
                )

        if self.use_vector_db and self.vector_db:
            semantic_result = self._check_semantic_similarity(expr)
            if semantic_result and semantic_result.similarity_score >= self.threshold:
                return semantic_result

        return DeduplicationResult(
            is_duplicate=False,
            historical_report_id=None,
            similarity_score=0.0,
            match_type="none"
        )

    def _is_config_equivalent(self, config1: Dict, existing_record: DuplicateRecord) -> bool:
        return False

    def _check_semantic_similarity(self, expr: str) -> Optional[DeduplicationResult]:
        """检查语义相似度"""
        # 预留接口，暂时返回 None
        return None

    def register_factor(
        self,
        candidate_id: str,
        expr: str,
        config_dict: Dict[str, Any],
        report_id: str
    ) -> None:
        expr_hash = self.hash_dedup.compute_expr_hash(expr)
        candidate_hash = self.hash_dedup.compute_candidate_hash(expr, config_dict)

        record = DuplicateRecord(
            candidate_id=candidate_id,
            expr_hash=expr_hash,
            candidate_hash=candidate_hash,
            checked_at=datetime.now(timezone.utc).isoformat(),
            historical_report_id=report_id,
            similarity_score=1.0
        )
        self.hash_dedup.add_record(record)

        if self.use_vector_db and self.vector_db:
            self._add_to_vector_db(expr, report_id)

    def _add_to_vector_db(self, expr: str, report_id: str) -> None:
        pass

    def load_history(self, path: Path) -> None:
        self.hash_dedup.load_from_persist(path)

    def save_history(self, path: Path) -> None:
        self.hash_dedup.save_to_persist(path)

    def clear_cache(self) -> None:
        self.hash_dedup.clear()

    def get_cache_size(self) -> int:
        return self.hash_dedup.size()


def normalize_expression(expr: str) -> str:
    normalizer = ExpressionNormalizer()
    return normalizer.normalize(expr)


def compute_expression_hash(expr: str) -> str:
    dedup = HashDeduplicator()
    return dedup.compute_expr_hash(expr)


def compute_candidate_hash(expr: str, config: Dict[str, Any]) -> str:
    dedup = HashDeduplicator()
    return dedup.compute_candidate_hash(expr, config)