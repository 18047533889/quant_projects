"""从 experiment yaml 加载配置（对齐 miner_delivery_spec.md / CogAlpha split）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# spec §1.3 domain_slug ↔ domain_root
DOMAIN_ROOT_TO_SLUG: dict[str, str] = {
    "price_volume": "pv",
    "fundamental": "fundamental",
    "alternative": "alt",
    "microstructure": "micro",
    "mixed": "mixed",
}

# 与 CogAlpha ashare baseline / 投递规范对齐（test 截止随本地数据延伸到 2026-07-31）
DEFAULT_MINING_CONFIG: dict[str, list[str]] = {
    "train_period": ["2016-01-01", "2021-12-31"],
    "valid_period": ["2022-01-01", "2023-12-31"],
    "test_period": ["2024-01-01", "2026-07-31"],
}

DEFAULT_MINING_SCOPE: dict[str, list[str]] = {
    "primary_tables": ["StockDailyBar", "StockCapitalDaily"],
    "auxiliary_tables": ["StockList", "StockStatus", "StockIndustry", "Calendar"],
    "forbidden_tables": ["StockBalance", "StockIncome", "StockCashFlow"],
}

DEFAULT_DATA_SOURCE: dict[str, str] = {
    "local": "data/a_share/lqtp_data/",
    "cos": "cos://qs-cold/clean_data/ashare/lqtp_data/",
}


def factor_engine_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "factor_engine"
        if (candidate / "api").is_dir():
            return candidate
    raise FileNotFoundError("无法定位 quant_projects/factor_engine")


def _expand(path: str) -> str:
    return str(Path(path).expanduser())


def _mining_config_from_raw(
    raw: dict[str, Any],
    config_path: Path | None = None,
) -> dict[str, list[str]]:
    """split（CogAlpha）或 dataset.segments（QuantaAlpha）→ mining_config。"""
    split = raw.get("split") or {}
    if split.get("train_start") and split.get("train_end"):
        return {
            "train_period": [str(split["train_start"]), str(split["train_end"])],
            "valid_period": [str(split["valid_start"]), str(split["valid_end"])],
            "test_period": [str(split["test_start"]), str(split["test_end"])],
        }

    segments = (raw.get("dataset") or {}).get("segments") or {}
    if segments.get("train") and segments.get("valid") and segments.get("test"):
        return {
            "train_period": list(segments["train"]),
            "valid_period": list(segments["valid"]),
            "test_period": list(segments["test"]),
        }

    # backtest_config_path → QuantaAlpha segments
    bt_rel = raw.get("backtest_config_path") or (raw.get("delivery") or {}).get(
        "backtest_config_path"
    )
    if bt_rel and config_path is not None:
        bt_path = Path(str(bt_rel)).expanduser()
        if not bt_path.is_absolute():
            bt_path = (config_path.parent / bt_path).resolve()
        if bt_path.is_file():
            bt_raw = yaml.safe_load(bt_path.read_text(encoding="utf-8")) or {}
            seg = (bt_raw.get("dataset") or {}).get("segments") or {}
            if seg.get("train") and seg.get("valid") and seg.get("test"):
                return {
                    "train_period": list(seg["train"]),
                    "valid_period": list(seg["valid"]),
                    "test_period": list(seg["test"]),
                }

    # 兼容旧版 delivery.mining_config
    legacy = (raw.get("delivery") or {}).get("mining_config")
    if legacy:
        return {
            "train_period": list(
                legacy.get("train_period", DEFAULT_MINING_CONFIG["train_period"])
            ),
            "valid_period": list(
                legacy.get("valid_period", DEFAULT_MINING_CONFIG["valid_period"])
            ),
            "test_period": list(
                legacy.get("test_period", DEFAULT_MINING_CONFIG["test_period"])
            ),
        }
    return dict(DEFAULT_MINING_CONFIG)


@dataclass
class DeliverySettings:
    """disk.v1 投递块（与 spec §0.5 CogAlpha delivery 同构）。"""

    campaign_id: str
    generator_name: str = "alphaprobe"
    generator_version: str = "0.1.0"
    mined_by: str = "sunhaiwei"
    market: str = "ashare"
    universe_id: str = "A_SHARE_ALL_A_EX_ST"
    domain_root: str = "price_volume"
    domain: str = "price_volume"
    frequency_bucket: str = "daily"
    signal_structure: str = "cross_sectional"
    asset_class: str = "equity"
    operator_policy: str = "lqtp_pv_daily"
    output_dir: str = "~/quant_projects/data/factor_pools/candidate_pool"
    mining_scope: dict[str, Any] = field(default_factory=dict)
    data_source: dict[str, str] = field(default_factory=dict)
    mining_config: dict[str, list[str]] = field(default_factory=dict)
    expression_type: str = "dsl"
    ic_export_threshold: float = 0.006
    # 运行时从环境变量注入，不进 yaml
    llm_model: str | None = None
    llm_provider: str | None = None
    llm_base_url: str | None = None
    filter_note: str | None = None

    def resolved_campaign_id(self) -> str:
        text = str(self.campaign_id or "").strip()
        if text:
            return text
        slug = DOMAIN_ROOT_TO_SLUG.get(self.domain_root, "pv")
        stamp = datetime.now().strftime("%Y%m%d%H%M")
        return f"{self.generator_name}_{self.market}_{slug}_{stamp}"

    def campaign_created_at(self) -> str:
        cid = self.resolved_campaign_id()
        suffix = cid.rsplit("_", 1)[-1]
        if len(suffix) == 12 and suffix.isdigit():
            dt = datetime.strptime(suffix, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:00Z")
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def output_campaign_dir(self) -> Path:
        return Path(_expand(self.output_dir)) / self.resolved_campaign_id()


@dataclass
class DataSettings:
    """数据路径：ashare_data_dir 相对 quant_projects 根。"""

    source: str = "ashare"
    ashare_data_dir: str = "data/a_share/lqtp_data"
    max_files: int | None = None
    instruments: str = "csi300"
    # ashare_pv | ashare_pv_valuation（默认估值 composite，可用 pe/pb/turnover）
    fe_profile: str = "ashare_pv_valuation"

    def lqtp_root_path(self) -> Path:
        root = factor_engine_root().parent
        rel = Path(self.ashare_data_dir)
        if rel.is_absolute():
            return rel.expanduser().resolve()
        return (root / rel).resolve()

    def bar_root_path(self) -> Path:
        return self.lqtp_root_path() / "StockDailyBar"


@dataclass
class MiningSettings:
    """算法超参（不进 disk.v1 mining_config）。"""

    label_days: int = 20
    search_time: int = 20
    pool_capacity: int = 50
    ic_threshold: float = 0.006
    ic_export_threshold: float = 0.006
    max_files: int | None = None


@dataclass
class ExperimentConfig:
    delivery: DeliverySettings
    data: DataSettings
    mining: MiningSettings
    raw: dict[str, Any]
    has_delivery_block: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        config_path = Path(path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        delivery_raw = dict(raw.get("delivery") or {})
        data_raw = dict(raw.get("data") or {})
        mining_raw = dict(raw.get("mining") or {})

        mining_config = _mining_config_from_raw(raw, config_path)

        if not delivery_raw:
            raise ValueError(
                f"缺少 delivery 块（见 miner_delivery_spec.md §0.5 CogAlpha delivery 示例）: {path}"
            )

        campaign_id_raw = delivery_raw.get("campaign_id")
        continuous_enabled = bool((raw.get("continuous") or {}).get("enabled", False))
        if campaign_id_raw is None:
            raise ValueError(
                "delivery.campaign_id 必填（7×24 连续模式可设为空字符串，由每轮自动生成）"
            )
        campaign_id = str(campaign_id_raw).strip()
        if not campaign_id and not continuous_enabled:
            raise ValueError(
                "delivery.campaign_id 不能为空；单轮挖掘请填写 "
                "alphaprobe_ashare_pv_YYYYMMDDHHMM，或启用 continuous 块"
            )

        delivery = DeliverySettings(
            campaign_id=campaign_id,
            generator_name=str(delivery_raw.get("generator_name", "alphaprobe")),
            generator_version=str(delivery_raw.get("generator_version", "0.1.0")),
            mined_by=str(delivery_raw.get("mined_by", "sunhaiwei")),
            market=str(delivery_raw.get("market", "ashare")),
            universe_id=str(delivery_raw.get("universe_id", "A_SHARE_ALL_A_EX_ST")),
            domain_root=str(delivery_raw.get("domain_root", "price_volume")),
            domain=str(
                delivery_raw.get("domain", delivery_raw.get("domain_root", "price_volume"))
            ),
            frequency_bucket=str(delivery_raw.get("frequency_bucket", "daily")),
            signal_structure=str(delivery_raw.get("signal_structure", "cross_sectional")),
            asset_class=str(delivery_raw.get("asset_class", "equity")),
            operator_policy=str(delivery_raw.get("operator_policy", "lqtp_pv_daily")),
            output_dir=str(
                delivery_raw.get(
                    "output_dir",
                    "~/quant_projects/data/factor_pools/candidate_pool",
                )
            ),
            mining_scope=dict(delivery_raw.get("mining_scope") or DEFAULT_MINING_SCOPE),
            data_source=dict(delivery_raw.get("data_source") or DEFAULT_DATA_SOURCE),
            mining_config=mining_config,
            expression_type=str(delivery_raw.get("expression_type", "dsl")),
            ic_export_threshold=float(
                mining_raw.get(
                    "ic_export_threshold",
                    delivery_raw.get("ic_export_threshold", 0.006),
                )
            ),
        )

        data = DataSettings(
            source=str(data_raw.get("source", "ashare")),
            ashare_data_dir=str(data_raw.get("ashare_data_dir", "data/a_share/lqtp_data")),
            max_files=mining_raw.get("max_files", data_raw.get("max_files")),
            instruments=str(data_raw.get("instruments", "csi300")),
            fe_profile=str(data_raw.get("fe_profile", "ashare_pv_valuation")),
        )

        mining = MiningSettings(
            label_days=int(mining_raw.get("label_days", 20)),
            search_time=int(mining_raw.get("search_time", 20)),
            pool_capacity=int(mining_raw.get("pool_capacity", 50)),
            ic_threshold=float(mining_raw.get("ic_threshold", 0.006)),
            ic_export_threshold=float(mining_raw.get("ic_export_threshold", 0.006)),
            max_files=mining_raw.get("max_files", data_raw.get("max_files")),
        )
        if mining.max_files is not None:
            data.max_files = mining.max_files

        return cls(
            delivery=delivery,
            data=data,
            mining=mining,
            raw=raw,
            has_delivery_block=True,
        )

    def factor_engine_data_source_config(
        self,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """A 股数据源：默认价量+估值 composite（COS 镜像 StockDailyBar + StockValuationDaily）。"""
        from alphaprobe.fe_bridge.paths import ensure_factor_engine_importable

        ensure_factor_engine_importable()
        profile = str(
            (self.raw.get("data") or {}).get("fe_profile")
            or getattr(self.data, "fe_profile", None)
            or "ashare_pv_valuation"
        ).strip()
        if profile in {"ashare_pv", "pv"}:
            from factor_engine.api.mining_integration import default_ashare_pv_data_source_config

            return default_ashare_pv_data_source_config(
                max_files=self.data.max_files,
                start_date=start_date,
                end_date=end_date,
            )
        from factor_engine.api.mining_integration import default_ashare_pv_valuation_data_source_config

        # valuation composite 不支持 max_files 限流；smoke 时请设 data.fe_profile=ashare_pv
        return default_ashare_pv_valuation_data_source_config(
            start_date=start_date,
            end_date=end_date,
        )
