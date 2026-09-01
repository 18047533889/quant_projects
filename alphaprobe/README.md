# alphaprobe

Deep-learning / reinforcement-learning factor mining framework for A-share
research. Runs a knowledge-pooled trainer over a cold-start factor library,
evaluates candidates through **factor_engine** (DSL), and delivers the
`candidate_pool` for downstream admission.

**Version:** 0.1.0 ｜ **Repo:** https://github.com/HKUST-QUANT-SOCIETY/alphaprobe (private)

> ⚠️ **Status: code complete, NOT yet run in production.** The known blocker is a
> missing cold-start library (`COLD_START_LIBRARY_SRC`). LLM quota must be
> configured via `.env` (DeepSeek). See platform handover doc for details.

## Layout

```
ashare/alphaprobe-dev/
├── src/alphaprobe/
│   ├── runner.py            # single-round mining: cold-start → train → deliver
│   ├── continuous.py        # 7×24 continuous mining loop (campaigns, resume, quota)
│   ├── cold_start/          # cold-start library loading (ALPHA101/158/191, V9+expand)
│   ├── trainer/             # AlphaKnowledgePool, AlphaKnowledgeTrainer, checkpoints
│   ├── delivery/            # DiskV1DeliveryExporter → candidate_pool (miner_delivery_spec)
│   └── fe_bridge/           # expression → FE DSL, FE evaluation, knowledge-graph patch
├── src/baselines/           # GAN, FQF/IQN/QRDQN, AlphaGFN, GPlearn
├── src/shared/              # alphagen / alphagen_qlib / utils / data_collection
├── apps/alphaprobe/run_continuous.py
├── configs/alphaprobe/      # experiment_ashare_pv_7x24.yaml (A-share 7×24)
├── configs/prompts/         # LLM prompts (generation / validity / separation / FE-DSL)
├── deploy/                  # run_continuous.sh + alphaprobe-continuous.service (systemd)
└── scripts/                 # FE prompt checks, delivery export
```

## Key entry points

```bash
# single round
PYTHONPATH=src python -m alphaprobe.runner --experiment_config configs/alphaprobe/experiment_ashare_pv_7x24.yaml
# continuous (7×24)
bash deploy/run_continuous.sh
```

- `enable_factor_engine_evaluation()` — patches the knowledge graph to evaluate
  mined expressions through factor_engine (DSL) instead of the legacy local
  calculator.
- `evaluate_via_factor_engine(expr, ...)` — metrics via FE + `calc_ic_ir`.
- Delivery writes per `factor_engine/docs/miner_delivery_spec.md`
  (campaign_id, formula, universe, domain, operator_policy, data_source local/cos).

## Data & labels

- Data: A-share via `data_access` (or local `lqtp_data`), `fe_profile:
  ashare_pv_valuation` (price-volume + pe/pb/turnover/market_cap).
- Label: **vwap-to-vwap** (`mining.label_price_pair: vwap_to_vwap`,
  `label_days: 20`). Split: train 2016–2021, valid 2022–2023, test 2024–2026-07.
- Operator policy: `lqtp_pv_daily`; forbidden fundamental tables (balance/income/cashflow).

## Requirements

Python 3.11, torch (CUDA), qlib, gymnasium, stable-baselines3, torchgfn,
transformers, openai (DeepSeek). See `pyproject.toml`. LLM keys in `.env`
(never committed).

## Related repos

- **factor_engine** — the DSL evaluation backend (fe_bridge)
- **data_access** — the data backend
- **factor_assets / factor_optimizer** — downstream admission / search of the
  delivered candidate pool
