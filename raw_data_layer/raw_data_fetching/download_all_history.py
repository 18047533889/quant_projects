from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from massive import RESTClient
from tqdm.auto import tqdm


class MassiveRestDownloader:
    """Unified downloader for Massive datasets (excluding 8K/10K by default)."""

    DATASETS = {
        # fundamentals
        "balance_sheet": {
            "category": "fundamentals",
            "source": "sdk",
            "m": "list_financials_balance_sheets",
            "sub": "balance_sheet",
            "p": "balance_sheet",
            "mode": "fiscal",
            "sort": "period_end.asc",
        },
        "income_statement": {
            "category": "fundamentals",
            "source": "sdk",
            "m": "list_financials_income_statements",
            "sub": "income_statement",
            "p": "income_statement",
            "mode": "fiscal",
            "sort": "period_end.asc",
        },
        "cash_flow_statement": {
            "category": "fundamentals",
            "source": "sdk",
            "m": "list_financials_cash_flow_statements",
            "sub": "cash_flow_statement",
            "p": "cash_flow_statement",
            "mode": "fiscal",
            "sort": "period_end.asc",
        },
        "financials_ratios": {
            "category": "fundamentals",
            "source": "sdk",
            "m": "list_financials_ratios",
            "sub": "financials_ratios",
            "p": "financials_ratios",
            "mode": "all",
            "sort": "ticker.asc",
        },
        "stocks_floats": {
            "category": "fundamentals",
            "source": "raw",
            "endpoint": "/stocks/vX/float",
            "sub": "stocks_floats",
            "p": "stocks_floats",
            "mode": "all",
            "sort": "ticker.asc",
        },
        "short_interest": {
            "category": "fundamentals",
            "source": "sdk",
            "m": "list_short_interest",
            "sub": "short_interest",
            "p": "short_interest",
            "mode": "date",
            "date_fields": ["settlement_date", "date", "as_of_date"],
            "sort": "settlement_date.asc,ticker.asc",
        },
        "short_volume": {
            "category": "fundamentals",
            "source": "sdk",
            "m": "list_short_volume",
            "sub": "short_volume",
            "p": "short_volume",
            "mode": "date",
            "date_fields": ["date", "settlement_date", "as_of_date"],
            "sort": "date.asc,ticker.asc",
        },
        # aggregate_bars
        "aggs_daily_market_summary": {
            "category": "aggregate_bars",
            "source": "raw",
            "endpoint_template": "/v2/aggs/grouped/locale/us/market/stocks/{date}",
            "sub": "daily_market_summary",
            "p": "daily_market_summary",
            "mode": "calendar_year",
            "start_year": 2004,
            "page_pause_s": 0.05,
            "include_otc": "false",
            "adjusted": "true",
        },
        # filing (8K/10K intentionally excluded)
        "filing_sec_edgar_index": {
            "category": "filing",
            "source": "raw",
            "endpoint": "/stocks/filings/vX/index",
            "sub": "sec_edgar_index",
            "p": "sec_edgar_index",
            "mode": "all",
            "sort": "filing_date.desc",
        },
        "filing_risk_categories": {
            "category": "filing",
            "source": "raw",
            "endpoint": "/stocks/taxonomies/vX/risk-factors",
            "sub": "risk_categories",
            "p": "risk_categories",
            "mode": "all",
            "sort": "taxonomy.desc",
        },
        "filing_risk_factors": {
            "category": "filing",
            "source": "auto",
            "m": "list_stocks_filings_risk_factors",
            "endpoint": "/stocks/filings/vX/risk-factors",
            "sub": "risk_factors",
            "p": "risk_factors",
            "mode": "date",
            "date_fields": ["filing_date"],
            "sort": "filing_date.desc",
        },
        # corporate_actions
        "corp_dividends": {
            "category": "corporate_actions",
            "source": "raw",
            "endpoint": "/stocks/v1/dividends",
            "sub": "dividends",
            "p": "dividends",
            "mode": "date",
            "date_fields": ["ex_dividend_date", "pay_date", "declaration_date"],
            "sort": "ex_dividend_date.desc",
        },
        "corp_splits": {
            "category": "corporate_actions",
            "source": "raw",
            "endpoint": "/stocks/v1/splits",
            "sub": "splits",
            "p": "splits",
            "mode": "date",
            "date_fields": ["execution_date"],
            "sort": "execution_date.desc",
        },
        "corp_ipos": {
            "category": "corporate_actions",
            "source": "raw",
            "endpoint": "/vX/reference/ipos",
            "sub": "ipos",
            "p": "ipos",
            "mode": "all",
            "max_limit": 1000,
        },
        # news
        "news_all": {
            "category": "news",
            "source": "raw",
            "endpoint": "/v2/reference/news",
            "sub": "news",
            "p": "news",
            "mode": "all",
        },
        # market_operations
        "market_holidays": {
            "category": "market_operations",
            "source": "raw",
            "endpoint": "/v1/marketstatus/upcoming",
            "sub": "market_holidays",
            "p": "market_holidays",
            "mode": "all",
            "no_limit": True,
            "sort": "date.asc",
        },
        "market_exchanges": {
            "category": "market_operations",
            "source": "raw",
            "endpoint": "/v3/reference/exchanges",
            "sub": "exchanges",
            "p": "exchanges",
            "mode": "all",
            "no_limit": True,
            "sort": "name.asc",
        },
        "market_condition_codes": {
            "category": "market_operations",
            "source": "raw",
            "endpoint": "/v3/reference/conditions",
            "sub": "condition_codes",
            "p": "condition_codes",
            "mode": "all",
            "no_limit": True,
        },
        # tickers
        "tickers_all": {
            "category": "tickers",
            "source": "raw",
            "endpoint": "/v3/reference/tickers",
            "sub": "all_tickers",
            "p": "all_tickers",
            "mode": "all",
            "max_limit": 1000,
        },
        "tickers_types": {
            "category": "tickers",
            "source": "raw",
            "endpoint": "/v3/reference/tickers/types",
            "sub": "ticker_types",
            "p": "ticker_types",
            "mode": "all",
            "no_limit": True,
            "sort": "code.asc",
        },
    }

    def __init__(
        self,
        client: RESTClient,
        root_dir: str,
        request_max_retries: int = 12,
        request_backoff_base: float = 1.7,
        request_backoff_cap: float = 90.0,
        request_jitter: float = 0.6,
        cursor_retry_floor_s: float = 6.0,
    ) -> None:
        self.client = client
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.request_max_retries = int(request_max_retries)
        self.request_backoff_base = float(request_backoff_base)
        self.request_backoff_cap = float(request_backoff_cap)
        self.request_jitter = float(request_jitter)
        self.cursor_retry_floor_s = float(cursor_retry_floor_s)

    @staticmethod
    def _marker_path(fp: Path) -> Path:
        return Path(str(fp) + ".ok")

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        s = str(exc).lower()
        retry_tokens = [
            "502",
            "503",
            "504",
            "429",
            "too many",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "connection reset",
            "max retries exceeded",
        ]
        return any(t in s for t in retry_tokens)

    def _raw_get_with_retry(self, path: str, params: dict) -> object:
        attempt = 0
        while True:
            try:
                return self.client._get(path=path, params=params, raw=True)
            except Exception as exc:
                attempt += 1
                if (not self._is_retryable_error(exc)) or (attempt > self.request_max_retries):
                    raise
                sleep_s = min(
                    self.request_backoff_cap,
                    self.request_backoff_base ** attempt + random.uniform(0.0, self.request_jitter),
                )
                if params and params.get("cursor"):
                    sleep_s = max(sleep_s, self.cursor_retry_floor_s + random.uniform(0.0, 1.0))
                print(
                    f"[retry] path={path} attempt={attempt}/{self.request_max_retries} "
                    f"sleep={sleep_s:.1f}s err={exc}"
                )
                time.sleep(sleep_s)

    def _iter_raw(self, endpoint: str, max_pages: int | None = None, page_pause_s: float = 0.0, **kwargs):
        path = endpoint
        params = dict(kwargs)
        page = 0
        while True:
            resp = self._raw_get_with_retry(path=path, params=params)
            payload = json.loads(resp.data.decode("utf-8"))
            page += 1

            if isinstance(payload, list):
                rows = payload
            else:
                rows = payload.get("results", [])
            if isinstance(rows, dict):
                rows = [rows]
            for row in rows:
                yield row

            if max_pages is not None and page >= int(max_pages):
                break

            next_url = payload.get("next_url") if isinstance(payload, dict) else None
            if not next_url:
                break

            if page_pause_s and float(page_pause_s) > 0:
                time.sleep(float(page_pause_s))

            parsed = urlparse(next_url)
            path = parsed.path
            params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

    def _iter(self, ds: str, max_pages: int | None = None, **kwargs):
        cfg = self.DATASETS[ds]
        source = cfg.get("source")

        if source == "raw":
            return self._iter_raw(
                cfg["endpoint"],
                max_pages=max_pages,
                page_pause_s=cfg.get("page_pause_s", 0.0),
                **kwargs,
            )

        method_name = cfg.get("m")
        if source == "sdk":
            if method_name and hasattr(self.client, method_name):
                return getattr(self.client, method_name)(**kwargs)
            raise AttributeError(f"RESTClient has no method: {method_name}")

        if source == "auto":
            if method_name and hasattr(self.client, method_name):
                return getattr(self.client, method_name)(**kwargs)
            return self._iter_raw(
                cfg["endpoint"],
                max_pages=max_pages,
                page_pause_s=cfg.get("page_pause_s", 0.0),
                **kwargs,
            )

        raise ValueError(f"Unknown source for dataset {ds}: {source}")

    @staticmethod
    def _to_dict(x: object) -> dict:
        if hasattr(x, "model_dump"):
            return x.model_dump()
        if hasattr(x, "__dict__"):
            return x.__dict__
        return dict(x)

    def _detect_date_field(self, ds: str) -> str:
        fields = self.DATASETS[ds].get("date_fields", [])
        first = next(self._iter(ds, limit=1, max_pages=1), None)
        if first is None:
            raise RuntimeError(f"No data returned for {ds}")

        rec = self._to_dict(first)
        for f_name in fields:
            if f_name in rec and rec.get(f_name) is not None:
                return f_name
        raise RuntimeError(f"Cannot find valid date field for {ds}. Candidates={fields}")

    def _partitions(self, ds: str):
        c = self.DATASETS[ds]

        if c["mode"] == "all":
            return ["all"], None

        if c["mode"] == "calendar_year":
            start_y = int(c.get("start_year", 2004))
            end_y = int(pd.Timestamp.today().year)
            return list(range(start_y, end_y + 1)), None

        if c["mode"] == "fiscal":
            a = self._to_dict(next(self._iter(ds, limit=1, sort="fiscal_year.asc,period_end.asc", max_pages=1), None))
            b = self._to_dict(next(self._iter(ds, limit=1, sort="fiscal_year.desc,period_end.desc", max_pages=1), None))
            return list(range(int(float(a["fiscal_year"])), int(float(b["fiscal_year"])) + 1)), None

        date_field = self._detect_date_field(ds)
        a = self._to_dict(next(self._iter(ds, limit=1, sort=f"{date_field}.asc", max_pages=1), None))
        b = self._to_dict(next(self._iter(ds, limit=1, sort=f"{date_field}.desc", max_pages=1), None))

        freq = c.get("partition_freq", "year")
        if freq == "month":
            start_month = pd.to_datetime(a[date_field]).to_period("M")
            end_month = pd.to_datetime(b[date_field]).to_period("M")
            months = pd.period_range(start=start_month, end=end_month, freq="M")
            return [str(m) for m in months], date_field

        ya = pd.to_datetime(a[date_field]).year
        yb = pd.to_datetime(b[date_field]).year
        return list(range(int(ya), int(yb) + 1)), date_field

    def _query(self, ds: str, part, limit: int | None, sort: str | None, date_field: str | None) -> dict:
        c = self.DATASETS[ds]
        eff_limit = None if c.get("no_limit", False) or limit is None else int(min(int(limit), int(c.get("max_limit", limit))))

        q = {"sort": sort or c.get("sort", "")}
        if eff_limit is not None:
            q["limit"] = eff_limit

        if c["mode"] == "fiscal":
            q["fiscal_year"] = int(part)

        elif c["mode"] == "date" and part != "all" and date_field is not None:
            if isinstance(part, str) and len(part) == 7 and part[4] == "-":
                start_ts = pd.Period(part, freq="M").start_time
                end_ts = (pd.Period(part, freq="M") + 1).start_time
                start_s = start_ts.strftime("%Y-%m-%d")
                end_s = end_ts.strftime("%Y-%m-%d")
            else:
                start_s = f"{part}-01-01"
                end_s = f"{int(part) + 1}-01-01"

            if c.get("source") == "raw":
                q[f"{date_field}.gte"] = start_s
                q[f"{date_field}.lt"] = end_s
            else:
                q[f"{date_field}_gte"] = start_s
                q[f"{date_field}_lt"] = end_s

        return {k: v for k, v in q.items() if v not in ("", None)}

    def _stream_write_df_chunks(self, fp: Path, records_iter, ds: str, chunk_size: int = 20000) -> int:
        rows = 0
        chunk = []
        writer = None
        base_cols = None

        if fp.exists():
            fp.unlink()

        def flush_chunk(records: list[dict]) -> None:
            nonlocal rows, writer, base_cols
            if not records:
                return

            df = pd.DataFrame(records)

            # Normalize known drifting columns to keep parquet schema stable across pages.
            if ds == "corp_dividends" and "declaration_date" in df.columns:
                df["declaration_date"] = df["declaration_date"].fillna("").astype(str)
            if ds == "filing_risk_factors" and "ticker" in df.columns:
                df["ticker"] = df["ticker"].fillna("").astype(str)
            if ds == "income_statement" and "extraordinary_items" in df.columns:
                df["extraordinary_items"] = pd.to_numeric(df["extraordinary_items"], errors="coerce")
            if ds == "income_statement" and "equity_in_affiliates" in df.columns:
                df["equity_in_affiliates"] = pd.to_numeric(df["equity_in_affiliates"], errors="coerce")
            if ds == "tickers_all":
                for c_name in ["currency_symbol", "base_currency_symbol", "base_currency_name"]:
                    if c_name in df.columns:
                        df[c_name] = df[c_name].fillna("").astype(str)
            if ds == "cash_flow_statement":
                for c_name in [
                    "income_loss_from_discontinued_operations",
                    "net_cash_from_financing_activities_discontinued_operations",
                    "net_cash_from_investing_activities_discontinued_operations",
                    "net_cash_from_operating_activities_discontinued_operations",
                    "other_cash_adjustments",
                ]:
                    if c_name in df.columns:
                        df[c_name] = pd.to_numeric(df[c_name], errors="coerce")
            if ds == "aggs_daily_market_summary" and "n" in df.columns:
                df["n"] = pd.to_numeric(df["n"], errors="coerce").astype("float64")

            if base_cols is None:
                base_cols = list(df.columns)
            else:
                for c_name in base_cols:
                    if c_name not in df.columns:
                        df[c_name] = None
                extra_cols = [x for x in df.columns if x not in base_cols]
                if extra_cols:
                    df = df.drop(columns=extra_cols)
                df = df.reindex(columns=base_cols)

            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(fp), table.schema, compression="snappy")
            writer.write_table(table)
            rows += len(df)

        try:
            for rec in records_iter:
                chunk.append(self._to_dict(rec))
                if len(chunk) >= int(chunk_size):
                    flush_chunk(chunk)
                    chunk = []
            if chunk:
                flush_chunk(chunk)
        finally:
            if writer is not None:
                writer.close()

        return rows

    def _stream_write_calendar_year(self, ds: str, year: int, fp: Path, chunk_size: int = 20000) -> int:
        cfg = self.DATASETS[ds]
        year_i = int(year)
        start = pd.Timestamp(year=year_i, month=1, day=1)
        end = pd.Timestamp.today().normalize() if year_i == int(pd.Timestamp.today().year) else pd.Timestamp(year=year_i, month=12, day=31)
        biz_days = pd.bdate_range(start=start, end=end)

        def records():
            pbar = tqdm(biz_days, desc=f"{ds}:{year_i}", unit="day", leave=False)
            for d in pbar:
                date_s = d.strftime("%Y-%m-%d")
                path = cfg["endpoint_template"].format(date=date_s)
                params = {"adjusted": cfg.get("adjusted", "true"), "include_otc": cfg.get("include_otc", "false")}
                payload = json.loads(self._raw_get_with_retry(path=path, params=params).data.decode("utf-8"))
                for rec in payload.get("results", []) or []:
                    row = rec if isinstance(rec, dict) else self._to_dict(rec)
                    row["trade_date"] = date_s
                    yield row
                pause_s = float(cfg.get("page_pause_s", 0.0))
                if pause_s > 0:
                    time.sleep(pause_s)

        return self._stream_write_df_chunks(fp, records(), ds=ds, chunk_size=chunk_size)

    def _stream_write_partition(self, ds: str, query: dict, fp: Path, chunk_size: int = 20000, max_pages: int | None = None) -> int:
        cfg = self.DATASETS[ds]
        if cfg.get("source") == "sdk":
            max_rows = None
            if max_pages is not None:
                max_rows = int(max_pages) * int(query.get("limit", 0))

            def sdk_records():
                rows_seen = 0
                for rec in self._iter(ds, **query):
                    yield rec
                    rows_seen += 1
                    if max_rows is not None and rows_seen >= max_rows:
                        break

            return self._stream_write_df_chunks(fp, sdk_records(), ds=ds, chunk_size=chunk_size)

        return self._stream_write_df_chunks(fp, self._iter(ds, max_pages=max_pages, **query), ds=ds, chunk_size=chunk_size)

    def download_dataset(
        self,
        ds: str,
        years=None,
        workers: int = 8,
        limit: int = 50000,
        sort: str | None = None,
        skip_existing: bool = True,
        delete_empty: bool = True,
        max_pages: int | None = None,
        chunk_size: int = 20000,
        emit_logs: bool = True,
    ) -> pd.DataFrame:
        c = self.DATASETS[ds]
        out = self.root / c["category"] / c["sub"]
        out.mkdir(parents=True, exist_ok=True)

        parts, date_field = self._partitions(ds)
        if years is not None and c["mode"] in ("fiscal", "date", "calendar_year") and parts != ["all"]:
            parts = sorted(list(years))

        def one(part):
            suffix = part if part != "all" else "all"
            fp = out / f"{c['p']}_{suffix}.parquet"
            marker_fp = self._marker_path(fp)

            if skip_existing and marker_fp.exists():
                return {
                    "dataset": ds,
                    "category": c["category"],
                    "partition": suffix,
                    "rows": None,
                    "status": "skipped",
                    "file": str(fp),
                    "marker": str(marker_fp),
                }

            if marker_fp.exists():
                marker_fp.unlink()

            if c["mode"] == "calendar_year":
                rows_local = self._stream_write_calendar_year(ds=ds, year=part, fp=fp, chunk_size=chunk_size)
            else:
                query = self._query(ds, part, limit, sort, date_field)
                rows_local = self._stream_write_partition(ds=ds, query=query, fp=fp, chunk_size=chunk_size, max_pages=max_pages)

            if rows_local == 0:
                if delete_empty and fp.exists():
                    fp.unlink()
                if marker_fp.exists():
                    marker_fp.unlink()
                return {
                    "dataset": ds,
                    "category": c["category"],
                    "partition": suffix,
                    "rows": 0,
                    "status": "empty",
                    "file": str(fp),
                    "marker": str(marker_fp),
                }

            marker_payload = {
                "dataset": ds,
                "partition": str(suffix),
                "rows": int(rows_local),
                "updated_at": pd.Timestamp.utcnow().isoformat(),
            }
            marker_fp.write_text(json.dumps(marker_payload, ensure_ascii=False), encoding="utf-8")
            return {
                "dataset": ds,
                "category": c["category"],
                "partition": suffix,
                "rows": rows_local,
                "status": "ok",
                "file": str(fp),
                "marker": str(marker_fp),
            }

        res = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fs = [ex.submit(one, p) for p in parts]
            pbar = tqdm(total=len(fs), desc=f"{c['category']}:{ds}", unit="part", disable=not emit_logs)
            for f in as_completed(fs):
                x = f.result()
                res.append(x)
                pbar.update(1)
                if emit_logs:
                    print(f"[{x['category']}][{ds}][{x['partition']}] {x['status']} rows={x['rows']}")
            pbar.close()

        return pd.DataFrame(res).sort_values("partition").reset_index(drop=True)

    def download_all(
        self,
        datasets=None,
        years=None,
        dataset_workers: int = 4,
        partition_workers: int = 8,
        limit: int = 50000,
        sort: str | None = None,
        skip_existing: bool = True,
        delete_empty: bool = True,
        max_pages: int | None = None,
        chunk_size: int = 20000,
        emit_logs: bool = True,
    ):
        dss = datasets or list(self.DATASETS.keys())
        out = {}

        with ThreadPoolExecutor(max_workers=dataset_workers) as ex:
            mp = {
                ex.submit(
                    self.download_dataset,
                    ds,
                    years,
                    partition_workers,
                    limit,
                    sort,
                    skip_existing,
                    delete_empty,
                    max_pages,
                    chunk_size,
                    emit_logs,
                ): ds
                for ds in dss
            }
            for f in as_completed(mp):
                ds = mp[f]
                try:
                    out[ds] = f.result()
                except Exception as e:
                    cfg = self.DATASETS[ds]
                    out[ds] = pd.DataFrame(
                        [
                            {
                                "dataset": ds,
                                "category": cfg["category"],
                                "partition": "all",
                                "rows": None,
                                "status": "failed",
                                "error": str(e),
                            }
                        ]
                    )
                    if emit_logs:
                        print(f"[{cfg['category']}][{ds}] failed: {e}")

        all_summary = pd.concat([out[k] for k in out], ignore_index=True)
        return out, all_summary


DEFAULT_DATASETS = [
    "balance_sheet",
    "income_statement",
    "cash_flow_statement",
    "financials_ratios",
    "stocks_floats",
    "short_interest",
    "short_volume",
    "aggs_daily_market_summary",
    "filing_sec_edgar_index",
    "filing_risk_categories",
    "filing_risk_factors",
    "corp_dividends",
    "corp_splits",
    "corp_ipos",
    "news_all",
    "market_holidays",
    "market_exchanges",
    "market_condition_codes",
    "tickers_all",
    "tickers_types",
]


def _status_counts(all_summary: pd.DataFrame) -> dict[str, int]:
    if all_summary.empty or "status" not in all_summary.columns:
        return {}
    counts = all_summary["status"].value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def run_download_all_history(
    *,
    api_key: str,
    root_dir: str,
    dataset_workers: int = 8,
    partition_workers: int = 12,
    limit: int = 5000,
    chunk_size: int = 10000,
    max_pages: int | None = None,
    skip_existing: bool = True,
    results_csv: str = "download_results.csv",
    datasets: Sequence[str] | None = None,
    emit_logs: bool = True,
) -> dict[str, Any]:
    if not api_key:
        raise ValueError("Missing API key. Set MASSIVE_API_KEY or pass --api-key.")

    client = RESTClient(api_key)
    downloader = MassiveRestDownloader(
        client,
        root_dir=root_dir,
        request_max_retries=12,
        request_backoff_base=1.7,
        request_backoff_cap=90.0,
        request_jitter=0.6,
        cursor_retry_floor_s=6.0,
    )

    datasets_to_download = list(datasets or DEFAULT_DATASETS)
    _, all_summary = downloader.download_all(
        datasets=datasets_to_download,
        years=None,
        dataset_workers=dataset_workers,
        partition_workers=partition_workers,
        limit=limit,
        skip_existing=skip_existing,
        delete_empty=True,
        max_pages=max_pages,
        chunk_size=chunk_size,
        emit_logs=emit_logs,
    )

    all_summary = all_summary.sort_values(["category", "dataset", "partition"]).reset_index(drop=True)
    results_csv_path = Path(results_csv).resolve()
    results_csv_path.parent.mkdir(parents=True, exist_ok=True)
    all_summary.to_csv(results_csv_path, index=False)

    return {
        "command": "download-all-history",
        "root_dir": str(Path(root_dir).resolve()),
        "results_csv": str(results_csv_path),
        "datasets": datasets_to_download,
        "summary_rows": int(len(all_summary)),
        "status_counts": _status_counts(all_summary),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download Massive historical datasets (excluding filing 8K/10K).")
    p.add_argument("--api-key", default=os.getenv("MASSIVE_API_KEY"), help="Massive API key. Default reads MASSIVE_API_KEY env.")
    p.add_argument("--root-dir", default="/home/yluel/share/projects/massive_parquet", help="Output root directory for parquet files.")
    p.add_argument("--dataset-workers", type=int, default=8, help="Parallel workers across datasets.")
    p.add_argument("--partition-workers", type=int, default=12, help="Parallel workers within one dataset.")
    p.add_argument("--limit", type=int, default=5000, help="Page size limit for pageable endpoints.")
    p.add_argument("--chunk-size", type=int, default=10000, help="Rows per parquet write chunk.")
    p.add_argument("--max-pages", type=int, default=None, help="Optional page cap (debug). Default full crawl.")
    p.add_argument("--skip-existing", action="store_true", default=True, help="Skip partitions that already have .ok marker.")
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing", help="Force re-download existing partitions.")
    p.add_argument("--results-csv", default="download_results.csv", help="Output summary csv path.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = run_download_all_history(
            api_key=args.api_key,
            root_dir=args.root_dir,
            dataset_workers=args.dataset_workers,
            partition_workers=args.partition_workers,
            limit=args.limit,
            chunk_size=args.chunk_size,
            max_pages=args.max_pages,
            skip_existing=args.skip_existing,
            results_csv=args.results_csv,
            emit_logs=True,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Done. Summary rows={result['summary_rows']}")
    if result["status_counts"]:
        print(pd.Series(result["status_counts"]).to_string())
    print(f"Saved summary to: {Path(args.results_csv).resolve()}")


if __name__ == "__main__":
    main()