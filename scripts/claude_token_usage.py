#!/usr/bin/env python3
"""Claude Code token usage tracker — aggregates across all sessions.

Scans .claude/ for session JSONL files, extracts usage records,
and produces a summary with timestamps, model, input/output tokens,
cache stats, and estimated cost.

Usage:
    python3 scripts/claude_token_usage.py [--sessions-dir DIR] [--since DATE]
"""
import json
import sys
import os
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# Pricing per million tokens (approximate, as of 2026-08)
PRICING = {
    "claude-opus-5": {"input": 15.0, "output": 75.0, "cache_read": 1.5, "cache_write": 18.75},
    "claude-sonnet-5": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.0, "cache_read": 0.08, "cache_write": 1.0},
    "glm-5.3": {"input": 0.50, "output": 2.0, "cache_read": 0.05, "cache_write": 0.60},
    "default": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_write": 3.75},
}

def get_pricing(model: str) -> dict:
    model_lower = model.lower()
    for key, prices in PRICING.items():
        if key in model_lower:
            return prices
    return PRICING["default"]

def extract_usage_from_session(filepath: Path, since: str = None) -> list:
    """Extract usage records from a Claude session JSONL file."""
    records = []
    session_start = None
    session_end = None

    try:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                ts = entry.get("timestamp")
                if ts and session_start is None:
                    session_start = ts
                if ts:
                    session_end = ts

                # Look for usage data in assistant messages
                msg = entry.get("message", {})
                if msg.get("role") == "assistant" and "usage" in msg:
                    usage = msg["usage"]
                    model = msg.get("model", "unknown")

                    # Filter by date if specified
                    if since and ts and ts[:10] < since:
                        continue

                    records.append({
                        "timestamp": ts,
                        "model": model,
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
                        "cache_write_tokens": usage.get("cache_creation_input_tokens", 0),
                        "session_file": filepath.name,
                    })
    except Exception as e:
        print(f"  Warning: Could not read {filepath}: {e}", file=sys.stderr)

    return records, session_start, session_end

def format_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Claude Code token usage tracker")
    parser.add_argument("--sessions-dir", default=os.path.expanduser("~/.claude/projects/-home-shw"),
                        help="Directory containing session JSONL files")
    parser.add_argument("--since", default=None, help="Only include sessions since DATE (YYYY-MM-DD)")
    parser.add_argument("--archives-dir", default=os.path.expanduser("~/.claude/archives/sessions"),
                        help="Directory containing archived session files")
    args = parser.parse_args()

    sessions_dir = Path(args.sessions_dir)
    archives_dir = Path(args.archives_dir)

    all_records = []
    session_summaries = []

    # Scan current sessions
    for jsonl_file in sorted(sessions_dir.glob("*.jsonl")):
        records, start, end = extract_usage_from_session(jsonl_file, args.since)
        if records:
            all_records.extend(records)
            total_in = sum(r["input_tokens"] for r in records)
            total_out = sum(r["output_tokens"] for r in records)
            total_cache_read = sum(r["cache_read_tokens"] for r in records)
            total_cache_write = sum(r["cache_write_tokens"] for r in records)
            models = set(r["model"] for r in records)
            session_summaries.append({
                "file": jsonl_file.name,
                "start": start,
                "end": end,
                "messages": len(records),
                "input_tokens": total_in,
                "output_tokens": total_out,
                "cache_read": total_cache_read,
                "cache_write": total_cache_write,
                "models": models,
            })

    # Scan archived sessions
    if archives_dir.exists():
        for jsonl_file in sorted(archives_dir.glob("*.jsonl")):
            records, start, end = extract_usage_from_session(jsonl_file, args.since)
            if records:
                all_records.extend(records)
                total_in = sum(r["input_tokens"] for r in records)
                total_out = sum(r["output_tokens"] for r in records)
                total_cache_read = sum(r["cache_read_tokens"] for r in records)
                total_cache_write = sum(r["cache_write_tokens"] for r in records)
                models = set(r["model"] for r in records)
                session_summaries.append({
                    "file": f"archive:{jsonl_file.name}",
                    "start": start,
                    "end": end,
                    "messages": len(records),
                    "input_tokens": total_in,
                    "output_tokens": total_out,
                    "cache_read": total_cache_read,
                    "cache_write": total_cache_write,
                    "models": models,
                })

    if not all_records:
        print("No usage records found.")
        return

    # Aggregate by model
    by_model = defaultdict(lambda: {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "count": 0})
    for r in all_records:
        m = r["model"]
        by_model[m]["input"] += r["input_tokens"]
        by_model[m]["output"] += r["output_tokens"]
        by_model[m]["cache_read"] += r["cache_read_tokens"]
        by_model[m]["cache_write"] += r["cache_write_tokens"]
        by_model[m]["count"] += 1

    # Overall totals
    total_input = sum(r["input_tokens"] for r in all_records)
    total_output = sum(r["output_tokens"] for r in all_records)
    total_cache_read = sum(r["cache_read_tokens"] for r in all_records)
    total_cache_write = sum(r["cache_write_tokens"] for r in all_records)

    # Calculate cost
    total_cost = 0.0
    for model, stats in by_model.items():
        pricing = get_pricing(model)
        cost = (
            stats["input"] / 1_000_000 * pricing["input"]
            + stats["output"] / 1_000_000 * pricing["output"]
            + stats["cache_read"] / 1_000_000 * pricing["cache_read"]
            + stats["cache_write"] / 1_000_000 * pricing["cache_write"]
        )
        total_cost += cost

    # Time range
    timestamps = [r["timestamp"] for r in all_records if r["timestamp"]]
    if timestamps:
        first_ts = min(timestamps)
        last_ts = max(timestamps)
    else:
        first_ts = last_ts = "unknown"

    # Print report
    print("=" * 70)
    print("  CLAUDE CODE TOKEN USAGE REPORT")
    print("=" * 70)
    print(f"  Time range:    {first_ts} → {last_ts}")
    print(f"  Sessions:      {len(session_summaries)}")
    print(f"  API calls:     {len(all_records)}")
    print()

    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  TOTAL USAGE                                              │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    print(f"  │  Input tokens:        {format_tokens(total_input):>12}                     │")
    print(f"  │  Output tokens:       {format_tokens(total_output):>12}                     │")
    print(f"  │  Cache read tokens:   {format_tokens(total_cache_read):>12}                     │")
    print(f"  │  Cache write tokens:  {format_tokens(total_cache_write):>12}                     │")
    print(f"  │  ─────────────────────────────────────────────────────── │")
    print(f"  │  ESTIMATED COST:      ${total_cost:>10.2f}                     │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  BY MODEL                                                 │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    for model, stats in sorted(by_model.items()):
        pricing = get_pricing(model)
        cost = (
            stats["input"] / 1_000_000 * pricing["input"]
            + stats["output"] / 1_000_000 * pricing["output"]
            + stats["cache_read"] / 1_000_000 * pricing["cache_read"]
            + stats["cache_write"] / 1_000_000 * pricing["cache_write"]
        )
        print(f"  │  {model:<20} │ in={format_tokens(stats['input']):>8} out={format_tokens(stats['output']):>8} │ ${cost:>8.2f} │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  PER-SESSION SUMMARY (top 10 by tokens)                   │")
    print("  ├─────────────────────────────────────────────────────────────┤")
    sorted_sessions = sorted(session_summaries, key=lambda s: s["input_tokens"] + s["output_tokens"], reverse=True)
    for s in sorted_sessions[:10]:
        total = s["input_tokens"] + s["output_tokens"]
        models_str = ", ".join(s["models"])
        print(f"  │  {s['file'][:30]:<30} │ {format_tokens(total):>8} tokens │ {s['messages']:>4} msgs │ {models_str[:15]:<15} │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()

    # Pricing reference
    print("  Pricing reference (per million tokens):")
    print("  ┌──────────────────┬────────┬────────┬──────────┬───────────┐")
    print("  │ Model            │  Input │ Output │ Cache R  │ Cache W   │")
    print("  ├──────────────────┼────────┼────────┼──────────┼───────────┤")
    for model, prices in PRICING.items():
        if model != "default":
            print(f"  │ {model:<16} │ ${prices['input']:>5.2f} │ ${prices['output']:>5.2f} │ ${prices['cache_read']:>6.2f}  │ ${prices['cache_write']:>7.2f} │")
    print("  └──────────────────┴────────┴────────┴──────────┴───────────┘")

if __name__ == "__main__":
    main()
