"""Notebook helpers for factor_engine - IPython display, progress bars, formatting."""

from __future__ import annotations

import time
from typing import Any, Optional
import pandas as pd
import numpy as np

try:
    from IPython.display import display, HTML, clear_output
    from IPython.core.display import DisplayObject
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
    display = print
    HTML = str
    clear_output = lambda wait=False: None


class ProgressBar:
    """Simple progress bar for notebooks."""

    def __init__(self, total: int, desc: str = "", width: int = 40):
        self.total = total
        self.current = 0
        self.desc = desc
        self.width = width
        self.start_time = time.time()

    def update(self, n: int = 1):
        """Update progress by n steps."""
        self.current += n
        self._render()

    def _render(self):
        """Render the progress bar."""
        if not IPYTHON_AVAILABLE:
            return

        pct = self.current / self.total if self.total > 0 else 0
        filled = int(self.width * pct)
        bar = "█" * filled + "░" * (self.width - filled)

        elapsed = time.time() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0

        html = f"""
        <div style="margin: 10px 0;">
            <div style="font-family: monospace; margin-bottom: 5px;">
                {self.desc}: {self.current}/{self.total} [{pct:.1%}]
            </div>
            <div style="background: #e0e0e0; border-radius: 4px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #3B6DFF, #38BDF8);
                            width: {pct*100}%; height: 24px; transition: width 0.3s;"></div>
            </div>
            <div style="font-family: monospace; font-size: 12px; color: #666; margin-top: 5px;">
                {rate:.1f} it/s | ETA: {eta:.0f}s
            </div>
        </div>
        """
        clear_output(wait=True)
        display(HTML(html))

    def close(self):
        """Mark progress as complete."""
        self.current = self.total
        self._render()


def display_factor_result(result: pd.DataFrame, title: str = "Factor Result"):
    """Display factor evaluation result with formatting."""
    if not IPYTHON_AVAILABLE:
        print(f"\n{title}")
        print(result)
        return

    html = f"""
    <div style="margin: 20px 0; padding: 15px; background: #0F131C;
                border-radius: 8px; border: 1px solid #1E2636;">
        <h3 style="margin: 0 0 15px 0; color: #38BDF8; font-weight: 600;">
            {title}
        </h3>
        {result.to_html(max_rows=20, float_format=lambda x: f'{x:.4f}')}
    </div>
    """
    display(HTML(html))


def display_metrics(metrics: dict[str, float], title: str = "Performance Metrics"):
    """Display metrics as styled cards."""
    if not IPYTHON_AVAILABLE:
        print(f"\n{title}")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        return

    cards = ""
    for key, value in metrics.items():
        # Color based on metric type
        if "ic" in key.lower() or "corr" in key.lower():
            color = "#6EE7B7" if value > 0 else "#E9A568"
        elif "sharpe" in key.lower():
            color = "#6EE7B7" if value > 1 else "#38BDF8"
        else:
            color = "#38BDF8"

        cards += f"""
        <div style="flex: 1; min-width: 150px; padding: 20px; background: #161D2B;
                    border-radius: 12px; margin: 8px; border: 1px solid #1E2636;">
            <div style="color: #94A3B8; font-size: 13px; text-transform: uppercase;
                        letter-spacing: 0.5px; margin-bottom: 8px;">
                {key.replace('_', ' ')}
            </div>
            <div style="color: {color}; font-size: 28px; font-weight: 600;">
                {value:.4f}
            </div>
        </div>
        """

    html = f"""
    <div style="margin: 20px 0;">
        <h3 style="color: #38BDF8; margin-bottom: 15px;">{title}</h3>
        <div style="display: flex; flex-wrap: wrap; margin: -8px;">
            {cards}
        </div>
    </div>
    """
    display(HTML(html))


def display_dataframe_summary(df: pd.DataFrame, name: str = "DataFrame"):
    """Display summary statistics for a dataframe."""
    if not IPYTHON_AVAILABLE:
        print(f"\n{name} Summary")
        print(df.describe())
        return

    stats = df.describe()

    html = f"""
    <div style="margin: 20px 0; padding: 20px; background: #0F131C;
                border-radius: 12px; border: 1px solid #1E2636;">
        <h3 style="color: #38BDF8; margin: 0 0 10px 0;">{name}</h3>
        <div style="color: #94A3B8; font-family: monospace; margin-bottom: 15px;">
            Shape: {df.shape[0]} rows × {df.shape[1]} columns |
            Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
        </div>
        {stats.to_html(float_format=lambda x: f'{x:.4f}')}
    </div>
    """
    display(HTML(html))


def display_warning(message: str):
    """Display a warning message."""
    if not IPYTHON_AVAILABLE:
        print(f"WARNING: {message}")
        return

    html = f"""
    <div style="margin: 15px 0; padding: 15px 15px 15px 45px;
                background: #1E2636; border-left: 4px solid #E9A568;
                border-radius: 4px; position: relative;">
        <div style="position: absolute; left: 15px; top: 15px;
                    font-size: 20px; color: #E9A568;">⚠</div>
        <div style="color: #E2E8F0; font-weight: 500;">
            {message}
        </div>
    </div>
    """
    display(HTML(html))


def display_success(message: str):
    """Display a success message."""
    if not IPYTHON_AVAILABLE:
        print(f"SUCCESS: {message}")
        return

    html = f"""
    <div style="margin: 15px 0; padding: 15px 15px 15px 45px;
                background: #0F131C; border-left: 4px solid #6EE7B7;
                border-radius: 4px; position: relative;">
        <div style="position: absolute; left: 15px; top: 15px;
                    font-size: 20px; color: #6EE7B7;">✓</div>
        <div style="color: #E2E8F0; font-weight: 500;">
            {message}
        </div>
    </div>
    """
    display(HTML(html))


def format_large_number(n: float) -> str:
    """Format large numbers with K/M/B suffixes."""
    if abs(n) >= 1e9:
        return f"{n/1e9:.2f}B"
    elif abs(n) >= 1e6:
        return f"{n/1e6:.2f}M"
    elif abs(n) >= 1e3:
        return f"{n/1e3:.2f}K"
    else:
        return f"{n:.2f}"


class NotebookTimer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        display_success(f"{self.name} completed in {self.elapsed:.2f}s")
