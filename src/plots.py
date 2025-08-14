from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity(df: pd.DataFrame, title: str) -> None:
    plt.figure(figsize=(12, 5))
    df["equity"].plot()
    plt.title(title)
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_price_with_signals(df: pd.DataFrame, title: str) -> None:
    plt.figure(figsize=(12, 5))
    df["close"].plot(label="Close", color="black", alpha=0.7)
    if "sma_short" in df.columns and "sma_long" in df.columns:
        df["sma_short"].plot(label="SMA short", alpha=0.8)
        df["sma_long"].plot(label="SMA long", alpha=0.8)
    if "rsi" in df.columns:
        # Secondary axis for RSI over time (optional: separate plot)
        pass
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
