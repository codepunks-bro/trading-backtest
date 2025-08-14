from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict

import pandas as pd

from src.data import FetchParams, fetch_ohlcv
from src.backtest import BacktestConfig, run_vectorized_backtest, summarize_performance
from src.strategies import (
    add_sma_crossover_signals,
    add_ema_crossover_signals,
    add_rsi_mean_reversion_signals,
    add_macd_signals,
    add_bbands_mean_reversion_signals,
    add_donchian_breakout_signals,
)


STRATEGIES: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "sma": add_sma_crossover_signals,
    "ema": add_ema_crossover_signals,
    "rsi": add_rsi_mean_reversion_signals,
    "macd": add_macd_signals,
    "bbands": add_bbands_mean_reversion_signals,
    "breakout": add_donchian_breakout_signals,
}


@dataclass
class RunAllParams:
    symbol: str = "SOL/USDT"
    timeframe: str = "15m"
    years: int = 3
    fee_rate: float = 0.0004
    position_mode: str = "long_only"


def run_all(params: RunAllParams) -> pd.DataFrame:
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=365 * params.years)

    df = fetch_ohlcv(
        FetchParams(
            symbol=params.symbol,
            timeframe=params.timeframe,
            since=int(start.timestamp() * 1000),
            until=int(end.timestamp() * 1000),
        )
    )

    records = []

    for name, fn in STRATEGIES.items():
        df_sig = fn(df)
        bt = run_vectorized_backtest(
            df_sig,
            BacktestConfig(fee_rate=params.fee_rate, position_mode=params.position_mode),
        )
        stats = summarize_performance(bt["equity"], bt["returns"])
        stats_row = {"strategy": name, **stats}
        records.append(stats_row)

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    result = run_all(RunAllParams())
    # Pretty print
    with pd.option_context("display.max_columns", None):
        print(result.sort_values("sharpe", ascending=False))
