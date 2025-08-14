from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt

from src.data import FetchParams, fetch_ohlcv
from src.strategies import (
    add_donchian_breakout_signals,
    add_rsi_mean_reversion_signals,
    add_sma_crossover_signals,
    add_ema_crossover_signals,
    add_macd_signals,
    add_bbands_mean_reversion_signals,
)
from src.backtest import BacktestConfig, run_vectorized_backtest, summarize_performance
from src.plots import plot_equity, plot_price_with_signals


STRATEGIES = {
    "sma": add_sma_crossover_signals,
    "ema": add_ema_crossover_signals,
    "rsi": add_rsi_mean_reversion_signals,
    "macd": add_macd_signals,
    "bbands": add_bbands_mean_reversion_signals,
    "breakout": add_donchian_breakout_signals,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crypto 15m backtester")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--years", type=int, default=3, help="How many years to fetch")
    parser.add_argument("--strategy", choices=list(STRATEGIES.keys()), default="sma")
    parser.add_argument("--position-mode", choices=["long_only", "long_short"], default="long_only")
    parser.add_argument("--fee", type=float, default=0.0004, help="Fee rate per trade")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=365 * args.years)

    params = FetchParams(
        symbol=args.symbol,
        timeframe=args.timeframe,
        since=int(start.timestamp() * 1000),
        until=int(end.timestamp() * 1000),
    )

    df = fetch_ohlcv(params)

    # Strategy signals
    signal_fn = STRATEGIES[args.strategy]
    df_sig = signal_fn(df)

    # Backtest
    cfg = BacktestConfig(fee_rate=args.fee, position_mode=args.position_mode)
    bt = run_vectorized_backtest(df_sig, cfg)
    stats = summarize_performance(bt["equity"], bt["returns"])

    print("Strategy:", args.strategy)
    print("Symbol:", args.symbol, args.timeframe)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    if args.plot:
        plot_price_with_signals(df_sig, title=f"{args.symbol} {args.timeframe} price & signals")
        plot_equity(bt, title=f"Equity curve: {args.strategy}")
        plt.show()


if __name__ == "__main__":
    main()
