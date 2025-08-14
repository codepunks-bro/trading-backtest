from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    fee_rate: float = 0.0004  # 4 bps per trade
    slippage_bps: float = 0.0
    position_mode: Literal["long_only", "long_short"] = "long_only"
    initial_equity: float = 10_000.0


def _apply_costs(returns: pd.Series, position_changes: pd.Series, fee_rate: float) -> pd.Series:
    trade_costs = position_changes.abs() * fee_rate
    return returns - trade_costs


def run_vectorized_backtest(
    df_with_signal: pd.DataFrame,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    df = df_with_signal.copy()

    if "signal" not in df.columns:
        raise ValueError("df_with_signal must contain 'signal' column")

    # Ensure no look-ahead bias by entering at next bar close
    df["position"] = df["signal"].shift(1).fillna(0)
    if cfg.position_mode == "long_only":
        df["position"] = df["position"].clip(lower=0)

    price = df["close"].astype(float)
    returns = price.pct_change().fillna(0)

    gross_strategy_returns = df["position"] * returns

    position_changes = df["position"].diff().fillna(df["position"]).abs()
    net_strategy_returns = _apply_costs(
        returns=gross_strategy_returns,
        position_changes=position_changes,
        fee_rate=cfg.fee_rate,
    )

    equity = (1 + net_strategy_returns).cumprod() * cfg.initial_equity

    df_out = df.copy()
    df_out["returns"] = net_strategy_returns
    df_out["equity"] = equity
    return df_out


def summarize_performance(equity_series: pd.Series, returns_series: pd.Series) -> dict:
    cum_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1
    max_equity = equity_series.cummax()
    drawdown = equity_series / max_equity - 1
    max_drawdown = drawdown.min()

    # Annualized metrics assume 15m bars => 96 bars/day, ~ 96*365 per year
    bars_per_year = 96 * 365
    ann_return = (1 + returns_series.mean()) ** bars_per_year - 1
    ann_vol = returns_series.std() * np.sqrt(bars_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    wins = (returns_series > 0).sum()
    losses = (returns_series < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else np.nan

    return {
        "cumulative_return": float(cum_return),
        "max_drawdown": float(max_drawdown),
        "annual_return": float(ann_return),
        "annual_volatility": float(ann_vol),
        "sharpe": float(sharpe),
        "win_rate": float(win_rate),
    }
