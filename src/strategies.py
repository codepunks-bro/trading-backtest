from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands


def add_sma_crossover_signals(df: pd.DataFrame, short: int = 20, long: int = 50) -> pd.DataFrame:
    out = df.copy()
    out["sma_short"] = out["close"].rolling(window=short, min_periods=short).mean()
    out["sma_long"] = out["close"].rolling(window=long, min_periods=long).mean()
    out["signal"] = 0
    out.loc[out["sma_short"] > out["sma_long"], "signal"] = 1
    out.loc[out["sma_short"] < out["sma_long"], "signal"] = -1
    return out


def add_rsi_mean_reversion_signals(
    df: pd.DataFrame, period: int = 14, low_th: int = 30, high_th: int = 70
) -> pd.DataFrame:
    out = df.copy()
    rsi = RSIIndicator(close=out["close"], window=period).rsi()
    out["rsi"] = rsi
    out["signal"] = 0
    out.loc[out["rsi"] < low_th, "signal"] = 1  # oversold -> long
    out.loc[out["rsi"] > high_th, "signal"] = -1  # overbought -> short
    return out


def add_donchian_breakout_signals(
    df: pd.DataFrame, channel: int = 20
) -> pd.DataFrame:
    out = df.copy()
    out["donchian_high"] = out["high"].rolling(window=channel, min_periods=channel).max()
    out["donchian_low"] = out["low"].rolling(window=channel, min_periods=channel).min()
    out["signal"] = 0
    # Breakout long when close breaks above previous channel high
    out.loc[out["close"] > out["donchian_high"].shift(1), "signal"] = 1
    # Breakout short when close breaks below previous channel low
    out.loc[out["close"] < out["donchian_low"].shift(1), "signal"] = -1
    return out


def add_ema_crossover_signals(df: pd.DataFrame, short: int = 20, long: int = 50) -> pd.DataFrame:
    out = df.copy()
    out["ema_short"] = out["close"].ewm(span=short, adjust=False).mean()
    out["ema_long"] = out["close"].ewm(span=long, adjust=False).mean()
    out["signal"] = 0
    out.loc[out["ema_short"] > out["ema_long"], "signal"] = 1
    out.loc[out["ema_short"] < out["ema_long"], "signal"] = -1
    return out


def add_macd_signals(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_window: int = 9
) -> pd.DataFrame:
    out = df.copy()
    macd = MACD(close=out["close"], window_fast=fast, window_slow=slow, window_sign=signal_window)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["signal"] = 0
    out.loc[out["macd"] > out["macd_signal"], "signal"] = 1
    out.loc[out["macd"] < out["macd_signal"], "signal"] = -1
    return out


def add_bbands_mean_reversion_signals(
    df: pd.DataFrame, window: int = 20, n_std: float = 2.0
) -> pd.DataFrame:
    out = df.copy()
    bb = BollingerBands(close=out["close"], window=window, window_dev=n_std)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()
    out["signal"] = 0
    # Mean reversion: long when price pierces below lower band, short when above upper band
    out.loc[out["close"] < out["bb_low"], "signal"] = 1
    out.loc[out["close"] > out["bb_high"], "signal"] = -1
    return out
