from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import ccxt
import pandas as pd
from dateutil import parser as date_parser
from tqdm import tqdm

# Map 15m timeframe to milliseconds
TIMEFRAME_TO_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
}


def _parse_time(value: str | int | float | datetime) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, datetime):
        return int(value.replace(tzinfo=timezone.utc).timestamp() * 1000)
    # assume str
    dt = date_parser.parse(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


@dataclass
class FetchParams:
    symbol: str = "BTC/USDT"
    timeframe: str = "15m"
    since: Optional[int] = None  # ms
    until: Optional[int] = None  # ms
    limit_per_call: int = 1000
    exchange: str = "binance"


def fetch_ohlcv(params: FetchParams) -> pd.DataFrame:
    ex = getattr(ccxt, params.exchange)({
        "enableRateLimit": True,
        # increase default timeouts
        "timeout": 30000,
        # set retries via built-in if available
        # no direct retries config in ccxt, we'll implement manual retry below
    })
    ms = TIMEFRAME_TO_MS[params.timeframe]

    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    since = params.since if params.since is not None else now_ms - 365 * 24 * 60 * 60 * 1000
    until = params.until if params.until is not None else now_ms

    all_rows: list[list[float]] = []
    cursor = since

    # Estimate loops for progress bar
    total_loops = max(1, math.ceil((until - since) / (params.limit_per_call * ms)))
    pbar = tqdm(total=total_loops, desc=f"Downloading {params.symbol} {params.timeframe}")

    while True:
        # manual retries for transient network issues
        attempts = 0
        last_exception: Exception | None = None
        while attempts < 5:
            try:
                candles = ex.fetch_ohlcv(
                    params.symbol,
                    timeframe=params.timeframe,
                    since=cursor,
                    limit=params.limit_per_call,
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_exception = exc
                attempts += 1
                ex.sleep(1000 * attempts)  # backoff in ms
        else:
            raise last_exception  # type: ignore[misc]

        if not candles:
            break
        all_rows.extend(candles)
        last_ts = candles[-1][0]
        cursor = last_ts + ms
        pbar.update(1)
        if cursor >= until:
            break

    pbar.close()

    if not all_rows:
        raise RuntimeError("No data fetched. Try different date range or symbol.")

    df = pd.DataFrame(
        all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("datetime", inplace=True)
    return df
