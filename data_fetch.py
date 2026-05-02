"""
Binance BTCUSDT data fetcher
Uses data-api.binance.vision (geo-unblocked, works in India)
"""

import time
import requests
import numpy as np
from typing import List, Tuple

BASE = "https://data-api.binance.vision/api/v3/klines"


def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1m",
    limit: int = 500,
) -> List[dict]:
    """
    Fetch the last `limit` closed 1-minute candles.
    Returns list of dicts with keys: open_time, open, high, low, close, volume, close_time
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    resp = requests.get(BASE, params=params, timeout=15)
    resp.raise_for_status()
    raw = resp.json()

    candles = []
    for k in raw:
        candles.append({
            "open_time":  int(k[0]),
            "open":       float(k[1]),
            "high":       float(k[2]),
            "low":        float(k[3]),
            "close":      float(k[4]),
            "volume":     float(k[5]),
            "close_time": int(k[6]),
        })

    # Drop the last candle if it's still open (close_time in the future)
    now_ms = int(time.time() * 1000)
    if candles and candles[-1]["close_time"] > now_ms:
        candles = candles[:-1]

    return candles


def fetch_30d_klines(symbol: str = "BTCUSDT") -> List[dict]:
    """
    Fetch ~720 1-hour candles (30 days).
    Binance max per request is 1000 so one call is fine.
    """
    return fetch_klines(symbol=symbol, interval="1h", limit=750)


def closes_array(candles: List[dict]) -> np.ndarray:
    return np.array([c["close"] for c in candles], dtype=np.float64)


def timestamps_array(candles: List[dict]) -> np.ndarray:
    return np.array([c["open_time"] for c in candles], dtype=np.int64)
