"""
Walk-forward backtest — Part A
================================
For each bar i in [warm_up, N-1):
  - use closes[0..i] (inclusive) to predict closes[i+1]
  - reveal closes[i+1] and score the prediction

NO peeking: the slice passed to fit_and_predict never includes bar i+1.

Outputs:
  - backtest_results.json  (one JSON line per prediction)
  - prints coverage, avg_width, mean_winkler
"""

import json
import time
import numpy as np
from pathlib import Path

from data_fetch import fetch_30d_klines, closes_array, timestamps_array
from model import fit_and_predict


# ── Winkler score ──────────────────────────────────────────────────────────────
def winkler_score(lower: float, upper: float, actual: float, alpha: float = 0.05) -> float:
    """
    Winkler interval score (lower = better).
      - If actual inside [l, u]: score = width
      - If actual below l:       score = width + (2/alpha) * (l - actual)
      - If actual above u:       score = width + (2/alpha) * (actual - u)
    """
    width = upper - lower
    if actual < lower:
        return width + (2 / alpha) * (lower - actual)
    elif actual > upper:
        return width + (2 / alpha) * (actual - upper)
    else:
        return width


def evaluate(predictions: list) -> dict:
    """
    predictions: list of dicts with keys lower_95, upper_95, actual_close
    Returns dict with coverage, avg_width, mean_winkler
    """
    hits, widths, scores = [], [], []
    for p in predictions:
        l, u, act = p["lower_95"], p["upper_95"], p["actual_close"]
        hits.append(int(l <= act <= u))
        widths.append(u - l)
        scores.append(winkler_score(l, u, act))
    return {
        "coverage_95":    float(np.mean(hits)),
        "avg_width":      float(np.mean(widths)),
        "mean_winkler_95": float(np.mean(scores)),
        "n_predictions":  len(predictions),
    }


# ── Main backtest ──────────────────────────────────────────────────────────────
def run_backtest(
    warm_up: int = 50,       # minimum bars needed before first prediction
    vol_window: int = 24,
    output_path: str = "backtest_results.json",
    verbose: bool = True,
) -> dict:
    if verbose:
        print("Fetching 30-day BTCUSDT 1h candles …")
    candles = fetch_30d_klines()
    closes = closes_array(candles)
    timestamps = timestamps_array(candles)

    n = len(closes)
    if verbose:
        print(f"  Got {n} candles. Running walk-forward backtest …")

    results = []
    start_idx = max(warm_up, vol_window + 1)

    for i in range(start_idx, n - 1):
        # ── STRICT NO-PEEKING: only close[0..i] ──
        hist = closes[:i + 1]

        lower, upper, sigma = fit_and_predict(
            hist,
            vol_window=vol_window,
        )

        actual = float(closes[i + 1])
        pred_ts = int(timestamps[i])
        actual_ts = int(timestamps[i + 1])

        record = {
            "predict_bar_open_time": pred_ts,
            "actual_bar_open_time":  actual_ts,
            "lower_95":              round(lower, 2),
            "upper_95":              round(upper, 2),
            "current_price":         round(float(closes[i]), 2),
            "actual_close":          round(actual, 2),
            "sigma_used":            round(sigma, 8),
            "hit":                   int(lower <= actual <= upper),
        }
        results.append(record)

        if verbose and (i - start_idx) % 100 == 0:
            pct = (i - start_idx) / (n - 1 - start_idx) * 100
            print(f"  {pct:5.1f}%  bar {i}/{n-2}", end="\r")

    if verbose:
        print()

    # ── Write JSONL ──
    out = Path(output_path)
    with out.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    metrics = evaluate(results)
    metrics["output_file"] = str(out.resolve())

    if verbose:
        print("\n── Backtest Metrics ──────────────────────────────")
        print(f"  Coverage (target 0.95):  {metrics['coverage_95']:.4f}")
        print(f"  Average width ($):       {metrics['avg_width']:.2f}")
        print(f"  Mean Winkler score:      {metrics['mean_winkler_95']:.2f}")
        print(f"  N predictions:           {metrics['n_predictions']}")
        print(f"  Results saved → {output_path}")

    return metrics


if __name__ == "__main__":
    run_backtest()
