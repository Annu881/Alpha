"""
BTC GBM Forecaster
==================
Geometric Brownian Motion with:
  - Student-t fat tails (no normal bell curve)
  - Volatility clustering via rolling window
  - Strict no-peeking enforcement
"""

import numpy as np
from scipy.stats import t as student_t
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Prediction:
    timestamp: int          # Unix ms of the candle being predicted
    lower_95: float
    upper_95: float
    current_price: float
    sigma_used: float       # annualised vol used


def fit_and_predict(
    closes: np.ndarray,
    n_sims: int = 10_000,
    vol_window: int = 24,       # hours of recent data for vol estimate
    df_t: float = 4.0,          # degrees of freedom for Student-t (fat tails)
    horizon: int = 1,           # bars ahead
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Given a 1-D array of CLOSING prices (oldest→newest, NO future bars),
    simulate `n_sims` GBM paths one bar forward and return the
    (lower, upper) confidence interval and current sigma.

    Parameters
    ----------
    closes      : price array, must have at least vol_window+1 elements
    n_sims      : Monte Carlo paths
    vol_window  : rolling window to estimate recent volatility (clustering)
    df_t        : Student-t degrees of freedom — DO NOT set >30 (fat tails matter)
    horizon     : bars ahead (always 1 for this challenge)
    confidence  : interval level (0.95 → 95 % CI)
    """
    assert len(closes) >= vol_window + 1, "Not enough bars for vol estimate"

    rng = np.random.default_rng(seed)

    # --- 1. Log returns (use only recent window for volatility clustering) ---
    log_rets = np.diff(np.log(closes))
    recent_rets = log_rets[-vol_window:]          # last `vol_window` returns

    # --- 2. Estimate per-bar mu and sigma from recent returns ---
    mu_bar = float(np.mean(recent_rets))
    sigma_bar = float(np.std(recent_rets, ddof=1))

    # Guard against degenerate sigma
    sigma_bar = max(sigma_bar, 1e-6)

    # --- 3. Simulate paths using Student-t scaled shocks ---
    # GBM: S_t+1 = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*Z*sqrt(dt))
    # Z ~ Student-t(df) scaled to unit variance → Z_scaled = t / sqrt(df/(df-2))
    scale_factor = np.sqrt(df_t / (df_t - 2))          # variance correction
    raw_z = rng.standard_t(df=df_t, size=(n_sims, horizon))
    z = raw_z / scale_factor                             # unit-variance t shocks

    drift = (mu_bar - 0.5 * sigma_bar ** 2) * horizon
    diffusion = sigma_bar * np.sqrt(horizon) * z        # shape (n_sims, horizon)

    log_path = drift + diffusion.sum(axis=1)            # total log-return
    S0 = closes[-1]
    final_prices = S0 * np.exp(log_path)

    # --- 4. Read off the CI ---
    alpha = (1 - confidence) / 2
    lower = float(np.quantile(final_prices, alpha))
    upper = float(np.quantile(final_prices, 1 - alpha))

    return lower, upper, sigma_bar
