"""
structural_breaks.py — CUSUM-based structural break detection on beta.

Detects when the BTC/ETH relationship fundamentally changed by monitoring
the cumulative sum of recursive residuals from the OLS regression.
"""

import numpy as np
import pandas as pd


def cusum_test(
    df: pd.DataFrame, rolling_beta_series: pd.Series
) -> dict | None:
    """
    CUSUM (Cumulative Sum) test for structural breaks in the beta relationship.

    Uses the rolling beta deviations from the full-sample beta to identify
    periods where the relationship significantly shifted.

    Returns dict with:
      - cusum: pd.Series of CUSUM values
      - upper_bound, lower_bound: pd.Series of 5% confidence bands
      - break_dates: list of dates where CUSUM crossed the boundary
      - has_break: bool
    """
    beta_series = rolling_beta_series.dropna()
    if len(beta_series) < 30:
        return None

    # Overall beta
    full_beta = beta_series.mean()
    std_beta = beta_series.std()

    if std_beta == 0:
        return None

    # Standardised deviations
    deviations = (beta_series - full_beta) / std_beta
    cusum = deviations.cumsum()

    # 5% significance boundaries (Brown–Durbin–Evans)
    n = len(beta_series)
    k = np.arange(1, n + 1)
    boundary = 0.948 * np.sqrt(n) + 2 * 0.948 * k / np.sqrt(n)

    upper = pd.Series(boundary, index=beta_series.index, name="Upper")
    lower = pd.Series(-boundary, index=beta_series.index, name="Lower")

    # Find break dates (where |CUSUM| exceeds boundary)
    breaks = beta_series.index[(cusum.abs() > upper).values].tolist()

    return {
        "cusum": cusum,
        "upper_bound": upper,
        "lower_bound": lower,
        "break_dates": breaks,
        "has_break": len(breaks) > 0,
    }


def detect_beta_regimes(
    rolling_beta_series: pd.Series, n_segments: int = 3
) -> pd.DataFrame:
    """
    Split the beta time-series into segments and compute stats for each.

    Returns DataFrame with columns: Start, End, Mean Beta, Std, Trend.
    """
    beta = rolling_beta_series.dropna()
    if len(beta) < n_segments * 10:
        return pd.DataFrame()

    chunk_size = len(beta) // n_segments
    rows = []

    for i in range(n_segments):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < n_segments - 1 else len(beta)
        segment = beta.iloc[start_idx:end_idx]

        # Linear trend direction
        x = np.arange(len(segment))
        slope = np.polyfit(x, segment.values, 1)[0] if len(segment) > 1 else 0

        rows.append({
            "Period": f"{segment.index[0].strftime('%Y-%m')}-{segment.index[-1].strftime('%Y-%m')}",
            "Start": segment.index[0],
            "End": segment.index[-1],
            "Mean β": round(float(segment.mean()), 4),
            "Std": round(float(segment.std()), 4),
            "Trend": "↑ Rising" if slope > 0.001 else ("↓ Falling" if slope < -0.001 else "→ Stable"),
        })

    return pd.DataFrame(rows)
