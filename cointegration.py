"""
cointegration.py — Cointegration tests for crypto return pairs.

Tests whether BTC and ETH prices share a long-run equilibrium using
Engle-Granger (2-step) and Johansen methods.
"""

import pandas as pd
import numpy as np


def engle_granger_test(df: pd.DataFrame) -> dict | None:
    """
    Engle-Granger two-step cointegration test on Asset_Price vs Benchmark_Price.

    Returns dict with:
      - adf_stat, p_value, critical_values
      - cointegrated (bool), interpretation (str)
      - hedge_ratio (cointegrating coefficient)
    """
    try:
        from statsmodels.tsa.stattools import adfuller
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant
    except ImportError:
        return None

    y = df["Asset_Price"].values
    x = df["Benchmark_Price"].values

    # Step 1: OLS regression  y = α + β·x + ε
    X = add_constant(x)
    ols = OLS(y, X).fit()
    residuals = ols.resid
    hedge_ratio = ols.params[1]

    # Step 2: ADF test on residuals
    adf_result = adfuller(residuals, maxlag=None, autolag="AIC")
    adf_stat = adf_result[0]
    p_value = adf_result[1]
    crit_vals = adf_result[4]

    cointegrated = p_value < 0.05

    return {
        "adf_stat": adf_stat,
        "p_value": p_value,
        "critical_values": {k: round(v, 4) for k, v in crit_vals.items()},
        "hedge_ratio": hedge_ratio,
        "cointegrated": cointegrated,
        "interpretation": (
            f"Cointegrated (p={p_value:.4f}). Long-run equilibrium exists."
            if cointegrated
            else f"Not cointegrated (p={p_value:.4f}). No stable long-run relationship."
        ),
    }


def johansen_test(df: pd.DataFrame, det_order: int = 0) -> dict | None:
    """
    Johansen cointegration test on Asset_Price and Benchmark_Price.

    Returns dict with:
      - trace_stats, max_eigen_stats (arrays)
      - critical_values_trace, critical_values_max_eigen
      - n_cointegrating (number of cointegrating relationships at 5%)
      - interpretation
    """
    try:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
    except ImportError:
        return None

    prices = df[["Asset_Price", "Benchmark_Price"]].dropna().values

    if len(prices) < 50:
        return None

    result = coint_johansen(prices, det_order=det_order, k_ar_diff=1)

    trace_stats = result.lr1
    max_eigen_stats = result.lr2
    crit_trace_5 = result.cvt[:, 1]  # 5% critical values
    crit_max_5 = result.cvm[:, 1]

    n_coint = sum(1 for t, c in zip(trace_stats, crit_trace_5) if t > c)

    return {
        "trace_stats": trace_stats.tolist(),
        "max_eigen_stats": max_eigen_stats.tolist(),
        "crit_trace_5pct": crit_trace_5.tolist(),
        "crit_max_eigen_5pct": crit_max_5.tolist(),
        "n_cointegrating": n_coint,
        "interpretation": (
            f"{n_coint} cointegrating relationship(s) found at 5% level."
            if n_coint > 0
            else "No cointegrating relationships found at 5% level."
        ),
    }
