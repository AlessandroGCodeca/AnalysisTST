"""
granger.py — Granger Causality Test for crypto return pairs.

Tests whether lagged values of one return series help predict another,
using the statsmodels implementation.
"""

import pandas as pd
import numpy as np


def granger_causality_test(
    df: pd.DataFrame, max_lag: int = 5
) -> dict | None:
    """
    Run pair-wise Granger-causality tests between Asset and Benchmark returns.

    Returns dict with keys:
      - asset_causes_bench: list of (lag, F-stat, p-value) tuples
      - bench_causes_asset: list of (lag, F-stat, p-value) tuples
      - best_direction: str describing the dominant causal direction
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        return None

    ret_a = df["Asset_Returns"].dropna()
    ret_b = df["Benchmark_Returns"].dropna()
    combined = pd.concat([ret_a, ret_b], axis=1).dropna()

    if len(combined) < max_lag + 20:
        return None

    results = {
        "asset_causes_bench": [],
        "bench_causes_asset": [],
    }

    # Test: does Asset Granger-cause Benchmark?
    try:
        gc_ab = grangercausalitytests(
            combined[["Benchmark_Returns", "Asset_Returns"]],
            maxlag=max_lag, verbose=False,
        )
        for lag in range(1, max_lag + 1):
            f_stat = gc_ab[lag][0]["ssr_ftest"][0]
            p_val = gc_ab[lag][0]["ssr_ftest"][1]
            results["asset_causes_bench"].append((lag, f_stat, p_val))
    except Exception:
        pass

    # Test: does Benchmark Granger-cause Asset?
    try:
        gc_ba = grangercausalitytests(
            combined[["Asset_Returns", "Benchmark_Returns"]],
            maxlag=max_lag, verbose=False,
        )
        for lag in range(1, max_lag + 1):
            f_stat = gc_ba[lag][0]["ssr_ftest"][0]
            p_val = gc_ba[lag][0]["ssr_ftest"][1]
            results["bench_causes_asset"].append((lag, f_stat, p_val))
    except Exception:
        pass

    # Determine dominant direction (best p-value across lags)
    best_ab = min(results["asset_causes_bench"], key=lambda x: x[2], default=(0, 0, 1.0))
    best_ba = min(results["bench_causes_asset"], key=lambda x: x[2], default=(0, 0, 1.0))

    if best_ab[2] < 0.05 and best_ba[2] < 0.05:
        results["best_direction"] = "Bidirectional (both influence each other)"
    elif best_ab[2] < 0.05:
        results["best_direction"] = "Asset → Benchmark (asset leads)"
    elif best_ba[2] < 0.05:
        results["best_direction"] = "Benchmark → Asset (benchmark leads)"
    else:
        results["best_direction"] = "No significant Granger causality detected"

    return results
