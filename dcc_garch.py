"""
dcc_garch.py — DCC-GARCH time-varying beta estimation.

Uses the `arch` library to fit univariate GARCH(1,1) models on each return
series, then estimates a DCC model on the standardised residuals to extract
time-varying conditional covariances and betas.
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


def fit_dcc_garch(df: pd.DataFrame) -> dict | None:
    """
    Fit a DCC-GARCH(1,1) model to Asset_Returns and Benchmark_Returns.

    Returns
    -------
    dict with keys:
        - dcc_beta : pd.Series   — time-varying beta
        - cond_corr : pd.Series  — conditional correlation
        - asset_vol : pd.Series  — GARCH conditional volatility (asset)
        - bench_vol : pd.Series  — GARCH conditional volatility (benchmark)
    or None if the `arch` library is not available / fitting fails.
    """
    try:
        from arch import arch_model
        from arch.univariate import ConstantMean, GARCH
    except ImportError:
        print("⚠ `arch` package not installed. DCC-GARCH will be unavailable.")
        return None

    try:
        asset_ret = df["Asset_Returns"] * 100   # arch expects %
        bench_ret = df["Benchmark_Returns"] * 100

        # ── Step 1: Fit univariate GARCH(1,1) ──────────────────────────
        am_asset = arch_model(asset_ret, vol="Garch", p=1, q=1, mean="Constant")
        res_asset = am_asset.fit(disp="off", show_warning=False)

        am_bench = arch_model(bench_ret, vol="Garch", p=1, q=1, mean="Constant")
        res_bench = am_bench.fit(disp="off", show_warning=False)

        # Standardised residuals
        std_resid_a = res_asset.std_resid
        std_resid_b = res_bench.std_resid

        # Conditional volatilities (convert back from %)
        asset_vol = res_asset.conditional_volatility / 100
        bench_vol = res_bench.conditional_volatility / 100

        # ── Step 2: DCC via exponential smoothing ──────────────────────
        # Simplified DCC: use EWMA on standardised residuals
        # This avoids the heavier DCC optimisation but captures the
        # time-varying correlation structure well.
        decay = 0.94  # RiskMetrics-style decay factor

        n = len(std_resid_a)
        q11 = np.ones(n)
        q22 = np.ones(n)
        q12 = np.full(n, std_resid_a.values @ std_resid_b.values / n)

        for t in range(1, n):
            q11[t] = decay * q11[t - 1] + (1 - decay) * std_resid_a.iloc[t - 1] ** 2
            q22[t] = decay * q22[t - 1] + (1 - decay) * std_resid_b.iloc[t - 1] ** 2
            q12[t] = decay * q12[t - 1] + (1 - decay) * (
                std_resid_a.iloc[t - 1] * std_resid_b.iloc[t - 1]
            )

        # Conditional correlation
        cond_corr = q12 / np.sqrt(q11 * q22)
        cond_corr = np.clip(cond_corr, -1, 1)

        # ── Step 3: Time-varying beta ──────────────────────────────────
        # beta_t = rho_t * (sigma_asset_t / sigma_bench_t)
        dcc_beta = cond_corr * (asset_vol.values / bench_vol.values)

        idx = df.index
        return {
            "dcc_beta": pd.Series(dcc_beta, index=idx, name="DCC_Beta"),
            "cond_corr": pd.Series(cond_corr, index=idx, name="Cond_Corr"),
            "asset_vol": pd.Series(asset_vol.values, index=idx, name="Asset_Vol"),
            "bench_vol": pd.Series(bench_vol.values, index=idx, name="Bench_Vol"),
        }

    except Exception as e:
        print(f"⚠ DCC-GARCH fitting failed: {e}")
        return None
