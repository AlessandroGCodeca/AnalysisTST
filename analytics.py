"""
analytics.py — Statistical computations: beta, rolling beta, risk metrics, diagnostics.
"""

import numpy as np
import pandas as pd
import scipy.stats as sp_stats


# ── Core Beta ──────────────────────────────────────────────────────────────

def calculate_beta(df: pd.DataFrame) -> dict:
    """
    OLS regression of Asset_Returns on Benchmark_Returns.
    Returns dict with beta, alpha, r_value, r_squared, p_value, std_err.
    """
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(
        df["Benchmark_Returns"], df["Asset_Returns"]
    )
    return {
        "beta": slope,
        "alpha": intercept,
        "r_value": r_value,
        "r_squared": r_value ** 2,
        "p_value": p_value,
        "std_err": std_err,
        "n_obs": len(df),
    }


def beta_confidence_interval(
    beta: float, std_err: float, n: int, confidence: float = 0.95
) -> tuple:
    """Return (lower, upper) CI for beta using t-distribution."""
    df_resid = n - 2
    t_crit = sp_stats.t.ppf((1 + confidence) / 2, df_resid)
    margin = t_crit * std_err
    return (beta - margin, beta + margin)


# ── Rolling Beta ───────────────────────────────────────────────────────────

def rolling_beta(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """Compute rolling beta using covariance / variance method."""
    cov = df["Asset_Returns"].rolling(window).cov(df["Benchmark_Returns"])
    var = df["Benchmark_Returns"].rolling(window).var()
    return (cov / var).rename("Rolling_Beta")


# ── Risk Metrics ───────────────────────────────────────────────────────────

def max_drawdown(prices: pd.Series) -> float:
    """Maximum drawdown (as a negative fraction) from a price series."""
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return drawdown.min()


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365,
) -> float:
    """Annualised Sharpe ratio."""
    excess = returns - risk_free_rate / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def value_at_risk(
    returns: pd.Series, confidence: float = 0.95
) -> float:
    """Parametric VaR (negative number = potential loss)."""
    z = sp_stats.norm.ppf(1 - confidence)
    return float(returns.mean() + z * returns.std())


def conditional_var(
    returns: pd.Series, confidence: float = 0.95
) -> float:
    """Expected Shortfall / CVaR — average loss beyond VaR."""
    var = value_at_risk(returns, confidence)
    return float(returns[returns <= var].mean())


def drawdown_series(prices: pd.Series) -> pd.Series:
    """Full drawdown time-series (for plotting)."""
    cummax = prices.cummax()
    return ((prices - cummax) / cummax).rename("Drawdown")


# ── Diagnostics ────────────────────────────────────────────────────────────

def ols_residuals(df: pd.DataFrame, stats_dict: dict) -> pd.Series:
    """Compute OLS residuals from the regression."""
    predicted = stats_dict["beta"] * df["Benchmark_Returns"] + stats_dict["alpha"]
    return (df["Asset_Returns"] - predicted).rename("Residuals")


def breusch_pagan_test(df: pd.DataFrame, stats_dict: dict) -> dict:
    """
    Simplified Breusch-Pagan test for heteroscedasticity.
    Returns dict with test_stat, p_value, and interpretation.
    """
    resid = ols_residuals(df, stats_dict)
    resid_sq = resid ** 2

    # Regress squared residuals on benchmark returns
    slope, intercept, r_val, p_val, se = sp_stats.linregress(
        df["Benchmark_Returns"], resid_sq
    )
    n = len(df)
    bp_stat = n * r_val ** 2
    bp_p = 1 - sp_stats.chi2.cdf(bp_stat, df=1)

    return {
        "test_stat": bp_stat,
        "p_value": bp_p,
        "heteroscedastic": bp_p < 0.05,
        "interpretation": (
            "Evidence of heteroscedasticity (p < 0.05). OLS beta may be inefficient."
            if bp_p < 0.05
            else "No significant heteroscedasticity detected."
        ),
    }


# ── Multi-Asset Beta ───────────────────────────────────────────────────────

def multi_asset_beta_matrix(
    multi_df: pd.DataFrame, benchmark_col: str = "benchmark"
) -> pd.DataFrame:
    """
    Compute beta for every asset column against the benchmark column.

    Parameters
    ----------
    multi_df : DataFrame with one column per asset (returns) and one
               benchmark column.
    benchmark_col : name of the benchmark column.

    Returns
    -------
    DataFrame with index=asset names, columns=[Beta, Alpha, R², Sharpe].
    """
    bench_ret = multi_df[benchmark_col]
    results = []

    for col in multi_df.columns:
        if col == benchmark_col:
            continue
        asset_ret = multi_df[col].dropna()
        common = asset_ret.index.intersection(bench_ret.dropna().index)
        if len(common) < 10:
            continue

        slope, intercept, r_val, p_val, se = sp_stats.linregress(
            bench_ret.loc[common], asset_ret.loc[common]
        )
        results.append({
            "Asset": col.replace("-", " ").title(),
            "Beta": round(slope, 4),
            "Alpha": round(intercept, 6),
            "R²": round(r_val ** 2, 4),
            "Sharpe": round(sharpe_ratio(asset_ret.loc[common]), 2),
        })

    return pd.DataFrame(results).set_index("Asset").sort_values("Beta", ascending=False)


# ── Summary builder ───────────────────────────────────────────────────────

def compute_all_stats(df: pd.DataFrame, window: int = 30) -> dict:
    """Compute everything and return a single dict."""
    beta_stats = calculate_beta(df)
    ci_lo, ci_hi = beta_confidence_interval(
        beta_stats["beta"], beta_stats["std_err"], beta_stats["n_obs"]
    )
    roll = rolling_beta(df, window)
    bp = breusch_pagan_test(df, beta_stats)

    return {
        **beta_stats,
        "beta_ci_lower": ci_lo,
        "beta_ci_upper": ci_hi,
        "asset_sharpe": sharpe_ratio(df["Asset_Returns"]),
        "benchmark_sharpe": sharpe_ratio(df["Benchmark_Returns"]),
        "asset_max_dd": max_drawdown(df["Asset_Price"]),
        "benchmark_max_dd": max_drawdown(df["Benchmark_Price"]),
        "asset_var_95": value_at_risk(df["Asset_Returns"], 0.95),
        "asset_cvar_95": conditional_var(df["Asset_Returns"], 0.95),
        "benchmark_var_95": value_at_risk(df["Benchmark_Returns"], 0.95),
        "breusch_pagan": bp,
        "rolling_beta": roll,
    }
