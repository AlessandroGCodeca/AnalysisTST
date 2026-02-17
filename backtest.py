"""
backtest.py — Simple beta-hedging backtest engine.

Computes hedged portfolio returns and performance statistics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from analytics import sharpe_ratio, max_drawdown, value_at_risk


def hedge_portfolio(
    df: pd.DataFrame, hedge_ratio: float | None = None
) -> pd.DataFrame:
    """
    Compute hedged vs unhedged portfolio returns.

    Parameters
    ----------
    df : DataFrame with Asset_Returns and Benchmark_Returns columns.
    hedge_ratio : float, optional. If None, uses the OLS beta from the data.

    Returns
    -------
    DataFrame with columns:
        Unhedged, Hedged, Cum_Unhedged, Cum_Hedged
    """
    if hedge_ratio is None:
        import scipy.stats as sp_stats
        slope, *_ = sp_stats.linregress(
            df["Benchmark_Returns"], df["Asset_Returns"]
        )
        hedge_ratio = slope

    result = pd.DataFrame(index=df.index)
    result["Unhedged"] = df["Asset_Returns"]
    result["Hedged"] = df["Asset_Returns"] - hedge_ratio * df["Benchmark_Returns"]
    result["Cum_Unhedged"] = (1 + result["Unhedged"]).cumprod() - 1
    result["Cum_Hedged"] = (1 + result["Hedged"]).cumprod() - 1

    return result


def backtest_stats(returns: pd.Series) -> dict:
    """Compute key performance stats for a return series."""
    cum = (1 + returns).cumprod()
    total_ret = cum.iloc[-1] - 1

    # Win rate
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

    return {
        "Total Return": f"{total_ret:.2%}",
        "Sharpe Ratio": f"{sharpe_ratio(returns):.2f}",
        "Max Drawdown": f"{max_drawdown(cum):.2%}",
        "VaR (95%)": f"{value_at_risk(returns):.2%}",
        "Volatility (ann.)": f"{returns.std() * np.sqrt(365):.2%}",
        "Win Rate": f"{win_rate:.1%}",
        "Best Day": f"{returns.max():.2%}",
        "Worst Day": f"{returns.min():.2%}",
    }


def plot_backtest(
    bt_df: pd.DataFrame,
    hedge_ratio: float,
    asset_name: str = "BTC",
    bench_name: str = "ETH",
) -> plt.Figure:
    """Plot cumulative returns: unhedged vs beta-hedged."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

    # ── Cumulative returns ─────────────────────────────────────────────
    ax1.plot(bt_df.index, bt_df["Cum_Unhedged"] * 100,
             color="#f7931a", linewidth=1.8, label=f"{asset_name} (Unhedged)")
    ax1.plot(bt_df.index, bt_df["Cum_Hedged"] * 100,
             color="#2ecc71", linewidth=1.8, label=f"Beta-Hedged (β={hedge_ratio:.2f})")
    ax1.axhline(0, color="grey", linestyle="--", alpha=0.4)
    ax1.fill_between(bt_df.index, bt_df["Cum_Hedged"] * 100, 0,
                     where=bt_df["Cum_Hedged"] > 0,
                     alpha=0.1, color="#2ecc71")
    ax1.fill_between(bt_df.index, bt_df["Cum_Hedged"] * 100, 0,
                     where=bt_df["Cum_Hedged"] <= 0,
                     alpha=0.1, color="#e74c3c")
    ax1.set_ylabel("Cumulative Return (%)", fontsize=12)
    ax1.set_title(f"Backtest: {asset_name} Unhedged vs Beta-Hedged ({bench_name})",
                  fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ── Daily hedge P&L ────────────────────────────────────────────────
    colors = ["#2ecc71" if x > 0 else "#e74c3c" for x in bt_df["Hedged"]]
    ax2.bar(bt_df.index, bt_df["Hedged"] * 100, color=colors, alpha=0.6, width=1.0)
    ax2.set_ylabel("Daily Hedged Return (%)", fontsize=10)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
