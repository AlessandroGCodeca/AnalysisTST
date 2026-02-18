"""
backtest.py — Simple beta-hedging backtest engine.

Computes hedged portfolio returns and performance statistics.
All charts use Plotly for interactive visualization.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analytics import sharpe_ratio, max_drawdown, value_at_risk


BG_DARK = "#0e1117"
CARD_DARK = "#1a1d23"
GRID_COLOR = "#2d3139"
TEXT_COLOR = "#fafafa"


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
) -> go.Figure:
    """Plot cumulative returns: unhedged vs beta-hedged (Plotly)."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3],
        subplot_titles=[
            f"Backtest: {asset_name} Unhedged vs Beta-Hedged ({bench_name})",
            "Daily Hedged P&L",
        ],
        vertical_spacing=0.08,
    )

    # ── Cumulative returns ─────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=bt_df.index, y=bt_df["Cum_Unhedged"] * 100,
        mode="lines", line=dict(color="#f7931a", width=2),
        name=f"{asset_name} (Unhedged)",
        hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=bt_df.index, y=bt_df["Cum_Hedged"] * 100,
        mode="lines", line=dict(color="#2ecc71", width=2),
        name=f"Beta-Hedged (β={hedge_ratio:.2f})",
        hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.4, row=1, col=1)

    # Fill: green above 0, red below 0
    fig.add_trace(go.Scatter(
        x=bt_df.index, y=bt_df["Cum_Hedged"].clip(lower=0) * 100,
        mode="lines", line=dict(width=0), showlegend=False,
        fill="tozeroy", fillcolor="rgba(46,204,113,0.1)",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=bt_df.index, y=bt_df["Cum_Hedged"].clip(upper=0) * 100,
        mode="lines", line=dict(width=0), showlegend=False,
        fill="tozeroy", fillcolor="rgba(231,76,60,0.1)",
    ), row=1, col=1)

    # ── Daily hedge P&L ────────────────────────────────────────────────
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in bt_df["Hedged"]]

    fig.add_trace(go.Bar(
        x=bt_df.index, y=bt_df["Hedged"] * 100,
        marker_color=colors, opacity=0.6,
        name="Daily Hedged Return",
        hovertemplate="%{x|%Y-%m-%d}<br>Return: %{y:.2f}%<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_DARK,
        plot_bgcolor=CARD_DARK,
        font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
        margin=dict(l=60, r=30, t=50, b=40),
        height=550,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
    )

    fig.update_yaxes(title_text="Cumulative Return (%)", gridcolor=GRID_COLOR, row=1, col=1)
    fig.update_yaxes(title_text="Daily Return (%)", gridcolor=GRID_COLOR, row=2, col=1)
    fig.update_xaxes(gridcolor=GRID_COLOR)

    return fig
