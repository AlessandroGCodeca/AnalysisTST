"""
visualizations.py — Enhanced matplotlib / seaborn charts.
All public functions return a matplotlib Figure for Streamlit embedding.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_theme(style="darkgrid", palette="muted")

# ── Color palette ──────────────────────────────────────────────────────────

ASSET_COLOR = "#f7931a"      # BTC orange
BENCH_COLOR = "#627eea"      # ETH blue
ACCENT_RED = "#e74c3c"
ACCENT_GREEN = "#2ecc71"


# ── 1. Scatter + Regression ────────────────────────────────────────────────

def plot_scatter_regression(
    df: pd.DataFrame,
    stats_dict: dict,
    asset_name: str = "BTC",
    bench_name: str = "ETH",
) -> plt.Figure:
    """Scatter plot with regression line, density contours, colour-coded by date."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Date-based colour map
    dates_num = mdates.date2num(df.index)
    sc = ax.scatter(
        df["Benchmark_Returns"],
        df["Asset_Returns"],
        c=dates_num,
        cmap="viridis",
        alpha=0.6,
        s=28,
        edgecolors="none",
        label="Daily Returns",
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    cbar.set_label("Date", fontsize=10)

    # Density contours
    try:
        sns.kdeplot(
            x=df["Benchmark_Returns"],
            y=df["Asset_Returns"],
            levels=5,
            color="white",
            linewidths=0.8,
            alpha=0.5,
            ax=ax,
        )
    except Exception:
        pass  # graceful fallback if too few points

    # Regression line
    beta = stats_dict["beta"]
    alpha = stats_dict["alpha"]
    x_range = np.linspace(
        df["Benchmark_Returns"].min(), df["Benchmark_Returns"].max(), 100
    )
    ax.plot(x_range, beta * x_range + alpha, color=ACCENT_RED, linewidth=2,
            label=f"β = {beta:.3f}  (R² = {stats_dict['r_squared']:.3f})")

    ax.set_xlabel(f"{bench_name} Returns", fontsize=12)
    ax.set_ylabel(f"{asset_name} Returns", fontsize=12)
    ax.set_title(f"{asset_name} vs {bench_name}  —  Beta Regression", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    return fig


# ── 2. Rolling Beta ───────────────────────────────────────────────────────

def plot_rolling_beta(
    rolling_beta: pd.Series,
    window: int = 30,
    asset_name: str = "BTC",
    bench_name: str = "ETH",
) -> plt.Figure:
    """Time-series chart of rolling beta."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(rolling_beta.index, rolling_beta.values,
            color=ASSET_COLOR, linewidth=1.8, label=f"{window}-day Rolling β")
    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.6, label="β = 1.0")
    ax.axhline(rolling_beta.mean(), color=BENCH_COLOR, linestyle=":",
               alpha=0.8, label=f"Mean β = {rolling_beta.mean():.2f}")

    ax.fill_between(rolling_beta.index, rolling_beta.values, 1.0,
                    where=rolling_beta.values > 1.0,
                    color=ACCENT_RED, alpha=0.15, label="Above 1")
    ax.fill_between(rolling_beta.index, rolling_beta.values, 1.0,
                    where=rolling_beta.values <= 1.0,
                    color=ACCENT_GREEN, alpha=0.15, label="Below 1")

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Beta", fontsize=12)
    ax.set_title(f"{asset_name}/{bench_name}  —  {window}-Day Rolling Beta",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    return fig


# ── 3. Price Chart (dual axis) ────────────────────────────────────────────

def plot_price_chart(
    df: pd.DataFrame,
    asset_name: str = "BTC",
    bench_name: str = "ETH",
) -> plt.Figure:
    """Dual-axis price chart."""
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(df.index, df["Asset_Price"], color=ASSET_COLOR, linewidth=1.5,
             label=f"{asset_name} Price")
    ax1.set_ylabel(f"{asset_name} Price (USD)", color=ASSET_COLOR, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=ASSET_COLOR)

    ax2 = ax1.twinx()
    ax2.plot(df.index, df["Benchmark_Price"], color=BENCH_COLOR, linewidth=1.5,
             label=f"{bench_name} Price")
    ax2.set_ylabel(f"{bench_name} Price (USD)", color=BENCH_COLOR, fontsize=12)
    ax2.tick_params(axis="y", labelcolor=BENCH_COLOR)

    ax1.set_title(f"{asset_name} & {bench_name}  —  Price History",
                  fontsize=14, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)
    fig.tight_layout()
    return fig


# ── 4. Returns Distribution ───────────────────────────────────────────────

def plot_returns_distribution(
    df: pd.DataFrame,
    asset_name: str = "BTC",
    bench_name: str = "ETH",
) -> plt.Figure:
    """Overlaid KDE + histogram of returns."""
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(df["Asset_Returns"], kde=True, color=ASSET_COLOR, alpha=0.5,
                 label=f"{asset_name}", stat="density", ax=ax, bins=50)
    sns.histplot(df["Benchmark_Returns"], kde=True, color=BENCH_COLOR, alpha=0.5,
                 label=f"{bench_name}", stat="density", ax=ax, bins=50)

    ax.set_xlabel("Daily Return", fontsize=12)
    ax.set_title("Returns Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()
    return fig


# ── 5. Residual Diagnostics ───────────────────────────────────────────────

def plot_residuals(
    df: pd.DataFrame, stats_dict: dict
) -> plt.Figure:
    """Residual scatter + histogram."""
    predicted = stats_dict["beta"] * df["Benchmark_Returns"] + stats_dict["alpha"]
    residuals = df["Asset_Returns"] - predicted

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter
    ax1.scatter(predicted, residuals, alpha=0.5, s=20, color=BENCH_COLOR)
    ax1.axhline(0, color=ACCENT_RED, linestyle="--", linewidth=1)
    ax1.set_xlabel("Predicted Returns", fontsize=11)
    ax1.set_ylabel("Residuals", fontsize=11)
    ax1.set_title("Residuals vs. Predicted", fontsize=13, fontweight="bold")

    # Histogram
    sns.histplot(residuals, kde=True, color=BENCH_COLOR, ax=ax2, bins=40)
    ax2.set_xlabel("Residual", fontsize=11)
    ax2.set_title("Residual Distribution", fontsize=13, fontweight="bold")

    fig.tight_layout()
    return fig


# ── 6. Drawdown Chart ─────────────────────────────────────────────────────

def plot_drawdown(
    df: pd.DataFrame,
    asset_name: str = "BTC",
    bench_name: str = "ETH",
) -> plt.Figure:
    """Drawdown time-series for both asset and benchmark."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for col, name, color in [
        ("Asset_Price", asset_name, ASSET_COLOR),
        ("Benchmark_Price", bench_name, BENCH_COLOR),
    ]:
        cummax = df[col].cummax()
        dd = (df[col] - cummax) / cummax
        ax.fill_between(df.index, dd, 0, alpha=0.35, color=color, label=name)
        ax.plot(df.index, dd, color=color, linewidth=0.8)

    ax.set_ylabel("Drawdown (%)", fontsize=12)
    ax.set_title("Drawdown from Peak", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()
    return fig


# ── 7. Correlation Heatmap ─────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Correlation matrix of returns and prices."""
    cols = [c for c in df.columns if "Returns" in c or "Price" in c]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig
