"""
visualizations.py — Interactive Plotly charts for Streamlit embedding.
All public functions return a plotly.graph_objects.Figure.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Theme ──────────────────────────────────────────────────────────────────

BTC_ORANGE = "#f7931a"
ETH_BLUE = "#627eea"
ACCENT_PURPLE = "#9b59b6"
ACCENT_RED = "#e74c3c"
ACCENT_GREEN = "#2ecc71"
BG_DARK = "#0e1117"
CARD_DARK = "#1a1d23"
GRID_COLOR = "#2d3139"
TEXT_COLOR = "#fafafa"

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=BG_DARK,
    plot_bgcolor=CARD_DARK,
    font=dict(color=TEXT_COLOR, family="Inter, sans-serif"),
    margin=dict(l=60, r=30, t=50, b=40),
    xaxis=dict(gridcolor=GRID_COLOR, showgrid=True),
    yaxis=dict(gridcolor=GRID_COLOR, showgrid=True),
)


def _apply_layout(fig, **kwargs):
    """Apply dark theme defaults to a figure."""
    merged = {**LAYOUT_DEFAULTS, **kwargs}
    fig.update_layout(**merged)
    return fig


# ── 1. Scatter + Regression ────────────────────────────────────────────────

def plot_scatter_regression(
    df: pd.DataFrame,
    stats_dict: dict,
    asset_name: str = "BTC",
    bench_name: str = "ETH",
):
    """Scatter plot with regression line, colour-coded by date."""
    x = df["Benchmark_Returns"]
    y = df["Asset_Returns"]

    fig = go.Figure()

    # Scatter points coloured by date (ordinal)
    dates = df.index
    date_nums = (dates - dates.min()).days

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(
            size=5, color=date_nums, colorscale="Sunset",
            opacity=0.6, colorbar=dict(title="Days", len=0.6),
        ),
        name="Daily Returns",
        hovertemplate=f"{bench_name}: %{{x:.3%}}<br>{asset_name}: %{{y:.3%}}<br><extra></extra>",
    ))

    # Regression line
    beta = stats_dict["beta"]
    alpha = stats_dict["alpha"]
    x_range = np.linspace(x.min(), x.max(), 100)
    y_line = alpha + beta * x_range

    fig.add_trace(go.Scatter(
        x=x_range, y=y_line, mode="lines",
        line=dict(color=ACCENT_RED, width=2, dash="dash"),
        name=f"β={beta:.3f}, α={alpha:.5f}",
    ))

    _apply_layout(fig,
        title=f"{asset_name} vs {bench_name} — Returns Scatter",
        xaxis_title=f"{bench_name} Returns",
        yaxis_title=f"{asset_name} Returns",
        xaxis_tickformat=".1%", yaxis_tickformat=".1%",
        height=450,
    )
    return fig


# ── 2. Rolling Beta ───────────────────────────────────────────────────────

def plot_rolling_beta(
    rolling_beta: pd.Series,
    window: int = 30,
    asset_name: str = "BTC",
    bench_name: str = "ETH",
):
    """Time-series chart of rolling beta."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rolling_beta.index, y=rolling_beta.values,
        mode="lines", line=dict(color=BTC_ORANGE, width=2),
        name=f"{window}d Rolling β",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>β = %{y:.3f}<extra></extra>",
    ))

    mean_val = rolling_beta.mean()
    fig.add_hline(y=mean_val, line_dash="dot", line_color="grey",
                  annotation_text=f"Mean β = {mean_val:.3f}")
    fig.add_hline(y=1.0, line_dash="dot", line_color=ACCENT_RED, opacity=0.4)

    _apply_layout(fig,
        title=f"{asset_name}/{bench_name} — {window}-Day Rolling Beta",
        yaxis_title="Beta (β)",
        height=400,
    )
    return fig


# ── 3. Price Chart (dual axis) ────────────────────────────────────────────

def plot_price_chart(
    df: pd.DataFrame,
    asset_name: str = "BTC",
    bench_name: str = "ETH",
):
    """Dual-axis price chart."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Asset_Price"], mode="lines",
        line=dict(color=BTC_ORANGE, width=2), name=f"{asset_name} Price",
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Benchmark_Price"], mode="lines",
        line=dict(color=ETH_BLUE, width=2), name=f"{bench_name} Price",
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
    ), secondary_y=True)

    _apply_layout(fig,
        title=f"{asset_name} & {bench_name} — Price History",
        height=450,
    )
    fig.update_yaxes(title_text=f"{asset_name} Price (USD)", secondary_y=False,
                     gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text=f"{bench_name} Price (USD)", secondary_y=True,
                     gridcolor=GRID_COLOR)
    return fig


# ── 4. Returns Distribution ───────────────────────────────────────────────

def plot_returns_distribution(
    df: pd.DataFrame,
    asset_name: str = "BTC",
    bench_name: str = "ETH",
):
    """Overlaid histogram of returns."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df["Asset_Returns"], name=asset_name, opacity=0.65,
        marker_color=BTC_ORANGE, nbinsx=50,
    ))
    fig.add_trace(go.Histogram(
        x=df["Benchmark_Returns"], name=bench_name, opacity=0.55,
        marker_color=ETH_BLUE, nbinsx=50,
    ))

    _apply_layout(fig,
        title="Returns Distribution",
        xaxis_title="Daily Return", yaxis_title="Count",
        xaxis_tickformat=".1%",
        barmode="overlay", height=400,
    )
    return fig


# ── 5. Residual Diagnostics ───────────────────────────────────────────────

def plot_residuals(df: pd.DataFrame, stats_dict: dict):
    """Residual scatter + histogram in subplots."""
    predicted = stats_dict["beta"] * df["Benchmark_Returns"] + stats_dict["alpha"]
    residuals = df["Asset_Returns"] - predicted

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Residuals vs Predicted", "Residual Distribution"])

    fig.add_trace(go.Scatter(
        x=predicted, y=residuals, mode="markers",
        marker=dict(color=ETH_BLUE, size=4, opacity=0.5),
        name="Residuals",
        hovertemplate="Predicted: %{x:.3%}<br>Residual: %{y:.3%}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color=ACCENT_RED, row=1, col=1)

    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=40, marker_color=ETH_BLUE, opacity=0.7,
        name="Distribution",
    ), row=1, col=2)

    _apply_layout(fig, height=380, showlegend=False)
    return fig


# ── 6. Drawdown Chart ─────────────────────────────────────────────────────

def plot_drawdown(
    df: pd.DataFrame,
    asset_name: str = "BTC",
    bench_name: str = "ETH",
):
    """Drawdown time-series for both asset and benchmark."""
    def _dd(prices):
        cummax = prices.cummax()
        return (prices - cummax) / cummax

    dd_asset = _dd(df["Asset_Price"])
    dd_bench = _dd(df["Benchmark_Price"])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dd_asset.index, y=dd_asset.values, mode="lines",
        fill="tozeroy", fillcolor="rgba(247,147,26,0.2)",
        line=dict(color=BTC_ORANGE, width=1.5), name=asset_name,
        hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dd_bench.index, y=dd_bench.values, mode="lines",
        fill="tozeroy", fillcolor="rgba(98,126,234,0.15)",
        line=dict(color=ETH_BLUE, width=1.5), name=bench_name,
        hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.1%}<extra></extra>",
    ))

    _apply_layout(fig,
        title="Drawdown", yaxis_title="Drawdown",
        yaxis_tickformat=".0%", height=400,
    )
    return fig


# ── 7. Correlation Heatmap ─────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation matrix of returns and prices."""
    cols = ["Asset_Price", "Benchmark_Price", "Asset_Returns", "Benchmark_Returns"]
    corr = df[cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=cols, y=cols,
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=corr.values.round(3), texttemplate="%{text}",
        textfont=dict(size=12),
    ))

    _apply_layout(fig, title="Correlation Matrix", height=400)
    return fig


# ── 8. Multi-Asset Beta Heatmap ───────────────────────────────────────────

def plot_beta_heatmap(beta_matrix):
    """Bar chart of multi-asset betas."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=beta_matrix.index, y=beta_matrix["Beta"],
        marker_color=[BTC_ORANGE if b > 0 else ACCENT_RED for b in beta_matrix["Beta"]],
        text=[f"{b:.3f}" for b in beta_matrix["Beta"]],
        textposition="outside",
        hovertemplate="Asset: %{x}<br>Beta: %{y:.4f}<extra></extra>",
    ))

    fig.add_hline(y=1.0, line_dash="dot", line_color="grey",
                  annotation_text="β = 1.0")

    _apply_layout(fig,
        title="Multi-Asset Beta Matrix",
        yaxis_title="Beta (β)", height=400,
    )
    return fig


# ── 9. DCC-GARCH vs Rolling Beta ─────────────────────────────────────────

def plot_dcc_vs_rolling(
    dcc_beta: "pd.Series",
    rolling_beta_series: "pd.Series",
    window: int = 30,
    asset_name: str = "BTC",
    bench_name: str = "ETH",
):
    """Compare DCC-GARCH beta with simple rolling beta."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dcc_beta.index, y=dcc_beta.values, mode="lines",
        line=dict(color=ACCENT_PURPLE, width=2), name="DCC-GARCH β",
    ))
    fig.add_trace(go.Scatter(
        x=rolling_beta_series.index, y=rolling_beta_series.values,
        mode="lines", line=dict(color=BTC_ORANGE, width=1.5, dash="dot"),
        name=f"{window}d Rolling β", opacity=0.7,
    ))

    fig.add_hline(y=1.0, line_dash="dot", line_color="grey", opacity=0.4)

    _apply_layout(fig,
        title=f"{asset_name}/{bench_name} — DCC-GARCH vs Rolling Beta",
        yaxis_title="Beta (β)", height=400,
    )
    return fig


# ── 10. Regime Chart ─────────────────────────────────────────────────────

def plot_regime_chart(
    df: pd.DataFrame,
    regimes: "pd.Series",
    asset_name: str = "BTC",
):
    """Price chart with regime-coloured background bands."""
    regime_colors = {
        "Bull": "rgba(46,204,113,0.15)",
        "Sideways": "rgba(241,196,15,0.12)",
        "Bear": "rgba(231,76,60,0.15)",
    }

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Asset_Price"], mode="lines",
        line=dict(color=BTC_ORANGE, width=2), name=f"{asset_name} Price",
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
    ))

    # Add regime bands as shapes
    if regimes is not None and len(regimes) > 0:
        current_regime = regimes.iloc[0]
        start = regimes.index[0]

        for i in range(1, len(regimes)):
            if regimes.iloc[i] != current_regime or i == len(regimes) - 1:
                color = regime_colors.get(current_regime, "rgba(128,128,128,0.1)")
                fig.add_vrect(
                    x0=start, x1=regimes.index[i],
                    fillcolor=color, layer="below", line_width=0,
                )
                current_regime = regimes.iloc[i]
                start = regimes.index[i]

    _apply_layout(fig,
        title=f"{asset_name} — Market Regimes (HMM)",
        yaxis_title="Price (USD)", height=450,
    )
    return fig


# ── 11. Rolling Beta with Event Annotations ──────────────────────────────

def plot_rolling_beta_with_events(
    rolling_beta_series: "pd.Series",
    events_df: "pd.DataFrame",
    window: int = 30,
    asset_name: str = "BTC",
    bench_name: str = "ETH",
):
    """Rolling beta chart with key event annotations."""
    fig = plot_rolling_beta(rolling_beta_series, window, asset_name, bench_name)

    if events_df is not None and not events_df.empty:
        for _, row in events_df.iterrows():
            event_date = row.get("Date", row.name)
            event_label = row.get("Label", row.get("Event", ""))
            fig.add_vline(
                x=event_date, line_dash="dot", line_color="rgba(255,255,255,0.4)",
                line_width=1,
            )
            fig.add_annotation(
                x=event_date,
                y=rolling_beta_series.max() * 0.95,
                text=str(event_label)[:20],
                showarrow=True, arrowhead=2, arrowsize=0.8,
                font=dict(size=9, color=TEXT_COLOR),
                bgcolor="rgba(30,30,30,0.8)",
                bordercolor="rgba(255,255,255,0.3)",
            )

    return fig


# ── 12. CUSUM Structural Breaks ──────────────────────────────────────────

def plot_cusum(cusum_result: dict, asset_name: str = "BTC", bench_name: str = "ETH"):
    """CUSUM chart with confidence bands and break markers."""
    cusum = cusum_result["cusum"]
    upper = cusum_result["upper_bound"]
    lower = cusum_result["lower_bound"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cusum.index, y=cusum.values, mode="lines",
        line=dict(color=BTC_ORANGE, width=2), name="CUSUM",
    ))
    fig.add_trace(go.Scatter(
        x=upper.index, y=upper.values, mode="lines",
        line=dict(color=ACCENT_RED, width=1, dash="dash"), name="5% boundary",
    ))
    fig.add_trace(go.Scatter(
        x=lower.index, y=lower.values, mode="lines",
        line=dict(color=ACCENT_RED, width=1, dash="dash"), name="_lower",
        showlegend=False,
    ))

    # Mark break dates
    for bd in cusum_result["break_dates"][:10]:  # limit to 10
        fig.add_vline(x=bd, line_dash="dot", line_color=ACCENT_GREEN, line_width=1)

    _apply_layout(fig,
        title=f"{asset_name}/{bench_name} — CUSUM Test for Structural Breaks",
        yaxis_title="CUSUM", height=400,
    )
    return fig


# ── 13. Fear & Greed Overlay ─────────────────────────────────────────────

def plot_fear_greed_overlay(df_merged: pd.DataFrame, asset_name: str = "BTC"):
    """Price chart with Fear & Greed index as colour overlay."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # F&G as area chart
    fig.add_trace(go.Scatter(
        x=df_merged.index, y=df_merged["FG_Value"],
        mode="lines", fill="tozeroy",
        fillcolor="rgba(241,196,15,0.15)",
        line=dict(color="#f1c40f", width=1),
        name="Fear & Greed",
        hovertemplate="%{x|%Y-%m-%d}<br>F&G: %{y}<extra></extra>",
    ), secondary_y=True)

    # Price line
    fig.add_trace(go.Scatter(
        x=df_merged.index, y=df_merged["Asset_Price"],
        mode="lines", line=dict(color=BTC_ORANGE, width=2),
        name=f"{asset_name} Price",
        hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
    ), secondary_y=False)

    # F&G zones
    fig.add_hrect(y0=0, y1=25, fillcolor="rgba(231,76,60,0.08)",
                  layer="below", line_width=0, secondary_y=True)
    fig.add_hrect(y0=75, y1=100, fillcolor="rgba(46,204,113,0.08)",
                  layer="below", line_width=0, secondary_y=True)

    _apply_layout(fig,
        title=f"{asset_name} Price vs Fear & Greed Index",
        height=450,
    )
    fig.update_yaxes(title_text=f"{asset_name} Price (USD)", secondary_y=False,
                     gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text="Fear & Greed (0-100)", secondary_y=True,
                     gridcolor=GRID_COLOR, range=[0, 100])
    return fig


# ── 14. CAPM Security Market Line ────────────────────────────────────────

def plot_sml(stats: dict, asset_name: str = "BTC", bench_name: str = "ETH"):
    """Security Market Line with actual vs expected return."""
    beta = stats["beta"]
    actual_return = stats["jensens_alpha"] + stats["capm_expected_return"]
    expected_return = stats["capm_expected_return"]

    # SML line: E(R) = Rf + β·(Rm - Rf)
    betas = np.linspace(0, 2, 50)
    # Market excess return per unit beta
    rm = expected_return / beta if beta != 0 else 0
    sml_returns = rm * betas

    fig = go.Figure()

    # SML line
    fig.add_trace(go.Scatter(
        x=betas, y=sml_returns * 100, mode="lines",
        line=dict(color="grey", width=2, dash="dash"),
        name="Security Market Line",
    ))

    # Actual position
    fig.add_trace(go.Scatter(
        x=[beta], y=[actual_return * 100], mode="markers+text",
        marker=dict(color=BTC_ORANGE, size=14, symbol="diamond"),
        text=[f"{asset_name}"], textposition="top center",
        name=f"{asset_name} (actual)",
        hovertemplate=f"β = {beta:.3f}<br>Return = {actual_return*100:.2f}%<extra></extra>",
    ))

    # Expected position
    fig.add_trace(go.Scatter(
        x=[beta], y=[expected_return * 100], mode="markers",
        marker=dict(color=ACCENT_GREEN, size=10, symbol="circle"),
        name=f"{asset_name} (CAPM expected)",
        hovertemplate=f"β = {beta:.3f}<br>Expected = {expected_return*100:.2f}%<extra></extra>",
    ))

    # Market portfolio point
    fig.add_trace(go.Scatter(
        x=[1.0], y=[rm * 100], mode="markers+text",
        marker=dict(color=ETH_BLUE, size=10), text=["Market"],
        textposition="top center", name="Market",
    ))

    _apply_layout(fig,
        title="CAPM — Security Market Line",
        xaxis_title="Beta (β)",
        yaxis_title="Annualised Return (%)",
        height=420,
    )
    return fig


# ── 15. Wavelet Coherence Heatmap ────────────────────────────────────────

def plot_wavelet_coherence(
    wavelet_result: dict,
    asset_name: str = "BTC",
    bench_name: str = "ETH",
):
    """
    Wavelet coherence heatmap: time × frequency.
    High coherence (yellow/red) = strong co-movement at that period.
    """
    coherence = wavelet_result["coherence"]
    periods = wavelet_result["periods"]
    times = wavelet_result["times"]

    fig = go.Figure(data=go.Heatmap(
        z=coherence,
        x=times,
        y=periods,
        colorscale="Inferno",
        zmin=0, zmax=1,
        colorbar=dict(title="Coherence", len=0.7),
        hovertemplate=(
            "Date: %{x|%Y-%m-%d}<br>"
            "Period: %{y:.0f} days<br>"
            "Coherence: %{z:.3f}<extra></extra>"
        ),
    ))

    _apply_layout(fig,
        title=f"{asset_name}/{bench_name} — Wavelet Coherence",
        xaxis_title="Date",
        yaxis_title="Period (days)",
        yaxis_type="log",
        height=480,
    )
    fig.update_yaxes(
        autorange="reversed",
        gridcolor=GRID_COLOR,
        tickvals=[4, 7, 14, 30, 60, 120],
        ticktext=["4d", "7d", "14d", "30d", "60d", "120d"],
    )
    return fig

