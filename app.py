"""
BTC Beta Analytics Dashboard
─────────────────────────────
Interactive Streamlit app with 4 tabs: Overview, Beta Analysis,
Risk Metrics, and Diagnostics.
"""

import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Crypto Beta Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports (after st.set_page_config) ─────────────────────────────────────
from data import build_merged_df, ASSET_CHOICES, BENCHMARK_CHOICES
from analytics import (
    compute_all_stats,
    rolling_beta,
    ols_residuals,
    drawdown_series,
)
from visualizations import (
    plot_scatter_regression,
    plot_rolling_beta,
    plot_price_chart,
    plot_returns_distribution,
    plot_residuals,
    plot_drawdown,
    plot_correlation_heatmap,
)


# ── Sidebar ────────────────────────────────────────────────────────────────

st.sidebar.markdown("## ⚙️ Configuration")

asset = st.sidebar.selectbox(
    "Asset",
    ASSET_CHOICES,
    index=0,
    help="The crypto asset to analyse.",
)
benchmark = st.sidebar.selectbox(
    "Benchmark",
    BENCHMARK_CHOICES,
    index=0,
    help="The benchmark to compute beta against.",
)
days = st.sidebar.slider("Look-back (days)", 30, 730, 365, step=30)
window = st.sidebar.slider("Rolling window (days)", 14, 120, 30, step=7)
freq = st.sidebar.radio("Return frequency", ["daily", "weekly"], horizontal=True)

asset_label = asset.replace("-", " ").title()
bench_label = benchmark.replace("_", " ").title()

# ── Data loading (cached) ─────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching data…", ttl=3600)
def load_data(asset_id, benchmark_id, days, freq):
    return build_merged_df(asset_id, benchmark_id, days, freq)


try:
    df = load_data(asset, benchmark, days, freq)
except Exception as e:
    st.error(f"**Data fetch failed:** {e}")
    st.info("CoinGecko's free API is rate-limited. Wait a minute and reload.")
    st.stop()

stats = compute_all_stats(df, window)
roll = stats.pop("rolling_beta")

# ── Header ─────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    # 📊 Crypto Beta Analytics
    **{asset_label}** vs **{bench_label}**  ·  {days} days  ·  {freq} returns
    """,
    unsafe_allow_html=True,
)

# ── Tabs ───────────────────────────────────────────────────────────────────

tab_overview, tab_beta, tab_risk, tab_diag = st.tabs(
    ["📈 Overview", "🎯 Beta Analysis", "⚠️ Risk Metrics", "🔬 Diagnostics"]
)

# ·· Tab 1 — Overview ·····················································

with tab_overview:
    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Beta (β)", f"{stats['beta']:.3f}")
    c2.metric("Alpha (α)", f"{stats['alpha']:.5f}")
    c3.metric("R²", f"{stats['r_squared']:.3f}")
    c4.metric(f"{asset_label} Sharpe", f"{stats['asset_sharpe']:.2f}")
    c5.metric(f"{bench_label} Sharpe", f"{stats['benchmark_sharpe']:.2f}")

    st.markdown("---")

    # Price chart
    st.pyplot(plot_price_chart(df, asset_label, bench_label))
    plt.close("all")

    # DataFrame preview
    with st.expander("📋 Raw Data Preview"):
        st.dataframe(df.tail(20), use_container_width=True)

# ·· Tab 2 — Beta Analysis ················································

with tab_beta:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Scatter + Regression")
        st.pyplot(plot_scatter_regression(df, stats, asset_label, bench_label))
        plt.close("all")

    with col_right:
        st.markdown("### Beta Statistics")
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | **Beta** | {stats['beta']:.4f} |
        | **Alpha** | {stats['alpha']:.6f} |
        | **95% CI** | [{stats['beta_ci_lower']:.4f}, {stats['beta_ci_upper']:.4f}] |
        | **R²** | {stats['r_squared']:.4f} |
        | **p-value** | {stats['p_value']:.2e} |
        | **Std Error** | {stats['std_err']:.4f} |
        | **Observations** | {stats['n_obs']} |
        """)

        if stats["p_value"] < 0.01:
            st.success("✅ Beta is **highly significant** (p < 0.01)")
        elif stats["p_value"] < 0.05:
            st.warning("⚠️ Beta is **significant** (p < 0.05)")
        else:
            st.error("❌ Beta is **not significant** (p ≥ 0.05)")

    st.markdown("---")
    st.markdown(f"### {window}-Day Rolling Beta")
    st.pyplot(plot_rolling_beta(roll, window, asset_label, bench_label))
    plt.close("all")

# ·· Tab 3 — Risk Metrics ·················································

with tab_risk:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{asset_label} Max DD", f"{stats['asset_max_dd']:.1%}")
    c2.metric(f"{bench_label} Max DD", f"{stats['benchmark_max_dd']:.1%}")
    c3.metric(f"{asset_label} VaR (95%)", f"{stats['asset_var_95']:.2%}")
    c4.metric(f"{asset_label} CVaR (95%)", f"{stats['asset_cvar_95']:.2%}")

    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.pyplot(plot_returns_distribution(df, asset_label, bench_label))
        plt.close("all")
    with col_r:
        st.pyplot(plot_drawdown(df, asset_label, bench_label))
        plt.close("all")

# ·· Tab 4 — Diagnostics ··················································

with tab_diag:
    bp = stats["breusch_pagan"]

    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.pyplot(plot_residuals(df, stats))
        plt.close("all")
    with col_r:
        st.markdown("### Breusch-Pagan Test")
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Test statistic | {bp['test_stat']:.4f} |
        | p-value | {bp['p_value']:.4f} |
        """)
        if bp["heteroscedastic"]:
            st.warning(f"⚠️ {bp['interpretation']}")
        else:
            st.success(f"✅ {bp['interpretation']}")

    st.markdown("---")
    st.markdown("### Correlation Matrix")
    st.pyplot(plot_correlation_heatmap(df))
    plt.close("all")

# ── Footer ─────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption("Data: CoinGecko (free API) & Yahoo Finance  ·  Built with Streamlit")
