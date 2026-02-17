"""
BTC Beta Analytics Dashboard
─────────────────────────────
Interactive Streamlit app with 8 tabs: Overview, Beta Analysis, DCC-GARCH,
Multi-Asset, Regimes, Risk Metrics, Backtesting, and Diagnostics.
"""

import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(
    page_title="Crypto Beta Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ────────────────────────────────────────────────────────────────
from data import (
    build_merged_df, fetch_multi_assets,
    ASSET_CHOICES, BENCHMARK_CHOICES,
)
from analytics import (
    compute_all_stats, rolling_beta, multi_asset_beta_matrix,
)
from visualizations import (
    plot_scatter_regression, plot_rolling_beta, plot_price_chart,
    plot_returns_distribution, plot_residuals, plot_drawdown,
    plot_correlation_heatmap, plot_beta_heatmap, plot_dcc_vs_rolling,
    plot_regime_chart, plot_rolling_beta_with_events,
)
from events import filter_events
from export import generate_pdf, to_csv_bytes


# ── Sidebar ────────────────────────────────────────────────────────────────

st.sidebar.markdown("## ⚙️ Configuration")

asset = st.sidebar.selectbox("Asset", ASSET_CHOICES, index=0)
benchmark = st.sidebar.selectbox("Benchmark", BENCHMARK_CHOICES, index=0)
days = st.sidebar.slider("Look-back (days)", 30, 730, 365, step=30)
window = st.sidebar.slider("Rolling window (days)", 14, 120, 30, step=7)
freq = st.sidebar.radio("Return frequency", ["daily", "weekly"], horizontal=True)
granularity = st.sidebar.radio("Granularity", ["daily", "hourly"], horizontal=True,
                                help="Hourly is limited to ≤90 days")
show_events = st.sidebar.toggle("📰 Event Overlay", value=True)

asset_label = asset.replace("-", " ").title()
bench_label = benchmark.replace("_", " ").title()


# ── Data loading ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching data…", ttl=3600)
def load_data(asset_id, benchmark_id, days, freq, granularity):
    return build_merged_df(asset_id, benchmark_id, days, freq, granularity)


try:
    df = load_data(asset, benchmark, days, freq, granularity)
except Exception as e:
    st.error(f"**Data fetch failed:** {e}")
    st.info("CoinGecko's free API is rate-limited. Wait a minute and reload.")
    st.stop()

stats = compute_all_stats(df, window)
roll = stats.pop("rolling_beta")

# Collect figures for PDF export
pdf_figures = []


# ── Header ─────────────────────────────────────────────────────────────────

st.markdown(
    f"# 📊 Crypto Beta Analytics\n"
    f"**{asset_label}** vs **{bench_label}**  ·  {days} days  ·  {freq}  ·  {granularity}"
)


# ── Tabs ───────────────────────────────────────────────────────────────────

tab_overview, tab_beta, tab_dcc, tab_multi, tab_regime, tab_risk, tab_bt, tab_diag = st.tabs([
    "📈 Overview", "🎯 Beta Analysis", "🧬 DCC-GARCH", "🏛️ Multi-Asset",
    "🔀 Regimes", "⚠️ Risk Metrics", "📊 Backtesting", "🔬 Diagnostics",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ═══════════════════════════════════════════════════════════════════════════

with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Beta (β)", f"{stats['beta']:.3f}")
    c2.metric("Alpha (α)", f"{stats['alpha']:.5f}")
    c3.metric("R²", f"{stats['r_squared']:.3f}")
    c4.metric(f"{asset_label} Sharpe", f"{stats['asset_sharpe']:.2f}")
    c5.metric(f"{bench_label} Sharpe", f"{stats['benchmark_sharpe']:.2f}")

    st.markdown("---")

    fig_price = plot_price_chart(df, asset_label, bench_label)
    st.pyplot(fig_price)
    pdf_figures.append(fig_price)
    plt.close("all")

    with st.expander("📋 Raw Data Preview"):
        st.dataframe(df.tail(20), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Beta Analysis
# ═══════════════════════════════════════════════════════════════════════════

with tab_beta:
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("### Scatter + Regression")
        fig_scatter = plot_scatter_regression(df, stats, asset_label, bench_label)
        st.pyplot(fig_scatter)
        pdf_figures.append(fig_scatter)
        plt.close("all")

    with col_r:
        st.markdown("### Beta Statistics")
        st.markdown(f"""
| Metric | Value |
|---|---|
| **Beta** | {stats['beta']:.4f} |
| **Alpha** | {stats['alpha']:.6f} |
| **95% CI** | [{stats['beta_ci_lower']:.4f}, {stats['beta_ci_upper']:.4f}] |
| **R²** | {stats['r_squared']:.4f} |
| **p-value** | {stats['p_value']:.2e} |
| **Std Error** | {stats['std_err']:.4f} |
| **Observations** | {stats['n_obs']} |
        """)
        if stats["p_value"] < 0.01:
            st.success("✅ Highly significant (p < 0.01)")
        elif stats["p_value"] < 0.05:
            st.warning("⚠️ Significant (p < 0.05)")
        else:
            st.error("❌ Not significant (p ≥ 0.05)")

    st.markdown("---")
    st.markdown(f"### {window}-Day Rolling Beta")

    if show_events:
        evts = filter_events(df.index.min(), df.index.max())
        fig_roll_ev = plot_rolling_beta_with_events(roll, evts, window, asset_label, bench_label)
        st.pyplot(fig_roll_ev)
        pdf_figures.append(fig_roll_ev)
    else:
        fig_roll = plot_rolling_beta(roll, window, asset_label, bench_label)
        st.pyplot(fig_roll)
        pdf_figures.append(fig_roll)
    plt.close("all")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — DCC-GARCH
# ═══════════════════════════════════════════════════════════════════════════

with tab_dcc:
    st.markdown("### 🧬 DCC-GARCH Time-Varying Beta")
    st.caption("GARCH(1,1) + Dynamic Conditional Correlation (EWMA)")

    with st.spinner("Fitting DCC-GARCH model…"):
        from dcc_garch import fit_dcc_garch
        dcc_result = fit_dcc_garch(df)

    if dcc_result is not None:
        dcc_beta = dcc_result["dcc_beta"]
        cond_corr = dcc_result["cond_corr"]
        asset_vol = dcc_result["asset_vol"]
        bench_vol = dcc_result["bench_vol"]

        c1, c2, c3 = st.columns(3)
        c1.metric("DCC β (latest)", f"{dcc_beta.iloc[-1]:.3f}")
        c2.metric("Cond. Correlation", f"{cond_corr.iloc[-1]:.3f}")
        c3.metric("Asset Vol (ann.)", f"{asset_vol.iloc[-1] * (365**0.5):.1%}")

        st.markdown("---")

        fig_dcc = plot_dcc_vs_rolling(dcc_beta, roll, window, asset_label, bench_label)
        st.pyplot(fig_dcc)
        pdf_figures.append(fig_dcc)
        plt.close("all")

        # Conditional correlation chart
        st.markdown("### Conditional Correlation")
        fig_corr, ax_corr = plt.subplots(figsize=(12, 4))
        ax_corr.plot(cond_corr.index, cond_corr.values, color="#9b59b6", linewidth=1.5)
        ax_corr.axhline(cond_corr.mean(), color="grey", linestyle=":", alpha=0.7,
                        label=f"Mean = {cond_corr.mean():.3f}")
        ax_corr.set_ylabel("Correlation", fontsize=11)
        ax_corr.set_title(f"{asset_label}/{bench_label} — Conditional Correlation",
                          fontsize=13, fontweight="bold")
        ax_corr.legend()
        ax_corr.set_ylim(-0.2, 1.1)
        fig_corr.tight_layout()
        st.pyplot(fig_corr)
        pdf_figures.append(fig_corr)
        plt.close("all")

        # GARCH volatility comparison
        st.markdown("### GARCH Conditional Volatility")
        fig_vol, ax_vol = plt.subplots(figsize=(12, 4))
        ax_vol.plot(asset_vol.index, asset_vol.values * (365**0.5) * 100,
                    color="#f7931a", linewidth=1.3, label=f"{asset_label}")
        ax_vol.plot(bench_vol.index, bench_vol.values * (365**0.5) * 100,
                    color="#627eea", linewidth=1.3, label=f"{bench_label}")
        ax_vol.set_ylabel("Annualised Vol (%)", fontsize=11)
        ax_vol.set_title("GARCH Conditional Volatility", fontsize=13, fontweight="bold")
        ax_vol.legend()
        fig_vol.tight_layout()
        st.pyplot(fig_vol)
        pdf_figures.append(fig_vol)
        plt.close("all")
    else:
        st.warning("DCC-GARCH model could not be fitted. Install the `arch` package or check data.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Multi-Asset
# ═══════════════════════════════════════════════════════════════════════════

with tab_multi:
    st.markdown("### 🏛️ Multi-Asset Beta Matrix")

    multi_coins = st.multiselect(
        "Select assets to compare",
        ASSET_CHOICES,
        default=["bitcoin", "ethereum", "solana", "cardano", "dogecoin"],
    )

    if len(multi_coins) >= 2:
        with st.spinner(f"Fetching {len(multi_coins)} assets…"):
            @st.cache_data(show_spinner=False, ttl=3600)
            def load_multi(coins, bench, d):
                prices_df = fetch_multi_assets(coins, days=d)
                # Compute returns
                returns_df = prices_df.pct_change().dropna()
                # Add benchmark column
                if bench in returns_df.columns:
                    returns_df.rename(columns={bench: "benchmark"}, inplace=True)
                else:
                    # Fetch separately
                    from data import fetch_coin_data, BENCHMARK_MAP
                    if bench in BENCHMARK_MAP:
                        bdf = BENCHMARK_MAP[bench](d)
                    else:
                        bdf = fetch_coin_data(bench, days=d)
                    bdf.index = bdf.index.normalize()
                    bdf = bdf[~bdf.index.duplicated(keep="first")]
                    bdf_ret = bdf["Price"].pct_change().dropna()
                    returns_df["benchmark"] = bdf_ret
                    returns_df.dropna(inplace=True)
                return returns_df

            try:
                multi_returns = load_multi(multi_coins, benchmark, days)
                beta_mat = multi_asset_beta_matrix(multi_returns, "benchmark")

                col_l, col_r = st.columns([2, 1])
                with col_l:
                    fig_bheat = plot_beta_heatmap(beta_mat)
                    st.pyplot(fig_bheat)
                    pdf_figures.append(fig_bheat)
                    plt.close("all")
                with col_r:
                    st.markdown("### Beta Table")
                    st.dataframe(beta_mat.style.format({"Beta": "{:.4f}", "Alpha": "{:.6f}",
                                                        "R²": "{:.4f}", "Sharpe": "{:.2f}"}),
                                 use_container_width=True)
            except Exception as e:
                st.error(f"Multi-asset fetch failed: {e}")
    else:
        st.info("Select at least 2 assets to compare.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — Regimes
# ═══════════════════════════════════════════════════════════════════════════

with tab_regime:
    st.markdown("### 🔀 Market Regime Detection")
    st.caption("Gaussian Hidden Markov Model (3 states)")

    with st.spinner("Fitting HMM…"):
        from regime import detect_regimes, regime_beta
        regimes = detect_regimes(df["Asset_Returns"])

    if regimes is not None:
        # Regime distribution
        regime_counts = regimes.value_counts()
        c1, c2, c3 = st.columns(3)
        for i, (regime, count) in enumerate(regime_counts.items()):
            pct = count / len(regimes) * 100
            [c1, c2, c3][i % 3].metric(f"{regime} Days", f"{count} ({pct:.0f}%)")

        st.markdown("---")

        fig_regime = plot_regime_chart(df, regimes, asset_label)
        st.pyplot(fig_regime)
        pdf_figures.append(fig_regime)
        plt.close("all")

        st.markdown("### Per-Regime Beta")
        rbeta = regime_beta(df, regimes)
        st.dataframe(rbeta, use_container_width=True)
    else:
        st.warning("Regime detection unavailable. Install `hmmlearn`.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — Risk Metrics
# ═══════════════════════════════════════════════════════════════════════════

with tab_risk:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{asset_label} Max DD", f"{stats['asset_max_dd']:.1%}")
    c2.metric(f"{bench_label} Max DD", f"{stats['benchmark_max_dd']:.1%}")
    c3.metric(f"{asset_label} VaR (95%)", f"{stats['asset_var_95']:.2%}")
    c4.metric(f"{asset_label} CVaR (95%)", f"{stats['asset_cvar_95']:.2%}")

    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        fig_dist = plot_returns_distribution(df, asset_label, bench_label)
        st.pyplot(fig_dist)
        pdf_figures.append(fig_dist)
        plt.close("all")
    with col_r:
        fig_dd = plot_drawdown(df, asset_label, bench_label)
        st.pyplot(fig_dd)
        pdf_figures.append(fig_dd)
        plt.close("all")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 7 — Backtesting
# ═══════════════════════════════════════════════════════════════════════════

with tab_bt:
    st.markdown("### 📊 Beta-Hedging Backtest")
    st.caption("Compare unhedged asset returns vs a beta-hedged portfolio")

    from backtest import hedge_portfolio, backtest_stats, plot_backtest

    hedge_mode = st.radio(
        "Hedge ratio",
        ["Use computed beta", "Custom"],
        horizontal=True,
    )
    if hedge_mode == "Custom":
        custom_ratio = st.slider("Custom hedge ratio", 0.0, 2.0, float(stats["beta"]), step=0.05)
        h_ratio = custom_ratio
    else:
        h_ratio = stats["beta"]

    bt_df = hedge_portfolio(df, h_ratio)

    # Stats comparison
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"#### Unhedged ({asset_label})")
        unhedged_s = backtest_stats(bt_df["Unhedged"])
        for k, v in unhedged_s.items():
            st.markdown(f"**{k}:** {v}")
    with col_r:
        st.markdown(f"#### Beta-Hedged (β={h_ratio:.2f})")
        hedged_s = backtest_stats(bt_df["Hedged"])
        for k, v in hedged_s.items():
            st.markdown(f"**{k}:** {v}")

    st.markdown("---")

    fig_bt = plot_backtest(bt_df, h_ratio, asset_label, bench_label)
    st.pyplot(fig_bt)
    pdf_figures.append(fig_bt)
    plt.close("all")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 8 — Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

with tab_diag:
    bp = stats["breusch_pagan"]

    col_l, col_r = st.columns([2, 1])
    with col_l:
        fig_resid = plot_residuals(df, stats)
        st.pyplot(fig_resid)
        pdf_figures.append(fig_resid)
        plt.close("all")
    with col_r:
        st.markdown("### Breusch-Pagan Test")
        st.markdown(f"""
| Metric | Value |
|---|---|
| Test statistic | {bp['test_stat']:.4f} |
| p-value | {bp['p_value']:.4f} |
        """)
        if bp["heteroscedastic"]:
            st.warning(f"⚠️ {bp['interpretation']}")
        else:
            st.success(f"✅ {bp['interpretation']}")

    st.markdown("---")
    st.markdown("### Correlation Matrix")
    fig_corr_mat = plot_correlation_heatmap(df)
    st.pyplot(fig_corr_mat)
    pdf_figures.append(fig_corr_mat)
    plt.close("all")


# ═══════════════════════════════════════════════════════════════════════════
# Export Sidebar
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("---")
st.sidebar.markdown("## 📤 Export")

# PDF download
if st.sidebar.button("📄 Generate PDF Report"):
    with st.spinner("Generating PDF…"):
        pdf_bytes = generate_pdf(pdf_figures, stats, asset_label, bench_label, days)
    st.sidebar.download_button(
        "⬇️ Download PDF",
        data=pdf_bytes,
        file_name=f"beta_report_{asset}_{benchmark}_{days}d.pdf",
        mime="application/pdf",
    )

# CSV download
csv_bytes = to_csv_bytes(df)
st.sidebar.download_button(
    "📊 Download CSV",
    data=csv_bytes,
    file_name=f"data_{asset}_{benchmark}_{days}d.csv",
    mime="text/csv",
)


# ── Footer ─────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption("Data: CoinGecko (free API) & Yahoo Finance  ·  Built with Streamlit")
st.caption("☁️ Deploy: `streamlit deploy app.py` or push to Streamlit Community Cloud")
