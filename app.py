"""
BTC Beta Analytics Dashboard
─────────────────────────────
Interactive Streamlit app with 7 tabs: Overview, Beta Analysis, DCC-GARCH,
Regimes, Backtesting, Risk & Diagnostics, and Multi-Asset.

All charts are interactive Plotly figures with hover, zoom, and range sliders.
"""

import streamlit as st
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
    plot_cusum, plot_fear_greed_overlay, plot_sml,
    plot_wavelet_coherence,
)
from events import filter_events
from export import generate_pdf, to_csv_bytes


# ── Sidebar ────────────────────────────────────────────────────────────────

st.sidebar.markdown("## ⚙️ Configuration")

asset = st.sidebar.selectbox("Asset", ["bitcoin", "ethereum"], index=0)
benchmark = st.sidebar.selectbox("Benchmark", ["ethereum", "bitcoin"], index=0)
days = st.sidebar.slider("Look-back (days)", 30, 730, 365, step=30)
window = st.sidebar.slider("Rolling window (days)", 14, 120, 30, step=7)
freq = st.sidebar.radio("Return frequency", ["daily", "weekly"], horizontal=True)
show_events = st.sidebar.toggle("📰 Event Overlay", value=True)
show_fg = st.sidebar.toggle("😱 Fear & Greed Overlay", value=True)

asset_label = asset.replace("-", " ").title()
bench_label = benchmark.replace("_", " ").title()


# ── Data loading ───────────────────────────────────────────────────────────

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

# Collect figures for PDF export
pdf_figures = []


# ── Header ─────────────────────────────────────────────────────────────────

st.markdown(
    f"# 📊 Crypto Beta Analytics\n"
    f"**{asset_label}** vs **{bench_label}**  ·  {days} days  ·  {freq}"
)


# ── Tabs ───────────────────────────────────────────────────────────────────

tab_overview, tab_beta, tab_dcc, tab_regime, tab_wavelet, tab_bt, tab_riskdiag, tab_multi = st.tabs([
    "📈 Overview", "🎯 Beta Analysis", "🧬 DCC-GARCH",
    "🔀 Regimes", "🌊 Wavelet", "📊 Backtesting", "⚠️ Risk & Diagnostics", "🏛️ Multi-Asset",
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
    st.plotly_chart(fig_price, use_container_width=True)
    pdf_figures.append(fig_price)

    # Fear & Greed overlay (if enabled)
    if show_fg:
        from fear_greed import fetch_fear_greed, merge_fear_greed
        fg = fetch_fear_greed(days)
        if fg is not None:
            df_fg = merge_fear_greed(df, fg)
            fig_fg = plot_fear_greed_overlay(df_fg, asset_label)
            st.plotly_chart(fig_fg, use_container_width=True)
            pdf_figures.append(fig_fg)
        else:
            st.caption("⚠️ Fear & Greed data unavailable.")

    with st.expander("📋 Raw Data Preview"):
        st.dataframe(df.tail(20), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Beta Analysis + CAPM
# ═══════════════════════════════════════════════════════════════════════════

with tab_beta:
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("### Scatter + Regression")
        fig_scatter = plot_scatter_regression(df, stats, asset_label, bench_label)
        st.plotly_chart(fig_scatter, use_container_width=True)
        pdf_figures.append(fig_scatter)

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
        st.plotly_chart(fig_roll_ev, use_container_width=True)
        pdf_figures.append(fig_roll_ev)
    else:
        fig_roll = plot_rolling_beta(roll, window, asset_label, bench_label)
        st.plotly_chart(fig_roll, use_container_width=True)
        pdf_figures.append(fig_roll)

    # ── CAPM Section ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📐 CAPM Framework")

    c1, c2, c3 = st.columns(3)
    c1.metric("Treynor Ratio", f"{stats['treynor']:.4f}")
    c2.metric("Jensen's Alpha", f"{stats['jensens_alpha']:.4%}")
    c3.metric("CAPM Expected Return", f"{stats['capm_expected_return']:.2%}")

    col_l, col_r = st.columns([2, 1])
    with col_l:
        fig_sml = plot_sml(stats, asset_label, bench_label)
        st.plotly_chart(fig_sml, use_container_width=True)
        pdf_figures.append(fig_sml)
    with col_r:
        st.markdown(f"""
**Treynor Ratio** measures excess return per unit of *systematic* (market) risk.
Higher = better risk-adjusted performance.

**Jensen's Alpha** shows the return above/below what CAPM predicts for this beta.
Positive α means the asset outperformed its CAPM-implied return.

**SML Chart** plots the Security Market Line. Assets above the line have positive
Jensen's alpha (outperforming), below = underperforming.
        """)


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
        import plotly.graph_objects as go

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
        st.plotly_chart(fig_dcc, use_container_width=True)
        pdf_figures.append(fig_dcc)

        # Conditional correlation chart
        st.markdown("### Conditional Correlation")
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Scatter(
            x=cond_corr.index, y=cond_corr.values,
            mode="lines", line=dict(color="#9b59b6", width=1.5),
            name="Conditional Correlation",
            hovertemplate="%{x|%Y-%m-%d}<br>Corr: %{y:.3f}<extra></extra>",
        ))
        fig_corr.add_hline(y=cond_corr.mean(), line_dash="dot", line_color="grey",
                           annotation_text=f"Mean = {cond_corr.mean():.3f}")
        fig_corr.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#1a1d23",
            title=f"{asset_label}/{bench_label} — Conditional Correlation",
            yaxis_title="Correlation", yaxis_range=[-0.2, 1.1], height=380,
            font=dict(color="#fafafa"),
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        pdf_figures.append(fig_corr)

        # GARCH volatility comparison
        st.markdown("### GARCH Conditional Volatility")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=asset_vol.index, y=asset_vol.values * (365**0.5) * 100,
            mode="lines", line=dict(color="#f7931a", width=1.5),
            name=asset_label,
        ))
        fig_vol.add_trace(go.Scatter(
            x=bench_vol.index, y=bench_vol.values * (365**0.5) * 100,
            mode="lines", line=dict(color="#627eea", width=1.5),
            name=bench_label,
        ))
        fig_vol.update_layout(
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#1a1d23",
            title="GARCH Conditional Volatility",
            yaxis_title="Annualised Vol (%)", height=380,
            font=dict(color="#fafafa"),
        )
        st.plotly_chart(fig_vol, use_container_width=True)
        pdf_figures.append(fig_vol)
    else:
        st.warning("DCC-GARCH model could not be fitted. Install the `arch` package or check data.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Regimes
# ═══════════════════════════════════════════════════════════════════════════

with tab_regime:
    st.markdown("### 🔀 Market Regime Detection")
    st.caption("Gaussian Hidden Markov Model (3 states)")

    with st.spinner("Fitting HMM…"):
        from regime import detect_regimes, regime_beta
        regimes = detect_regimes(df["Asset_Returns"])

    if regimes is not None:
        regime_counts = regimes.value_counts()
        c1, c2, c3 = st.columns(3)
        for i, (regime, count) in enumerate(regime_counts.items()):
            pct = count / len(regimes) * 100
            [c1, c2, c3][i % 3].metric(f"{regime} Days", f"{count} ({pct:.0f}%)")

        st.markdown("---")

        fig_regime = plot_regime_chart(df, regimes, asset_label)
        st.plotly_chart(fig_regime, use_container_width=True)
        pdf_figures.append(fig_regime)

        st.markdown("### Per-Regime Beta")
        rbeta = regime_beta(df, regimes)
        st.dataframe(rbeta, use_container_width=True)
    else:
        st.warning("Regime detection unavailable. Install `hmmlearn`.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — Wavelet Coherence
# ═══════════════════════════════════════════════════════════════════════════

with tab_wavelet:
    st.markdown("### 🌊 Wavelet Coherence Analysis")
    st.caption(
        "Frequency-domain decomposition of co-movement. "
        "Bright colours = strong coherence at that period & time."
    )

    with st.spinner("Computing wavelet transform…"):
        from wavelet import compute_wavelet_coherence
        wc = compute_wavelet_coherence(df)

    fig_wc = plot_wavelet_coherence(wc, asset_label, bench_label)
    st.plotly_chart(fig_wc, use_container_width=True)
    pdf_figures.append(fig_wc)

    # Frequency band summary table
    st.markdown("### Frequency Band Summary")
    st.caption("Average coherence across key periodicities")

    import pandas as pd
    fb_df = pd.DataFrame(wc["freq_bands"])
    if not fb_df.empty:
        st.dataframe(fb_df, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough data for frequency band analysis.")

    with st.expander("📖 How to Read This Chart"):
        st.markdown("""
**X-axis**: Time  
**Y-axis**: Period (in days, log scale) — lower = faster cycles, higher = slower cycles  
**Colour**: Coherence from 0 (no co-movement, dark) to 1 (perfect co-movement, bright)

**Interpretation:**
- **Bright band at 7–14d**: BTC and ETH move together on weekly cycles
- **Bright band at 30–60d**: Monthly macro cycles drive co-movement
- **Dark patches**: Temporary decoupling at that frequency
- **Bright across all frequencies**: Strong full-spectrum co-movement (typical in risk-off events)
        """)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — Backtesting
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
    st.plotly_chart(fig_bt, use_container_width=True)
    pdf_figures.append(fig_bt)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — Risk & Diagnostics (merged)
# ═══════════════════════════════════════════════════════════════════════════

with tab_riskdiag:

    # ── Risk Metrics ──────────────────────────────────────────────────────
    st.markdown("### ⚠️ Risk Metrics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{asset_label} Max DD", f"{stats['asset_max_dd']:.1%}")
    c2.metric(f"{bench_label} Max DD", f"{stats['benchmark_max_dd']:.1%}")
    c3.metric(f"{asset_label} VaR (95%)", f"{stats['asset_var_95']:.2%}")
    c4.metric(f"{asset_label} CVaR (95%)", f"{stats['asset_cvar_95']:.2%}")

    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        fig_dist = plot_returns_distribution(df, asset_label, bench_label)
        st.plotly_chart(fig_dist, use_container_width=True)
        pdf_figures.append(fig_dist)
    with col_r:
        fig_dd = plot_drawdown(df, asset_label, bench_label)
        st.plotly_chart(fig_dd, use_container_width=True)
        pdf_figures.append(fig_dd)

    # ── Diagnostics ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔬 Diagnostics")

    bp = stats["breusch_pagan"]

    col_l, col_r = st.columns([2, 1])
    with col_l:
        fig_resid = plot_residuals(df, stats)
        st.plotly_chart(fig_resid, use_container_width=True)
        pdf_figures.append(fig_resid)
    with col_r:
        st.markdown("#### Breusch-Pagan Test")
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
    st.plotly_chart(fig_corr_mat, use_container_width=True)
    pdf_figures.append(fig_corr_mat)

    # ── Structural Breaks ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📐 Structural Break Detection (CUSUM)")
    st.caption("Tests whether the beta relationship has structurally changed over time")

    from structural_breaks import cusum_test, detect_beta_regimes
    cusum_result = cusum_test(df, roll)

    if cusum_result is not None:
        if cusum_result["has_break"]:
            n_breaks = len(cusum_result["break_dates"])
            st.error(f"🔴 **{n_breaks} structural break(s) detected** — beta relationship is unstable")
        else:
            st.success("✅ No structural breaks — beta relationship is stable over this period")

        fig_cusum = plot_cusum(cusum_result, asset_label, bench_label)
        st.plotly_chart(fig_cusum, use_container_width=True)
        pdf_figures.append(fig_cusum)

        st.markdown("#### Beta Period Analysis")
        beta_periods = detect_beta_regimes(roll)
        if not beta_periods.empty:
            st.dataframe(
                beta_periods[["Period", "Mean β", "Std", "Trend"]],
                use_container_width=True, hide_index=True,
            )
    else:
        st.info("Not enough data for structural break detection.")

    # ── Cointegration ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔗 Cointegration Tests")
    st.caption(f"Do {asset_label} and {bench_label} share a long-run equilibrium?")

    from cointegration import engle_granger_test, johansen_test

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Engle-Granger (2-step)")
        eg = engle_granger_test(df)
        if eg is not None:
            st.markdown(f"""
| Metric | Value |
|---|---|
| ADF Statistic | {eg['adf_stat']:.4f} |
| p-value | {eg['p_value']:.4f} |
| Hedge Ratio | {eg['hedge_ratio']:.4f} |
            """)
            for level, crit in eg["critical_values"].items():
                st.caption(f"Critical value ({level}): {crit}")
            if eg["cointegrated"]:
                st.success(f"✅ {eg['interpretation']}")
            else:
                st.warning(f"⚠️ {eg['interpretation']}")
        else:
            st.info("Install `statsmodels` for cointegration tests.")

    with col_r:
        st.markdown("#### Johansen Test")
        joh = johansen_test(df)
        if joh is not None:
            for i in range(len(joh["trace_stats"])):
                sig = "✅" if joh["trace_stats"][i] > joh["crit_trace_5pct"][i] else "❌"
                st.markdown(
                    f"**r ≤ {i}:** Trace = {joh['trace_stats'][i]:.2f} "
                    f"(5% crit = {joh['crit_trace_5pct'][i]:.2f}) {sig}"
                )
            st.markdown(f"**Result:** {joh['interpretation']}")
        else:
            st.info("Install `statsmodels` for Johansen test.")

    # ── Granger Causality ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧪 Granger Causality Test")
    st.caption(f"Does {asset_label} lead {bench_label}, or vice versa?")

    from granger import granger_causality_test
    gc = granger_causality_test(df, max_lag=5)

    if gc is not None:
        st.markdown(f"**Dominant direction:** {gc['best_direction']}")

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"#### {asset_label} → {bench_label}")
            gc_ab_df = pd.DataFrame(
                gc["asset_causes_bench"], columns=["Lag", "F-stat", "p-value"],
            )
            gc_ab_df["Significant"] = gc_ab_df["p-value"].apply(
                lambda p: "✅" if p < 0.05 else "❌"
            )
            st.dataframe(
                gc_ab_df.style.format({"F-stat": "{:.3f}", "p-value": "{:.4f}"}),
                use_container_width=True, hide_index=True,
            )

        with col_r:
            st.markdown(f"#### {bench_label} → {asset_label}")
            gc_ba_df = pd.DataFrame(
                gc["bench_causes_asset"], columns=["Lag", "F-stat", "p-value"],
            )
            gc_ba_df["Significant"] = gc_ba_df["p-value"].apply(
                lambda p: "✅" if p < 0.05 else "❌"
            )
            st.dataframe(
                gc_ba_df.style.format({"F-stat": "{:.3f}", "p-value": "{:.4f}"}),
                use_container_width=True, hide_index=True,
            )
    else:
        st.info("Install `statsmodels` for Granger causality tests.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 7 — Multi-Asset (power-user, at the end)
# ═══════════════════════════════════════════════════════════════════════════

with tab_multi:
    st.markdown("### 🏛️ Multi-Asset Beta Matrix")
    st.caption("Compare beta across multiple coins against the selected benchmark")

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
                returns_df = prices_df.pct_change().dropna()
                if bench in returns_df.columns:
                    returns_df.rename(columns={bench: "benchmark"}, inplace=True)
                else:
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
                    st.plotly_chart(fig_bheat, use_container_width=True)
                    pdf_figures.append(fig_bheat)
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
st.caption("Data: CoinGecko (free API) & Yahoo Finance  ·  Built with Streamlit + Plotly")
st.caption("☁️ Deploy: `streamlit deploy app.py` or push to Streamlit Community Cloud")
