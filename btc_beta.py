#!/usr/bin/env python3
"""
btc_beta.py — CLI for BTC Beta Analytics.

Usage:
    python btc_beta.py --asset bitcoin --benchmark ethereum --days 365 --window 30
    python btc_beta.py --asset bitcoin --benchmark sp500 --days 180
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import build_merged_df, ASSET_CHOICES, BENCHMARK_CHOICES
from analytics import compute_all_stats
from visualizations import (
    plot_scatter_regression,
    plot_rolling_beta,
    plot_price_chart,
    plot_returns_distribution,
    plot_drawdown,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute beta and risk metrics for a crypto asset."
    )
    p.add_argument("--asset", default="bitcoin", choices=ASSET_CHOICES,
                   help="CoinGecko asset ID (default: bitcoin)")
    p.add_argument("--benchmark", default="ethereum", choices=BENCHMARK_CHOICES,
                   help="Benchmark ID (default: ethereum)")
    p.add_argument("--days", type=int, default=365,
                   help="Look-back in days (default: 365)")
    p.add_argument("--window", type=int, default=30,
                   help="Rolling beta window in days (default: 30)")
    p.add_argument("--freq", default="daily", choices=["daily", "weekly"],
                   help="Return frequency (default: daily)")
    p.add_argument("--output", default="btc_beta_plot.png",
                   help="Output filename for the plot (default: btc_beta_plot.png)")
    return p.parse_args()


def main():
    args = parse_args()
    asset_label = args.asset.replace("-", " ").title()
    bench_label = args.benchmark.replace("_", " ").title()

    print(f"{'='*60}")
    print(f"  Crypto Beta Analytics")
    print(f"  {asset_label} vs {bench_label}  ·  {args.days} days  ·  {args.freq}")
    print(f"{'='*60}\n")

    # ── Fetch data ─────────────────────────────────────────────────────
    print("📡 Fetching data...")
    try:
        df = build_merged_df(args.asset, args.benchmark, args.days, args.freq)
    except Exception as e:
        print(f"❌ Data fetch failed: {e}")
        sys.exit(1)

    print(f"✅ Loaded {len(df)} data points\n")

    # ── Compute stats ──────────────────────────────────────────────────
    print("🔢 Computing statistics...")
    stats = compute_all_stats(df, args.window)
    roll = stats.pop("rolling_beta")
    bp = stats["breusch_pagan"]

    print(f"""
┌──────────────────────────────────────┐
│  Beta (β)         {stats['beta']:>10.4f}         │
│  Alpha (α)        {stats['alpha']:>10.6f}       │
│  R²               {stats['r_squared']:>10.4f}         │
│  p-value          {stats['p_value']:>10.2e}       │
│  95% CI           [{stats['beta_ci_lower']:.4f}, {stats['beta_ci_upper']:.4f}]  │
├──────────────────────────────────────┤
│  {asset_label} Sharpe      {stats['asset_sharpe']:>10.2f}         │
│  {bench_label} Sharpe      {stats['benchmark_sharpe']:>10.2f}         │
│  {asset_label} Max DD      {stats['asset_max_dd']:>10.1%}         │
│  {asset_label} VaR (95%)   {stats['asset_var_95']:>10.2%}         │
│  {asset_label} CVaR (95%)  {stats['asset_cvar_95']:>10.2%}         │
├──────────────────────────────────────┤
│  Breusch-Pagan p  {bp['p_value']:>10.4f}         │
│  Heteroscedastic  {'   YES' if bp['heteroscedastic'] else '    NO':>10}         │
└──────────────────────────────────────┘
""")

    # ── Save composite plot ────────────────────────────────────────────
    print(f"📊 Generating plots → {args.output}")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Top-left: scatter + regression
    fig_scatter = plot_scatter_regression(df, stats, asset_label, bench_label)
    fig_scatter.savefig("_tmp_scatter.png", dpi=100, bbox_inches="tight")
    plt.close(fig_scatter)

    # Top-right: rolling beta
    fig_roll = plot_rolling_beta(roll, args.window, asset_label, bench_label)
    fig_roll.savefig("_tmp_roll.png", dpi=100, bbox_inches="tight")
    plt.close(fig_roll)

    # Bottom-left: returns distribution
    fig_dist = plot_returns_distribution(df, asset_label, bench_label)
    fig_dist.savefig("_tmp_dist.png", dpi=100, bbox_inches="tight")
    plt.close(fig_dist)

    # Bottom-right: drawdown
    fig_dd = plot_drawdown(df, asset_label, bench_label)
    fig_dd.savefig("_tmp_dd.png", dpi=100, bbox_inches="tight")
    plt.close(fig_dd)

    # Composite
    from PIL import Image
    import numpy as np

    imgs = [Image.open(f"_tmp_{n}.png") for n in ["scatter", "roll", "dist", "dd"]]
    w = max(im.width for im in imgs)
    h = max(im.height for im in imgs)
    composite = Image.new("RGB", (w * 2, h * 2), "white")
    composite.paste(imgs[0], (0, 0))
    composite.paste(imgs[1], (w, 0))
    composite.paste(imgs[2], (0, h))
    composite.paste(imgs[3], (w, h))
    composite.save(args.output, dpi=(150, 150))

    # Clean up temp files
    import os
    for n in ["scatter", "roll", "dist", "dd"]:
        os.remove(f"_tmp_{n}.png")

    print(f"✅ Saved to {args.output}")
    print(f"\n💡 For an interactive dashboard run:  streamlit run app.py")


if __name__ == "__main__":
    main()
