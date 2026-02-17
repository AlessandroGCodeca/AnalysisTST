"""
events.py — Key crypto / macro event annotations for chart overlays.
"""

import pandas as pd
from datetime import datetime


# ── Curated event list ─────────────────────────────────────────────────────
# Each entry: (date_str, label, category)
# Categories: "regulatory", "macro", "hack", "protocol", "etf", "halving"

EVENTS = [
    # 2024
    ("2024-01-10", "BTC Spot ETF Approved", "etf"),
    ("2024-01-31", "Fed Holds Rates", "macro"),
    ("2024-03-05", "BTC Hits $69k ATH (pre-halving)", "protocol"),
    ("2024-03-20", "Fed Holds Rates", "macro"),
    ("2024-04-20", "BTC Halving #4", "halving"),
    ("2024-05-01", "Fed Holds Rates", "macro"),
    ("2024-05-23", "ETH Spot ETF Approved", "etf"),
    ("2024-06-12", "Fed Holds Rates", "macro"),
    ("2024-07-05", "Mt. Gox Repayments Begin", "regulatory"),
    ("2024-07-31", "Fed Holds Rates", "macro"),
    ("2024-09-18", "Fed Cuts 50bps", "macro"),
    ("2024-11-05", "US Election Day", "macro"),
    ("2024-11-07", "Fed Cuts 25bps", "macro"),
    ("2024-12-05", "BTC Breaks $100k", "protocol"),
    ("2024-12-18", "Fed Cuts 25bps", "macro"),

    # 2025
    ("2025-01-20", "Trump Inauguration", "macro"),
    ("2025-01-29", "Fed Holds Rates", "macro"),
    ("2025-02-03", "Trump Tariffs Shock", "macro"),
    ("2025-02-21", "Bybit $1.5B Hack", "hack"),
    ("2025-03-02", "Trump Crypto Reserve Announcement", "regulatory"),
    ("2025-03-07", "WH Crypto Summit", "regulatory"),
    ("2025-03-19", "Fed Holds Rates", "macro"),
    ("2025-05-07", "Fed Holds Rates", "macro"),
    ("2025-06-18", "Fed Holds Rates", "macro"),
    ("2025-07-30", "Fed Holds Rates (expected)", "macro"),

    # 2026
    ("2026-01-28", "Fed Meeting", "macro"),
]

# Convert to DataFrame
_events_df = pd.DataFrame(EVENTS, columns=["Date", "Label", "Category"])
_events_df["Date"] = pd.to_datetime(_events_df["Date"])


CATEGORY_COLORS = {
    "regulatory": "#9b59b6",
    "macro": "#3498db",
    "hack": "#e74c3c",
    "protocol": "#f39c12",
    "etf": "#2ecc71",
    "halving": "#e67e22",
}

CATEGORY_STYLES = {
    "regulatory": "--",
    "macro": ":",
    "hack": "-",
    "protocol": "-.",
    "etf": "-",
    "halving": "-",
}


def filter_events(
    start: pd.Timestamp, end: pd.Timestamp, categories: list | None = None
) -> pd.DataFrame:
    """Return events within date range, optionally filtered by category."""
    mask = (_events_df["Date"] >= start) & (_events_df["Date"] <= end)
    if categories:
        mask &= _events_df["Category"].isin(categories)
    return _events_df[mask].copy()


def annotate_chart(ax, events_df: pd.DataFrame, ypos: str = "top"):
    """
    Add vertical lines + rotated labels for events on a matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib Axes
    events_df : DataFrame from filter_events()
    ypos : 'top' or 'bottom' — where to place labels
    """
    ylim = ax.get_ylim()
    y_text = ylim[1] * 0.95 if ypos == "top" else ylim[0] * 1.05

    for _, row in events_df.iterrows():
        color = CATEGORY_COLORS.get(row["Category"], "#888888")
        style = CATEGORY_STYLES.get(row["Category"], "--")

        ax.axvline(
            row["Date"], color=color, linestyle=style,
            alpha=0.6, linewidth=1, zorder=1,
        )
        ax.text(
            row["Date"], y_text, f"  {row['Label']}",
            rotation=90, fontsize=7, color=color,
            va="top" if ypos == "top" else "bottom",
            alpha=0.8, zorder=2,
        )
