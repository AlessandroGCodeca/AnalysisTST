"""
fear_greed.py — Crypto Fear & Greed Index fetcher.

Fetches from alternative.me public API and returns a time-series
suitable for chart overlays.
"""

import pandas as pd
import requests
import os
import hashlib

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def fetch_fear_greed(days: int = 365) -> pd.DataFrame | None:
    """
    Fetch the Crypto Fear & Greed Index.

    Returns DataFrame with columns: Date (index), Value (0-100), Classification.
    Labels: Extreme Fear (0-24), Fear (25-49), Neutral (50), Greed (51-74), Extreme Greed (75-100).
    """
    cache_key = hashlib.md5(f"fg_{days}".encode()).hexdigest()[:10]
    cache_path = os.path.join(CACHE_DIR, f"fear_greed_{cache_key}.csv")
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Check cache (1h TTL)
    if os.path.exists(cache_path):
        import time
        if time.time() - os.path.getmtime(cache_path) < 3600:
            df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
            return df

    try:
        url = f"https://api.alternative.me/fng/?limit={days}&format=json"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json().get("data", [])

        if not data:
            return None

        rows = []
        for entry in data:
            rows.append({
                "Date": pd.to_datetime(int(entry["timestamp"]), unit="s"),
                "FG_Value": int(entry["value"]),
                "FG_Class": entry["value_classification"],
            })

        df = pd.DataFrame(rows)
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)

        df.to_csv(cache_path)
        return df

    except Exception:
        # Return cached data if available, even if expired
        if os.path.exists(cache_path):
            return pd.read_csv(cache_path, index_col="Date", parse_dates=True)
        return None


def merge_fear_greed(df: pd.DataFrame, fg: pd.DataFrame) -> pd.DataFrame:
    """
    Merge Fear & Greed data with the main analysis DataFrame.
    Aligns on date index using nearest join.
    """
    fg_daily = fg[["FG_Value", "FG_Class"]].copy()
    fg_daily.index = fg_daily.index.normalize()
    fg_daily = fg_daily[~fg_daily.index.duplicated(keep="first")]

    merged = df.join(fg_daily, how="left")
    merged["FG_Value"] = merged["FG_Value"].ffill().bfill()
    merged["FG_Class"] = merged["FG_Class"].ffill().bfill()
    return merged
