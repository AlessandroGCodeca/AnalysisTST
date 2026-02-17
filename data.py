"""
data.py — Robust data fetching with retries, caching, and multiple benchmarks.
"""

import os
import time
import hashlib
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

# ── HTTP Session with retry logic ──────────────────────────────────────────

def _get_session() -> requests.Session:
    """Create a requests session with automatic retries and back-off."""
    session = requests.Session()
    retries = Retry(
        total=4,
        backoff_factor=1,          # 1s, 2s, 4s, 8s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"Accept": "application/json"})
    return session


# ── Caching helpers ────────────────────────────────────────────────────────

def _cache_path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{safe}.csv")


def _cache_is_fresh(path: str, max_age_hours: float = 12) -> bool:
    if not os.path.exists(path):
        return False
    age = time.time() - os.path.getmtime(path)
    return age < max_age_hours * 3600


# ── CoinGecko fetchers ────────────────────────────────────────────────────

def fetch_coin_data(coin_id: str, vs_currency: str = "usd",
                    days: int = 365) -> pd.DataFrame:
    """
    Fetch historical price data for a coin from CoinGecko.
    Returns a DataFrame with DatetimeIndex and a 'Price' column.
    """
    cache_key = f"coin_{coin_id}_{vs_currency}_{days}"
    cp = _cache_path(cache_key)

    if _cache_is_fresh(cp):
        return pd.read_csv(cp, index_col="Date", parse_dates=True)

    url = (
        f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        f"/market_chart?vs_currency={vs_currency}&days={days}"
    )
    session = _get_session()
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    prices = resp.json()["prices"]

    df = pd.DataFrame(prices, columns=["Date", "Price"])
    df["Date"] = pd.to_datetime(df["Date"], unit="ms")
    df.set_index("Date", inplace=True)

    df.to_csv(cp)
    return df


def fetch_total_market_cap(vs_currency: str = "usd",
                           days: int = 365) -> pd.DataFrame:
    """
    Fetch total crypto market-cap from CoinGecko /global endpoint.
    Returns DataFrame with DatetimeIndex and 'Price' column (market cap).
    """
    cache_key = f"total_mcap_{vs_currency}_{days}"
    cp = _cache_path(cache_key)

    if _cache_is_fresh(cp):
        return pd.read_csv(cp, index_col="Date", parse_dates=True)

    url = (
        f"https://api.coingecko.com/api/v3/coins/bitcoin"
        f"/market_chart?vs_currency={vs_currency}&days={days}"
    )
    # CoinGecko free tier doesn't expose total-market-cap time-series
    # directly, so we approximate with a multi-coin weighted approach.
    # For a cleaner proxy we fetch the top coins and sum.
    coins = ["bitcoin", "ethereum", "tether", "ripple", "solana"]
    frames = []
    session = _get_session()

    for coin in coins:
        coin_url = (
            f"https://api.coingecko.com/api/v3/coins/{coin}"
            f"/market_chart?vs_currency={vs_currency}&days={days}"
        )
        try:
            r = session.get(coin_url, timeout=30)
            r.raise_for_status()
            data = r.json()["market_caps"]
            cdf = pd.DataFrame(data, columns=["Date", coin])
            cdf["Date"] = pd.to_datetime(cdf["Date"], unit="ms")
            cdf.set_index("Date", inplace=True)
            frames.append(cdf)
            time.sleep(1.5)  # rate-limit courtesy
        except Exception as e:
            print(f"⚠ Could not fetch {coin} market cap: {e}")

    if not frames:
        raise RuntimeError("Failed to fetch any market-cap data.")

    combined = pd.concat(frames, axis=1, sort=True)
    combined.dropna(inplace=True)
    combined["Price"] = combined.sum(axis=1)
    result = combined[["Price"]]

    result.to_csv(cp)
    return result


def fetch_sp500(days: int = 365) -> pd.DataFrame:
    """
    Fetch S&P 500 (^GSPC) data via yfinance.
    Returns DataFrame with DatetimeIndex and 'Price' column.
    """
    import yfinance as yf

    cache_key = f"sp500_{days}"
    cp = _cache_path(cache_key)

    if _cache_is_fresh(cp):
        return pd.read_csv(cp, index_col="Date", parse_dates=True)

    period_map = {30: "1mo", 90: "3mo", 180: "6mo", 365: "1y", 730: "2y"}
    period = period_map.get(days, "1y")

    ticker = yf.Ticker("^GSPC")
    hist = ticker.history(period=period)
    df = hist[["Close"]].rename(columns={"Close": "Price"})
    df.index.name = "Date"
    # Remove timezone info for consistency
    df.index = df.index.tz_localize(None)

    df.to_csv(cp)
    return df


# ── High-level builder ─────────────────────────────────────────────────────

BENCHMARK_MAP = {
    "ethereum":   lambda days: fetch_coin_data("ethereum", days=days),
    "solana":     lambda days: fetch_coin_data("solana", days=days),
    "total_mcap": lambda days: fetch_total_market_cap(days=days),
    "sp500":      lambda days: fetch_sp500(days=days),
}

ASSET_CHOICES = [
    "bitcoin", "ethereum", "solana", "cardano", "ripple",
    "dogecoin", "polkadot", "avalanche-2", "chainlink", "litecoin",
]

BENCHMARK_CHOICES = list(BENCHMARK_MAP.keys())


def build_merged_df(
    asset_id: str = "bitcoin",
    benchmark_id: str = "ethereum",
    days: int = 365,
    freq: str = "daily",
) -> pd.DataFrame:
    """
    Fetch asset + benchmark, merge, and compute returns.

    Parameters
    ----------
    asset_id : str      CoinGecko coin id for the asset.
    benchmark_id : str   Key from BENCHMARK_MAP.
    days : int           Look-back window.
    freq : str           'daily' or 'weekly'.

    Returns
    -------
    pd.DataFrame with columns:
        Asset_Price, Benchmark_Price, Asset_Returns, Benchmark_Returns
    """
    # Fetch asset
    asset_df = fetch_coin_data(asset_id, days=days)
    asset_df.rename(columns={"Price": "Asset_Price"}, inplace=True)

    # Fetch benchmark
    if benchmark_id in BENCHMARK_MAP:
        bench_df = BENCHMARK_MAP[benchmark_id](days)
    else:
        bench_df = fetch_coin_data(benchmark_id, days=days)
    bench_df.rename(columns={"Price": "Benchmark_Price"}, inplace=True)

    # Merge on date (normalize to daily)
    asset_df.index = asset_df.index.normalize()
    bench_df.index = bench_df.index.normalize()

    # Drop duplicate dates (keep first)
    asset_df = asset_df[~asset_df.index.duplicated(keep="first")]
    bench_df = bench_df[~bench_df.index.duplicated(keep="first")]

    df = pd.concat([asset_df, bench_df], axis=1, sort=True)
    df.dropna(inplace=True)

    # Resample to weekly if requested
    if freq == "weekly":
        df = df.resample("W").last()
        df.dropna(inplace=True)

    # Compute returns
    df["Asset_Returns"] = df["Asset_Price"].pct_change()
    df["Benchmark_Returns"] = df["Benchmark_Price"].pct_change()
    df.dropna(inplace=True)

    return df
