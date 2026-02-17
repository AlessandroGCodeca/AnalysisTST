"""
regime.py — Hidden Markov Model regime detection for crypto returns.

Classifies each day into Bull / Sideways / Bear regimes using a Gaussian HMM,
then computes per-regime beta statistics.
"""

import numpy as np
import pandas as pd
import scipy.stats as sp_stats


REGIME_LABELS = {0: "Bear", 1: "Sideways", 2: "Bull"}
REGIME_COLORS = {"Bear": "#e74c3c", "Sideways": "#95a5a6", "Bull": "#2ecc71"}


def detect_regimes(
    returns: pd.Series, n_states: int = 3
) -> pd.Series | None:
    """
    Fit a Gaussian HMM to returns and assign regime labels.

    Returns a Series of regime labels aligned to the returns index,
    or None if hmmlearn is unavailable.
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        print("⚠ `hmmlearn` not installed. Regime detection unavailable.")
        return None

    try:
        X = returns.dropna().values.reshape(-1, 1)

        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        model.fit(X)
        hidden_states = model.predict(X)

        # Sort states by mean return so 0=Bear, 2=Bull
        means = model.means_.flatten()
        order = np.argsort(means)
        state_map = {old: new for new, old in enumerate(order)}
        mapped = np.array([state_map[s] for s in hidden_states])

        labels = pd.Series(
            [REGIME_LABELS[s] for s in mapped],
            index=returns.dropna().index,
            name="Regime",
        )
        return labels

    except Exception as e:
        print(f"⚠ Regime detection failed: {e}")
        return None


def regime_beta(df: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    """
    Compute OLS beta separately for each regime.

    Returns DataFrame with columns: Regime, Beta, Alpha, R², N_obs, Sharpe.
    """
    from analytics import sharpe_ratio

    results = []
    for regime in REGIME_LABELS.values():
        mask = regimes == regime
        sub = df.loc[mask.index[mask]]
        if len(sub) < 10:
            results.append({
                "Regime": regime, "Beta": None, "Alpha": None,
                "R²": None, "N_obs": len(sub), "Sharpe": None,
            })
            continue

        slope, intercept, r_val, p_val, se = sp_stats.linregress(
            sub["Benchmark_Returns"], sub["Asset_Returns"]
        )
        results.append({
            "Regime": regime,
            "Beta": round(slope, 4),
            "Alpha": round(intercept, 6),
            "R²": round(r_val ** 2, 4),
            "N_obs": len(sub),
            "Sharpe": round(sharpe_ratio(sub["Asset_Returns"]), 2),
        })

    return pd.DataFrame(results)
