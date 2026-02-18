"""
wavelet.py — Wavelet coherence analysis for frequency-domain co-movement.

Uses continuous wavelet transform (Morlet) to decompose the BTC/ETH
relationship across time and frequency, revealing which periodicities
(e.g. 7-day, 30-day, 90-day cycles) drive co-movement.
"""

import numpy as np
import pandas as pd
import pywt


def _morlet_cwt(signal: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Compute CWT using complex Morlet wavelet."""
    # pywt uses 'cmor' wavelet: cmor{bandwidth}-{center_freq}
    coefficients, _ = pywt.cwt(signal, scales, "cmor1.5-1.0")
    return coefficients


def compute_wavelet_coherence(
    df: pd.DataFrame,
    min_period: int = 4,
    max_period: int = 128,
    n_scales: int = 64,
) -> dict:
    """
    Compute wavelet coherence between asset and benchmark returns.

    Parameters
    ----------
    df : DataFrame with Asset_Returns and Benchmark_Returns columns.
    min_period : minimum period in days to analyse.
    max_period : maximum period in days (capped at len/4).
    n_scales : number of frequency scales to compute.

    Returns
    -------
    dict with keys:
        coherence : 2D array (n_scales × n_time), values in [0, 1]
        phase : 2D array of phase differences (radians)
        periods : 1D array of periods in days
        times : DatetimeIndex
        power_asset : 2D wavelet power for asset
        power_bench : 2D wavelet power for benchmark
        freq_bands : dict summarising coherence in key frequency bands
    """
    x = df["Asset_Returns"].values
    y = df["Benchmark_Returns"].values
    n = len(x)

    # Cap max_period at data length / 4
    max_period = min(max_period, n // 4)

    # Define scales — logarithmically spaced periods
    periods = np.logspace(
        np.log10(min_period), np.log10(max_period), n_scales
    )
    # Morlet wavelet: scale ≈ period / (2π × center_freq)
    # For cmor1.5-1.0, center_freq = 1.0, so scale ≈ period / (2π)
    scales = periods / (2 * np.pi)

    # CWT for both signals
    Wx = _morlet_cwt(x, scales)
    Wy = _morlet_cwt(y, scales)

    # Cross-wavelet spectrum
    Wxy = Wx * np.conj(Wy)

    # Smoothing kernel for coherence (Gaussian along time axis)
    from scipy.ndimage import gaussian_filter1d
    smooth_sigma = 5  # smoothing bandwidth

    S_xx = gaussian_filter1d(np.abs(Wx) ** 2, sigma=smooth_sigma, axis=1)
    S_yy = gaussian_filter1d(np.abs(Wy) ** 2, sigma=smooth_sigma, axis=1)
    S_xy = gaussian_filter1d(Wxy, sigma=smooth_sigma, axis=1)

    # Wavelet coherence: |S(Wxy)|² / (S(|Wx|²) · S(|Wy|²))
    numerator = np.abs(S_xy) ** 2
    denominator = S_xx * S_yy
    # Avoid division by zero
    denominator = np.where(denominator < 1e-15, 1e-15, denominator)
    coherence = numerator / denominator
    coherence = np.clip(coherence, 0, 1)

    # Phase difference
    phase = np.angle(S_xy)

    # Cone of influence — edges are unreliable
    coi = np.minimum(np.arange(n), np.arange(n)[::-1])
    coi = coi.astype(float)

    # Frequency band summaries
    freq_bands = _frequency_band_summary(coherence, periods)

    return {
        "coherence": coherence,
        "phase": phase,
        "periods": periods,
        "times": df.index,
        "power_asset": np.abs(Wx) ** 2,
        "power_bench": np.abs(Wy) ** 2,
        "coi": coi,
        "freq_bands": freq_bands,
    }


def _frequency_band_summary(coherence: np.ndarray, periods: np.ndarray) -> list:
    """Summarise mean coherence in key frequency bands."""
    bands = [
        ("Short-term (4–7d)", 4, 7),
        ("Weekly (7–14d)", 7, 14),
        ("Bi-weekly (14–30d)", 14, 30),
        ("Monthly (30–60d)", 30, 60),
        ("Quarterly (60–120d)", 60, 120),
    ]

    results = []
    for name, lo, hi in bands:
        mask = (periods >= lo) & (periods <= hi)
        if mask.any():
            mean_coh = coherence[mask].mean()
            max_coh = coherence[mask].max()
            results.append({
                "Band": name,
                "Mean Coherence": round(float(mean_coh), 3),
                "Max Coherence": round(float(max_coh), 3),
                "Interpretation": _interpret_coherence(mean_coh),
            })

    return results


def _interpret_coherence(coh: float) -> str:
    """Human-readable coherence interpretation."""
    if coh > 0.8:
        return "Very strong co-movement"
    elif coh > 0.6:
        return "Strong co-movement"
    elif coh > 0.4:
        return "Moderate co-movement"
    elif coh > 0.2:
        return "Weak co-movement"
    else:
        return "Negligible co-movement"
