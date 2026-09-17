"""
emg_detector.py
---------------
Pure signal processing module for EMG detection.

Responsibility: given a preprocessed ECG window as a numpy array,
compute the spectral power ratio between the EMG band (20-40 Hz)
and the QRS band (5-15 Hz).

This module has NO knowledge of CACHET-CADB or any file system.
It receives numpy arrays and returns floats. All data loading,
threshold calibration, and window flagging happens in dataset.py
which knows how to read the dataset.

The only function intended for external use is:
    compute_emg_ratio(ecg_window) -> float
"""

import numpy as np
from scipy.signal import welch

from config import (
    ECG_FS       as FS,
    EMG_QRS_BAND as QRS_BAND,
    EMG_EMG_BAND as EMG_BAND,
)


def band_power(signal: np.ndarray, fs: int, fmin: float, fmax: float) -> float:
    """
    Compute mean power in a frequency band using Welch's method.

    Welch is preferred over a single FFT here because it reduces
    variance in the power estimate by averaging over overlapping
    sub-windows — important for short noisy segments.

    Args:
        signal:  1D array, single ECG channel
        fs:      sampling frequency in Hz
        fmin:    lower band edge in Hz
        fmax:    upper band edge in Hz

    Returns:
        Mean power spectral density in the band (uV^2/Hz)
    """
    # nperseg = 2s sub-windows → 0.5 Hz resolution, enough for our bands
    nperseg = min(2 * fs, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not band_mask.any():
        return 0.0

    # Integrate PSD over band using trapezoidal rule
    return float(np.trapezoid(psd[band_mask], freqs[band_mask]))


def compute_emg_ratio(
    ecg_window: np.ndarray,
    fs: int = FS,
    qrs_band: tuple = QRS_BAND,
    emg_band: tuple = EMG_BAND,
    epsilon: float = 1e-10,
) -> float:
    """
    Compute the EMG-to-QRS spectral power ratio for a single 10s window.

    A ratio >> 1 means more energy in the EMG band than the QRS band,
    indicating likely muscle artifact contamination.

    Args:
        ecg_window:  1D array of length WINDOW_SAMPLES (preprocessed ECG)
        fs:          sampling frequency
        qrs_band:    (fmin, fmax) in Hz for QRS power measurement
        emg_band:    (fmin, fmax) in Hz for EMG power measurement
        epsilon:     small constant to avoid division by zero

    Returns:
        EMG ratio (float >= 0). Values > threshold indicate EMG.
    """
    assert ecg_window.ndim == 1, "Expected a single-channel 1D ECG window"

    p_qrs = band_power(ecg_window, fs, *qrs_band)
    p_emg = band_power(ecg_window, fs, *emg_band)

    return p_emg / (p_qrs + epsilon)



# ── smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(42)
    t   = np.linspace(0, 10, 10 * FS)

    clean = (
        0.5 * np.sin(2 * np.pi * 1.2 * t)
        + 0.3 * np.sin(2 * np.pi * 8.0 * t)
        + 0.02 * rng.standard_normal(len(t))
    )
    noisy = clean + 0.8 * rng.standard_normal(len(t)) + 0.6 * np.sin(2 * np.pi * 28 * t)

    r_clean = compute_emg_ratio(clean)
    r_noisy = compute_emg_ratio(noisy)
    print(f"EMG ratio  clean : {r_clean:.6f}")
    print(f"EMG ratio  noisy : {r_noisy:.6f}")
    print(f"Separation       : {r_noisy / r_clean:.0f}x")
    print("compute_emg_ratio working correctly.")
