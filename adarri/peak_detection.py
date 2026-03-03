"""R-peak detection using Pan-Tompkins algorithm.

Wraps neurokit2's implementation of the Pan-Tompkins (1985) algorithm,
matching the original MATLAB fcn_pan_tompkin.m used in the paper.

ECG recordings in the study were sampled at 240 Hz.
"""

import numpy as np
import neurokit2 as nk


SAMPLING_RATE = 240  # Hz, per Section 2.2 of the paper


def detect_r_peaks(ecg_signal, sampling_rate=SAMPLING_RATE):
    """Detect R-peaks in an ECG signal using Pan-Tompkins algorithm.

    Args:
        ecg_signal: 1D array of ECG amplitude values.
        sampling_rate: Sampling frequency in Hz (default 240).

    Returns:
        r_peaks: 1D integer array of sample indices where R-peaks are detected.
    """
    ecg_signal = np.asarray(ecg_signal, dtype=np.float64)

    # Clean the signal
    ecg_cleaned = nk.ecg_clean(
        ecg_signal, sampling_rate=sampling_rate, method="pantompkins1985"
    )

    # Detect peaks
    _, info = nk.ecg_peaks(
        ecg_cleaned, sampling_rate=sampling_rate, method="pantompkins1985"
    )

    r_peaks = np.array(info["ECG_R_Peaks"], dtype=int)
    return r_peaks
