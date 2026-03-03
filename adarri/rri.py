"""RR interval (RRI) and absolute difference (adRRI) computation.

Mirrors the MATLAB pipeline:
    fcn_get_rr_interval.m -> spline interpolation -> abs(diff(RRI))
"""

import numpy as np
from scipy.interpolate import CubicSpline


def compute_rri(r_peaks, sampling_rate=240):
    """Compute RR intervals from R-peak locations.

    Equation (1) from paper: RRI_n = R_{n+1} - R_n

    Args:
        r_peaks: Array of R-peak sample indices.
        sampling_rate: Sampling rate in Hz.

    Returns:
        rri: RR intervals in seconds (length = len(r_peaks) - 1).
        times: Time of each RRI in seconds (at the first R-peak of each pair).
    """
    r_peaks = np.asarray(r_peaks, dtype=np.float64)
    times = r_peaks[:-1] / sampling_rate  # seconds
    rri = np.diff(r_peaks) / sampling_rate  # seconds
    return rri, times


def interpolate_rri(rri, times, step_s=0.01):
    """Spline-interpolate RRI onto a uniform time grid.

    Mirrors the MATLAB code: tt = t(1):0.01:t(end); yy = spline(t, rri, tt);

    Args:
        rri: RR intervals in milliseconds.
        times: Time points corresponding to each RRI.
        step_s: Interpolation step in seconds (default 0.01 = 10 ms).

    Returns:
        rri_interp: Interpolated RRI values in milliseconds.
        times_interp: Uniform time grid.
    """
    times = np.asarray(times, dtype=np.float64)
    rri = np.asarray(rri, dtype=np.float64)

    cs = CubicSpline(times, rri)
    times_interp = np.arange(times[0], times[-1], step_s)
    rri_interp = cs(times_interp)

    return rri_interp, times_interp


def compute_adrri(rri):
    """Compute absolute difference of adjacent RR intervals (adRRI).

    Equation (2) from paper: adRRI_n = |RRI_n - RRI_{n+1}|

    Args:
        rri: Array of RR intervals (any units).

    Returns:
        adrri: Array of adRRI values (length = len(rri) - 1).
    """
    return np.abs(np.diff(rri))


def process_epoch(ecg_signal, sampling_rate=240, step_ms=10):
    """Full pipeline: ECG -> R-peaks -> RRI -> interpolated RRI -> adRRI.

    This mirrors fcnRealSampleData.m: for each epoch, detect R-peaks,
    compute RRI, spline-interpolate at 10ms steps, then abs(diff).

    Args:
        ecg_signal: 1D ECG signal array.
        sampling_rate: Hz.
        step_ms: Interpolation step in milliseconds.

    Returns:
        dict with keys:
            'rri_ms': interpolated RRI in ms
            'adrri_ms': abs(diff(rri)) in ms
            'r_peaks': detected R-peak indices
            'rri_raw': raw (non-interpolated) RRI in seconds
            'times_raw': time points for raw RRI in seconds
    """
    from .peak_detection import detect_r_peaks

    r_peaks = detect_r_peaks(ecg_signal, sampling_rate)

    if len(r_peaks) < 3:
        return None

    rri, times = compute_rri(r_peaks, sampling_rate)

    # Convert to ms for interpolation (matching MATLAB: rri=interval*1000)
    rri_ms = rri * 1000.0
    times_s = times  # keep in seconds for interpolation

    # Spline interpolation at step_ms intervals
    step_s = step_ms / 1000.0
    rri_interp, times_interp = interpolate_rri(rri_ms, times_s, step_s)

    # Absolute difference of interpolated RRI
    adrri = compute_adrri(rri_interp)

    return {
        "rri_ms": rri_interp,
        "adrri_ms": adrri,
        "r_peaks": r_peaks,
        "rri_raw": rri,
        "times_raw": times,
    }
