"""Clifford's artifact detection method (Method C).

Implements the percentage-based threshold from:
    Clifford GD, McSharry PE, Tarassenko L. Characterizing artefact in
    the normal human 24-hour RR time series to aid identification and
    artificial replication of circadian variations in human beat to beat
    heart rate using a simple threshold. Computers in Cardiology. 2002.

Faithfully ported from fcn_clean_hrv4.m.
"""

import numpy as np


def clean_hrv(times, rri, pc=80):
    """Clean HRV data by removing artifact beats using Clifford's method.

    Removes any RR intervals that differ by more than (100-pc)% from the
    previous valid interval. Default pc=80 means removal of beats with
    >20% change. For consecutive removed beats, the threshold is relaxed
    by 3% per beat removed.

    Ported from fcn_clean_hrv4.m.

    Args:
        times: Array of R-peak times (in ms).
        rri: Array of RR intervals (in ms).
        pc: Percentage tolerance (default 80 = 20% deviation allowed).

    Returns:
        clean_times: Times of kept beats.
        clean_rri: RR intervals of kept beats.
        artifact_mask: Boolean array (True = artifact/removed).
    """
    times = np.asarray(times, dtype=np.float64)
    rri = np.asarray(rri, dtype=np.float64)
    n = len(rri)

    if n < 2:
        return times.copy(), rri.copy(), np.zeros(n, dtype=bool)

    # Find first valid sample (close to mean and close to next sample)
    mean_rri = np.mean(rri)
    j = 0
    while j < n - 1:
        if abs(rri[j] - mean_rri) <= 10 or abs(rri[j] - rri[j + 1]) <= 10:
            break
        j += 1

    if j >= n - 1:
        # No valid sample found
        return np.array([]), np.array([]), np.ones(n, dtype=bool)

    # Initialize with first valid beat
    clean_times_list = [times[j]]
    clean_rri_list = [rri[j]]
    kept_indices = [j]
    consecutive_removed = 0

    # Mirror last sample for boundary handling
    rri_ext = np.append(rri, rri[-2] if n > 1 else rri[-1])

    for i in range(1, n - j):
        idx = j + i
        if idx >= n:
            break

        # Current and reference heart rates (in bpm)
        current_hr = 60000.0 / rri[idx] if rri[idx] > 0 else 0
        last_valid_hr = 60000.0 / clean_rri_list[-1] if clean_rri_list[-1] > 0 else 0

        # Next beat HR for look-ahead
        next_idx = min(idx + 1, len(rri_ext) - 1)
        next_hr = 60000.0 / rri_ext[next_idx] if rri_ext[next_idx] > 0 else 0

        # Adaptive threshold: relax by 3% per consecutive removed beat
        effective_pc = pc - consecutive_removed * 3
        tolerance = (1.0 - effective_pc / 100.0) * last_valid_hr

        # Keep if change from last valid OR change to next is within tolerance
        if abs(current_hr - last_valid_hr) < tolerance or abs(current_hr - next_hr) < tolerance:
            clean_times_list.append(times[idx])
            clean_rri_list.append(rri[idx])
            kept_indices.append(idx)
            consecutive_removed = 0
        else:
            consecutive_removed += 1

    # Build artifact mask
    artifact_mask = np.ones(n, dtype=bool)
    for idx in kept_indices:
        artifact_mask[idx] = False

    return (
        np.array(clean_times_list),
        np.array(clean_rri_list),
        artifact_mask,
    )


def detect_artifacts_clifford(times, rri, pc=80):
    """Detect artifacts using Clifford's method (Method C).

    Args:
        times: Array of R-peak times (in ms).
        rri: Array of RR intervals (in ms).
        pc: Percentage tolerance (default 80).

    Returns:
        flags: Boolean array of length len(rri). True = artifact.
    """
    _, _, artifact_mask = clean_hrv(times, rri, pc)
    return artifact_mask


def classify_epochs_clifford(rri_epochs, times_epochs):
    """Classify each epoch using Clifford's method.

    An epoch is classified as artifact if any beat within it is flagged.

    Args:
        rri_epochs: List of RRI arrays (one per epoch, in ms).
        times_epochs: List of time arrays (one per epoch, in ms).

    Returns:
        predictions: Array of predicted labels (0=valid, 1=artifact).
    """
    predictions = np.zeros(len(rri_epochs), dtype=int)
    for j, (rri, times) in enumerate(zip(rri_epochs, times_epochs)):
        flags = detect_artifacts_clifford(times, rri)
        if np.any(flags):
            predictions[j] = 1
    return predictions
