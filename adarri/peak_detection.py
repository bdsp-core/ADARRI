"""R-peak detection using Pan-Tompkins algorithm.

Faithful port of the MATLAB fcn_pan_tompkin.m used in the paper.

Implements the complete Pan-Tompkins (1985) algorithm including:
- Bandpass filtering (5-15 Hz)
- Derivative filtering
- Squaring and moving window integration
- Adaptive thresholding with search-back for missed QRS
- T-wave discrimination
- Refractory period enforcement (200ms minimum between detections)
- Dual verification on bandpass filtered signal

ECG recordings in the study were sampled at 240 Hz.
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


SAMPLING_RATE = 240  # Hz, per Section 2.2 of the paper


def detect_r_peaks(ecg_signal, sampling_rate=SAMPLING_RATE):
    """Detect R-peaks in an ECG signal using Pan-Tompkins algorithm.

    Faithful port of the MATLAB fcn_pan_tompkin.m. For non-200 Hz signals,
    uses Butterworth bandpass filtering (5-15 Hz, order 3) with zero-phase
    filtering (filtfilt).

    Args:
        ecg_signal: 1D array of ECG amplitude values.
        sampling_rate: Sampling frequency in Hz (default 240).

    Returns:
        r_peaks: 1D integer array of sample indices where R-peaks are detected.
    """
    ecg = np.asarray(ecg_signal, dtype=np.float64).ravel()
    fs = sampling_rate

    if len(ecg) < fs:
        return np.array([], dtype=int)

    # ===== Stage 1: Bandpass filtering (5-15 Hz) =====
    if fs == 200:
        # Original Pan-Tompkins integer-coefficient filters for 200 Hz
        from scipy.signal import lfilter

        # Low pass: H(z) = ((1 - z^(-6))^2) / (1 - z^(-1))^2
        b_lp = np.array([1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1],
                        dtype=np.float64)
        a_lp = np.array([1, -2, 1], dtype=np.float64)
        h_l = lfilter(b_lp, a_lp, np.concatenate([[1], np.zeros(12)]))
        ecg_l = np.convolve(ecg, h_l)
        max_val = np.max(np.abs(ecg_l))
        if max_val > 0:
            ecg_l = ecg_l / max_val

        # High pass: H(z) = (-1 + 32z^(-16) + z^(-32)) / (1 + z^(-1))
        b_hp = np.zeros(33, dtype=np.float64)
        b_hp[0] = -1
        b_hp[16] = 32
        b_hp[17] = -32
        b_hp[32] = 1
        a_hp = np.array([1, -1], dtype=np.float64)
        h_h = lfilter(b_hp, a_hp, np.concatenate([[1], np.zeros(32)]))
        ecg_h = np.convolve(ecg_l, h_h)
        max_val = np.max(np.abs(ecg_h))
        if max_val > 0:
            ecg_h = ecg_h / max_val
    else:
        # Butterworth bandpass for other sampling frequencies
        f1, f2 = 5.0, 15.0
        Wn = np.array([f1, f2]) * 2 / fs
        N = 3
        b, a = butter(N, Wn, btype='band')
        ecg_h = filtfilt(b, a, ecg)
        max_val = np.max(np.abs(ecg_h))
        if max_val > 0:
            ecg_h = ecg_h / max_val

    # ===== Stage 2: Derivative filter =====
    # H(z) = (1/8T)(-z^(-2) - 2z^(-1) + 2z + z^(2))
    h_d = np.array([-1, -2, 0, 2, 1]) * (1.0 / 8)
    ecg_d = np.convolve(ecg_h, h_d)
    max_val = np.max(ecg_d)
    if max_val > 0:
        ecg_d = ecg_d / max_val

    # ===== Stage 3: Squaring =====
    ecg_s = ecg_d ** 2

    # ===== Stage 4: Moving window integration (150ms window) =====
    win_len = round(0.150 * fs)
    ecg_m = np.convolve(ecg_s, np.ones(win_len) / win_len)

    # ===== Stage 5: Find peaks with minimum distance 200ms =====
    min_dist = round(0.2 * fs)
    peak_indices, _ = find_peaks(ecg_m, distance=min_dist)
    pks = ecg_m[peak_indices]
    locs = peak_indices

    if len(locs) == 0:
        return np.array([], dtype=int)

    # ===== Stage 6: Initialize thresholds (2-second training phase) =====
    train_end = min(2 * fs, len(ecg_m))

    # MWI signal thresholds
    THR_SIG = np.max(ecg_m[:train_end]) * (1.0 / 3)
    THR_NOISE = np.mean(ecg_m[:train_end]) * 0.5
    SIG_LEV = THR_SIG
    NOISE_LEV = THR_NOISE

    # Bandpass filtered signal thresholds
    train_end_h = min(2 * fs, len(ecg_h))
    THR_SIG1 = np.max(ecg_h[:train_end_h]) * (1.0 / 3)
    THR_NOISE1 = np.mean(ecg_h[:train_end_h]) * 0.5
    SIG_LEV1 = THR_SIG1
    NOISE_LEV1 = THR_NOISE1

    # Output arrays
    qrs_c = []       # QRS amplitudes in MWI signal
    qrs_i = []       # QRS indices in MWI signal
    qrs_i_raw = []   # QRS indices in bandpass filtered signal (final output)
    qrs_amp_raw = []  # QRS amplitudes in bandpass filtered signal

    # State variables
    m_selected_RR = 0
    mean_RR = 0
    skip = 0
    not_nois = 0
    ser_back = 0
    test_m = 0

    win = round(0.150 * fs)  # lookback window for bandpass peak location

    # ===== Stage 7: Adaptive thresholding loop =====
    for i in range(len(pks)):

        # --- (a) Locate corresponding peak in bandpass filtered signal ---
        y_i = 0.0
        x_i = 0

        if locs[i] - win >= 0 and locs[i] < len(ecg_h):
            seg = ecg_h[locs[i] - win:locs[i] + 1]
            x_i = int(np.argmax(seg))
            y_i = seg[x_i]
        else:
            if i == 0:
                seg = ecg_h[:locs[i] + 1]
                x_i = int(np.argmax(seg))
                y_i = seg[x_i]
                ser_back = 1
            elif locs[i] >= len(ecg_h):
                start_idx = max(0, locs[i] - win)
                seg = ecg_h[start_idx:]
                x_i = int(np.argmax(seg))
                y_i = seg[x_i]

        # --- (b) Update heart rate estimates ---
        if len(qrs_c) >= 9:
            diffRR = np.diff(np.array(qrs_i[-9:]))
            mean_RR = np.mean(diffRR)
            comp = qrs_i[-1] - qrs_i[-2]

            if comp <= 0.92 * mean_RR or comp >= 1.16 * mean_RR:
                # Irregular rhythm — lower thresholds to detect better
                THR_SIG = 0.5 * THR_SIG
                THR_SIG1 = 0.5 * THR_SIG1
            else:
                m_selected_RR = mean_RR  # latest regular beats mean

        # Determine which RR mean to use for searchback
        if m_selected_RR:
            test_m = m_selected_RR
        elif mean_RR and m_selected_RR == 0:
            test_m = mean_RR
        else:
            test_m = 0

        # --- (c) Searchback for missed QRS complexes ---
        if test_m and len(qrs_i) > 0:
            if (locs[i] - qrs_i[-1]) >= round(1.66 * test_m):
                # QRS possibly missed — search back
                sb_start = qrs_i[-1] + round(0.200 * fs)
                sb_end = locs[i] - round(0.200 * fs)

                if sb_start < sb_end and sb_start >= 0:
                    sb_end_clip = min(sb_end + 1, len(ecg_m))
                    sb_seg = ecg_m[sb_start:sb_end_clip]

                    if len(sb_seg) > 0:
                        pks_temp_idx = int(np.argmax(sb_seg))
                        pks_temp = sb_seg[pks_temp_idx]
                        locs_temp = sb_start + pks_temp_idx

                        if pks_temp > THR_NOISE:
                            qrs_c.append(pks_temp)
                            qrs_i.append(locs_temp)

                            # Find location in bandpass filtered signal
                            bp_start = max(0, locs_temp - win)
                            if locs_temp < len(ecg_h):
                                bp_seg = ecg_h[bp_start:locs_temp + 1]
                            else:
                                bp_seg = ecg_h[bp_start:]

                            if len(bp_seg) > 0:
                                x_i_t = int(np.argmax(bp_seg))
                                y_i_t = bp_seg[x_i_t]

                                if y_i_t > THR_NOISE1:
                                    qrs_i_raw.append(bp_start + x_i_t)
                                    qrs_amp_raw.append(y_i_t)
                                    SIG_LEV1 = 0.25 * y_i_t + 0.75 * SIG_LEV1

                            not_nois = 1
                            SIG_LEV = 0.25 * pks_temp + 0.75 * SIG_LEV
            else:
                not_nois = 0

        # --- (d,e) QRS detection with T-wave discrimination ---
        if pks[i] >= THR_SIG:
            # Check for T-wave: if within 360ms of previous QRS
            if len(qrs_c) >= 3:
                if (locs[i] - qrs_i[-1]) <= round(0.360 * fs):
                    slope_win = round(0.075 * fs)

                    # Mean slope of current candidate
                    s1_start = max(0, locs[i] - slope_win)
                    if s1_start < locs[i] and locs[i] < len(ecg_m):
                        Slope1 = np.mean(np.diff(
                            ecg_m[s1_start:locs[i] + 1]))
                    else:
                        Slope1 = 0

                    # Mean slope of previous QRS
                    s2_start = max(0, qrs_i[-1] - slope_win)
                    s2_end = min(qrs_i[-1] + 1, len(ecg_m))
                    if s2_start < qrs_i[-1] and s2_end > s2_start + 1:
                        Slope2 = np.mean(np.diff(
                            ecg_m[s2_start:s2_end]))
                    else:
                        Slope2 = 0

                    if abs(Slope1) <= abs(0.5 * Slope2):
                        # T wave detected — classify as noise
                        skip = 1
                        NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
                        NOISE_LEV = 0.125 * pks[i] + 0.875 * NOISE_LEV
                    else:
                        skip = 0

            if skip == 0:
                # QRS complex confirmed
                qrs_c.append(pks[i])
                qrs_i.append(locs[i])

                # (f) Verify on bandpass filtered signal
                if y_i >= THR_SIG1:
                    if ser_back:
                        qrs_i_raw.append(x_i)
                    else:
                        qrs_i_raw.append(locs[i] - win + x_i)
                    qrs_amp_raw.append(y_i)
                    SIG_LEV1 = 0.125 * y_i + 0.875 * SIG_LEV1

                # Update signal level
                SIG_LEV = 0.125 * pks[i] + 0.875 * SIG_LEV

        elif THR_NOISE <= pks[i] < THR_SIG:
            # Between noise and signal thresholds — update noise levels
            NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
            NOISE_LEV = 0.125 * pks[i] + 0.875 * NOISE_LEV

        else:
            # Below noise threshold
            NOISE_LEV1 = 0.125 * y_i + 0.875 * NOISE_LEV1
            NOISE_LEV = 0.125 * pks[i] + 0.875 * NOISE_LEV

        # --- Adjust thresholds based on updated signal/noise levels ---
        if NOISE_LEV != 0 or SIG_LEV != 0:
            THR_SIG = NOISE_LEV + 0.25 * abs(SIG_LEV - NOISE_LEV)
            THR_NOISE = 0.5 * THR_SIG

        if NOISE_LEV1 != 0 or SIG_LEV1 != 0:
            THR_SIG1 = NOISE_LEV1 + 0.25 * abs(SIG_LEV1 - NOISE_LEV1)
            THR_NOISE1 = 0.5 * THR_SIG1

        # Reset per-iteration state
        skip = 0
        not_nois = 0
        ser_back = 0

    # Sort output (searchback can insert out-of-order)
    r_peaks = np.sort(np.array(qrs_i_raw, dtype=int))

    # Ensure indices are within bounds of original signal
    r_peaks = r_peaks[(r_peaks >= 0) & (r_peaks < len(ecg))]

    return r_peaks
